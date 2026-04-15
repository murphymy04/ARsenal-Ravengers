"""Networking primitives for the custom INMO Air 3 stream protocol.

Ported from streamtest/serveraudioplaybuffer.py. VideoReceiver is refactored
to keep an ordered ring of recent frames so the pairing loop can consume
frames in sequence rather than only seeing the latest one.

Excluded from this port: JitterBuffer, AudioPlayer (playback-only),
CommandServer (handled via a separate websocket), LaptopServer.
"""

import json
import socket
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np
from pipeline.config import HUD_BROADCAST_PORT, HUD_BROADCAST_ENABLED

AUDIO_PORT = 5003
DISCOVERY_PORT = 5002
VIDEO_PORT = 5000
COMMAND_PORT = 5001


@dataclass
class FrameData:
    seq: int
    timestamp: int
    width: int
    height: int
    data: np.ndarray
    received_at: float


@dataclass
class AudioChunk:
    seq: int
    timestamp_start: int
    timestamp_end: int
    sample_rate: int
    num_samples: int
    data: np.ndarray
    received_at: float


@dataclass
class GlassesConnection:
    ip: str
    tcp_socket: Optional[socket.socket] = None
    connected: bool = False


class DiscoveryService:
    def __init__(self, on_glasses_found: Callable[[str], None]):
        self.on_glasses_found = on_glasses_found
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.sock.bind(("0.0.0.0", DISCOVERY_PORT))
        self.sock.settimeout(1.0)
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.discovered_ips: set[str] = set()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.thread.start()
        print(f"[Discovery] Listening for glasses on port {DISCOVERY_PORT}")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        self.sock.close()

    def _listen_loop(self):
        while self.running:
            try:
                data, addr = self.sock.recvfrom(1024)
                self._handle_discovery(data, addr)
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"[Discovery] Error: {e}")

    def _handle_discovery(self, data: bytes, addr: tuple):
        try:
            msg = json.loads(data.decode("utf-8"))
        except json.JSONDecodeError:
            print(f"[Discovery] Invalid JSON from {addr}")
            return

        if msg.get("type") != "discover":
            return

        glasses_ip = addr[0]
        glasses_name = msg.get("name", "unknown")
        print(f"[Discovery] Found glasses '{glasses_name}' at {glasses_ip}")

        response = json.dumps(
            {
                "type": "discover_response",
                "video_port": VIDEO_PORT,
                "command_port": COMMAND_PORT,
                "audio_port": AUDIO_PORT,
                "hud_broadcast_enabled": HUD_BROADCAST_ENABLED,
                "hud_broadcast_port": HUD_BROADCAST_PORT,
            }
        ).encode("utf-8")
        self.sock.sendto(response, addr)

        if glasses_ip not in self.discovered_ips:
            self.discovered_ips.add(glasses_ip)
            self.on_glasses_found(glasses_ip)


class AudioReceiver:
    HEADER_SIZE = 28

    def __init__(self, max_chunks: int = 1000):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("0.0.0.0", AUDIO_PORT))
        self.sock.listen(1)
        self.sock.settimeout(1.0)

        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.client_socket: Optional[socket.socket] = None

        self._chunks: list[AudioChunk] = []
        self._chunks_lock = threading.Lock()
        self._max_chunks = max_chunks

        self.chunks_received = 0
        self.start_time: Optional[float] = None

        self.on_connect_callback: Optional[Callable[[], None]] = None

    def start(self):
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._accept_loop, daemon=True)
        self.thread.start()
        print(f"[Audio] TCP server listening on port {AUDIO_PORT}")

    def stop(self):
        self.running = False
        if self.client_socket:
            try:
                self.client_socket.close()
            except Exception:
                pass
        if self.thread:
            self.thread.join(timeout=2)
        self.sock.close()

    def get_audio_range(self, ts_start: int, ts_end: int) -> list[AudioChunk]:
        with self._chunks_lock:
            return [
                chunk
                for chunk in self._chunks
                if chunk.timestamp_end >= ts_start and chunk.timestamp_start <= ts_end
            ]

    def get_latest_audio_timestamp(self) -> Optional[int]:
        with self._chunks_lock:
            if not self._chunks:
                return None
            return self._chunks[-1].timestamp_end

    def get_next_chunk(self, after_seq: int) -> Optional[AudioChunk]:
        with self._chunks_lock:
            for chunk in self._chunks:
                if chunk.seq > after_seq:
                    return chunk
        return None

    def get_stats(self) -> dict:
        elapsed = time.time() - self.start_time if self.start_time else 0
        return {
            "chunks_received": self.chunks_received,
            "chunks_buffered": len(self._chunks),
            "chunks_per_sec": self.chunks_received / elapsed if elapsed > 0 else 0,
            "connected": self.client_socket is not None,
        }

    def _accept_loop(self):
        while self.running:
            try:
                client, addr = self.sock.accept()
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"[Audio] Accept error: {e}")
                continue

            print(f"[Audio] Glasses connected from {addr}")
            self.client_socket = client
            self.client_socket.settimeout(1.0)

            with self._chunks_lock:
                self._chunks = []
            self.chunks_received = 0

            if self.on_connect_callback:
                self.on_connect_callback()

            self._receive_loop()

    def _receive_loop(self):
        while self.running:
            try:
                length_bytes = self._recv_exact(4)
                if not length_bytes:
                    break

                length = struct.unpack("<I", length_bytes)[0]
                if length > 1024 * 1024:
                    print(f"[Audio] Invalid length: {length}")
                    break

                packet = self._recv_exact(length)
                if not packet:
                    break

                self._handle_packet(packet)
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"[Audio] Receive error: {e}")
                break

        print("[Audio] Glasses disconnected")
        self.client_socket = None

    def _recv_exact(self, n: int) -> Optional[bytes]:
        data = b""
        while len(data) < n:
            try:
                chunk = self.client_socket.recv(n - len(data))
                if not chunk:
                    return None
                data += chunk
            except socket.timeout:
                if not self.running:
                    return None
                continue
        return data

    def _handle_packet(self, packet: bytes):
        if len(packet) < self.HEADER_SIZE:
            return

        seq = struct.unpack("<I", packet[0:4])[0]
        ts_start = struct.unpack("<Q", packet[4:12])[0]
        ts_end = struct.unpack("<Q", packet[12:20])[0]
        sample_rate = struct.unpack("<I", packet[20:24])[0]
        num_samples = struct.unpack("<I", packet[24:28])[0]

        audio_bytes = packet[self.HEADER_SIZE :]
        expected_bytes = num_samples * 2

        if len(audio_bytes) != expected_bytes:
            print(
                f"[Audio] Size mismatch: got {len(audio_bytes)}, expected {expected_bytes}"
            )
            return

        audio_data = np.frombuffer(audio_bytes, dtype=np.int16)

        chunk = AudioChunk(
            seq=seq,
            timestamp_start=ts_start,
            timestamp_end=ts_end,
            sample_rate=sample_rate,
            num_samples=num_samples,
            data=audio_data,
            received_at=time.time(),
        )

        with self._chunks_lock:
            self._chunks.append(chunk)
            if len(self._chunks) > self._max_chunks:
                self._chunks = self._chunks[-self._max_chunks :]

        self.chunks_received += 1


class VideoReceiver:
    HEADER_SIZE = 20

    def __init__(self, max_frames: int = 120):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        self.sock.bind(("0.0.0.0", VIDEO_PORT))
        self.sock.settimeout(1.0)

        self.running = False
        self.thread: Optional[threading.Thread] = None

        self._frames: deque[FrameData] = deque(maxlen=max_frames)
        self._frame_lock = threading.Lock()

        self._fragments: dict[int, dict[int, bytes]] = {}
        self._fragment_info: dict[int, tuple[int, int, int, int]] = {}
        self._fragment_lock = threading.Lock()

        self.frames_received = 0
        self.fragments_received = 0
        self.last_seq = -1
        self.frames_dropped = 0
        self.incomplete_dropped = 0
        self.start_time: Optional[float] = None

    def start(self):
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.thread.start()
        print(f"[Video] Listening on port {VIDEO_PORT}")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        self.sock.close()

    def get_frame_after(self, seq: int) -> Optional[FrameData]:
        with self._frame_lock:
            for frame in self._frames:
                if frame.seq > seq:
                    return frame
        return None

    def get_stats(self) -> dict:
        elapsed = time.time() - self.start_time if self.start_time else 0
        return {
            "frames_received": self.frames_received,
            "frames_dropped": self.frames_dropped,
            "incomplete_dropped": self.incomplete_dropped,
            "frames_buffered": len(self._frames),
            "fps": self.frames_received / elapsed if elapsed > 0 else 0,
            "elapsed": elapsed,
        }

    def _receive_loop(self):
        while self.running:
            try:
                data, _ = self.sock.recvfrom(65535)
                self._handle_packet(data)
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"[Video] Error: {e}")

    def _handle_packet(self, data: bytes):
        if len(data) < self.HEADER_SIZE:
            return

        seq = struct.unpack("<I", data[0:4])[0]
        timestamp = struct.unpack("<Q", data[4:12])[0]
        width = struct.unpack("<H", data[12:14])[0]
        height = struct.unpack("<H", data[14:16])[0]
        frag_index = struct.unpack("<H", data[16:18])[0]
        frag_total = struct.unpack("<H", data[18:20])[0]
        payload = data[self.HEADER_SIZE :]

        self.fragments_received += 1

        if frag_total == 1:
            self._complete_frame(seq, timestamp, width, height, payload)
            return

        with self._fragment_lock:
            if seq not in self._fragments:
                self._fragments[seq] = {}
                self._fragment_info[seq] = (frag_total, timestamp, width, height)

            self._fragments[seq][frag_index] = payload

            if len(self._fragments[seq]) == frag_total:
                full_data = b"".join(self._fragments[seq][i] for i in range(frag_total))
                _, ts, w, h = self._fragment_info[seq]
                del self._fragments[seq]
                del self._fragment_info[seq]
                self._complete_frame(seq, ts, w, h, full_data)

            stale_seqs = [s for s in self._fragments.keys() if s < seq - 10]
            for stale_seq in stale_seqs:
                del self._fragments[stale_seq]
                del self._fragment_info[stale_seq]
                self.incomplete_dropped += 1

    def _complete_frame(
        self, seq: int, timestamp: int, width: int, height: int, jpeg_data: bytes
    ):
        if self.last_seq >= 0 and seq > self.last_seq + 1:
            self.frames_dropped += seq - self.last_seq - 1
        self.last_seq = seq

        frame = cv2.imdecode(np.frombuffer(jpeg_data, np.uint8), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            return

        frame_data = FrameData(
            seq=seq,
            timestamp=timestamp,
            width=width,
            height=height,
            data=frame,
            received_at=time.time(),
        )

        with self._frame_lock:
            self._frames.append(frame_data)

        self.frames_received += 1
