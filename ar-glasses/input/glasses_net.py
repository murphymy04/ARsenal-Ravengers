"""Networking primitives for the custom INMO Air 3 stream protocol.

Audio is TCP (lossless), video is UDP (lossy with fragment reassembly), discovery
is UDP broadcast. Each receiver runs its own thread and exposes `reset()` so the
pairing loop can wipe stale state on reconnect or silence.
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

from pipeline.config import HUD_BROADCAST_ENABLED, HUD_BROADCAST_PORT

AUDIO_PORT = 5003
DISCOVERY_PORT = 5002
VIDEO_PORT = 5000
COMMAND_PORT = 5001

VIDEO_SILENCE_TIMEOUT_SEC = 2.0


@dataclass
class FrameData:
    seq: int
    timestamp: int
    width: int
    height: int
    data: np.ndarray


@dataclass
class AudioChunk:
    seq: int
    timestamp_start: int
    timestamp_end: int
    sample_rate: int
    num_samples: int
    data: np.ndarray


class DiscoveryService:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.sock.bind(("0.0.0.0", DISCOVERY_PORT))
        self.sock.settimeout(1.0)
        self.running = False
        self.thread: Optional[threading.Thread] = None

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

        glasses_name = msg.get("name", "unknown")
        print(f"[Discovery] Reply to glasses '{glasses_name}' at {addr[0]}")

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

        self._chunks: deque[AudioChunk] = deque(maxlen=max_chunks)
        self._chunks_lock = threading.Lock()

        self.chunks_received = 0
        self.on_reconnect_callback: Optional[Callable[[], None]] = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._accept_loop, daemon=True)
        self.thread.start()
        print(f"[Audio] TCP server listening on port {AUDIO_PORT}")

    def stop(self):
        self.running = False
        self._close_client()
        if self.thread:
            self.thread.join(timeout=2)
        self.sock.close()

    def reset(self):
        with self._chunks_lock:
            self._chunks.clear()
        self.chunks_received = 0

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
            self._enable_keepalive(client)
            client.settimeout(1.0)
            self.client_socket = client

            self.reset()
            if self.on_reconnect_callback:
                self.on_reconnect_callback()

            self._receive_loop()
            self._close_client()

    @staticmethod
    def _enable_keepalive(sock: socket.socket):
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        if hasattr(socket, "TCP_KEEPIDLE"):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 5)
        if hasattr(socket, "TCP_KEEPINTVL"):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 2)
        if hasattr(socket, "TCP_KEEPCNT"):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3)
        if hasattr(socket, "TCP_KEEPALIVE"):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPALIVE, 5)

    def _close_client(self):
        if self.client_socket:
            try:
                self.client_socket.close()
            except Exception:
                pass
            self.client_socket = None

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

        chunk = AudioChunk(
            seq=seq,
            timestamp_start=ts_start,
            timestamp_end=ts_end,
            sample_rate=sample_rate,
            num_samples=num_samples,
            data=np.frombuffer(audio_bytes, dtype=np.int16),
        )

        with self._chunks_lock:
            self._chunks.append(chunk)
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

        self._state_lock = threading.Lock()
        self._frames: deque[FrameData] = deque(maxlen=max_frames)
        self._fragments: dict[int, dict[int, bytes]] = {}
        self._fragment_info: dict[int, tuple[int, int, int, int]] = {}

        self.frames_received = 0
        self.frames_dropped = 0
        self.last_seq = -1

        self._last_packet_at: Optional[float] = None
        self._silent = True
        self.on_silence_callback: Optional[Callable[[], None]] = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.thread.start()
        print(f"[Video] Listening on port {VIDEO_PORT}")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        self.sock.close()

    def reset(self):
        with self._state_lock:
            self._frames.clear()
            self._fragments.clear()
            self._fragment_info.clear()
            self.frames_received = 0
            self.frames_dropped = 0
            self.last_seq = -1
        self._last_packet_at = None
        self._silent = True

    def get_frame_after(self, seq: int) -> Optional[FrameData]:
        with self._state_lock:
            for frame in self._frames:
                if frame.seq > seq:
                    return frame
        return None

    def _receive_loop(self):
        while self.running:
            try:
                data, _ = self.sock.recvfrom(65535)
                self._last_packet_at = time.monotonic()
                if self._silent:
                    self._silent = False
                    print("[Video] Stream active")
                self._handle_packet(data)
            except socket.timeout:
                self._check_silence()
            except Exception as e:
                if self.running:
                    print(f"[Video] Error: {e}")

    def _check_silence(self):
        if self._silent or self._last_packet_at is None:
            return
        if time.monotonic() - self._last_packet_at < VIDEO_SILENCE_TIMEOUT_SEC:
            return

        print(f"[Video] Stream silent for >{VIDEO_SILENCE_TIMEOUT_SEC:.1f}s")
        self._silent = True
        self.reset()
        if self.on_silence_callback:
            self.on_silence_callback()

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

        completed: Optional[tuple[int, int, int, int, bytes]] = None
        with self._state_lock:
            if frag_total == 1:
                completed = (seq, timestamp, width, height, payload)
            else:
                if seq not in self._fragments:
                    self._fragments[seq] = {}
                    self._fragment_info[seq] = (frag_total, timestamp, width, height)
                self._fragments[seq][frag_index] = payload

                if len(self._fragments[seq]) == frag_total:
                    full_data = b"".join(
                        self._fragments[seq][i] for i in range(frag_total)
                    )
                    _, ts, w, h = self._fragment_info[seq]
                    del self._fragments[seq]
                    del self._fragment_info[seq]
                    completed = (seq, ts, w, h, full_data)

                stale_seqs = [s for s in self._fragments if s < seq - 10]
                for stale_seq in stale_seqs:
                    del self._fragments[stale_seq]
                    del self._fragment_info[stale_seq]

        if completed is not None:
            self._complete_frame(*completed)

    def _complete_frame(
        self, seq: int, timestamp: int, width: int, height: int, jpeg_data: bytes
    ):
        frame = cv2.imdecode(np.frombuffer(jpeg_data, np.uint8), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            return

        frame_data = FrameData(
            seq=seq,
            timestamp=timestamp,
            width=width,
            height=height,
            data=frame,
        )
        with self._state_lock:
            if self.last_seq >= 0 and seq > self.last_seq + 1:
                self.frames_dropped += seq - self.last_seq - 1
            self.last_seq = seq
            self._frames.append(frame_data)
            self.frames_received += 1
