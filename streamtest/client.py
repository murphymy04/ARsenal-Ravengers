# glasses_client.py

import argparse
import socket
import struct
import threading
import json
import time
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional
from queue import Queue, Empty

# ============== Configuration ==============
AUDIO_PORT = 5003
AUDIO_CHUNK_MS = 100
AUDIO_SAMPLE_RATE = 16000

DISCOVERY_PORT = 5002
DISCOVERY_BROADCAST = '255.255.255.255'
DISCOVERY_INTERVAL = 1.0
DISCOVERY_TIMEOUT = 30.0

FRAGMENT_SIZE = 1400
HEADER_SIZE = 20

JPEG_QUALITY = 95


# ============== Data Classes ==============

@dataclass
class ServerInfo:
    ip: str
    video_port: int
    command_port: int
    audio_port: int  # Add this

@dataclass
class Command:
    cmd_id: int
    cmd: str
    data: dict
    timestamp: int

# ============== Latest Frame Slot (Thread-Safe) ==============

class LatestFrameSlot:
    def __init__(self):
        self._frame: Optional[bytes] = None
        self._timestamp: int = 0
        self._seq: int = 0
        self._width: int = 0
        self._height: int = 0
        self._lock = threading.Lock()
        self._new_frame_event = threading.Event()
        
    def put(self, frame_bytes: bytes, timestamp: int, seq: int, width: int, height: int):
        with self._lock:
            self._frame = frame_bytes
            self._timestamp = timestamp
            self._seq = seq
            self._width = width
            self._height = height
        self._new_frame_event.set()
        
    def get(self, timeout: float = 1.0) -> Optional[tuple]:
        if self._new_frame_event.wait(timeout=timeout):
            with self._lock:
                frame = self._frame
                timestamp = self._timestamp
                seq = self._seq
                width = self._width
                height = self._height
                self._frame = None
            self._new_frame_event.clear()
            
            if frame is not None:
                return (frame, timestamp, seq, width, height)
        return None
# ============== Discovery Client ==============

class DiscoveryClient:
    def __init__(self, glasses_name: str = "glasses-poc"):
        self.glasses_name = glasses_name
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.settimeout(0.5)
        self.sock.bind(('0.0.0.0', 0))
        
    def discover(self, timeout: float = DISCOVERY_TIMEOUT) -> Optional[ServerInfo]:
        """Broadcast discovery until laptop found or timeout"""
        print(f"[Discovery] Searching for laptop...")
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Send broadcast
            message = json.dumps({
                'type': 'discover',
                'name': self.glasses_name
            }).encode('utf-8')
            
            try:
                self.sock.sendto(message, (DISCOVERY_BROADCAST, DISCOVERY_PORT))
                print(f"[Discovery] Broadcast sent...")
            except Exception as e:
                print(f"[Discovery] Broadcast failed: {e}")
                
            # Wait for response
            try:
                data, addr = self.sock.recvfrom(1024)
                response = json.loads(data.decode('utf-8'))
                
                if response.get('type') == 'discover_response':
                    server_info = ServerInfo(
                        ip=addr[0],
                        video_port=response.get('video_port', 5000),
                        command_port=response.get('command_port', 5001),
                        audio_port=response.get('audio_port', 5003)  # Add this
                    )
                    print(f"[Discovery] Found laptop at {server_info.ip}")
                    return server_info
            except socket.timeout:
                continue
            except json.JSONDecodeError:
                continue
                
            time.sleep(DISCOVERY_INTERVAL)
            
        print("[Discovery] Timeout - laptop not found")
        return None
        
    def close(self):
        self.sock.close()
class AudioChunkSlot:
    def __init__(self):
        self._data: Optional[bytes] = None
        self._timestamp_start: int = 0
        self._timestamp_end: int = 0
        self._seq: int = 0
        self._lock = threading.Lock()
        self._new_chunk_event = threading.Event()

    def put(self, pcm_data: bytes, ts_start: int, ts_end: int, seq: int):
        with self._lock:
            self._data = pcm_data
            self._timestamp_start = ts_start
            self._timestamp_end = ts_end
            self._seq = seq
        self._new_chunk_event.set()

    def get(self, timeout: float = 1.0) -> Optional[tuple]:
        if self._new_chunk_event.wait(timeout=timeout):
            with self._lock:
                data = self._data
                ts_start = self._timestamp_start
                ts_end = self._timestamp_end
                seq = self._seq
                self._data = None
            self._new_chunk_event.clear()
            if data is not None:
                return (data, ts_start, ts_end, seq)
        return None

class AudioProducer:
    def __init__(self, video_path: str, chunk_slot: AudioChunkSlot, shared_clock, video_producer):
        self.video_path = video_path
        self.chunk_slot = chunk_slot
        self.shared_clock = shared_clock
        self.video_producer = video_producer  # Reference to video producer
        self.sample_rate = AUDIO_SAMPLE_RATE
        self.chunk_ms = AUDIO_CHUNK_MS
        
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.seq_counter = 0
        self.chunks_produced = 0
        
        # Pre-extract all audio
        self.audio_data: Optional[np.ndarray] = None
        self.audio_duration_ms = 0
        
    def start(self):
        # Extract full audio first
        self._extract_full_audio()
        
        self.running = True
        self.thread = threading.Thread(target=self._produce_loop, daemon=True)
        self.thread.start()
        print(f"[AudioProducer] Started - synced to video")
        
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        print(f"[AudioProducer] Stopped - produced {self.chunks_produced} chunks")
        
    def _extract_full_audio(self):
        """Extract entire audio track into memory"""
        import subprocess
        
        cmd = [
            'ffmpeg',
            '-i', self.video_path,
            '-f', 's16le',
            '-acodec', 'pcm_s16le',
            '-ar', str(self.sample_rate),
            '-ac', '1',
            '-v', 'quiet',
            '-'
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=60)
            pcm_bytes = result.stdout
            
            if len(pcm_bytes) > 0:
                self.audio_data = np.frombuffer(pcm_bytes, dtype=np.int16)
                self.audio_duration_ms = (len(self.audio_data) * 1000) // self.sample_rate
                print(f"[AudioProducer] Extracted {self.audio_duration_ms}ms of audio")
            else:
                print("[AudioProducer] No audio in video, using silence")
                self.audio_data = None
                
        except Exception as e:
            print(f"[AudioProducer] ffmpeg error: {e}")
            self.audio_data = None
            
    def _produce_loop(self):
        samples_per_chunk = (self.sample_rate * self.chunk_ms) // 1000
        chunk_interval = self.chunk_ms / 1000.0
        
        last_video_pos_ms = -1
        
        while self.running:
            loop_start = time.time()
            
            # Get current video position from video producer
            video_pos_ms = self.video_producer.get_position_ms()
            
            # Skip if video hasn't advanced
            if video_pos_ms == last_video_pos_ms:
                time.sleep(0.01)
                continue
                
            last_video_pos_ms = video_pos_ms
            
            # Get audio chunk for this video position
            ts_start = self.shared_clock.elapsed_ms()
            pcm_data = self._get_audio_at_position(video_pos_ms, samples_per_chunk)
            ts_end = self.shared_clock.elapsed_ms()
            
            self.chunk_slot.put(pcm_data, ts_start, ts_end, self.seq_counter)
            self.seq_counter += 1
            self.chunks_produced += 1
            
            # Rate limit
            elapsed = time.time() - loop_start
            sleep_time = chunk_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
                
    def _get_audio_at_position(self, pos_ms: int, num_samples: int) -> bytes:
        """Get audio samples starting at video position"""
        if self.audio_data is None:
            return bytes(num_samples * 2)  # Silence
            
        # Calculate sample offset
        sample_offset = (pos_ms * self.sample_rate) // 1000
        
        # Handle wraparound (video looped)
        sample_offset = sample_offset % len(self.audio_data)
        
        # Extract samples
        end_offset = sample_offset + num_samples
        
        if end_offset <= len(self.audio_data):
            samples = self.audio_data[sample_offset:end_offset]
        else:
            # Wrap around
            part1 = self.audio_data[sample_offset:]
            part2 = self.audio_data[:end_offset - len(self.audio_data)]
            samples = np.concatenate([part1, part2])
            
        return samples.tobytes()
        
    def get_stats(self) -> dict:
        return {
            'chunks_produced': self.chunks_produced,
            'seq': self.seq_counter
        }

class AudioConsumer:
    HEADER_SIZE = 28
    
    def __init__(self, chunk_slot: AudioChunkSlot, server_info: ServerInfo):
        self.chunk_slot = chunk_slot
        self.server_ip = server_info.ip
        self.server_port = server_info.audio_port
        self.sample_rate = AUDIO_SAMPLE_RATE
        
        self.sock: Optional[socket.socket] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
        self.chunks_sent = 0
        self.bytes_sent = 0
        self.start_time: Optional[float] = None
        
    def start(self):
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._send_loop, daemon=True)
        self.thread.start()
        print(f"[AudioConsumer] Started - sending to {self.server_ip}:{self.server_port}")
        
    def stop(self):
        self.running = False
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
        if self.thread:
            self.thread.join(timeout=2)
        print(f"[AudioConsumer] Stopped - sent {self.chunks_sent} chunks")
        
    def _send_loop(self):
        while self.running and not self._connect():
            time.sleep(2)
            
        while self.running:
            result = self.chunk_slot.get(timeout=1.0)
            
            if result is None:
                continue
                
            pcm_data, ts_start, ts_end, seq = result
            
            try:
                self._send_chunk(pcm_data, ts_start, ts_end, seq)
                self.chunks_sent += 1
            except Exception as e:
                print(f"[AudioConsumer] Send error: {e}")
                # Reconnect
                self.sock.close()
                while self.running and not self._connect():
                    time.sleep(2)
                    
    def _connect(self) -> bool:
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.server_ip, self.server_port))
            print(f"[AudioConsumer] TCP connected")
            return True
        except Exception as e:
            print(f"[AudioConsumer] Connect failed: {e}")
            return False
            
    def _send_chunk(self, pcm_data: bytes, ts_start: int, ts_end: int, seq: int):
        num_samples = len(pcm_data) // 2
        
        # Header: seq(4) + ts_start(8) + ts_end(8) + sample_rate(4) + num_samples(4) = 28
        header = struct.pack('<I', seq)
        header += struct.pack('<Q', ts_start)
        header += struct.pack('<Q', ts_end)
        header += struct.pack('<I', self.sample_rate)
        header += struct.pack('<I', num_samples)
        
        packet = header + pcm_data
        
        # Length prefix
        length_prefix = struct.pack('<I', len(packet))
        
        self.sock.sendall(length_prefix + packet)
        self.bytes_sent += len(length_prefix) + len(packet)
        
    def get_stats(self) -> dict:
        elapsed = time.time() - self.start_time if self.start_time else 0
        return {
            'chunks_sent': self.chunks_sent,
            'bytes_sent': self.bytes_sent,
            'cps': self.chunks_sent / elapsed if elapsed > 0 else 0
        }
class SharedClock:
    def __init__(self):
        self._start = time.time()
        
    def elapsed_ms(self) -> int:
        return int((time.time() - self._start) * 1000)

# ============== Video Producer ==============


class VideoProducer:
    """
    Reads frames from video file and puts them in the latest frame slot.
    Simulates camera capture on glasses.
    """
    
    def __init__(self, video_path: str, frame_slot: LatestFrameSlot, target_fps: int = 30):
        self.video_path = video_path
        self.frame_slot = frame_slot
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
        self.seq_counter = 0
        self.frames_produced = 0
        self.start_time: Optional[float] = None

        self.current_position_ms = 0
        self._position_lock = threading.Lock()

    def get_position_ms(self) -> int:
        """Get current video playback position in milliseconds"""
        with self._position_lock:
            return self.current_position_ms

        
    def start(self):
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._produce_loop, daemon=True)
        self.thread.start()
        print(f"[Producer] Started - reading from {self.video_path}")
        
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        print(f"[Producer] Stopped - produced {self.frames_produced} frames")
        
    def _produce_loop(self):
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            print(f"[Producer] ERROR: Cannot open video file: {self.video_path}")
            return
            
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[Producer] Video: {total_frames} frames at {video_fps:.1f} FPS")
        
        while self.running:
            loop_start = time.time()
            
            ret, frame = cap.read()
            
            if not ret:
                print("[Producer] Video ended, looping...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                with self._position_lock:
                    self.current_position_ms = 0
                continue
            
            # Update position
            with self._position_lock:
                self.current_position_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            
            # ... rest of existing frame processing code ...
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            ret, jpeg_bytes = cv2.imencode('.jpg', gray, encode_params)
            
            if not ret:
                continue
                
            timestamp = self.shared_clock.elapsed_ms() if self.shared_clock else int(time.time() * 1000)
            self.frame_slot.put(jpeg_bytes.tobytes(), timestamp, self.seq_counter, width, height)
            
            self.seq_counter += 1
            self.frames_produced += 1
            
            elapsed = time.time() - loop_start
            sleep_time = self.frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
                
        cap.release() 

    def get_stats(self) -> dict:
        elapsed = time.time() - self.start_time if self.start_time else 0
        return {
            'frames_produced': self.frames_produced,
            'fps': self.frames_produced / elapsed if elapsed > 0 else 0,
            'seq': self.seq_counter
        }
# ============== Video Consumer (UDP Sender) ==============

class VideoConsumer:
    def __init__(self, frame_slot: LatestFrameSlot, server_info: ServerInfo):
        self.frame_slot = frame_slot
        self.server_ip = server_info.ip
        self.server_port = server_info.video_port
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
        self.frames_sent = 0
        self.fragments_sent = 0
        self.bytes_sent = 0
        self.start_time: Optional[float] = None
        
    def start(self):
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._consume_loop, daemon=True)
        self.thread.start()
        print(f"[VideoConsumer] Started - sending to {self.server_ip}:{self.server_port}")
        
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        self.sock.close()
        print(f"[VideoConsumer] Stopped - sent {self.frames_sent} frames, {self.fragments_sent} fragments, {self.bytes_sent / 1024 / 1024:.2f} MB")
        
    def _consume_loop(self):
        endpoint = (self.server_ip, self.server_port)
        
        while self.running:
            result = self.frame_slot.get(timeout=1.0)
            
            if result is None:
                continue
                
            frame_bytes, timestamp, seq, width, height = result
            
            try:
                self._send_frame(endpoint, seq, timestamp, width, height, frame_bytes)
                self.frames_sent += 1
            except Exception as e:
                print(f"[VideoConsumer] Send error: {e}")
                
    def _send_frame(self, endpoint: tuple, seq: int, timestamp: int, width: int, height: int, frame_data: bytes):
        # Header: seq(4) + timestamp(8) + width(2) + height(2) + fragIndex(2) + fragTotal(2) = 20 bytes
        max_payload = FRAGMENT_SIZE - HEADER_SIZE
        total_fragments = (len(frame_data) + max_payload - 1) // max_payload
        
        if total_fragments == 0:
            total_fragments = 1
        
        for i in range(total_fragments):
            offset = i * max_payload
            length = min(max_payload, len(frame_data) - offset)
            
            # Build header
            header = struct.pack('<I', seq)                    # seq (4 bytes)
            header += struct.pack('<Q', timestamp)             # timestamp (8 bytes)
            header += struct.pack('<H', width)                 # width (2 bytes)
            header += struct.pack('<H', height)                # height (2 bytes)
            header += struct.pack('<H', i)                     # frag_index (2 bytes)
            header += struct.pack('<H', total_fragments)       # frag_total (2 bytes)
            
            # Build packet
            payload = frame_data[offset:offset + length]
            packet = header + payload
            
            self.sock.sendto(packet, endpoint)
            self.fragments_sent += 1
            self.bytes_sent += len(packet)
                
    def get_stats(self) -> dict:
        elapsed = time.time() - self.start_time if self.start_time else 0
        return {
            'frames_sent': self.frames_sent,
            'fragments_sent': self.fragments_sent,
            'bytes_sent': self.bytes_sent,
            'fps': self.frames_sent / elapsed if elapsed > 0 else 0,
            'mbps': (self.bytes_sent * 8 / 1024 / 1024) / elapsed if elapsed > 0 else 0
        }
# ============== Command Receiver (TCP Client) ==============

class CommandReceiver:
    """
    Connects to laptop TCP server and receives commands.
    Commands are queued for main thread to process.
    """
    
    def __init__(self, server_info: ServerInfo):
        self.server_ip = server_info.ip
        self.server_port = server_info.command_port
        
        self.sock: Optional[socket.socket] = None
        self.running = False
        self.connected = False
        self.thread: Optional[threading.Thread] = None
        
        # Queue for received commands (main thread processes these)
        self.command_queue: Queue[Command] = Queue()
        
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._connect_and_receive, daemon=True)
        self.thread.start()
        print(f"[Commands] Started - connecting to {self.server_ip}:{self.server_port}")
        
    def stop(self):
        self.running = False
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
        if self.thread:
            self.thread.join(timeout=2)
        print("[Commands] Stopped")
        
    def get_command(self, timeout: float = 0) -> Optional[Command]:
        """Get next command from queue, non-blocking by default"""
        try:
            return self.command_queue.get(timeout=timeout)
        except Empty:
            return None
            
    def is_connected(self) -> bool:
        return self.connected
        
    def _connect_and_receive(self):
        while self.running:
            # Connect
            if not self._connect():
                time.sleep(2)
                continue
                
            # Receive loop
            self._receive_loop()
            
            # Disconnected, will retry
            self.connected = False
            if self.running:
                print("[Commands] Connection lost, reconnecting...")
                time.sleep(1)
                
    def _connect(self) -> bool:
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(5.0)
            self.sock.connect((self.server_ip, self.server_port))
            self.sock.settimeout(1.0)
            self.connected = True
            print(f"[Commands] Connected to laptop")
            return True
        except Exception as e:
            print(f"[Commands] Connection failed: {e}")
            return False
            
    def _receive_loop(self):
        while self.running and self.connected:
            try:
                # Read length prefix (4 bytes)
                length_bytes = self._recv_exact(4)
                if length_bytes is None:
                    break
                    
                length = struct.unpack('<I', length_bytes)[0]
                
                if length > 1024 * 1024:  # Sanity check
                    print(f"[Commands] Invalid message length: {length}")
                    break
                    
                # Read JSON
                json_bytes = self._recv_exact(length)
                if json_bytes is None:
                    break
                    
                message = json.loads(json_bytes.decode('utf-8'))
                self._handle_message(message)
                
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"[Commands] Receive error: {e}")
                break
                
    def _recv_exact(self, n: int) -> Optional[bytes]:
        """Receive exactly n bytes"""
        data = b''
        while len(data) < n:
            try:
                chunk = self.sock.recv(n - len(data))
                if not chunk:
                    return None
                data += chunk
            except socket.timeout:
                if not self.running:
                    return None
                continue
        return data
        
    def _handle_message(self, message: dict):
        cmd_id = message.get('cmd_id', 0)
        cmd = message.get('cmd', '')
        data = message.get('data', {})
        timestamp = message.get('timestamp', 0)
        
        command = Command(cmd_id=cmd_id, cmd=cmd, data=data, timestamp=timestamp)
        self.command_queue.put(command)
        
        # Send ACK
        self._send_ack(cmd_id)
        
    def _send_ack(self, cmd_id: int):
        try:
            ack = json.dumps({
                'type': 'ack',
                'cmd_id': cmd_id
            }).encode('utf-8')
            
            length_prefix = struct.pack('<I', len(ack))
            self.sock.sendall(length_prefix + ack)
        except Exception as e:
            print(f"[Commands] ACK send error: {e}")

# ============== Main Glasses Client ==============

class GlassesClient:
    def __init__(self, video_path: str, target_fps: int = 30):
        self.video_path = video_path
        self.target_fps = target_fps

        self.discovery: Optional[DiscoveryClient] = None
        self.server_info: Optional[ServerInfo] = None

        # Shared clock for sync
        self.shared_clock = SharedClock()

        # Video
        self.frame_slot = LatestFrameSlot()
        self.producer: Optional[VideoProducer] = None
        self.consumer: Optional[VideoConsumer] = None

        # Audio
        self.audio_slot = AudioChunkSlot()
        self.audio_producer: Optional[AudioProducer] = None
        self.audio_consumer: Optional[AudioConsumer] = None

        # Commands
        self.commands: Optional[CommandReceiver] = None

        self.running = False

    def discover_and_connect(self, timeout: float = 30.0) -> bool:
        self.discovery = DiscoveryClient()
        self.server_info = self.discovery.discover(timeout=timeout)
        self.discovery.close()

        if self.server_info is None:
            return False

        return True

    def start(self):
        if self.server_info is None:
            raise RuntimeError("Must call discover_and_connect() first")

        self.running = True

        # Video producer first (audio needs reference to it)
        self.producer = VideoProducer(
            self.video_path,
            self.frame_slot,
            target_fps=self.target_fps
        )
        self.producer.shared_clock = self.shared_clock
        self.producer.start()

        # Video consumer
        self.consumer = VideoConsumer(self.frame_slot, self.server_info)
        self.consumer.start()

        # Audio producer - pass video producer for sync
        self.audio_producer = AudioProducer(
            self.video_path,
            self.audio_slot,
            self.shared_clock,
            self.producer  # Video producer reference for sync
        )
        self.audio_producer.start()

        # Audio consumer
        self.audio_consumer = AudioConsumer(self.audio_slot, self.server_info)
        self.audio_consumer.start()

        # Commands
        self.commands = CommandReceiver(self.server_info)
        self.commands.start()

        print("[Glasses] All services started")

    def stop(self):
        self.running = False

        if self.producer:
            self.producer.stop()
        if self.consumer:
            self.consumer.stop()
        if self.audio_producer:
            self.audio_producer.stop()
        if self.audio_consumer:
            self.audio_consumer.stop()
        if self.commands:
            self.commands.stop()

        print("[Glasses] All services stopped")

    def _handle_command(self, cmd: Command):
        print(f"[Glasses] Received command: {cmd.cmd} (id={cmd.cmd_id})")
        print(f"         Data: {cmd.data}")

    def _print_stats(self):
        ps = self.producer.get_stats() if self.producer else {}
        vs = self.consumer.get_stats() if self.consumer else {}
        aps = self.audio_producer.get_stats() if self.audio_producer else {}
        acs = self.audio_consumer.get_stats() if self.audio_consumer else {}
        tcp = self.commands.is_connected() if self.commands else False

        print("-" * 50)
        print(f"[Video] Produced: {ps.get('frames_produced', 0)}, "
              f"Sent: {vs.get('frames_sent', 0)}, "
              f"FPS: {vs.get('fps', 0):.1f}")
        print(f"[Audio] Produced: {aps.get('chunks_produced', 0)}, "
              f"Sent: {acs.get('chunks_sent', 0)}")
        print(f"[TCP] Commands connected: {tcp}")
        print("-" * 50)

    def run_main_loop(self):
        print("[Glasses] Main loop started")
        print("Press Ctrl+C to stop")
        print("-" * 50)

        last_stats_time = time.time()
        stats_interval = 2.0

        while self.running:
            # Process received commands
            while True:
                cmd = self.commands.get_command()
                if cmd is None:
                    break
                self._handle_command(cmd)

            # Print stats periodically
            if time.time() - last_stats_time > stats_interval:
                self._print_stats()
                last_stats_time = time.time()

            # Small sleep to prevent busy loop
            time.sleep(0.01)


# ============== Entry Point ==============

def main():
    parser = argparse.ArgumentParser(description='Glasses Client POC')
    parser.add_argument('video', help='Path to video file')
    parser.add_argument('--fps', type=int, default=30, help='Target FPS (default: 30)')
    parser.add_argument('--timeout', type=float, default=30.0, help='Discovery timeout in seconds (default: 30)')

    args = parser.parse_args()

    print("=" * 50)
    print("Glasses Client - POC")
    print("=" * 50)
    print(f"Video: {args.video}")
    print(f"Target FPS: {args.fps}")
    print("=" * 50)

    client = GlassesClient(video_path=args.video, target_fps=args.fps)

    try:
        print("\n[1/2] Discovering laptop...")
        if not client.discover_and_connect(timeout=args.timeout):
            print("ERROR: Could not find laptop. Make sure laptop_server.py is running.")
            sys.exit(1)

        print("\n[2/2] Starting streaming...")
        client.start()

        client.run_main_loop()

    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        client.stop()


if __name__ == '__main__':
    main()
#class GlassesClient:
#    def __init__(self, video_path: str, audio_path: str = None, target_fps: int = 30):
#        self.video_path = video_path
#        self.target_fps = target_fps
#        
#        self.discovery: Optional[DiscoveryClient] = None
#        self.server_info: Optional[ServerInfo] = None
#        
#        # Shared clock for sync
#        self.shared_clock = SharedClock()
#        
#        # Video
#        self.frame_slot = LatestFrameSlot()
#        self.producer: Optional[VideoProducer] = None
#        self.consumer: Optional[VideoConsumer] = None
#        
#        # Audio
#        self.audio_slot = AudioChunkSlot()
#        self.audio_producer: Optional[AudioProducer] = None
#        self.audio_consumer: Optional[AudioConsumer] = None
#        
#        # Commands
#        self.commands: Optional[CommandReceiver] = None
#        
#        self.running = False
#
#        
#    def discover_and_connect(self, timeout: float = 30.0) -> bool:
#        """Discover laptop and setup connections"""
#        self.discovery = DiscoveryClient()
#        self.server_info = self.discovery.discover(timeout=timeout)
#        self.discovery.close()
#        
#        if self.server_info is None:
#            return False
#            
#        return True
#        
#    def start(self):
#        if self.server_info is None:
#            raise RuntimeError("Must call discover_and_connect() first")
#            
#        self.running = True
#        
#        # Video
#        self.producer = VideoProducer(
#            self.video_path, 
#            self.frame_slot, 
#            target_fps=self.target_fps
#        )
#        self.producer.shared_clock = self.shared_clock  # Share clock
#        self.producer.start()
#        
#        self.consumer = VideoConsumer(self.frame_slot, self.server_info)
#        self.consumer.start()
#        
#        # Audio
#        self.audio_producer = AudioProducer(
#            self.video_path,  # Same video file
#            self.audio_slot,
#            self.shared_clock
#        )
#        self.audio_producer.start()
#        
#        self.audio_consumer = AudioConsumer(self.audio_slot, self.server_info)
#        self.audio_consumer.start()
#        
#        # Commands
#        self.commands = CommandReceiver(self.server_info)
#        self.commands.start()
#        
#        print("[Glasses] All services started")
#
#        
#    def stop(self):
#        self.running = False
#        
#        if self.producer:
#            self.producer.stop()
#        if self.consumer:
#            self.consumer.stop()
#        if self.audio_producer:
#            self.audio_producer.stop()
#        if self.audio_consumer:
#            self.audio_consumer.stop()
#        if self.commands:
#            self.commands.stop()
#            
#        print("[Glasses] All services stopped")
#
#        
#    def _handle_command(self, cmd: Command):
#        """
#        Process a command from the laptop.
#        In real Unity app, this would update UI elements.
#        """
#        print(f"[Glasses] Received command: {cmd.cmd} (id={cmd.cmd_id})")
#        print(f"         Data: {cmd.data}")
#        
#        if cmd.cmd == 'highlight':
#            x = cmd.data.get('x', 0)
#            y = cmd.data.get('y', 0)
#            radius = cmd.data.get('radius', 50)
#            print(f"         -> Would highlight at ({x}, {y}) with radius {radius}")
#            
#        elif cmd.cmd == 'show_text':
#            text = cmd.data.get('text', '')
#            duration = cmd.data.get('duration', 3)
#            print(f"         -> Would show text '{text}' for {duration}s")
#            
#        elif cmd.cmd == 'clear':
#            print(f"         -> Would clear overlay")
#            
#        else:
#            print(f"         -> Unknown command")
#            
#    def _print_stats(self):
#        ps = self.producer.get_stats() if self.producer else {}
#        vs = self.consumer.get_stats() if self.consumer else {}
#        aps = self.audio_producer.get_stats() if self.audio_producer else {}
#        acs = self.audio_consumer.get_stats() if self.audio_consumer else {}
#        tcp = self.commands.is_connected() if self.commands else False
#        
#        print("-" * 50)
#        print(f"[Video] Produced: {ps.get('frames_produced', 0)}, Sent: {vs.get('frames_sent', 0)}")
#        print(f"[Audio] Produced: {aps.get('chunks_produced', 0)}, Sent: {acs.get('chunks_sent', 0)}")
#        print(f"[TCP] Commands: {tcp}")
#        print("-" * 50)
#        
#    def run_main_loop(self):
#        """
#        Main loop - processes commands and displays status.
#        In real Unity app, this would be Update().
#        """
#        print("[Glasses] Main loop started")
#        print("Press Ctrl+C to stop")
#        print("-" * 50)
#        
#        last_stats_time = time.time()
#        stats_interval = 2.0
#        
#        while self.running:
#            # Process received commands
#            while True:
#                cmd = self.commands.get_command()
#                if cmd is None:
#                    break
#                self._handle_command(cmd)
#                
#            # Print stats periodically
#            if time.time() - last_stats_time > stats_interval:
#                self._print_stats()
#                last_stats_time = time.time()
#                
#            # Small sleep to prevent busy loop
#            time.sleep(0.01)
#
#def main():
#    parser = argparse.ArgumentParser(description='Glasses Client POC')
#    parser.add_argument('video', help='Path to video file')
#    parser.add_argument('--audio', help='Path to audio file (WAV)', default=None)
#    parser.add_argument('--fps', type=int, default=30, help='Target FPS')
#    parser.add_argument('--timeout', type=float, default=30.0, help='Discovery timeout')
#
#    args = parser.parse_args()
#
#    print("=" * 50)
#    print("Glasses Client - POC")
#    print("=" * 50)
#    print(f"Video: {args.video}")
#    print(f"Audio: {args.audio or 'None (silence)'}")
#    print("=" * 50)
#
#    client = GlassesClient(
#        video_path=args.video,
#        audio_path=args.audio,
#        target_fps=args.fps
#    )
#
#    try:
#        if not client.discover_and_connect(timeout=args.timeout):
#            print("ERROR: Could not find laptop")
#            sys.exit(1)
#
#        client.start()
#        client.run_main_loop()
#
#    except KeyboardInterrupt:
#        print("\nShutting down...")
#    finally:
#        client.stop()
#
#if __name__ == '__main__':
#    main()
