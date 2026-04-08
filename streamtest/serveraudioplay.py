# laptop_server.py

import socket
import struct
import threading
import json
import time
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable
import queue
import pyaudio


# ============== Configuration ==============

AUDIO_PORT = 5003
DISCOVERY_PORT = 5002
VIDEO_PORT = 5000
COMMAND_PORT = 5001

# ============== Data Classes ==============

@dataclass
class FrameData:
    seq: int
    timestamp: int
    width: int
    height: int
    data: np.ndarray
    received_at: float

@dataclass 
class GlassesConnection:
    ip: str
    tcp_socket: Optional[socket.socket] = None
    connected: bool = False

@dataclass
class AudioChunk:
    seq: int
    timestamp_start: int
    timestamp_end: int
    sample_rate: int
    num_samples: int
    data: np.ndarray
    received_at: float


# ============== Discovery Service ==============

class DiscoveryService:
    def __init__(self, on_glasses_found: Callable[[str], None]):
        self.on_glasses_found = on_glasses_found
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.sock.bind(('0.0.0.0', DISCOVERY_PORT))
        self.sock.settimeout(1.0)
        self.running = False
        self.thread = None
        self.discovered_ips = set()
        
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
            msg = json.loads(data.decode('utf-8'))
            if msg.get('type') == 'discover':
                glasses_ip = addr[0]
                glasses_name = msg.get('name', 'unknown')
                print(f"[Discovery] Found glasses '{glasses_name}' at {glasses_ip}")

                response = json.dumps({
                    'type': 'discover_response',
                    'video_port': VIDEO_PORT,
                    'command_port': COMMAND_PORT,
                    'audio_port': AUDIO_PORT  # Add this
                }).encode('utf-8')
                self.sock.sendto(response, addr)

                if glasses_ip not in self.discovered_ips:
                    self.discovered_ips.add(glasses_ip)
                    self.on_glasses_found(glasses_ip)

        except json.JSONDecodeError:
            print(f"[Discovery] Invalid JSON from {addr}")


# ============== AudioReceiver Class - Add ==============

class AudioReceiver:
    HEADER_SIZE = 28

    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('0.0.0.0', AUDIO_PORT))
        self.sock.listen(1)
        self.sock.settimeout(1.0)

        self.running = False
        self.thread = None
        self.client_socket = None

        self._chunks = []
        self._chunks_lock = threading.Lock()
        self._max_chunks = 1000  # Keep last N chunks

        self.chunks_received = 0
        self.start_time = None

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
            except:
                pass
        if self.thread:
            self.thread.join(timeout=2)
        self.sock.close()

    def get_audio_for_timestamp(self, ts: int) -> Optional[tuple]:
        """Get audio data for a specific video timestamp"""
        with self._chunks_lock:
            for chunk in self._chunks:
                if chunk.timestamp_start <= ts <= chunk.timestamp_end:
                    progress = (ts - chunk.timestamp_start) / max(1, chunk.timestamp_end - chunk.timestamp_start)
                    sample_offset = int(progress * chunk.num_samples)
                    return (chunk.data, sample_offset, chunk.sample_rate)
        return None

    def get_audio_range(self, ts_start: int, ts_end: int) -> list:
        """Get all audio chunks overlapping timestamp range"""
        result = []
        with self._chunks_lock:
            for chunk in self._chunks:
                if chunk.timestamp_end >= ts_start and chunk.timestamp_start <= ts_end:
                    result.append(chunk)
        return result

    def get_latest_chunk(self) -> Optional[AudioChunk]:
        """Get most recent audio chunk"""
        with self._chunks_lock:
            if self._chunks:
                return self._chunks[-1]
        return None

    def get_stats(self) -> dict:
        elapsed = time.time() - self.start_time if self.start_time else 0
        return {
            'chunks_received': self.chunks_received,
            'chunks_buffered': len(self._chunks),
            'chunks_per_sec': self.chunks_received / elapsed if elapsed > 0 else 0,
            'connected': self.client_socket is not None
        }

    def _accept_loop(self):
        while self.running:
            try:
                client, addr = self.sock.accept()
                print(f"[Audio] Glasses connected from {addr}")
                self.client_socket = client
                self.client_socket.settimeout(1.0)
                self._receive_loop()
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"[Audio] Accept error: {e}")

    def _receive_loop(self):
        while self.running:
            try:
                length_bytes = self._recv_exact(4)
                if not length_bytes:
                    break

                length = struct.unpack('<I', length_bytes)[0]

                if length > 1024 * 1024:  # Sanity check 1MB
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
        data = b''
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

        seq = struct.unpack('<I', packet[0:4])[0]
        ts_start = struct.unpack('<Q', packet[4:12])[0]
        ts_end = struct.unpack('<Q', packet[12:20])[0]
        sample_rate = struct.unpack('<I', packet[20:24])[0]
        num_samples = struct.unpack('<I', packet[24:28])[0]

        audio_bytes = packet[self.HEADER_SIZE:]
        expected_bytes = num_samples * 2

        if len(audio_bytes) != expected_bytes:
            print(f"[Audio] Size mismatch: got {len(audio_bytes)}, expected {expected_bytes}")
            return

        # Convert bytes to numpy array (16-bit signed PCM)
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16)

        chunk = AudioChunk(
            seq=seq,
            timestamp_start=ts_start,
            timestamp_end=ts_end,
            sample_rate=sample_rate,
            num_samples=num_samples,
            data=audio_data,
            received_at=time.time()
        )

        with self._chunks_lock:
            self._chunks.append(chunk)
            # Trim old chunks
            if len(self._chunks) > self._max_chunks:
                self._chunks = self._chunks[-self._max_chunks:]

        self.chunks_received += 1


### audio playback ###########################
class AudioPlayer:
    def __init__(self, audio_receiver: AudioReceiver):
        self.audio_receiver = audio_receiver
        self.sample_rate = 16000

        self.running = False
        self.thread: Optional[threading.Thread] = None

        self.pyaudio = None
        self.stream = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._play_loop, daemon=True)
        self.thread.start()
        print("[AudioPlayer] Started")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.pyaudio:
            self.pyaudio.terminate()
        print("[AudioPlayer] Stopped")

    def _play_loop(self):
        self.pyaudio = pyaudio.PyAudio()
        self.stream = self.pyaudio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            output=True,
            frames_per_buffer=1600  # 100ms at 16kHz
        )

        last_seq = -1

        while self.running:
            chunk = self.audio_receiver.get_latest_chunk()

            if chunk is None:
                time.sleep(0.01)
                continue

            # Skip if already played
            if chunk.seq <= last_seq:
                time.sleep(0.01)
                continue

            last_seq = chunk.seq

            # Play audio
            self.stream.write(chunk.data.tobytes())
#end audio playback #########################

# ============== Video Receiver with Fragment Reassembly ==============

class VideoReceiver:
    HEADER_SIZE = 20  # seq(4) + timestamp(8) + width(2) + height(2) + fragIndex(2) + fragTotal(2)
    
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        self.sock.bind(('0.0.0.0', VIDEO_PORT))
        self.sock.settimeout(1.0)
        
        self.running = False
        self.thread = None
        
        self._latest_frame: Optional[FrameData] = None
        self._frame_lock = threading.Lock()
        
        self._fragments = {}
        self._fragment_info = {}
        self._fragment_lock = threading.Lock()
        
        self.frames_received = 0
        self.fragments_received = 0
        self.last_seq = -1
        self.frames_dropped = 0
        self.incomplete_dropped = 0
        self.start_time = None
        
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
        
    def get_latest_frame(self) -> Optional[FrameData]:
        with self._frame_lock:
            return self._latest_frame
            
    def get_stats(self) -> dict:
        elapsed = time.time() - self.start_time if self.start_time else 0
        return {
            'frames_received': self.frames_received,
            'frames_dropped': self.frames_dropped,
            'incomplete_dropped': self.incomplete_dropped,
            'fps': self.frames_received / elapsed if elapsed > 0 else 0,
            'elapsed': elapsed
        }
        
    def _receive_loop(self):
        while self.running:
            try:
                data, addr = self.sock.recvfrom(65535)
                self._handle_packet(data)
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"[Video] Error: {e}")
                    
    def _handle_packet(self, data: bytes):
        if len(data) < self.HEADER_SIZE:
            return
            
        seq = struct.unpack('<I', data[0:4])[0]
        timestamp = struct.unpack('<Q', data[4:12])[0]
        width = struct.unpack('<H', data[12:14])[0]
        height = struct.unpack('<H', data[14:16])[0]
        frag_index = struct.unpack('<H', data[16:18])[0]
        frag_total = struct.unpack('<H', data[18:20])[0]
        payload = data[self.HEADER_SIZE:]
        
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
                try:
                    full_data = b''.join(self._fragments[seq][i] for i in range(frag_total))
                    _, ts, w, h = self._fragment_info[seq]
                    del self._fragments[seq]
                    del self._fragment_info[seq]
                    self._complete_frame(seq, ts, w, h, full_data)
                except KeyError:
                    del self._fragments[seq]
                    del self._fragment_info[seq]
                    
            old_seqs = [s for s in self._fragments.keys() if s < seq - 10]
            for old_seq in old_seqs:
                del self._fragments[old_seq]
                del self._fragment_info[old_seq]
                self.incomplete_dropped += 1
                    
    def _complete_frame(self, seq: int, timestamp: int, width: int, height: int, jpeg_data: bytes):
        if self.last_seq >= 0 and seq > self.last_seq + 1:
            self.frames_dropped += seq - self.last_seq - 1
        self.last_seq = seq
        
        try:
            frame = cv2.imdecode(np.frombuffer(jpeg_data, np.uint8), cv2.IMREAD_GRAYSCALE)
            if frame is None:
                return
        except:
            return
            
        frame_data = FrameData(
            seq=seq,
            timestamp=timestamp,
            width=width,
            height=height,
            data=frame,
            received_at=time.time()
        )
        
        with self._frame_lock:
            self._latest_frame = frame_data
            
        self.frames_received += 1
# ============== Command Sender (TCP Server) ==============

class CommandServer:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('0.0.0.0', COMMAND_PORT))
        self.sock.listen(1)
        self.sock.settimeout(1.0)
        
        self.running = False
        self.thread = None
        
        self.client_socket: Optional[socket.socket] = None
        self._client_lock = threading.Lock()
        
        self.cmd_id_counter = 0
        
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._accept_loop, daemon=True)
        self.thread.start()
        print(f"[Command] TCP server listening on port {COMMAND_PORT}")
        
    def stop(self):
        self.running = False
        with self._client_lock:
            if self.client_socket:
                self.client_socket.close()
        if self.thread:
            self.thread.join(timeout=2)
        self.sock.close()
        
    def is_connected(self) -> bool:
        with self._client_lock:
            return self.client_socket is not None
            
    def send_command(self, cmd: str, data: dict = None) -> bool:
        """Send command to glasses. Returns True if sent successfully."""
        with self._client_lock:
            if not self.client_socket:
                print("[Command] No glasses connected")
                return False
                
            self.cmd_id_counter += 1
            message = {
                'cmd_id': self.cmd_id_counter,
                'cmd': cmd,
                'data': data or {},
                'timestamp': int(time.time() * 1000)
            }
            
            try:
                json_bytes = json.dumps(message).encode('utf-8')
                # Length-prefixed: 4 bytes length + JSON
                length_prefix = struct.pack('<I', len(json_bytes))
                self.client_socket.sendall(length_prefix + json_bytes)
                print(f"[Command] Sent: {cmd} (id={self.cmd_id_counter})")
                return True
            except Exception as e:
                print(f"[Command] Send error: {e}")
                self._disconnect_client()
                return False
                
    def _accept_loop(self):
        while self.running:
            try:
                client, addr = self.sock.accept()
                print(f"[Command] Glasses connected from {addr}")
                
                with self._client_lock:
                    if self.client_socket:
                        print("[Command] Closing previous connection")
                        self.client_socket.close()
                    self.client_socket = client
                    self.client_socket.settimeout(1.0)
                    
                # Start receive thread for this client
                recv_thread = threading.Thread(
                    target=self._receive_from_client, 
                    daemon=True
                )
                recv_thread.start()
                
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"[Command] Accept error: {e}")
                    
    def _receive_from_client(self):
        """Receive ACKs and messages from glasses"""
        while self.running:
            with self._client_lock:
                sock = self.client_socket
            if not sock:
                break
                
            try:
                # Read length prefix
                length_bytes = self._recv_exact(sock, 4)
                if not length_bytes:
                    break
                    
                length = struct.unpack('<I', length_bytes)[0]
                if length > 1024 * 1024:  # Sanity check: 1MB max
                    print(f"[Command] Invalid message length: {length}")
                    break
                    
                # Read JSON
                json_bytes = self._recv_exact(sock, length)
                if not json_bytes:
                    break
                    
                message = json.loads(json_bytes.decode('utf-8'))
                self._handle_glasses_message(message)
                
            except socket.timeout:
                continue
            except Exception as e:
                print(f"[Command] Receive error: {e}")
                break
                
        self._disconnect_client()
        
    def _recv_exact(self, sock: socket.socket, n: int) -> Optional[bytes]:
        """Receive exactly n bytes"""
        data = b''
        while len(data) < n:
            try:
                chunk = sock.recv(n - len(data))
                if not chunk:
                    return None
                data += chunk
            except socket.timeout:
                if not self.running:
                    return None
                continue
        return data
        
    def _handle_glasses_message(self, message: dict):
        msg_type = message.get('type', 'unknown')
        if msg_type == 'ack':
            cmd_id = message.get('cmd_id')
            print(f"[Command] ACK received for cmd_id={cmd_id}")
        elif msg_type == 'stats':
            print(f"[Command] Glasses stats: {message}")
        else:
            print(f"[Command] Received from glasses: {message}")
            
    def _disconnect_client(self):
        with self._client_lock:
            if self.client_socket:
                try:
                    self.client_socket.close()
                except:
                    pass
                self.client_socket = None
        print("[Command] Glasses disconnected")

# ============== Main Laptop Server ==============

class LaptopServer:
    def __init__(self):
        self.discovery = DiscoveryService(self._on_glasses_found)
        self.video = VideoReceiver()
        self.commands = CommandServer()
        self.audio = AudioReceiver()  # Add this

        self.audio_player = AudioPlayer(self.audio)  # Add audio player

                
        self.glasses_ip: Optional[str] = None
        self.running = False
 
    def _on_glasses_found(self, ip: str):
        print(f"[Server] Glasses discovered at {ip}")
        self.glasses_ip = ip
        
    def start(self):
        self.running = True
        self.discovery.start()
        self.video.start()
        self.commands.start()
        self.audio.start()  # Add this
        self.audio_player.start()  # Add
        print("[Server] All services started")

    def stop(self):
        self.running = False
        self.audio_player.stop()  # Add
        self.discovery.stop()
        self.video.stop()
        self.commands.stop()
        self.audio.stop()  # Add this
        print("[Server] All services stopped")
        
    def run_display_loop(self):
        cv2.namedWindow('Glasses Feed', cv2.WINDOW_NORMAL)

        while self.running:
            frame_data = self.video.get_latest_frame()

            if frame_data is not None:
                display = cv2.cvtColor(frame_data.data, cv2.COLOR_GRAY2BGR)

                # Video stats
                vstats = self.video.get_stats()
                y = 30
                cv2.putText(display, f"Res: {frame_data.width}x{frame_data.height}", (10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y += 25
                cv2.putText(display, f"Video FPS: {vstats['fps']:.1f}", (10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y += 25
                cv2.putText(display, f"Dropped: {vstats['frames_dropped']}", (10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y += 25

                # Audio stats
                astats = self.audio.get_stats()
                cv2.putText(display, f"Audio: {astats['chunks_received']} chunks", (10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y += 25
                audio_conn = "YES" if astats['connected'] else "NO"
                cv2.putText(display, f"Audio TCP: {audio_conn}", (10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y += 25

                # Command stats
                cmd_conn = "YES" if self.commands.is_connected() else "NO"
                cv2.putText(display, f"Cmd TCP: {cmd_conn}", (10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.imshow('Glasses Feed', display)
            else:
                waiting = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(waiting, "Waiting for glasses...", (150, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
                cv2.imshow('Glasses Feed', waiting)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('h'):
                self.commands.send_command('highlight', {'x': 320, 'y': 240, 'radius': 50})
            elif key == ord('t'):
                self.commands.send_command('show_text', {'text': 'Hello!', 'duration': 3})

        cv2.destroyAllWindows()


# ============== Entry Point ==============

def main():
    print("=" * 50)
    print("Laptop Server - Glasses Streaming POC")
    print("=" * 50)
    print("Controls:")
    print("  q - Quit")
    print("  h - Send highlight command")
    print("  t - Send text command")
    print("  c - Send clear command")
    print("  s - Print stats")
    print("=" * 50)
    
    server = LaptopServer()
    
    try:
        server.start()
        server.run_display_loop()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server.stop()
        
if __name__ == '__main__':
    main()
