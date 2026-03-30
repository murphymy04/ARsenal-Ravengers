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

# ============== Configuration ==============

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
                
                # Send response
                response = json.dumps({
                    'type': 'discover_response',
                    'video_port': VIDEO_PORT,
                    'command_port': COMMAND_PORT
                }).encode('utf-8')
                self.sock.sendto(response, addr)
                
                # Notify if new glasses
                if glasses_ip not in self.discovered_ips:
                    self.discovered_ips.add(glasses_ip)
                    self.on_glasses_found(glasses_ip)
                    
        except json.JSONDecodeError:
            print(f"[Discovery] Invalid JSON from {addr}")

# ============== Video Receiver ==============

class VideoReceiver:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('0.0.0.0', VIDEO_PORT))
        self.sock.settimeout(1.0)
        
        self.running = False
        self.thread = None
        
        # Latest frame (thread-safe access)
        self._latest_frame: Optional[FrameData] = None
        self._frame_lock = threading.Lock()
        
        # Stats
        self.frames_received = 0
        self.last_seq = -1
        self.frames_dropped = 0
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
        """Get latest frame, returns None if no frame available"""
        with self._frame_lock:
            return self._latest_frame
            
    def get_stats(self) -> dict:
        elapsed = time.time() - self.start_time if self.start_time else 0
        return {
            'frames_received': self.frames_received,
            'frames_dropped': self.frames_dropped,
            'fps': self.frames_received / elapsed if elapsed > 0 else 0,
            'elapsed': elapsed
        }
        
    def _receive_loop(self):
        while self.running:
            try:
                data, addr = self.sock.recvfrom(65535)
                self._handle_frame(data)
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"[Video] Error: {e}")
                    
    def _handle_frame(self, data: bytes):
        # Header: seq(4) + timestamp(8) + width(2) + height(2) = 16 bytes
        if len(data) < 16:
            return
            
        header = data[:16]
        jpeg_data = data[16:]
        
        seq, timestamp, width, height = struct.unpack('<IQHH', header)
        
        # Track dropped frames
        if self.last_seq >= 0:
            expected = self.last_seq + 1
            if seq > expected:
                dropped = seq - expected
                self.frames_dropped += dropped
                # Only log occasionally to avoid spam
                if dropped > 1:
                    print(f"[Video] Dropped {dropped} frames (seq {expected} to {seq-1})")
        self.last_seq = seq
        
        # Decode JPEG
        try:
            frame = cv2.imdecode(np.frombuffer(jpeg_data, np.uint8), cv2.IMREAD_GRAYSCALE)
            if frame is None:
                print("[Video] Failed to decode JPEG")
                return
        except Exception as e:
            print(f"[Video] Decode error: {e}")
            return
            
        # Store latest frame (overwrite, no queue)
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
        print("[Server] All services started")
        
    def stop(self):
        self.running = False
        self.discovery.stop()
        self.video.stop()
        self.commands.stop()
        print("[Server] All services stopped")
        
    def run_display_loop(self):
        """Main display loop - run on main thread"""
        cv2.namedWindow('Glasses Feed', cv2.WINDOW_NORMAL)
        
        last_stats_time = time.time()
        
        while self.running:
            # Get latest frame
            frame_data = self.video.get_latest_frame()
            
            if frame_data is not None:
                # Convert to color for display
                display = cv2.cvtColor(frame_data.data, cv2.COLOR_GRAY2BGR)
                
                # Add overlay info
                stats = self.video.get_stats()
                cv2.putText(display, f"FPS: {stats['fps']:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display, f"Seq: {frame_data.seq}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display, f"Dropped: {stats['frames_dropped']}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                connected = "YES" if self.commands.is_connected() else "NO"
                cv2.putText(display, f"TCP: {connected}", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Glasses Feed', display)
            else:
                # Show waiting screen
                waiting = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(waiting, "Waiting for glasses...", (150, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
                cv2.imshow('Glasses Feed', waiting)
            
            # Handle keyboard
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('h'):
                # Send highlight command
                self.commands.send_command('highlight', {'x': 320, 'y': 240, 'radius': 50})
            elif key == ord('t'):
                # Send text command
                self.commands.send_command('show_text', {'text': 'Hello from laptop!', 'duration': 3})
            elif key == ord('c'):
                # Send clear command  
                self.commands.send_command('clear')
            elif key == ord('s'):
                # Print stats
                print(f"Stats: {self.video.get_stats()}")
                
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
