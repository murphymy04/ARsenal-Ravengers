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

DISCOVERY_PORT = 5002
DISCOVERY_BROADCAST = '255.255.255.255'
DISCOVERY_INTERVAL = 1.0
DISCOVERY_TIMEOUT = 30.0

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
JPEG_QUALITY = 80

# ============== Data Classes ==============

@dataclass
class ServerInfo:
    ip: str
    video_port: int
    command_port: int

@dataclass
class Command:
    cmd_id: int
    cmd: str
    data: dict
    timestamp: int

# ============== Latest Frame Slot (Thread-Safe) ==============

class LatestFrameSlot:
    """
    Single-slot buffer that always holds the latest frame.
    Producer overwrites, consumer reads latest.
    No backup, no queue - if consumer is slow, frames are skipped.
    """
    
    def __init__(self):
        self._frame: Optional[bytes] = None
        self._timestamp: int = 0
        self._seq: int = 0
        self._lock = threading.Lock()
        self._new_frame_event = threading.Event()
        
    def put(self, frame_bytes: bytes, timestamp: int, seq: int):
        """Producer calls this - always overwrites"""
        with self._lock:
            self._frame = frame_bytes
            self._timestamp = timestamp
            self._seq = seq
        self._new_frame_event.set()
        
    def get(self, timeout: float = 1.0) -> Optional[tuple[bytes, int, int]]:
        """
        Consumer calls this - gets latest frame.
        Returns (frame_bytes, timestamp, seq) or None if timeout.
        """
        if self._new_frame_event.wait(timeout=timeout):
            with self._lock:
                frame = self._frame
                timestamp = self._timestamp
                seq = self._seq
                self._frame = None  # Mark as consumed
            self._new_frame_event.clear()
            
            if frame is not None:
                return (frame, timestamp, seq)
        return None
        
    def get_nonblocking(self) -> Optional[tuple[bytes, int, int]]:
        """Get latest frame without waiting"""
        with self._lock:
            if self._frame is None:
                return None
            frame = self._frame
            timestamp = self._timestamp
            seq = self._seq
            self._frame = None
        self._new_frame_event.clear()
        return (frame, timestamp, seq)

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
                        command_port=response.get('command_port', 5001)
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
        print(f"[Producer] Video: {total_frames} frames at {video_fps} FPS")
        
        last_frame_time = time.time()
        
        while self.running:
            loop_start = time.time()
            
            # Read frame
            ret, frame = cap.read()
            
            if not ret:
                # Loop video
                print("[Producer] Video ended, looping...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
                
            # Convert to grayscale and resize (simulating glasses camera)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (FRAME_WIDTH, FRAME_HEIGHT))
            
            # Encode to JPEG
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            ret, jpeg_bytes = cv2.imencode('.jpg', gray, encode_params)
            
            if not ret:
                print("[Producer] JPEG encode failed")
                continue
                
            # Put in slot (overwrites any previous frame)
            timestamp = int(time.time() * 1000)  # milliseconds
            self.frame_slot.put(jpeg_bytes.tobytes(), timestamp, self.seq_counter)
            
            self.seq_counter += 1
            self.frames_produced += 1
            
            # Rate limiting to target FPS
            elapsed = time.time() - loop_start
            sleep_time = self.frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
                
    def get_stats(self) -> dict:
        elapsed = time.time() - self.start_time if self.start_time else 0
        return {
            'frames_produced': self.frames_produced,
            'fps': self.frames_produced / elapsed if elapsed > 0 else 0,
            'seq': self.seq_counter
        }

# ============== Video Consumer (UDP Sender) ==============

class VideoConsumer:
    """
    Reads frames from the latest frame slot and sends them to laptop via UDP.
    If slot is empty, waits. If slot has frame, sends immediately.
    """
    
    def __init__(self, frame_slot: LatestFrameSlot, server_info: ServerInfo):
        self.frame_slot = frame_slot
        self.server_ip = server_info.ip
        self.server_port = server_info.video_port
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
        self.frames_sent = 0
        self.bytes_sent = 0
        self.start_time: Optional[float] = None
        
    def start(self):
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._consume_loop, daemon=True)
        self.thread.start()
        print(f"[Consumer] Started - sending to {self.server_ip}:{self.server_port}")
        
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        self.sock.close()
        print(f"[Consumer] Stopped - sent {self.frames_sent} frames, {self.bytes_sent / 1024 / 1024:.2f} MB")
        
    def _consume_loop(self):
        while self.running:
            # Get latest frame (blocks until available or timeout)
            result = self.frame_slot.get(timeout=1.0)
            
            if result is None:
                continue
                
            frame_bytes, timestamp, seq = result
            
            # Build packet
            # Header: seq(4) + timestamp(8) + width(2) + height(2) = 16 bytes
            header = struct.pack('<IQHH', seq, timestamp, FRAME_WIDTH, FRAME_HEIGHT)
            packet = header + frame_bytes
            
            # Send via UDP
            try:
                self.sock.sendto(packet, (self.server_ip, self.server_port))
                self.frames_sent += 1
                self.bytes_sent += len(packet)
            except Exception as e:
                print(f"[Consumer] Send error: {e}")
                
    def get_stats(self) -> dict:
        elapsed = time.time() - self.start_time if self.start_time else 0
        return {
            'frames_sent': self.frames_sent,
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
        
        self.frame_slot = LatestFrameSlot()
        self.producer: Optional[VideoProducer] = None
        self.consumer: Optional[VideoConsumer] = None
        self.commands: Optional[CommandReceiver] = None
        
        self.running = False
        
    def discover_and_connect(self, timeout: float = 30.0) -> bool:
        """Discover laptop and setup connections"""
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
        
        # Start producer (reads video, fills slot)
        self.producer = VideoProducer(
            self.video_path, 
            self.frame_slot, 
            target_fps=self.target_fps
        )
        self.producer.start()
        
        # Start consumer (reads slot, sends to laptop)
        self.consumer = VideoConsumer(self.frame_slot, self.server_info)
        self.consumer.start()
        
        # Start command receiver
        self.commands = CommandReceiver(self.server_info)
        self.commands.start()
        
        print("[Glasses] All services started")
        
    def stop(self):
        self.running = False
        
        if self.producer:
            self.producer.stop()
        if self.consumer:
            self.consumer.stop()
        if self.commands:
            self.commands.stop()
            
        print("[Glasses] All services stopped")
        
    def _handle_command(self, cmd: Command):
        """
        Process a command from the laptop.
        In real Unity app, this would update UI elements.
        """
        print(f"[Glasses] Received command: {cmd.cmd} (id={cmd.cmd_id})")
        print(f"         Data: {cmd.data}")
        
        if cmd.cmd == 'highlight':
            x = cmd.data.get('x', 0)
            y = cmd.data.get('y', 0)
            radius = cmd.data.get('radius', 50)
            print(f"         -> Would highlight at ({x}, {y}) with radius {radius}")
            
        elif cmd.cmd == 'show_text':
            text = cmd.data.get('text', '')
            duration = cmd.data.get('duration', 3)
            print(f"         -> Would show text '{text}' for {duration}s")
            
        elif cmd.cmd == 'clear':
            print(f"         -> Would clear overlay")
            
        else:
            print(f"         -> Unknown command")
            
    def _print_stats(self):
        """Print current stats"""
        producer_stats = self.producer.get_stats() if self.producer else {}
        consumer_stats = self.consumer.get_stats() if self.consumer else {}
        tcp_connected = self.commands.is_connected() if self.commands else False
        
        print("-" * 50)
        print(f"[Stats] Producer: {producer_stats.get('frames_produced', 0)} frames, "
              f"{producer_stats.get('fps', 0):.1f} FPS")
        print(f"[Stats] Consumer: {consumer_stats.get('frames_sent', 0)} frames, "
              f"{consumer_stats.get('fps', 0):.1f} FPS, "
              f"{consumer_stats.get('mbps', 0):.2f} Mbps")
        print(f"[Stats] TCP Connected: {tcp_connected}")
        
        # Show dropped frames (produced - sent)
        produced = producer_stats.get('frames_produced', 0)
        sent = consumer_stats.get('frames_sent', 0)
        dropped = produced - sent
        if dropped > 0:
            print(f"[Stats] Frames skipped: {dropped}")
        print("-" * 50)
        
    def run_main_loop(self):
        """
        Main loop - processes commands and displays status.
        In real Unity app, this would be Update().
        """
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
        # Discover laptop
        print("\n[1/2] Discovering laptop...")
        if not client.discover_and_connect(timeout=args.timeout):
            print("ERROR: Could not find laptop. Make sure laptop_server.py is running.")
            sys.exit(1)
            
        # Start streaming
        print("\n[2/2] Starting streaming...")
        client.start()
        
        # Run main loop
        client.run_main_loop()
        
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        client.stop()
        

if __name__ == '__main__':
    main()
