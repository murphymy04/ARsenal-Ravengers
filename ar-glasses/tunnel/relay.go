package main

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"os"
	"sync"
	"time"
)

const (
	discoveryPort  = 5002
	videoPort      = 5000
	commandPort    = 5001
	audioPort      = 5003
	hudPort        = 8765
	tunnelPort     = 55000
	audioBridgePort = 55003
	hudBridgePort   = 55008
	maxPacketBytes = 65536
)

type discoverResponse struct {
	Type                string `json:"type"`
	VideoPort           int    `json:"video_port"`
	CommandPort         int    `json:"command_port"`
	AudioPort           int    `json:"audio_port"`
	HudBroadcastEnabled bool   `json:"hud_broadcast_enabled"`
	HudBroadcastPort    int    `json:"hud_broadcast_port"`
}

func runGlasses() {
	go discoveryResponder()
	go videoRelay()
	go audioRelayGlasses()
	go hudRelayGlasses()
	select {}
}

func discoveryResponder() {
	conn, err := net.ListenUDP("udp", &net.UDPAddr{Port: discoveryPort})
	if err != nil {
		log.Fatalf("discovery bind :%d: %v", discoveryPort, err)
	}
	log.Printf("discovery listening on UDP :%d", discoveryPort)

	resp, _ := json.Marshal(discoverResponse{
		Type:                "discover_response",
		VideoPort:           videoPort,
		CommandPort:         commandPort,
		AudioPort:           audioPort,
		HudBroadcastEnabled: true,
		HudBroadcastPort:    hudPort,
	})

	buf := make([]byte, 4096)
	for {
		n, src, err := conn.ReadFromUDP(buf)
		if err != nil {
			log.Printf("discovery read: %v", err)
			continue
		}
		var req map[string]any
		if err := json.Unmarshal(buf[:n], &req); err != nil {
			continue
		}
		if req["type"] != "discover" {
			continue
		}
		log.Printf("discovery req from %s name=%v", src, req["name"])
		if _, err := conn.WriteToUDP(resp, src); err != nil {
			log.Printf("discovery resp: %v", err)
		}
	}
}

func videoRelay() {
	udpConn, err := net.ListenUDP("udp", &net.UDPAddr{Port: videoPort})
	if err != nil {
		log.Fatalf("video bind :%d: %v", videoPort, err)
	}
	if err := udpConn.SetReadBuffer(4 * 1024 * 1024); err != nil {
		log.Printf("video SetReadBuffer: %v", err)
	}
	log.Printf("video listening on UDP :%d", videoPort)

	var (
		mu      sync.Mutex
		writeMu sync.Mutex
		tcpConn net.Conn
	)

	dial := func() {
		for {
			c, err := net.DialTimeout("tcp", fmt.Sprintf("127.0.0.1:%d", tunnelPort), 2*time.Second)
			if err == nil {
				if tc, ok := c.(*net.TCPConn); ok {
					_ = tc.SetNoDelay(true)
					_ = tc.SetWriteBuffer(2 * 1024 * 1024)
				}
				mu.Lock()
				tcpConn = c
				mu.Unlock()
				log.Printf("tunnel connected to 127.0.0.1:%d", tunnelPort)
				return
			}
			log.Printf("tunnel dial failed: %v (retry 1s)", err)
			time.Sleep(time.Second)
		}
	}
	dial()

	go func() {
		hb := make([]byte, 4)
		binary.LittleEndian.PutUint32(hb, 0)
		t := time.NewTicker(time.Second)
		defer t.Stop()
		for range t.C {
			mu.Lock()
			c := tcpConn
			mu.Unlock()
			if c == nil {
				continue
			}
			writeMu.Lock()
			_, err := c.Write(hb)
			writeMu.Unlock()
			if err != nil {
				log.Printf("tunnel heartbeat: %v", err)
				c.Close()
				mu.Lock()
				if tcpConn == c {
					tcpConn = nil
				}
				mu.Unlock()
				go dial()
			}
		}
	}()

	buf := make([]byte, maxPacketBytes)
	hdr := make([]byte, 4)
	var forwarded, dropped uint64
	last := time.Now()

	for {
		n, _, err := udpConn.ReadFromUDP(buf)
		if err != nil {
			log.Printf("video read: %v", err)
			continue
		}

		mu.Lock()
		c := tcpConn
		mu.Unlock()
		if c == nil {
			dropped++
			continue
		}

		binary.LittleEndian.PutUint32(hdr, uint32(n))
		writeMu.Lock()
		var werr error
		if _, werr = c.Write(hdr); werr == nil {
			_, werr = c.Write(buf[:n])
		}
		writeMu.Unlock()
		if werr != nil {
			log.Printf("tunnel write: %v", werr)
			c.Close()
			mu.Lock()
			if tcpConn == c {
				tcpConn = nil
			}
			mu.Unlock()
			go dial()
			dropped++
			continue
		}
		forwarded++

		if time.Since(last) > 5*time.Second {
			log.Printf("video relay forwarded=%d dropped=%d", forwarded, dropped)
			last = time.Now()
		}
	}
}

func runMac() {
	go bridgeAcceptor("audio", audioBridgePort, audioPort)
	go bridgeAcceptor("hud", hudBridgePort, hudPort)

	listener, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", tunnelPort))
	if err != nil {
		log.Fatalf("tunnel listen :%d: %v", tunnelPort, err)
	}
	log.Printf("tunnel listening on TCP 127.0.0.1:%d", tunnelPort)

	udpDest, _ := net.ResolveUDPAddr("udp", fmt.Sprintf("127.0.0.1:%d", videoPort))
	udpOut, err := net.DialUDP("udp", nil, udpDest)
	if err != nil {
		log.Fatalf("video udp dial: %v", err)
	}
	log.Printf("video relay -> UDP 127.0.0.1:%d", videoPort)

	for {
		c, err := listener.Accept()
		if err != nil {
			log.Printf("accept: %v", err)
			continue
		}
		log.Printf("tunnel accepted from %s", c.RemoteAddr())
		go handleTunnel(c, udpOut)
	}
}

func audioRelayGlasses() {
	tcpBridgeListener("audio", audioPort, audioBridgePort)
}

func hudRelayGlasses() {
	tcpBridgeListener("hud", hudPort, hudBridgePort)
}

func tcpBridgeListener(name string, listenPort, tunnelDialPort int) {
	listener, err := net.Listen("tcp", fmt.Sprintf("0.0.0.0:%d", listenPort))
	if err != nil {
		log.Fatalf("%s bridge listen :%d: %v", name, listenPort, err)
	}
	log.Printf("%s bridge listening on TCP 0.0.0.0:%d -> tunnel 127.0.0.1:%d",
		name, listenPort, tunnelDialPort)
	for {
		client, err := listener.Accept()
		if err != nil {
			log.Printf("%s bridge accept: %v", name, err)
			continue
		}
		go handleBridgeClient(name, client, tunnelDialPort)
	}
}

func handleBridgeClient(name string, client net.Conn, tunnelDialPort int) {
	defer client.Close()
	log.Printf("%s bridge accepted client %s", name, client.RemoteAddr())

	upstream, err := net.DialTimeout("tcp",
		fmt.Sprintf("127.0.0.1:%d", tunnelDialPort), 5*time.Second)
	if err != nil {
		log.Printf("%s bridge dial upstream :%d: %v", name, tunnelDialPort, err)
		return
	}
	defer upstream.Close()
	log.Printf("%s bridge connected upstream", name)

	if tc, ok := client.(*net.TCPConn); ok {
		_ = tc.SetNoDelay(true)
	}
	if tc, ok := upstream.(*net.TCPConn); ok {
		_ = tc.SetNoDelay(true)
	}

	pipeBoth(name, client, upstream)
}

func bridgeAcceptor(name string, listenPort, dialPort int) {
	listener, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", listenPort))
	if err != nil {
		log.Fatalf("%s acceptor listen :%d: %v", name, listenPort, err)
	}
	log.Printf("%s bridge acceptor listening on TCP 127.0.0.1:%d -> 127.0.0.1:%d",
		name, listenPort, dialPort)
	for {
		incoming, err := listener.Accept()
		if err != nil {
			log.Printf("%s acceptor accept: %v", name, err)
			continue
		}
		go handleBridgeAcceptor(name, incoming, dialPort)
	}
}

func handleBridgeAcceptor(name string, incoming net.Conn, dialPort int) {
	defer incoming.Close()
	log.Printf("%s acceptor got tunnel connection from %s", name, incoming.RemoteAddr())

	deadline := time.Now().Add(15 * time.Second)
	var local net.Conn
	var err error
	for time.Now().Before(deadline) {
		local, err = net.DialTimeout("tcp",
			fmt.Sprintf("127.0.0.1:%d", dialPort), 1*time.Second)
		if err == nil {
			break
		}
		time.Sleep(500 * time.Millisecond)
	}
	if err != nil {
		log.Printf("%s acceptor dial local :%d: %v", name, dialPort, err)
		return
	}
	defer local.Close()
	log.Printf("%s acceptor connected local :%d", name, dialPort)

	if tc, ok := incoming.(*net.TCPConn); ok {
		_ = tc.SetNoDelay(true)
	}
	if tc, ok := local.(*net.TCPConn); ok {
		_ = tc.SetNoDelay(true)
	}

	pipeBoth(name, incoming, local)
}

func pipeBoth(name string, a, b net.Conn) {
	done := make(chan struct{}, 2)
	go func() {
		_, err := io.Copy(b, a)
		if err != nil && !errors.Is(err, io.EOF) {
			log.Printf("%s pipe a->b: %v", name, err)
		}
		done <- struct{}{}
	}()
	go func() {
		_, err := io.Copy(a, b)
		if err != nil && !errors.Is(err, io.EOF) {
			log.Printf("%s pipe b->a: %v", name, err)
		}
		done <- struct{}{}
	}()
	<-done
	log.Printf("%s bridge connection closed", name)
}

func handleTunnel(c net.Conn, udpOut *net.UDPConn) {
	defer c.Close()
	if tc, ok := c.(*net.TCPConn); ok {
		_ = tc.SetNoDelay(true)
	}
	hdr := make([]byte, 4)
	var forwarded uint64
	last := time.Now()
	for {
		_ = c.SetReadDeadline(time.Now().Add(5 * time.Second))
		if _, err := io.ReadFull(c, hdr); err != nil {
			if os.IsTimeout(err) || errors.Is(err, os.ErrDeadlineExceeded) {
				log.Printf("tunnel idle 5s, closing")
				return
			}
			if err != io.EOF {
				log.Printf("tunnel read hdr: %v", err)
			}
			log.Printf("tunnel closed; forwarded=%d", forwarded)
			return
		}
		n := binary.LittleEndian.Uint32(hdr)
		if n == 0 {
			continue
		}
		if n > maxPacketBytes {
			log.Printf("tunnel: bad packet len %d", n)
			return
		}
		buf := make([]byte, n)
		if _, err := io.ReadFull(c, buf); err != nil {
			log.Printf("tunnel read payload: %v", err)
			return
		}
		if _, err := udpOut.Write(buf); err != nil {
			log.Printf("video udp write: %v", err)
		}
		forwarded++
		if time.Since(last) > 5*time.Second {
			log.Printf("video relay forwarded=%d", forwarded)
			last = time.Now()
		}
	}
}

func main() {
	mode := flag.String("mode", "", "glasses or mac")
	flag.Parse()
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)
	switch *mode {
	case "glasses":
		runGlasses()
	case "mac":
		runMac()
	default:
		fmt.Fprintln(os.Stderr, "use -mode=glasses or -mode=mac")
		os.Exit(1)
	}
}
