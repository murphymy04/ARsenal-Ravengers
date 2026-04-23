#!/usr/bin/env bash
set -uo pipefail

RELAY_MAC_BIN="/Users/timurtakhtarov/Programming/ARsenal-Ravengers/ar-glasses/tunnel/relay-mac"
RELAY_MAC_LOG="/tmp/relay-mac.log"
GLASSES_RELAY_PATH="/data/local/tmp/relay"
GLASSES_RELAY_LOG="/data/local/tmp/relay.log"
POLL_INTERVAL_SECONDS=1
MAC_TUNNEL_PORTS=(55000 55003 55008)
PORT_FREE_TIMEOUT_SECONDS=5

relay_mac_pid=""
tunnel_state="down"

if [[ ! -x "$RELAY_MAC_BIN" ]]; then
    echo "error: relay-mac binary not found or not executable at $RELAY_MAC_BIN" >&2
    echo "build it with: cd tunnel && go build -o relay-mac ." >&2
    exit 1
fi

if ! command -v adb >/dev/null 2>&1; then
    echo "error: adb not found in PATH" >&2
    exit 1
fi

log() {
    echo "[tunneld $(date +%H:%M:%S)] $*"
}

wait_until_port_free() {
    local port=$1
    local deadline=$((SECONDS + PORT_FREE_TIMEOUT_SECONDS))
    while (( SECONDS < deadline )); do
        if ! lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
            return 0
        fi
        sleep 0.2
    done
    log "warning: port $port still in use after ${PORT_FREE_TIMEOUT_SECONDS}s"
    return 1
}

bring_tunnel_up() {
    log "glasses attached, bringing tunnel up"

    pkill -9 -f "$RELAY_MAC_BIN" 2>/dev/null || true
    adb shell "pkill -9 -f $GLASSES_RELAY_PATH" >/dev/null 2>&1 || true
    adb reverse --remove-all >/dev/null 2>&1 || true

    for port in "${MAC_TUNNEL_PORTS[@]}"; do
        wait_until_port_free "$port"
    done

    adb reverse tcp:55000 tcp:55000 >/dev/null
    adb reverse tcp:55003 tcp:55003 >/dev/null
    adb reverse tcp:55008 tcp:55008 >/dev/null

    adb shell "nohup $GLASSES_RELAY_PATH -mode=glasses > $GLASSES_RELAY_LOG 2>&1 &" >/dev/null

    "$RELAY_MAC_BIN" -mode=mac >"$RELAY_MAC_LOG" 2>&1 &
    relay_mac_pid=$!

    sleep 0.5
    if ! adb shell "pgrep -f $GLASSES_RELAY_PATH" >/dev/null 2>&1; then
        log "error: glasses-side relay failed to start, last log lines:"
        adb shell "tail -5 $GLASSES_RELAY_LOG" 2>&1 | while read -r line; do log "  $line"; done
        kill "$relay_mac_pid" 2>/dev/null || true
        wait "$relay_mac_pid" 2>/dev/null || true
        relay_mac_pid=""
        tunnel_state="down"
        return 1
    fi

    tunnel_state="up"
    log "tunnel up (relay-mac pid=$relay_mac_pid, log=$RELAY_MAC_LOG)"
}

tear_tunnel_down() {
    log "glasses detached, tearing down"

    if [[ -n "$relay_mac_pid" ]] && kill -0 "$relay_mac_pid" 2>/dev/null; then
        kill "$relay_mac_pid" 2>/dev/null || true
        wait "$relay_mac_pid" 2>/dev/null || true
    fi
    relay_mac_pid=""

    adb shell "pkill -f $GLASSES_RELAY_PATH" >/dev/null 2>&1 || true

    tunnel_state="down"
    log "tunnel down"
}

shutdown() {
    log "received signal, shutting down"
    if [[ "$tunnel_state" == "up" ]]; then
        tear_tunnel_down
    fi
    exit 0
}

trap shutdown INT TERM

log "watching adb device state (poll every ${POLL_INTERVAL_SECONDS}s)"

while true; do
    device_state="$(adb get-state 2>/dev/null || echo "unknown")"

    if [[ "$device_state" == "device" && "$tunnel_state" == "down" ]]; then
        bring_tunnel_up
    elif [[ "$device_state" != "device" && "$tunnel_state" == "up" ]]; then
        tear_tunnel_down
    elif [[ "$tunnel_state" == "up" && -n "$relay_mac_pid" ]] && ! kill -0 "$relay_mac_pid" 2>/dev/null; then
        log "relay-mac died unexpectedly (check $RELAY_MAC_LOG), tearing down"
        relay_mac_pid=""
        tear_tunnel_down
    fi

    sleep "$POLL_INTERVAL_SECONDS"
done
