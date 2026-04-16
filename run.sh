#!/usr/bin/env bash
# Unified startup script for ARsenal Ravengers backend.
#
# Services started (in order):
#   1. Neo4j          — docker compose  (bolt :7687, browser :7474)
#   2. Knowledge API  — knowledge/main.py via uv  (FastAPI :8000)
#   3. People API     — ar-glasses/api/api.py      (FastAPI :5000)
#   4. Dashboard      — ar-glasses/dashboard.py    (Flask  :5050)  [foreground]
#
# Usage:
#   ./run.sh [--debug] [--skip-neo4j] [--skip-knowledge] [--skip-api] [-- <dashboard args>]
#
# --debug              Show live logs from all background services with [prefix] labels.
#                      Without this flag, background service output is suppressed.
# Any args after --    Forwarded to dashboard.py instead of the default --glasses.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AR_GLASSES_DIR="$SCRIPT_DIR/ar-glasses"
KNOWLEDGE_DIR="$SCRIPT_DIR/knowledge"

# ── Flags ─────────────────────────────────────────────────────────────────────

DEBUG=false
SKIP_NEO4J=false
SKIP_KNOWLEDGE=false
SKIP_API=false
DASHBOARD_ARGS=()
PARSING_DASHBOARD_ARGS=false

for arg in "$@"; do
  if $PARSING_DASHBOARD_ARGS; then
    DASHBOARD_ARGS+=("$arg")
  elif [[ "$arg" == "--" ]]; then
    PARSING_DASHBOARD_ARGS=true
  elif [[ "$arg" == "--debug" ]]; then
    DEBUG=true
  elif [[ "$arg" == "--skip-neo4j" ]]; then
    SKIP_NEO4J=true
  elif [[ "$arg" == "--skip-knowledge" ]]; then
    SKIP_KNOWLEDGE=true
  elif [[ "$arg" == "--skip-api" ]]; then
    SKIP_API=true
  else
    echo "Unknown flag: $arg" >&2
    echo "Usage: $0 [--debug] [--skip-neo4j] [--skip-knowledge] [--skip-api] [-- <dashboard args>]" >&2
    exit 1
  fi
done

# Default dashboard args if none provided after --
if [[ ${#DASHBOARD_ARGS[@]} -eq 0 ]]; then
  DASHBOARD_ARGS=(--glasses)
fi

# ── Environment ───────────────────────────────────────────────────────────────

if [[ -f "$AR_GLASSES_DIR/.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$AR_GLASSES_DIR/.env"
  set +a
fi

# ── Process tracking & cleanup ────────────────────────────────────────────────

PIDS=()

# Kill any process listening on a given port using netstat + taskkill (Windows-safe)
kill_port() {
  local port="$1"
  if ! (echo > /dev/tcp/localhost/"$port") 2>/dev/null; then
    return 0
  fi
  python -c "
import subprocess
port = '$port'
r = subprocess.run(['netstat', '-ano'], capture_output=True, text=True)
for line in r.stdout.splitlines():
    parts = line.split()
    if len(parts) >= 5 and parts[3] == 'LISTENING' and parts[1].endswith(':' + port) and parts[4].isdigit():
        subprocess.run(['taskkill', '/F', '/PID', parts[4]], capture_output=True)
" 2>/dev/null || true
}

cleanup() {
  echo ""
  echo "Shutting down..."
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  # Kill by port to catch any child processes that outlive their subshell
  kill_port 8765  # HUD WebSocket broadcast
  kill_port 5050  # Flask dashboard
  kill_port 5000  # People API
  kill_port 8000  # Knowledge API
  if ! $SKIP_NEO4J; then
    echo "Stopping Neo4j..."
    docker compose -f "$KNOWLEDGE_DIR/docker-compose.yml" down
  fi
  echo "Done."
}
trap cleanup EXIT INT TERM

# ── Helpers ───────────────────────────────────────────────────────────────────

wait_for_port() {
  local name="$1"
  local port="$2"
  local retries="${3:-30}"
  echo -n "Waiting for $name (:$port)"
  for _ in $(seq 1 "$retries"); do
    if (echo > /dev/tcp/localhost/"$port") 2>/dev/null; then
      echo " ready"
      return 0
    fi
    echo -n "."
    sleep 1
  done
  echo " TIMEOUT — $name may not have started correctly" >&2
  return 1
}

free_port() {
  local port="$1"
  if ! (echo > /dev/tcp/localhost/"$port") 2>/dev/null; then
    return 0  # already free
  fi
  echo "Port $port already in use — killing existing process..."
  kill_port "$port"
  for _ in $(seq 1 15); do
    sleep 1
    if ! (echo > /dev/tcp/localhost/"$port") 2>/dev/null; then
      return 0
    fi
  done
  echo "Warning: port $port may not have been released" >&2
}

# Route a background service's output: prefixed to terminal in debug mode, silent otherwise
service_output() {
  local prefix="$1"
  if $DEBUG; then
    sed "s/^/[$prefix] /"
  else
    cat > /dev/null
  fi
}

# ── 1. Neo4j ──────────────────────────────────────────────────────────────────

if ! $SKIP_NEO4J; then
  echo "Starting Neo4j..."
  if $DEBUG; then
    docker compose -f "$KNOWLEDGE_DIR/docker-compose.yml" up -d
  else
    docker compose -f "$KNOWLEDGE_DIR/docker-compose.yml" up -d --quiet-pull 2>/dev/null
  fi
  echo -n "Waiting for Neo4j (:7474)"
  for _ in $(seq 1 60); do
    if curl -sf http://localhost:7474/ > /dev/null 2>&1; then
      echo " ready"
      break
    fi
    echo -n "."
    sleep 2
  done
fi

# ── 2. Knowledge API (port 8000) ──────────────────────────────────────────────

if ! $SKIP_KNOWLEDGE; then
  free_port 8000
  echo "Starting knowledge API..."
  (cd "$KNOWLEDGE_DIR" && uv run uvicorn main:app --host 0.0.0.0 --port 8000 2>&1 \
    | service_output "knowledge") &
  PIDS+=($!)
  wait_for_port "knowledge API" 8000 20
fi

# ── 3. People API (port 5000) ─────────────────────────────────────────────────

if ! $SKIP_API; then
  free_port 5000
  echo "Starting people API..."
  (cd "$AR_GLASSES_DIR" && python -m api.api 2>&1 \
    | service_output "api") &
  PIDS+=($!)
  wait_for_port "people API" 5000 20
fi

# ── 4. Dashboard (foreground) ─────────────────────────────────────────────────

free_port 5050   # Flask dashboard
free_port 8765   # HUD WebSocket broadcast (HUD_BROADCAST_PORT default)

echo ""
echo "Starting dashboard (http://localhost:5050)..."
echo "  Args: ${DASHBOARD_ARGS[*]}"
$DEBUG && echo "  Debug mode: background service logs enabled"
echo ""

cd "$AR_GLASSES_DIR"
PYTHONIOENCODING=utf-8 \
HUD_BROADCAST_ENABLED=true \
RETRIEVAL_ENABLED=true \
SAVE_TO_MEMORY=true \
  python dashboard.py "${DASHBOARD_ARGS[@]}"
