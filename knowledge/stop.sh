#!/usr/bin/env bash
set -e

if lsof -ti TCP:8000 &>/dev/null; then
  echo "Killing process on :8000..."
  kill $(lsof -ti TCP:8000)
fi

echo "Stopping Docker Compose services..."
docker compose down

echo "Done."
