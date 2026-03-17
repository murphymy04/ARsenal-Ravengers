#!/usr/bin/env bash
set -e

docker compose -f ../knowledge/docker-compose.yml up -d

SAVE_TO_MEMORY=true python main.py --mode live --camera 0
