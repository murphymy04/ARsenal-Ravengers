# Knowledge

Retrieval pipeline and HTTP API for the ArgusEye smart glasses app. Stores conversations in a knowledge graph and surfaces context about recognised people on demand.

## Stack

- [graphiti-core](https://github.com/getzep/graphiti) — knowledge graph (self-hosted)
- Neo4j 5 — graph backend (Docker)
- OpenAI `gpt-4o-mini` — episode extraction + context post-processing
- FastAPI + Uvicorn — HTTP API

## Prerequisites

- Docker Desktop running
- `uv` installed
- OpenAI API key in `.env`

## Setup

```bash
# 1. Add your OpenAI key
echo "OPENAI_API_KEY=sk-..." >> .env

# 2. Start Neo4j
docker compose up -d

# 3. Install dependencies
uv sync
```

---

## 1. Seed the knowledge graph (run once)

Adds the Will/Peter Nguyen conference conversation to the graph so there's something to query. This is just for test flow to see how this works with an empty db.

```bash
python seed.py
```

## 2. Run the retrieval PoC

Simulates a face recognition trigger for Peter Nguyen, queries the graph for relevant facts, generates a conversation prompt for Will, and asserts the output is correct. You can read the full conversation in seed.py.

```bash
python retrieve.py
```

Expected output:

```
[face recognised] → Peter Nguyen
Querying knowledge graph...

Raw facts (4):
- Peter Nguyen aims to publish Chillenium on Steam by summer 2025.
- Peter Nguyen has been using Cursor as an AI coding tool for a few months.
- Peter Nguyen built the game Chillenium during a 48-hour game jam last month.
- Peter Nguyen uses Claude under the hood as part of his AI coding tools.

──────────────────────────────────────────────────
[glasses display] → Hey Peter, how's the progress on Chillenium going since that intense game jam last month?
──────────────────────────────────────────────────

✓ 4 facts retrieved from graph
✓ Prompt references known topics: ['game', 'chillenium', 'jam']
✓ All assertions passed
```

## 3. Run the API server

```bash
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The server exposes two endpoints for the smart glasses system to interact with the knowledge graph programmatically. `POST /conversations` ingests a structured conversation (list of speaker/text messages with a date) and stores it as an episode in the graph. `GET /retrieve/{name}` accepts a recognised person's name, queries the graph for relevant facts about them, and returns those facts along with an LLM-generated conversation starter the wearer can use.

Interactive API docs (Swagger UI): http://localhost:8000/docs
