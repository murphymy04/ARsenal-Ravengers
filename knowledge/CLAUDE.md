# ARsenal Ravengers — Knowledge Subproject

## What this is

Retrieval pipeline proof-of-concept for the smart glasses app.
Demonstrates: face recognised → knowledge graph query → context prompt for wearer.

Full system design: `../docs/high_level.md`

## Stack

| Layer | Tech |
|---|---|
| Knowledge graph | Zep `graphiti-core` (open-source, self-hosted) |
| Graph backend | Neo4j 5 (Docker) |
| LLM (graph extraction + post-processing) | OpenAI (`gpt-4o-mini`) |
| Runtime | Python 3.10, uv |

## Files

```
knowledge/
  docker-compose.yml  — Neo4j (bolt :7687, browser :7474, auth neo4j/ravengers)
  seed.py             — adds Will/Peter Nguyen conference conversation to graphiti
  retrieve.py         — face trigger → graph query → LLM prompt → assertions
  pyproject.toml      — uv project, deps: graphiti-core, openai, python-dotenv
  .env                — OPENAI_API_KEY + Neo4j connection (gitignored)
```

## How to run

```bash
docker compose up -d          # start Neo4j
uv sync                       # install deps
python seed.py                # seed Peter Nguyen's episode (run once)
python retrieve.py            # run retrieval pipeline + assertions
```

## Test scenario

- **Wearer**: Will
- **Recognised person**: Peter Nguyen
- **Seeded episode**: conference conversation — starts on AI coding tools (Cursor/Claude),
  flows into Peter's game from Chillenium game jam (platformer, card-based boss mechanics),
  Steam launch planned for summer
- **Expected output**: prompt for Will referencing Peter's game/Steam/Chillenium

## Key API notes

- `Graphiti(uri, user, password)` — main client
- `await graphiti.build_indices_and_constraints()` — run once on fresh DB
- `await graphiti.add_episode(name, episode_body, source, source_description, reference_time)` — add episode
- `await graphiti.search(query)` — returns list of results, each with `.fact: str`
