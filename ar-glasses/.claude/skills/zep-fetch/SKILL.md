---
name: zep-fetch
description: Fetch facts about a person from the Zep knowledge graph (Graphiti search). Usage - /zep-fetch Person Name
user_invocable: true
---

# Fetch Zep Facts for a Person

Query the Graphiti knowledge graph for facts about a specific person.

The person name comes from `$ARGUMENTS`. If no argument is provided, ask the user who to look up.

## Steps

Run this Python snippet from the `ar-glasses/` directory, substituting the person name:

```python
import asyncio, os, sys
sys.path.insert(0, "/Users/timurtakhtarov/Programming/ARsenal-Ravengers/ar-glasses")
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "ravengers")
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from graphiti_core import Graphiti

PERSON = "$ARGUMENTS"

async def main():
    client = Graphiti(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    results = await client.search(f"{PERSON} projects work interests")
    print(f"{len(results)} facts about {PERSON}:")
    for r in results:
        print(f"  - {r.fact}")
    await client.close()

asyncio.run(main())
```

## Behavior
- Replace `$ARGUMENTS` with the person name the user provided.
- Run the script via Bash.
- Present the facts clearly. If no results, tell the user the graph may be empty or not seeded for that person.
- If Neo4j is not running, tell the user to start it with `docker compose -f ../knowledge/docker-compose.yml up -d`.
