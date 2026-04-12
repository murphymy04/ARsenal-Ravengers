---
name: zep-clear
description: Clear all data from the Zep knowledge graph (Neo4j). Deletes all nodes and relationships.
user_invocable: true
---

# Clear Zep Knowledge Graph

Delete all nodes and relationships from the Neo4j-backed Zep knowledge graph, then rebuild indices.

## Steps

Run this Python snippet from the `ar-glasses/` directory:

```python
import asyncio, os, sys
sys.path.insert(0, "/Users/timurtakhtarov/Programming/ARsenal-Ravengers/ar-glasses")
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "ravengers")
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from graphiti_core import Graphiti
from neo4j import AsyncGraphDatabase

async def main():
    driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    async with driver.session() as session:
        result = await session.run("MATCH (n) DETACH DELETE n")
        summary = await result.consume()
        print(f"Deleted {summary.counters.nodes_deleted} nodes, {summary.counters.relationships_deleted} relationships")
    await driver.close()

    client = Graphiti(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    await client.build_indices_and_constraints()
    await client.close()
    print("Indices rebuilt. Graph is empty and ready.")

asyncio.run(main())
```

## Behavior
- Run the script above via Bash.
- Report how many nodes and relationships were deleted.
- If Neo4j is not running, tell the user to start it with `docker compose -f ../knowledge/docker-compose.yml up -d`.
