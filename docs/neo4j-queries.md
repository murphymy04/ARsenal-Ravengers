# Neo4j / Knowledge Graph Cypher Reference

Useful queries for inspecting and clearing the knowledge graph during development.

Open the Neo4j browser at [http://localhost:7474](http://localhost:7474) (bolt: `bolt://localhost:7687`) and log in with user `neo4j`, password `ravengers`.

```cypher
-- Everything
MATCH (n) RETURN n

-- All entities (no episodes)
MATCH (n) WHERE NOT "Episodic" IN labels(n) RETURN n

-- All fact edges (the useful stuff)
MATCH (a)-[r]->(b) WHERE r.fact IS NOT NULL RETURN a, r, b

-- Facts as text
MATCH (a)-[r]->(b) WHERE r.fact IS NOT NULL
RETURN a.name AS from, type(r) AS rel, b.name AS to, r.fact AS fact

-- All edges including MENTIONS
MATCH (a)-[r]->(b) RETURN a.name, type(r), b.name, r.fact

-- Episodes with metadata
MATCH (e:Episodic) RETURN e.name, e.source_description, e.content

-- Entities by type
MATCH (n:Person) RETURN n.name, n.summary
MATCH (n:Product) RETURN n.name, n.summary
MATCH (n:Topic) RETURN n.name, n.summary

-- Everything about a specific person
MATCH (n {name: "Myles Murphy"})-[r]-(m) RETURN n, r, m

-- Node and relationship counts
MATCH (n) RETURN labels(n) AS labels, count(n) AS count
MATCH ()-[r]->() RETURN type(r) AS type, count(r) AS count

-- Nuke it
MATCH (n) DETACH DELETE n
```
