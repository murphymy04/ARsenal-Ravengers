# Editing Neo4j Entities & Embeddings in Graphiti

## 1. Renaming an Entity

Example: renaming "Person 10" to "Myles".

```cypher
-- Rename entity node
MATCH (n:Entity {name: 'Person 10'})
SET n.name = 'Myles',
    n.summary = replace(n.summary, 'Person 10', 'Myles')

-- Rename in relationship facts
MATCH (n:Entity {name: 'Myles'})-[r:RELATES_TO]-()
WHERE r.fact CONTAINS 'Person 10'
SET r.fact = replace(r.fact, 'Person 10', 'Myles')

-- Rename in episode content and names
MATCH (e:Episodic)
WHERE e.content CONTAINS 'Person 10' OR e.name CONTAINS 'Person 10'
SET e.content = replace(e.content, 'Person 10', 'Myles'),
    e.name = replace(e.name, 'Person 10', 'Myles')
```

## 2. Regenerating name_embedding After Rename

Graphiti uses OpenAI `text-embedding-3-small` (1024 dim). After renaming, the `name_embedding` vector is stale and must be regenerated.

```python
embedding = await graph.embedder.create(input_data=["Myles"])
await graph.driver.execute_query(
    "MATCH (n:Entity {uuid: $uuid}) SET n.name_embedding = $embedding",
    uuid="<entity-uuid>",
    embedding=embedding,
)
```

## 3. Neo4j Connection

- Bolt URI: `bolt://localhost:7687`
- Auth: `neo4j` / `ravengers`

## 4. Graphiti Internals

- Embedder: `graph.embedder` (instance of the OpenAI embedder)
- Entity node class: `graphiti_core.nodes.EntityNode`
- Embedding generation method: `EntityNode.generate_name_embedding(embedder)`
