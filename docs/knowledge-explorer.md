# Knowledge Explorer — Research & Plan

Goal: add a "Knowledge" tab to `dashboard.py` that (1) renders the Zep/Graphiti graph as an impressive-looking visualization and (2) lets you chat with it in natural language.

## TL;DR

- **No Zep-native JS viz library exists.** Graphiti stores in Neo4j, so we visualize the Neo4j backend but surface Zep-specific concepts (fact triples, bi-temporal validity windows, episodes) in the UI.
- **Recommended viz**: `3d-force-graph` (WebGL, vanilla JS, same author as `react-force-graph`). Looks cinematic, drops into a `<div>`, works in our Jinja/Flask dashboard without a React build step.
- **Recommended chat**: wrap `Graphiti.search()` (hybrid semantic + BM25 + graph traversal) → feed top facts to an LLM → stream conversational answer. Avoid raw text-to-Cypher; Graphiti's hybrid search already beats it and respects the temporal schema.

## 1. Visualization

### Library shortlist

| Library | Look | Fit | Notes |
|---|---|---|---|
| **3d-force-graph** / `react-force-graph-3d` | 3D WebGL, particle-flow links, bloom | **Top pick** for "impressive" | Vanilla JS variant drops into Flask with zero build. ThreeJS under the hood. |
| Cosmograph | 2D WebGL, scales to 1M+ nodes | Overkill (we'll have <10k facts) | Good if graph ever explodes. |
| NVL (Neo4j Visualization Library) | Bloom-style 2D | Safe/official | TypeScript, React wrapper, needs bundler. |
| neovis.js | 2D vis-network | Easiest | Fewer knobs; looks dated. |
| Cytoscape.js | 2D, many layouts | "Business" look | Less wow-factor. |

**Pick `3d-force-graph`** — it's the single biggest source of "impressive" for the least code.

### Zep-flavoured styling (what makes it a *Zep* viz, not a generic Neo4j one)

Graphiti's data model gives us dimensions generic Neo4j viz ignores:

- **Node types** (`Entity`, `Episode`, `Community`) → different shapes/icons. Faces from our pipeline map to `Entity` nodes — attach the person's thumbnail as the node sprite.
- **Edge validity windows** (`valid_at`, `invalid_at`) → color gradient:
  - active edges = bright, animated particles
  - superseded edges = dim/dashed
  - scrub a **time slider** to replay the graph's state at any past moment (this is the Zep "wow" moment — the bi-temporal model replayed visually)
- **Episode provenance** — clicking a fact edge highlights the `Episode` node it came from; show the raw transcript snippet in a side panel.
- **Per-person subgraphs** — our pipeline already has `person_id` / labeled names. Add a cluster focus: pick "Peter Nguyen" from a dropdown → camera zooms to their ego network.

### Data path

```
Neo4j (bolt://localhost:7687)
  ├─ cypher query (nodes + edges with valid_at/invalid_at)
  └─ Flask endpoint `/api/knowledge/graph` → JSON {nodes, links}
                                                 │
                            3d-force-graph in templates/knowledge.html
```

Keep it dead simple: one Cypher on load, incremental fetches when the user expands a node. Cache for 30s server-side.

## 2. Chat

### Approach comparison

| Approach | Pros | Cons | Verdict |
|---|---|---|---|
| **Graphiti `search()` + LLM summarizer** | Uses hybrid search (semantic + BM25 + graph traversal), temporal-aware, already in our stack (`graphiti-core`) | Doesn't do arbitrary aggregations ("how many times did we meet Peter?") | **Pick this.** Covers 95% of questions. |
| LangChain `GraphCypherQAChain` | Can answer aggregations | Text-to-Cypher has to learn Graphiti's opinionated schema; error-prone; duplicates what Graphiti already does well | Skip. |
| NeoConverse (Neo4j Labs) | Turnkey NL interface | Heavyweight, expects enterprise data model | Skip. |

### Shape

```python
async def ask(question: str) -> str:
    results = await graphiti.search(question, num_results=10)
    facts = [r.fact for r in results]

    system = "Answer using only the facts below. Cite fact IDs inline as [1], [2]."
    user = f"Question: {question}\n\nFacts:\n" + "\n".join(
        f"[{i}] {f}" for i, f in enumerate(facts, 1)
    )
    return await openai_chat(system, user)  # gpt-4o-mini
```

Front-end: a chat panel docked beside the graph. Each assistant reply has clickable `[1]`, `[2]` citations that select the corresponding edge in `3d-force-graph` and pan the camera to it. That tight coupling is what sells the demo.

### Nice extras (cheap)

- **Suggested questions** on empty state: "Who have I talked to this week?", "What did Peter say about his game?", "When did I first meet Myles?"
- **Temporal follow-ups** — pass the time slider's current value as a filter into `search()` so "what did we know at time T" actually works.
- **Streaming** — OpenAI streaming → Server-Sent Events → incremental typing in the UI.

## 3. Implementation sketch

```
ar-glasses/
  pages/knowledge.py          # Flask blueprint: /knowledge tab + /api/knowledge/*
  templates/knowledge.html    # 3d-force-graph + chat panel (vanilla JS, HTMX for chat)
  static/knowledge.js         # graph bootstrap, node click → fetch ego, chat → SSE
  pipeline/knowledge_query.py # ask() wrapper around graphiti.search + OpenAI
```

Dependencies: none new on Python side (graphiti-core + openai already present). Front-end: one `<script src="//unpkg.com/3d-force-graph">` tag. No build pipeline.

## Sources

- [getzep/graphiti](https://github.com/getzep/graphiti) — search API, hybrid retrieval, temporal model
- [Graphiti quick-start](https://help.getzep.com/graphiti/getting-started/quick-start)
- [3d-force-graph / react-force-graph](https://github.com/vasturiano/react-force-graph)
- [Neo4j Visualization Library](https://neo4j.com/docs/nvl/current/)
- [15 Neo4j viz tools roundup](https://neo4j.com/blog/graph-visualization/neo4j-graph-visualization-tools/)
- [NeoConverse — NL → Cypher](https://neo4j.com/labs/genai-ecosystem/neoconverse/)
- [LangChain GraphCypherQAChain](https://neo4j.com/labs/genai-ecosystem/langchain/)
