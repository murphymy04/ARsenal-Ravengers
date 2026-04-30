# 0002 — Knowledge graph (Zep Graphiti over Neo4j)

**Status:** accepted
**Date:** 2026-02-22

## Context

After each conversation we have a transcript with speaker labels (`Wearer:` / `Other:`). We need to turn that into queryable structured knowledge so that on the next encounter with that same person we can answer "what did we last talk about?" and "what's the most useful thing to ask?".

Constraints:

- The schema cannot be hand-curated. New conversation topics arrive constantly; an LLM has to extract entities and edges on the fly.
- We need **temporal awareness**: facts have a `valid_at` and may later be invalidated. "Peter is at Capital One" → "Peter left Capital One" should not produce two contradictory live facts.
- We need fast retrieval of "tell me everything about Person X" within ~1–2 seconds of recognition firing.
- Storage is local (single-user prototype). No multi-tenant requirement.

## Options considered

| Option | Pros | Cons |
|---|---|---|
| **Zep Graphiti on Neo4j** | Built for exactly this use case. Hybrid search (semantic + BM25 + graph traversal). Bi-temporal model with `valid_at`/`invalid_at` baked in. Custom entity types via Pydantic. Active project, OpenAI-backed extraction. | Couples us to Graphiti's schema conventions. LLM cost per `add_episode` call is non-trivial. Internals shift between releases. |
| **Plain Neo4j with hand-rolled extraction** | Full control over schema. No third-party LLM coupling. | We re-build everything Graphiti gives us: entity dedup, edge invalidation, hybrid search, temporal queries. Months of work for a senior-design project. |
| **Vector DB only (pgvector / Qdrant) on transcripts** | Simple. One-shot embedding of each transcript, retrieve top-k by similarity. | Loses entity structure entirely. Cannot answer "what does Peter promise to send me?" — only "find similar transcripts." Loses commitment tracking. |
| **LangChain `GraphCypherQAChain`** | NL → Cypher works for ad-hoc questions. | Text-to-Cypher has to learn the schema; brittle in practice. Doesn't supersede Graphiti — it would sit on top. |
| **OpenAI assistants API memory** | Hosted, no infra. | Closed schema, no temporal model, no graph queries, locks us into one vendor's memory abstraction. |

## Decision

Use **Graphiti** (Python `graphiti-core`) backed by **Neo4j 5.26** in Docker. Define four custom entity types (`Person`, `Product`, `Topic`, `Commitment`) and a permissive edge map ([pipeline/knowledge.py](../../ar-glasses/pipeline/knowledge.py)). On retrieval, call `Graphiti.search()` for top-3 facts plus `retrieve_episodes(last_n=50)` filtered by name for the most recent conversation; format with a small Groq-hosted Qwen model into the `last_spoke_about` / `ask_about` HUD payload ([pipeline/retrieval.py](../../ar-glasses/pipeline/retrieval.py)).

## Consequences

**Enables:**
- Two-line write path from transcript to graph: `save_to_memory(transcript, other_name=name)` ([pipeline/knowledge.py:258](../../ar-glasses/pipeline/knowledge.py)).
- Temporal queries — we can ask "what did we know about Peter as of three weeks ago?" by filtering on `valid_at`.
- Custom commitment tracking via the `Commitment` entity. Promises made in one conversation surface as `ask_about` reminders the next time.

**Costs:**
- Every conversation flush triggers an OpenAI `gpt-4.1` call (configurable via `GRAPHITI_LLM_MODEL`). Cost scales with conversation length. Mitigated by only flushing labeled people — auto-clusters stay in SQLite until a name lands.
- Initial Neo4j boot takes 20–30 s; the [run.sh](../../run.sh) startup is dominated by waiting on `:7474`.
- The Graphiti schema (`Episodic` nodes, `RELATES_TO`/`MENTIONS` edges, `name_embedding` properties) is opinionated. Renaming entities requires regenerating `name_embedding` (see [pipeline/NOTES.md](../../ar-glasses/pipeline/NOTES.md)).
- Speaker attribution is unreliable, so the extraction prompt explicitly tells the LLM to ignore speaker tags and infer attribution from content. See `build_extraction_instructions` in [pipeline/knowledge.py:79](../../ar-glasses/pipeline/knowledge.py).

**Locks us out of:**
- Lightweight deployment without Docker/Neo4j. The graph is the central artifact; replacing Neo4j means rebuilding around a different store.
- Schemaless free-text memory. We have a structured graph by construction, which makes some questions ("summarize this person's vibe") harder than a pure RAG approach would.
