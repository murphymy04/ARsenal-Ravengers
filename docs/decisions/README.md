# Architecture Decision Records (ADRs)

Each file here records a single non-obvious choice: what we picked, what we considered, and why. The point is so that someone joining the project six months from now can tell whether a decision is still load-bearing or just historical.

## Template

```markdown
# NNNN — <decision title>

**Status:** accepted | superseded by NNNN | deprecated
**Date:** YYYY-MM-DD

## Context
What problem are we solving? What constraints exist? What did we already try?

## Options considered
- Option A — pros / cons
- Option B — pros / cons
- Option C — pros / cons

## Decision
The choice, in one sentence.

## Consequences
What this enables, what it costs, what it locks us out of.
```

## Index

- [0001 — Face recognition pipeline (MediaPipe + EdgeFace)](0001-face-pipeline.md)
- [0002 — Knowledge graph (Zep Graphiti over Neo4j)](0002-knowledge-graph.md)
- [0003 — Transcription (Groq-hosted Whisper)](0003-transcription.md)
