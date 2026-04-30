# Operations

Day-to-day ops for ArgusEye: backups, resets, recovery, key rotation. The system is single-user and local, so most of this is "what to do when something gets weird," not formal SRE.

## 1. Where state lives

| Store | Location | Purpose |
|---|---|---|
| SQLite | `ar-glasses/data/people.db` | People, embeddings, interactions. Survives `git pull`. Gitignored. |
| Neo4j | Docker volume `neo4j_data` | Knowledge graph (entities, edges, episodes). Persists across `docker compose down`; nuked by `docker compose down -v`. |
| API keys | `ar-glasses/.env` | `OPENAI_API_KEY`, `GROQ_API_KEY`. Gitignored. |
| Cached embeddings / debug artifacts | `ar-glasses/data/`, `ar-glasses/cache/` | Regenerable. Safe to delete. |

If you've never run the pipeline, `data/people.db` and the `neo4j_data` volume don't exist yet — they're created on first launch.

## 2. Backups

### SQLite

```bash
# One-shot copy (safe while the API is running thanks to WAL mode)
cp ar-glasses/data/people.db ar-glasses/data/people.db.bak

# Or use SQLite's online backup
sqlite3 ar-glasses/data/people.db ".backup 'ar-glasses/data/people.db.bak'"
```

### Neo4j

Easiest path is a graph dump via the running container:

```bash
# Stop the DB cleanly first
docker compose -f ar-glasses/docker-compose.yml down

# Dump
docker run --rm -v ar-glasses_neo4j_data:/data -v "$PWD":/backup \
  neo4j:5.26 neo4j-admin database dump neo4j --to-path=/backup

# Restart
docker compose -f ar-glasses/docker-compose.yml up -d
```

The dump file lands at `./neo4j.dump`. Restore with `neo4j-admin database load neo4j --from-path=/backup --overwrite-destination=true`.

For ad-hoc snapshots during development you can also just `tar` the volume:

```bash
docker run --rm -v ar-glasses_neo4j_data:/data -v "$PWD":/backup \
  alpine tar czf /backup/neo4j_data.tar.gz -C /data .
```

## 3. Resets

### Wipe just the knowledge graph (keep face DB)

```bash
# From cypher-shell or the Neo4j browser at :7474
MATCH (n) DETACH DELETE n
```

Or use the bundled skill: `/zep-clear` (defined in `ar-glasses/.claude/skills/zep-clear/`).

The next labeled-person flush rebuilds the schema (`build_indices_and_constraints`) automatically.

### Wipe just the face DB (keep graph)

```bash
rm ar-glasses/data/people.db
```

Re-runs of the pipeline will recreate the schema and start enrolling fresh.

### Full reset

```bash
# Stop everything
./run.sh   # then Ctrl+C, or:
docker compose -f ar-glasses/docker-compose.yml down -v   # -v deletes the volume

# Remove SQLite + cached artifacts
rm -rf ar-glasses/data/people.db ar-glasses/cache/

# Restart
./run.sh
```

## 4. Resetting a single session

To clear the current conversation buffer without restarting (e.g. you walked into another room mid-recording):

- Stop the dashboard (`Ctrl+C` in the foreground terminal).
- Restart with `./run.sh`. Pipeline state (rolling buffers, RMS adaptation) is in-memory and resets cleanly.

To roll back a single bad knowledge-graph episode without nuking the graph:

```cypher
// Find episodes by date and inspect before deleting
MATCH (e:Episodic) WHERE e.valid_at >= datetime('2026-04-29T00:00:00')
RETURN e.name, e.valid_at, e.content
ORDER BY e.valid_at DESC
LIMIT 20

// Once you've identified the bad one, detach-delete it.
// Note: this leaves orphaned facts derived from this episode — if they bother
// you, also clean up `MENTIONS` edges pointing at this episode first.
MATCH (e:Episodic) WHERE e.name = '<episode name>' DETACH DELETE e
```

More Cypher recipes: [docs/neo4j-queries.md](neo4j-queries.md).

## 5. Renaming a person across the whole graph

Graphiti stores `name_embedding` on every `:Entity` node. Renaming via plain Cypher leaves this stale and breaks future entity matching. The full recipe is in [ar-glasses/pipeline/NOTES.md](../ar-glasses/pipeline/NOTES.md):

1. `MATCH (n:Entity {name: 'Old'}) SET n.name = 'New', n.summary = replace(n.summary, 'Old', 'New')`
2. Update related edge facts: `SET r.fact = replace(r.fact, 'Old', 'New')`
3. Update episode names and content: `SET e.name = replace(e.name, 'Old', 'New'), e.content = replace(e.content, 'Old', 'New')`
4. **Regenerate `name_embedding`** by calling `graph.embedder.create(input_data=['New'])` and writing it back to the entity.

For renaming an auto-cluster to a real name, prefer the proper path: `POST /api/people/{id}/label` — the API does the SQLite + interaction-rewrite + Graphiti re-flush in one shot.

## 6. Rotating API keys

```bash
# Edit
$EDITOR ar-glasses/.env

# Restart so the new keys are picked up
./run.sh   # Ctrl+C the previous run first
```

Keys are read at startup. Long-running services do not re-read `.env`.

## 7. Health checks

| Check | Command |
|---|---|
| Neo4j up? | `curl -fs http://localhost:7474/` (HTTP 200 = up) |
| People API up? | `curl -fs http://localhost:5000/` |
| HUD broadcast up? | `python -m websockets ws://localhost:8765` (connects, then waits) |
| Graph has data? | Cypher: `MATCH (n) RETURN labels(n) AS labels, count(n) AS count` |
| Face DB has data? | `sqlite3 ar-glasses/data/people.db 'SELECT person_id, name, is_labeled FROM people'` |

[run.sh](../run.sh) does these automatically on startup with `wait_for_port` and `curl` polling.

## 8. Common operational issues

**Port already in use**
[run.sh](../run.sh) calls `kill_port` to free `5000`, `5050`, `8765` on startup. If a port can't be released, restart your terminal or reboot Docker Desktop.

**Neo4j won't start**
- Make sure Docker Desktop is running.
- First boot can take 20–30 s — don't interrupt the wait loop.
- Auth failures usually mean a stale volume from a previous run with a different password. `docker compose down -v && docker compose up -d`.

**Knowledge graph appears empty even after a session**
- Are you in `--enroll` mode? `--retrieval` does not write.
- Was the person labeled before the flush? Auto-clusters are not flushed to Graphiti.
- Look for `[knowledge] Saved to knowledge graph: ...` in the logs. If you see `Knowledge graph error: ...`, your `OPENAI_API_KEY` is invalid/exhausted or `gpt-4.1` is unavailable.

**Mobile app sees no unlabeled clusters**
- Pipeline must have run with `AUTO_ENROLL_ENABLED=true` (default in `--enroll`).
- A face must be seen ≥ `MIN_SIGHTINGS_TO_CLUSTER` (12) times to be promoted.
- Check `GET /api/people/unlabeled` directly — empty list confirms no clusters exist yet.

**HUD never receives a context card**
- Was the person labeled? Auto-clusters do not trigger broadcasts.
- Has the cooldown elapsed (`RETRIEVAL_COOLDOWN_SECONDS`, default 10 s)?
- Confirm a client is connected: `[hud] client connected (1 total)` in logs.
- Test with `python -m websockets ws://localhost:8765`; if no message arrives, retrieval isn't firing — check `[retrieval]` logs.

## 9. What you can ignore

These are intentional, not bugs:

- `[hud] publish skipped — no connected clients` — the broadcast server is up but nobody is listening. Only matters during demos.
- `[knowledge] skipping Zep flush — knowledge support unavailable` — `pipeline/knowledge` failed to import (usually missing `OPENAI_API_KEY`). The pipeline keeps running; only graph writes are skipped.
- Two clusters with the same name — the People API auto-collapses these on read by primary `person_id`. You can also merge them explicitly via `POST /api/people/merge`.
