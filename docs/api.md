# API Reference

ArgusEye exposes three network surfaces: a REST API for the companion mobile app, a Flask dashboard for live debugging, and a WebSocket broadcast for the glasses HUD. None of them are authenticated — this is a single-user prototype meant to run on the wearer's LAN.

## 1. People API (FastAPI, port 5000)

Implemented in [ar-glasses/api/api.py](../ar-glasses/api/api.py). Auto-generated OpenAPI docs are available at `http://<host>:5000/docs` while the service is running.

### Health

#### `GET /`

```json
{ "status": "ok", "service": "ar-glasses-labeling-api" }
```

### People

#### `GET /api/people`

All enrolled people, deduplicated by name (case- and whitespace-insensitive — the lowest `person_id` wins).

Returns `PersonResponse[]`:

```json
[
  {
    "person_id": 1,
    "name": "Myles Murphy",
    "is_labeled": true,
    "embedding_count": 17,
    "notes": "",
    "created_at": "2026-04-01T18:22:11",
    "last_seen": "2026-04-29T09:14:02",
    "thumbnail_url": "/api/people/1/thumbnail"
  }
]
```

#### `GET /api/people/labeled`

Same shape as `/api/people`, restricted to `is_labeled = true`. Each entry inlines a base64 JPEG `thumbnail` field.

#### `GET /api/people/unlabeled`

Auto-clusters that the mobile app should label. The shape differs slightly — name is the auto-generated `Person N`, no `is_labeled` field needed:

```json
[
  {
    "person_id": 42,
    "name": "Person 42",
    "embedding_count": 15,
    "last_seen": "2026-04-29T09:10:00",
    "thumbnail": "<base64 jpeg>"
  }
]
```

#### `GET /api/people/{person_id}`

Single person. Returns `PersonResponse` or 404.

#### `GET /api/people/{person_id}/thumbnail?format=jpeg|base64`

- `format=jpeg` (default) → streams `image/jpeg` bytes.
- `format=base64` → returns JSON with a `thumbnail_data_url` for embedding in mobile UI.

Returns 404 if the person has no thumbnail.

### Labeling

#### `POST /api/people/{person_id}/label`

Assign a real name to an unlabeled cluster.

**Body:**
```json
{ "name": "Myles Murphy" }
```

**Behavior:**
1. Marks the cluster `is_labeled = true` and sets the name.
2. Rewrites every stored interaction's transcript: any non-`Wearer` speaker becomes the new name (`_update_interaction_speaker_names`).
3. Fires off `save_to_memory` on each rewritten interaction onto Graphiti's background loop. **Does not block** — returns within ~50 ms even if the Zep flush takes minutes.

**Errors:**
- `400` — empty name, or person is already labeled with a different name.
- `404` — `person_id` not found.

**Response:** `LabelResponse` with `action: "labeled" | "already_labeled"`.

#### `POST /api/people/{person_id}/notes`

**Body:** `{ "notes": "..." }`. Returns the updated `PersonResponse`.

### Admin

#### `POST /api/people/merge`

Merge two clusters. Moves all embeddings from `discard_person_id` onto `keep_person_id` and deletes the discard.

**Body:**
```json
{ "keep_person_id": 1, "discard_person_id": 42 }
```

**Errors:** `404` if either is missing, `400` if they're the same id.

#### `DELETE /api/people/{person_id}`

Irreversible. Deletes the person row and cascades to embeddings. Interactions are kept (FK is `SET NULL`).

### Interactions

#### `GET /api/interactions`

Every interaction across every person, newest first. The `person_id` field is rewritten to the **primary** `person_id` for that name, so the mobile app can collapse multiple clusters of the same person into one timeline.

```json
[
  {
    "interaction_id": 12,
    "person_id": 1,
    "timestamp": "2026-04-29T09:14:02",
    "transcript": "Wearer: hey...\nMyles Murphy: ...",
    "context": "",
    "person_name": "Myles Murphy"
  }
]
```

#### `GET /api/interactions/labeled`

Same as above, restricted to interactions whose person is `is_labeled = true`.

#### `GET /api/people/{person_id}/interactions?limit=20`

Recent interactions for one person. Default limit 20.

#### `POST /api/interactions`

Manually log an interaction. Body: `{ person_id, transcript, context }`. Mostly used by tests and seed scripts; the live pipeline writes through the SQLite layer directly, not over HTTP.

## 2. Dashboard (Flask, port 5050)

Defined in [ar-glasses/dashboard.py](../ar-glasses/dashboard.py). Intended for debugging and live demos, not for programmatic clients.

| Path | Purpose |
|---|---|
| `GET /` | Live dashboard — annotated video, captions, retrieval results panel. |
| `GET /knowledge` | Graph viewer (when enabled). See [docs/knowledge-explorer.md](knowledge-explorer.md). |
| `GET /api/knowledge/graph` | JSON `{nodes, links}` for the 3D force-graph viz. |

The dashboard internally talks to the People API on `:5000` and the local Neo4j on `:7687`. Its routes are not a stable contract — treat it as the "developer console," not as a public API.

## 3. HUD WebSocket (port 8765)

Run by [ar-glasses/pipeline/hud_broadcast.py](../ar-glasses/pipeline/hud_broadcast.py). Started automatically by the pipeline when `HUD_BROADCAST_ENABLED=true`.

### Connecting

```
ws://<laptop-ip>:8765
```

No authentication, no handshake message, no subprotocols. Connect and read JSON.

### Messages

A single message type today: `person_context`. Full schema and example in [docs/data-model.md §4](data-model.md#4-hud-websocket-payload).

```json
{
  "type": "person_context",
  "name": "Myles Murphy",
  "person_id": 1,
  "context": {
    "last_spoke": "3 days ago",
    "last_spoke_about": "...",
    "ask_about": "..." ,
    "raw_facts": ["...", "..."]
  }
}
```

### Guarantees

- One message per recognition event, after the per-person cooldown (`RETRIEVAL_COOLDOWN_SECONDS`, default 10 s).
- Only labeled people trigger broadcasts. Auto-clusters (`Person N`) never appear.
- Disconnected clients are silently dropped on the next broadcast — no error feedback.
- The server delivers to **all connected clients**, fan-out style. Multiple HUDs can listen simultaneously.

### Testing without a HUD

```bash
# Quick CLI listener
python -m websockets ws://localhost:8765
```

Trigger a broadcast by recognizing a labeled person on a video file:

```bash
RETRIEVAL_ENABLED=true HUD_BROADCAST_ENABLED=true \
  python pipeline/live.py test_videos/timur_myles_2.mp4
```

## 4. External services we call

These are not "our" APIs but the system depends on them.

| Service | Purpose | Configured by |
|---|---|---|
| Groq `whisper-large-v3` | Audio → transcript with timestamps | `GROQ_API_KEY` |
| Groq `qwen/qwen3-32b` | HUD context formatting (`last_spoke_about` / `ask_about`) | `GROQ_API_KEY` |
| OpenAI `gpt-4.1` (default) | Graphiti episode extraction (entities, edges, facts) | `OPENAI_API_KEY`, `GRAPHITI_LLM_MODEL` |
| OpenAI `text-embedding-3-small` | Graphiti `name_embedding` for entity matching | `OPENAI_API_KEY` |

Without `GROQ_API_KEY` the pipeline still runs but produces no transcript. Without `OPENAI_API_KEY` the knowledge graph cannot be written or searched.
