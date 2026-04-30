# Data Model

ArgusEye keeps state in two places: a local **SQLite** identity database and the **Neo4j** knowledge graph. This doc is the canonical reference for both, plus the in-flight `TranscriptSegment` and HUD WebSocket payload shapes.

## 1. SQLite — `ar-glasses/data/people.db`

Created and migrated by [storage/sqlite_database.py](../ar-glasses/storage/sqlite_database.py). Three tables.

### `people`

The roster of every face cluster the system has discovered, labeled or not.

| Column | Type | Notes |
|---|---|---|
| `person_id` | INTEGER PK AUTOINCREMENT | Stable identifier used everywhere downstream. |
| `name` | TEXT NOT NULL | `"Person {id}"` for auto-clusters; the real name once labeled. |
| `notes` | TEXT DEFAULT `''` | Free-form notes the wearer can attach via the mobile app. |
| `face_thumbnail` | BLOB | JPEG-encoded BGR image (stored via `cv2.imencode`). Optional. |
| `is_labeled` | INTEGER (0/1) | `0` = auto-discovered cluster awaiting a real name. `1` = user-confirmed. |
| `created_at` | TEXT (ISO datetime) | Set on insert. |
| `last_seen` | TEXT (ISO datetime) | Updated on recognition (`update_last_seen`). |

**Invariants:**
- `is_labeled = 0` ⇒ `name` follows the pattern `Person {person_id}`.
- Multiple rows can share a name (e.g. two clusters later identified as the same person). The People API treats the lowest `person_id` as the primary; merges should consolidate via `POST /api/people/merge`.

### `embeddings`

A bag of 512-d face vectors keyed to a `person_id`. Cosine-similarity matching is over this set.

| Column | Type | Notes |
|---|---|---|
| `embedding_id` | INTEGER PK AUTOINCREMENT | |
| `person_id` | INTEGER FK → `people.person_id` (CASCADE) | Many embeddings per person. |
| `vector` | BLOB | `np.float32` array of shape `(512,)`, raw bytes. |
| `model_name` | TEXT DEFAULT `'edgeface_xs_gamma_06'` | Tag for embedding provenance — useful if we ever swap models and need to invalidate. |
| `created_at` | TEXT (ISO datetime) | |

**Invariants:**
- Each person has ≥ 1 embedding (enforced at clustering time).
- Cap of `MAX_EMBEDDINGS_PER_PERSON` (20) per person; new embeddings are only added if cosine distance ≥ `EMBEDDING_DIVERSITY_THRESHOLD` (0.15) from existing ones, every `EMBEDDING_UPDATE_INTERVAL` (360) frames.

### `interactions`

Captured conversations, regardless of whether they've been flushed to the knowledge graph yet. Auto-cluster interactions sit here until the cluster is labeled, then are replayed into the graph.

| Column | Type | Notes |
|---|---|---|
| `interaction_id` | INTEGER PK AUTOINCREMENT | |
| `person_id` | INTEGER FK → `people.person_id` (SET NULL on delete) | Nullable so deleting a person doesn't lose history. |
| `timestamp` | TEXT (ISO datetime) | Set on insert. |
| `transcript` | TEXT | The full conversation, line-prefixed `Wearer:` / `<Name>:` / `Person {id}:`. |
| `context` | TEXT | Anything extra the pipeline wants to attach (currently unused). |

**Lifecycle:**
1. Pipeline flushes a window → `add_interaction(person_id, transcript)` writes here.
2. Mobile app labels person N → API rewrites every interaction's transcript to swap `Person N` for the real name (`_update_interaction_speaker_names`).
3. API queues each interaction onto Graphiti (`save_to_memory(transcript, other_name=name)`) — see [pipeline/knowledge.py](../ar-glasses/pipeline/knowledge.py).

## 2. Neo4j — knowledge graph (managed by Graphiti)

Connection: `bolt://localhost:7687`, user `neo4j`, password `ravengers` (overridable via `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD`).

We use Graphiti's bi-temporal data model and layer four custom entity types on top.

### Entity types

Defined in [pipeline/knowledge.py](../ar-glasses/pipeline/knowledge.py).

| Entity | Pydantic schema | Meaning |
|---|---|---|
| `Person` | `role: 'wearer' | 'other'` | Wearer of the glasses, or someone they spoke with. |
| `Product` | `category: str` | A product, device, technology, or tool mentioned. |
| `Topic` | (no extra fields) | An abstract topic, project, event, or idea. |
| `Commitment` | `description: str`, `status: 'open' | 'fulfilled'` | A promise/follow-up between two people. The thing that powers `ask_about` reminders. |

### Edge type map

Allowed edge labels by source/target type. Graphiti picks among these during extraction.

| From | To | Allowed edges |
|---|---|---|
| `Person` | `Person` | `SPOKE_WITH`, `KNOWS`, `MENTIONED` |
| `Person` | `Product` | `DISCUSSED`, `USES`, `OWNS`, `REVIEWED` |
| `Person` | `Topic` | `DISCUSSED`, `INTERESTED_IN`, `WORKS_ON` |
| `Person` | `Commitment` | `PROMISED`, `OWED` |
| `Commitment` | `Person` | `PROMISED_TO` |
| `Product` | `Topic` | `RELATES_TO` |
| `Product` | `Product` | `COMPARED_TO`, `ALTERNATIVE_TO`, `COMPONENT_OF` |

Every edge stores a `fact: str` summarizing what was extracted (the thing we surface in the HUD).

### Graphiti-managed nodes & properties

In addition to our custom entities, Graphiti adds:

- `:Episodic` nodes — one per `add_episode` call. Holds `name`, `content` (the raw transcript), `source_description`, `valid_at`.
- `:Entity` nodes — superclass label that all our `Person`/`Product`/`Topic`/`Commitment` nodes also carry.
- `:Community` nodes — clusters of related entities; we don't query these directly.
- `name_embedding` property on every `:Entity` — 1024-d vector from OpenAI `text-embedding-3-small`. **If you rename an entity by hand in Cypher, this becomes stale and matching breaks.** See [ar-glasses/pipeline/NOTES.md](../ar-glasses/pipeline/NOTES.md) for the regeneration recipe.

### Bi-temporal validity

Every fact edge has `valid_at` and `invalid_at` timestamps. When a new fact contradicts an old one, Graphiti sets `invalid_at` on the old edge rather than deleting it. This means:

- "Active" facts are those where `invalid_at IS NULL`.
- We can reconstruct what we knew at any past moment.

A starter Cypher reference is in [docs/neo4j-queries.md](neo4j-queries.md).

## 3. In-memory: `TranscriptSegment`

Defined in [ar-glasses/models.py](../ar-glasses/models.py). The unit of work that flows from Whisper through speaker assignment into the knowledge graph.

```python
@dataclass
class TranscriptSegment:
    text: str
    start_time: float       # seconds, relative to window start
    end_time: float
    speaker_label: str | None   # "Wearer" | "Other" | "<Name>" once resolved
    person_id: int | None       # filled by the identity pass
```

Other in-memory dataclasses worth knowing about (same file): `BoundingBox`, `DetectedFace`, `FaceEmbedding`, `IdentityMatch`, `Person`.

## 4. HUD WebSocket payload

The single shape any glasses client must understand. Sent by [pipeline/hud_broadcast.py](../ar-glasses/pipeline/hud_broadcast.py) when retrieval fires for a labeled person; clients connect to `ws://<laptop-ip>:8765`.

```json
{
  "type": "person_context",
  "name": "Myles Murphy",
  "person_id": 1,
  "context": {
    "last_spoke": "3 days ago",
    "last_spoke_about": "Senior design project and Capital One job",
    "ask_about": "How the Capital One onboarding is going",
    "raw_facts": [
      "Works at Capital One starting June",
      "Senior design project on smart glasses"
    ]
  }
}
```

| Field | Type | Notes |
|---|---|---|
| `type` | `"person_context"` | Reserved for future message types. Only this one exists today. |
| `name` | string | Resolved real name. Auto-clusters never trigger a broadcast. |
| `person_id` | int | The SQLite primary `person_id`. |
| `context.last_spoke` | string | Humanized time delta from `_humanize_delta` ("just now", "3 days ago", "1 year ago"). |
| `context.last_spoke_about` | string \| null | One-sentence summary, < 15 words. |
| `context.ask_about` | string \| null | Single suggested opener, < 12 words. `null` if nothing useful — clients must handle this. |
| `context.raw_facts` | string[] | Top-3 facts from `Graphiti.search()`. Useful for debugging, optional for display. |

**Rate limiting:** at most one message per `(name, RETRIEVAL_COOLDOWN_SECONDS)` window. The cooldown is name-keyed, not `person_id`-keyed, so two clusters labeled with the same name share one window.

## 5. Where to look in code

| Concern | File |
|---|---|
| SQLite schema, migrations, CRUD | [storage/sqlite_database.py](../ar-glasses/storage/sqlite_database.py) |
| In-memory dataclasses | [models.py](../ar-glasses/models.py) |
| Neo4j entity/edge definitions | [pipeline/knowledge.py](../ar-glasses/pipeline/knowledge.py) |
| Retrieval & HUD payload assembly | [pipeline/retrieval.py](../ar-glasses/pipeline/retrieval.py) |
| WebSocket broadcast | [pipeline/hud_broadcast.py](../ar-glasses/pipeline/hud_broadcast.py) |
| Useful Cypher | [docs/neo4j-queries.md](neo4j-queries.md) |
