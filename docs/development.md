# Developer Guide

This is the doc for someone who has cloned the repo, gotten `./run.sh` working once, and now wants to actually change something. The user-facing install/run guide is in [README.md](../README.md); this picks up where that leaves off.

## 1. Repo layout, once more — annotated

```
ar-glasses/
├── dashboard.py          ← Flask debug UI (:5050). Entry point you'll hit most often.
├── config.py             ← Top-level config; re-exports per-subsystem configs.
├── paths.py              ← Project paths (DB_PATH, EDGEFACE_CHECKPOINT, etc).
├── models.py             ← Shared dataclasses (BoundingBox, DetectedFace, Person, …).
├── docker-compose.yml    ← Neo4j 5.26 with APOC plugins.
│
├── api/                  ← FastAPI app (People API, :5000)
│   ├── api.py            ←   Entry: `python -m api.api`
│   ├── requests/         ←   Pydantic request models
│   └── responses/        ←   Pydantic response models
│
├── pipeline/             ← Streaming pipeline driver + side-channels
│   ├── live.py           ←   LivePipelineDriver — main loop
│   ├── driver.py         ←   PipelineDriver — batch (offline video) variant
│   ├── identity.py       ←   FullIdentity / NullIdentity modules
│   ├── retrieval.py      ←   Knowledge-graph retrieval worker (Graphiti search)
│   ├── knowledge.py      ←   Write side: save_to_memory + entity definitions
│   ├── knowledge_query.py←   Read side: ask() wrapper for the dashboard chat panel
│   ├── transcription.py  ←   Groq Whisper client
│   ├── diarization.py    ←   Frame-level speaker assignment glue
│   ├── hud_broadcast.py  ←   WebSocket server (:8765)
│   ├── conversation_end.py    ← Heuristic flush trigger
│   ├── flush_worker.py   ←   Background flush queue
│   ├── recording_buffer.py    ← Rolling A/V window
│   └── segments.py       ←   TranscriptSegment helpers
│
├── processing/           ← Per-frame ML
│   ├── face_detector.py  ←   OpenCV Res10-SSD (default) / MediaPipe variants
│   ├── face_embedder.py  ←   EdgeFace wrapper
│   ├── face_matcher.py   ←   Cosine similarity gallery match
│   ├── face_tracker.py   ←   Bbox proximity + temporal smoothing
│   └── vad_speaker.py    ←   Silero VAD + adaptive RMS speaker classifier
│
├── storage/              ← Persistence
│   ├── sqlite_database.py←   Schema, migrations, CRUD
│   ├── enrollment.py     ←   Auto-cluster promotion logic
│   └── speaking_log.py   ←   Per-track speaking timeline
│
├── edgeface/             ← Bundled face-embedding model + checkpoints
├── input/                ← Camera/file input wrappers
├── data/                 ← Runtime: people.db, cached embeddings (gitignored)
├── test_videos/          ← Sample clips
├── testing/              ← Smoke tests, seeds, evals
└── eval/                 ← Speaker-classification evaluation harness
    ├── evaluate.py
    ├── prepare_lfw.py
    └── transcripts_to_annotate.txt    ← Ground truth for VAD eval
```

## 2. Running individual services

The unified script ([run.sh](../run.sh)) is convenient but obscures what's happening. Each piece can be run on its own.

### Neo4j only

```bash
cd ar-glasses
docker compose up -d         # bolt :7687, browser :7474
docker compose down          # stop
docker compose down -v       # stop + delete the volume (nukes the graph)
```

### People API only

```bash
cd ar-glasses
python -m api.api                      # bind 0.0.0.0:5000 by default
python -m api.api --port 5001 --reload # dev mode with auto-reload
```

OpenAPI explorer: http://localhost:5000/docs.

### Pipeline only (no dashboard)

```bash
cd ar-glasses
python pipeline/live.py                              # live camera
python pipeline/live.py test_videos/timur_myles_2.mp4   # offline video
```

`pipeline/live.py` is the place to put a debugger when you want to step through identity resolution or speaker assignment.

### Dashboard only

```bash
cd ar-glasses
python dashboard.py                       # live camera, full speed
python dashboard.py --glasses             # source = glasses (over network)
python dashboard.py test_videos/clip.mp4  # offline video, audio-synced
python dashboard.py --fast video.mp4      # offline, accelerated, uses VISION_STRIDE
```

The dashboard imports the same pipeline modules under the hood — it's not a separate stack, it just adds a Flask UI.

## 3. The two run modes — what they actually toggle

| Mode | `AUTO_ENROLL_ENABLED` | `SAVE_TO_MEMORY` | `RETRIEVAL_ENABLED` | `HUD_BROADCAST_ENABLED` |
|---|---|---|---|---|
| `--enroll` (default) | true | true | true | true |
| `--retrieval` | false | false | true | true |

These are env vars read by [pipeline/config.py](../ar-glasses/pipeline/config.py) and [processing/config.py](../ar-glasses/processing/config.py). Setting them by hand bypasses [run.sh](../run.sh):

```bash
AUTO_ENROLL_ENABLED=true SAVE_TO_MEMORY=true RETRIEVAL_ENABLED=true \
HUD_BROADCAST_ENABLED=true \
  python pipeline/live.py test_videos/timur_myles_2.mp4
```

What each toggle does:

- `AUTO_ENROLL_ENABLED` — when an unknown face has been seen `MIN_SIGHTINGS_TO_CLUSTER` (12) times consistently, promote it to a `Person N` cluster in SQLite. Off → unknown faces stay labeled `Unknown` and never enter the DB.
- `SAVE_TO_MEMORY` — on flush, write the conversation to Graphiti. Auto-clusters are still gated out: only labeled people get flushed (see `_resolve_and_save` in `pipeline/live.py`).
- `RETRIEVAL_ENABLED` — when a labeled face appears, fire a Graphiti search and put the formatted result on the result queue. Independent of writes; `--retrieval` mode runs this without `SAVE_TO_MEMORY`.
- `HUD_BROADCAST_ENABLED` — start the WebSocket server on `:8765` and forward every retrieval result to connected clients.

Useful combos:

```bash
# Replay a recorded conversation into the graph without affecting recognition
AUTO_ENROLL_ENABLED=false SAVE_TO_MEMORY=true RETRIEVAL_ENABLED=false \
  python pipeline/live.py test_videos/timur_peter.mp4

# Demo: recognition only, no writes, no auto-clusters
./run.sh --retrieval

# Develop the HUD: set up canned data with seed_myles.py, then run retrieval-only
python testing/seed_myles.py
./run.sh --retrieval
```

## 4. Running on a recorded video

The `--fast` flag in `dashboard.py` (and equivalently the `VISION_STRIDE` config) is the trick that makes offline processing usable.

- **Default (live mode)**: every frame goes through detection + embedding. ~30 fps real-time.
- **`--fast`**: detection + embedding run on every Nth frame (`VISION_STRIDE`, default 5). Reused identity on skipped frames. Audio still processed continuously. Roughly 5× speedup on the vision side.

For a 5-minute test clip you typically want:

```bash
./run.sh -- --fast test_videos/timur_myles_2.mp4
```

The `--` separator tells [run.sh](../run.sh) to forward the rest to `dashboard.py`. The default dashboard arg is `--glasses` (live network camera); replace it like above when working from a file.

## 5. Adding a new pipeline stage

The pipeline is plain functions and a couple of small classes — there is no plugin framework. To add a stage:

1. **Add config knobs** in the matching `config.py` (`processing/`, `pipeline/`, or top-level). Magic numbers must live here, never in the stage itself ([processing/CLAUDE.md](../ar-glasses/processing/CLAUDE.md)).
2. **Write the stage** as a function or small stateful class in `processing/` (per-frame ML) or `pipeline/` (orchestration). Keep it pure where possible; pass models/state in explicitly.
3. **Wire it into the driver** in [pipeline/live.py](../ar-glasses/pipeline/live.py) at the appropriate point in the per-frame loop.
4. **Surface it in the dashboard** if it has anything visual: drop it into `templates/` and add a route to `dashboard.py`.
5. **Add a test or smoke run** in `testing/` — at minimum a script that runs the stage on a fixture video and prints something verifiable.

If you find yourself adding a base class with one subclass, or a factory function that returns one type, stop and just write the function. See [ar-glasses/CLAUDE.md](../ar-glasses/CLAUDE.md) for code-style ground rules.

## 6. Logs, where things go

- **Background services** (`run.sh` without `--debug`) — stdout is suppressed. Run with `--debug` to get prefixed `[api]`, `[pipeline]`, etc. lines on the terminal.
- **Dashboard** — runs in the foreground, logs straight to your terminal.
- **HUD broadcast** — prints connect/disconnect/broadcast events with a `[hud]` prefix.
- **Retrieval worker** — `[retrieval]` prefix. `QUEUED`, `FIRING`, `SENT`, `skip` events trace the lifecycle of every recognition.
- **Knowledge writes** — `[knowledge]` prefix. Saves are async; success/failure logs land out-of-band when the future resolves.
- **Neo4j** — `docker compose logs -f neo4j`.

## 7. Code style — quick rules

Full set lives in [ar-glasses/CLAUDE.md](../ar-glasses/CLAUDE.md), [pipeline/CLAUDE.md](../ar-glasses/pipeline/CLAUDE.md), and [processing/CLAUDE.md](../ar-glasses/processing/CLAUDE.md). Highlights:

- Names are the documentation. No comments except for non-obvious "why."
- No commented-out code. Ever.
- All Python must pass `ruff check` with zero violations. Run `ruff check --fix .` and `ruff format .` after editing.
- Magic numbers go in `config.py`, not in the stage that uses them.
- Always tag tensor/array shapes after non-obvious ops: `features = encoder(frame)  # (batch, 512)`.
- Frames default to BGR; rename to `frame_rgb` the moment you convert.

## 8. Common gotchas

- **`ModuleNotFoundError: api`** — you're running from `ARsenal-Ravengers/` instead of `ar-glasses/`. Most entry points expect `cwd = ar-glasses/`.
- **Neo4j authentication failed** — the volume from a previous run remembers the old password. `docker compose down -v` to reset, then `docker compose up -d`.
- **`UnicodeEncodeError` in terminal (Windows)** — set `PYTHONIOENCODING=utf-8`. [run.sh](../run.sh) already does this.
- **Retrieval fires but no HUD message** — the cluster is unlabeled. `[retrieval] skip ... (unlabeled or auto-cluster)` confirms it. Label it via the mobile app or the seed script.
- **Graphiti writes succeed but nothing in `:7474`** — you're looking at the wrong DB or container. Check `docker compose ps` and `bolt://localhost:7687`.
- **Vision falls behind real time** — bump `VISION_STRIDE` from 5 → 7 or 10. Audio stays continuous; you only lose temporal resolution on face detection, which conversational head-motion tolerates fine.
