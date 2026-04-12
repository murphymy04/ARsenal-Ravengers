# AR Glasses — Diarization & Speaker Detection Pipeline

## Code style

### Naming & Readability
- Names are the documentation. Use descriptive, intention-revealing names so the code reads without comments.
  - BAD: `d = get_data(src)` 
  - GOOD: `user_records = fetch_active_users(database)`
- No comments except for genuinely non-obvious "why" explanations. Never comment "what" the code does. If you need a comment to explain what code does, rename things until you don't.
- No commented-out code. Ever. That's what git is for.

### Structure & Simplicity
- Functions do one thing. If you're tempted to name it `process_and_save` or `validate_and_transform`, split it.
- Keep functions short — if it doesn't fit on one screen (~30 lines), break it up.
- Flat is better than nested. If you have 3+ levels of indentation, refactor with early returns, guard clauses, or extraction.
- Early returns are great, prefer them over deep nesting
- Use Pythonic idioms: list comprehensions, f-strings, `pathlib`, unpacking, `enumerate`, `zip`. But still make it skimmable

### No Over-Engineering
- No defensive code for situations that won't happen. Don't check types at runtime, don't add try/except around things that shouldn't fail, don't validate inputs that you control.
- No abstractions until there's a clear second use case. No base classes with one subclass. No factory functions that return one thing.
- Prefer plain functions over classes. Only use a class when you need to manage state across multiple methods.
- That being said, if there are two modules that do similar things, use a base class for a consistent interface

### Visual Cleanliness
- Group related code into visual blocks with one blank line between them. Use two blank lines between top-level functions/classes.
- Imports at the top, organized: stdlib → third-party → local. Let ruff handle ordering.
- Do not use lazy or inline imports inside functions unless there is a documented, unavoidable reason such as a hard optional dependency or import cycle. Default to top-level imports.
- No deeply nested data transformations on one line. If a list comprehension has a condition and a nested loop, break it into a regular loop.

## Code Quality
- Run `ruff check --fix .` and `ruff format .` after modifying Python files
- All code must pass `ruff check` with zero violations
- Follow the ruff config in pyproject.toml

## What this is

Real-time diarization pipeline for 1-on-1 conversations filmed through smart glasses.
Detects faces, tracks identity, and assigns speech to the wearer vs the other person.

## Pipeline architecture

```
Camera frames ──► FaceDetector (OpenCV DNN) ──► FaceTracker ──► IdentityModule (EdgeFace)
                                                                        │
Audio stream  ──► VadSpeaker (Silero VAD + RMS) ───────────────────────┘
                                                                        │
                                                               SpeakingLog ──► TranscriptionPipeline (Whisper)
                                                                        │
                                                               Combined segments (speaker + text)
```

### Per-frame cost breakdown

| Operation | ~Latency | Backend |
|---|---|---|
| Face detection | 5-10ms | OpenCV DNN Res10-SSD |
| Face embedding | 10-20ms | EdgeFace (per detected face) |
| VAD inference | 1-2ms | Silero VAD (512-sample chunks) |
| Face tracking | <1ms | Bbox proximity matching |

At 30fps this totals ~25-35ms/frame — roughly real-time.

## Vision stride (accelerated video processing)

`VISION_STRIDE` in `config.py` controls how many frames to skip between face detection/embedding runs.
Audio/VAD still processes every frame for continuity. On skipped frames, the last detected faces are reused.

- **stride=1**: full detection every frame (default for live camera)
- **stride=5**: detect every 5th frame (~5x faster, used for video simulation)

### Usage

```bash
# Live pipeline with video file — uses VISION_STRIDE automatically
python -m pipeline.live test_videos/clip.mp4

# Debug overlay — normal speed with audio sync
python testing/debug_video.py test_videos/clip.mp4

# Debug overlay — accelerated, no audio sync, uses VISION_STRIDE
python testing/debug_video.py --fast test_videos/clip.mp4
```

Tune `VISION_STRIDE` in `config.py`. Higher = faster but less temporal resolution on face detection.
For 1-on-1 conversations stride 3-5 is safe since faces move slowly.

## Retrieval pipeline

When a known face appears, an optional retrieval side-channel queries the Zep/Graphiti knowledge graph for context about that person. The diarization pipeline pushes track creation events to a queue; `pipeline/retrieval.py` consumes them, applies a per-person cooldown (`RETRIEVAL_COOLDOWN_SECONDS`), and queries Graphiti. Results are drained and printed at each flush window. Enabled via `RETRIEVAL_ENABLED=true` env var.

### Identity resolution & gates

- **Labeled-only retrieval**: retrieval skips auto-clusters ("Person N"). Only labeled people with real names trigger a Graphiti query.
- **Name-based cooldown**: cooldown is keyed by resolved name, not person_id. Multiple face clusters for the same person (e.g. clusters 2 and 3 both labeled "Peter Nguyen") share one cooldown window.
- **Dominant person_id filtering** (`_dominant_person_id` in `live.py`): when flushing a conversation buffer, picks the most-seen person_id and ignores any cluster with <10% of speaker assignments.
- **Labeled-only Zep flush** (`_resolve_and_save` in `live.py`): unlabeled people only get stored in the SQLite interactions table. Labeled people get their real name substituted into the transcript before the Zep episode flush.

### Testing retrieval

```bash
# Seed knowledge graph (requires Neo4j running via docker compose)
python testing/seed_myles.py

# Labeled person — retrieval fires once, Zep flush with resolved name
RETRIEVAL_ENABLED=True SAVE_TO_MEMORY=true python pipeline/live.py test_videos/timur_myles_2.mp4

# Unlabeled person — no retrieval, no Zep flush, interaction still stored in SQLite
RETRIEVAL_ENABLED=True SAVE_TO_MEMORY=true python pipeline/live.py test_videos/timur_peter.mp4
```

## Key files

| File | Purpose |
|---|---|
| `config.py` | All tunable constants |
| `pipeline/live.py` | Main streaming pipeline (LivePipelineDriver) |
| `pipeline/driver.py` | Batch video pipeline (PipelineDriver) |
| `testing/debug_video.py` | Visual debug overlay with face boxes |
| `processing/vad_speaker.py` | VAD + RMS speaker assignment |
| `processing/face_detector.py` | Face detection (3 backends) |
| `processing/face_tracker.py` | Temporal identity smoothing |
| `pipeline/identity.py` | NullIdentity / FullIdentity modules |
| `pipeline/retrieval.py` | Retrieval worker (cooldown + Graphiti search) |
| `testing/test_retrieval_e2e.py` | E2E test: seed knowledge graph then test retrieval |

## Running

```bash
cd ar-glasses

# Video file (auto-caches results)
python pipeline/live.py test_videos/clip.mp4

# Debug overlay
python testing/debug_video.py [--fast] test_videos/clip.mp4
```
