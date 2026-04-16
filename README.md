# ArgusEye

Smart glasses that recognize faces in real time, recall everything you've talked about with that person, and surface a conversation prompt directly on your heads-up display — so you never blank on someone's name or forget what you last discussed.

---

## What It Does

You wear a pair of smart glasses. As you move through the world and talk to people, the glasses are quietly doing three things at once:

1. **Watching** — the camera continuously detects and tracks faces. Each face gets converted into a unique fingerprint (called an embedding) and stored.
2. **Listening** — the microphone transcribes the conversation and figures out who said what (speaker diarization). That transcript gets fed into a knowledge graph that extracts and stores facts about the people you talk to.
3. **Remembering** — the next time a known face appears, the glasses look up everything stored about that person and push a context card to the HUD: their name, what you talked about last time, and a suggested conversation opener.

After each session, you open the **[ArgusEye Mobile App](https://github.com/MaxOrangeTabby/ARgusEye-mobile-app)** and put names to faces — linking the face fingerprints to real identities so the knowledge graph knows who is who.

> **The companion app is required for the full pipeline.** Without it, the system can detect and track faces but cannot associate them with real names. Retrieval, HUD prompts, and knowledge graph storage only activate for labeled (named) people — so faces that haven't been named through the app are invisible to the memory system.

---

## System Architecture

```
Smart Glasses (camera + mic)
        │
        ├─── Video stream ──► Face detection (MediaPipe)
        │                         │
        │                    EdgeFace embedding
        │                         │
        │                    SQLite identity DB ◄──── ArgusEye Mobile App (you label faces)
        │                         │
        │              [known face?] ──► Knowledge graph query (Neo4j)
        │                                       │
        │                               Context post-processing (GPT-4o-mini)
        │                                       │
        │                               HUD display (WebSocket → glasses)
        │
        └─── Audio stream ──► Voice Activity Detection (Silero VAD)
                                    │
                               WhisperX transcription + speaker diarization
                                    │
                               Zep Graphiti — extracts facts, entities, relationships
                                    │
                               Neo4j knowledge graph
```

### Services

| Service | What it does | Port |
|---|---|---|
| **Neo4j** | Stores the knowledge graph (facts, relationships, conversation history) | 7687 (bolt), 7474 (browser UI) |
| **Knowledge API** | FastAPI server — ingests transcripts into Neo4j, serves context for recognized people | 8000 |
| **People API** | FastAPI server — manages face identity database, serves the companion app | 5000 |
| **Dashboard** | Flask web app — live debug view of the video stream, captions, and retrieval results | 5050 |
| **HUD Broadcast** | WebSocket server — pushes real-time context cards to the Unity glasses app | 8765 |
| **[ArgusEye Mobile App](https://github.com/MaxOrangeTabby/ARgusEye-mobile-app)** ⚠️ **Required** | Android app — browse face clusters after a session and assign real names to them | connects to :5000 |

---

## Prerequisites

Before installing anything, make sure you have these tools installed on your machine.

### Required

| Tool | Why you need it | Install |
|---|---|---|
| **Python 3.10 or newer** | Runs all pipeline and API code | [python.org](https://www.python.org/downloads/) |
| **Docker Desktop** | Runs the Neo4j knowledge graph database | [docker.com](https://www.docker.com/products/docker-desktop/) |
| **Git Bash** *(Windows only)* | Required to run `.sh` scripts on Windows | Included with [Git for Windows](https://git-scm.com/download/win) |

### API Keys

You need to add these to `ar-glasses/.env`. A template is at `ar-glasses/.env.example` (copy it to `.env` and fill in your keys).

| Key | Where to get it | Used for |
|---|---|---|
| `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com/api-keys) | Knowledge graph extraction (GPT-4o-mini) and context post-processing |
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com/) | Fast Whisper transcription via Groq API |
| `HUGGINGFACE_TOKEN` | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) | Downloading WhisperX speaker diarization model |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/murphymy04/ARsenal-Ravengers
cd ARsenal-Ravengers
```

### 2. Create a Python virtual environment

```bash
cd ar-glasses
python -m venv .venv
```

Activate it:

```bash
# Windows (Git Bash)
source .venv/Scripts/activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

> **GPU acceleration (optional but recommended)**
>
> The above installs a CPU-only version of PyTorch. If you have an NVIDIA GPU:
>
> ```bash
> # CUDA 12.1
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```
>
> Check your CUDA version with `nvidia-smi` and pick the matching build at [pytorch.org](https://pytorch.org/get-started/locally/).

### 5. Configure environment variables

```bash
cp ar-glasses/.env.example ar-glasses/.env
```

Open `ar-glasses/.env` and fill in your API keys:

```env
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...
HUGGINGFACE_TOKEN=hf_...
```

### 6. Make the run script executable *(first time only)*

```bash
chmod +x run.sh
```

---

## Running the Full Stack

From the **repository root**:

```bash
./run.sh
```

This starts all five services in order, waits for each one to be healthy before starting the next, and shuts everything down cleanly when you press `Ctrl+C`.

### What the startup looks like

```
Starting Neo4j...
Waiting for Neo4j (:7474)...... ready
Starting knowledge API...
Waiting for knowledge API (:8000) ready
Starting people API...
Waiting for people API (:5000) ready

Starting dashboard (http://localhost:5050)...
  Args: --glasses
```

### Options

```bash
./run.sh                     # Standard run — glasses mode, background logs suppressed
./run.sh --debug             # Show live logs from all background services
./run.sh --skip-neo4j        # Skip Neo4j (useful if it's already running)
./run.sh --skip-knowledge    # Skip knowledge API
./run.sh --skip-api          # Skip people API
./run.sh -- --fast video.mp4 # Pass custom args to dashboard (e.g. run on a video file)
```

### Accessing the services

Once running, open these in your browser:

| URL | What you'll find |
|---|---|
| http://localhost:5050 | Live dashboard — annotated video, captions, retrieval results |
| http://localhost:8000/docs | Knowledge API — interactive API explorer |
| http://localhost:7474 | Neo4j browser — explore the knowledge graph directly |

Neo4j login: username `neo4j`, password `ravengers`.

---

## Seeding the Knowledge Graph

The knowledge graph starts empty. To test retrieval with pre-existing conversation history, run one of the seed scripts after the stack is up:

```bash
# Seed conversations between Timur and Myles Murphy
cd ar-glasses
python testing/seed_myles.py

# Or seed conversations between Will and Peter Nguyen
cd knowledge
python seed.py
```

Each seed script adds a few past conversations to Neo4j. The next time that person's face is recognized during a session, the system will retrieve those facts and generate a conversation prompt.

---

## How the Knowledge Graph Works

Every conversation gets turned into a set of **facts** stored in Neo4j. For example, after a conversation where Peter mentions his game jam project:

```
Peter Nguyen — worked on → Platformer game (Chillenium 2024)
Peter Nguyen — plans to → Launch on Steam (summer 2025)
Peter Nguyen — used tool → Cursor, Claude Code
```

When Peter's face appears again, the system queries these facts, passes them to GPT-4o-mini, and generates a single natural conversation starter like:

> *"Hey, how's the Steam launch prep going for your Chillenium game?"*

That prompt is pushed to the glasses HUD via WebSocket.

---

## HUD Broadcast (Unity / Smart Glasses Integration)

When a labeled person is recognized, a JSON message is sent over WebSocket to any connected glasses app:

```json
{
  "type": "person_context",
  "name": "Myles Murphy",
  "person_id": 1,
  "context": {
    "last_spoke": "3 days ago",
    "last_spoke_about": "Senior design project and Capital One job",
    "ask_about": "How the Capital One onboarding is going",
    "raw_facts": ["Works at Capital One starting June", "Senior design project on smart glasses"]
  }
}
```

**To connect a Unity app:** open a WebSocket connection to `ws://<laptop-ip>:8765`. Each `OnMessage` delivers one JSON object. Rate-limited to one message per person per 30 seconds (configurable via `RETRIEVAL_COOLDOWN_SECONDS`).

**To test without a glasses device:**

```bash
python -m websockets ws://localhost:8765
```

---

## Configuration Reference

All defaults live in the config files under `ar-glasses/`. Override any of them by setting the corresponding environment variable — either in `ar-glasses/.env` or by prefixing the `run.sh` command.

### Core toggles

| Variable | Default | What it controls |
|---|---|---|
| `SAVE_TO_MEMORY` | `false` | Save conversations to Neo4j after each session |
| `RETRIEVAL_ENABLED` | `false` | Query knowledge graph when a known face appears |
| `HUD_BROADCAST_ENABLED` | `false` | Start the WebSocket server for the glasses HUD |
| `RETRIEVAL_COOLDOWN_SECONDS` | `30` | Minimum seconds between retrieval queries for the same person |

> `run.sh` sets all three of the above to `true` automatically.

### Database connection

| Variable | Default | What it controls |
|---|---|---|
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection address |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `ravengers` | Neo4j password |

### Camera and audio

| Variable | Default | What it controls |
|---|---|---|
| `CAMERA_FPS` | `30` | Target frame rate from the glasses camera |
| `SAMPLE_RATE` | `16000` | Audio sample rate in Hz |
| `VISION_STRIDE` | `5` | Process faces every Nth frame (higher = faster, less precise) |
| `LIVE_BUFFER_SECONDS` | `10` | How many seconds of audio/video to process in each window |

### HUD broadcast

| Variable | Default | What it controls |
|---|---|---|
| `HUD_BROADCAST_HOST` | `0.0.0.0` | WebSocket server bind address |
| `HUD_BROADCAST_PORT` | `8765` | WebSocket server port |

---

## Project Structure

```
ARsenal-Ravengers/
├── run.sh                      ← Start (and stop) the full backend
├── ar-glasses/
│   ├── dashboard.py            ← Live debug dashboard (Flask, port 5050)
│   ├── pipeline/
│   │   ├── live.py             ← Main streaming pipeline driver
│   │   ├── retrieval.py        ← Knowledge graph query worker
│   │   └── hud_broadcast.py    ← WebSocket server for glasses HUD
│   ├── processing/
│   │   ├── face_detector.py    ← Face detection (MediaPipe / OpenCV)
│   │   ├── face_tracker.py     ← Temporal identity smoothing
│   │   └── vad_speaker.py      ← Voice activity detection + speaker assignment
│   ├── api/
│   │   └── api.py              ← People/identity REST API (FastAPI, port 5000)
│   ├── storage/
│   │   └── database.py         ← SQLite face identity database
│   ├── edgeface/               ← Face embedding model + checkpoints (bundled)
│   ├── testing/
│   │   └── seed_myles.py       ← Seed knowledge graph with sample conversations
│   ├── requirements.txt        ← Python dependencies
│   └── .env                    ← API keys (not committed — you create this)
├── knowledge/
│   ├── main.py                 ← Knowledge graph REST API (FastAPI, port 8000)
│   ├── docker-compose.yml      ← Neo4j 5.26 database container
│   └── seed.py                 ← Alternative seed script
└── docs/
    └── high_level.md           ← Full system design document
```

---

## Troubleshooting

**`run.sh: Permission denied`**
Run `chmod +x run.sh` to make the script executable.

**Neo4j never becomes ready**
Make sure Docker Desktop is running before you start. Neo4j can take 20–30 seconds to fully initialize on first launch.

**`ModuleNotFoundError` when starting**
Make sure your virtual environment is active (`source .venv/Scripts/activate` on Windows or `source .venv/bin/activate` on Mac/Linux) and that you ran `pip install -r requirements.txt` inside `ar-glasses/`.

**Port already in use warning**
`run.sh` automatically kills any process on the required ports before starting. If the warning says a port couldn't be released, manually restart the terminal or reboot Docker Desktop.

**No retrieval results / blank HUD**
Make sure you ran a seed script (see [Seeding the Knowledge Graph](#seeding-the-knowledge-graph)) and that the person's face has been labeled in the companion app. Retrieval only fires for people with a real name — not unnamed face clusters.

**`UnicodeEncodeError` in terminal**
This is a Windows terminal encoding issue. `run.sh` already sets `PYTHONIOENCODING=utf-8` to prevent it. If you're running dashboard scripts manually, prefix with `PYTHONIOENCODING=utf-8 python dashboard.py ...`.
