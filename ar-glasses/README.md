# AR Glasses Modular Prototype

Face recognition system that identifies people in real-time and links conversation context. Built on EdgeFace for lightweight, efficient face embeddings.

## Neo4j Queries

Neo4j browser: `http://localhost:7474` (bolt: `bolt://localhost:7687`, user: `neo4j`, password: `ravengers`)

```cypher
-- Everything
MATCH (n) RETURN n

-- All entities (no episodes)
MATCH (n) WHERE NOT "Episodic" IN labels(n) RETURN n

-- All fact edges (the useful stuff)
MATCH (a)-[r]->(b) WHERE r.fact IS NOT NULL RETURN a, r, b

-- Facts as text
MATCH (a)-[r]->(b) WHERE r.fact IS NOT NULL
RETURN a.name AS from, type(r) AS rel, b.name AS to, r.fact AS fact

-- All edges including MENTIONS
MATCH (a)-[r]->(b) RETURN a.name, type(r), b.name, r.fact

-- Episodes with metadata
MATCH (e:Episodic) RETURN e.name, e.source_description, e.content

-- Entities by type
MATCH (n:Person) RETURN n.name, n.summary
MATCH (n:Product) RETURN n.name, n.summary
MATCH (n:Topic) RETURN n.name, n.summary

-- Everything about a specific person
MATCH (n {name: "Myles Murphy"})-[r]-(m) RETURN n, r, m

-- Node and relationship counts
MATCH (n) RETURN labels(n) AS labels, count(n) AS count
MATCH ()-[r]->() RETURN type(r) AS type, count(r) AS count

-- Nuke it
MATCH (n) DETACH DELETE n
```

## Project Structure

```
ar-glasses/
    config.py              # constants, thresholds, paths
    models.py              # shared dataclasses
    main.py                # orchestrator (video loop + enrollment)
    edgeface/              # EdgeFace model code (self-contained, no external dependency)
        backbones/
            __init__.py    # get_model() factory
            timmfr.py      # EdgeNext backbone + LoRA compression
        checkpoints/
            edgeface_xs_gamma_06.pt   # primary model (6.9 MB, used by default)
            edgeface_s_gamma_05.pt    # medium model (15 MB)
            edgeface_xxs.pt           # extra-extra-small (4.8 MB)
            edgeface_base.pt          # full model (70 MB)
        LICENSE
    input/
        camera.py          # OpenCV VideoCapture wrapper
        microphone.py      # PyAudio capture (Phase 2 stub)
    processing/
        face_detector.py   # MediaPipe face detection → 112x112 crops
        face_embedder.py   # EdgeFace → 512-dim embeddings
        face_matcher.py    # cosine similarity matching
        audio_processor.py # WhisperX transcription (Phase 2 stub)
    storage/
        database.py        # SQLite CRUD for people, embeddings, interactions
        enrollment.py      # add new person workflow
    output/
        display.py         # OpenCV overlay (boxes, names, confidence)
        companion_app.py   # Flask web app (Phase 2 stub)
    data/
        people.db          # created at runtime
```

## Environment Setup

### 1. Create and activate a virtual environment

```bash
cd /mnt/c/Users/a11155/capstone/ar-glasses
python3 -m venv venv
source venv/bin/activate
```

### 2. Install PyTorch

Pick the command that matches your hardware.

**CPU only:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**CUDA (NVIDIA GPU):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Not sure which to pick? Run `nvidia-smi` — if it shows a GPU, use CUDA. Otherwise use CPU.

### 3. Install remaining dependencies

```bash
pip install -r requirements.txt
```

Or individually:
```bash
pip install opencv-python numpy scikit-learn timm mediapipe
```

### 4. Verify EdgeFace is ready

The checkpoints are already bundled in `edgeface/checkpoints/`. Confirm they're in place:

```bash
ls -lh edgeface/checkpoints/
```

You should see `edgeface_xs_gamma_06.pt` (~6.9 MB). No download needed — it's included in this repo.

### 5. (WSL only) Fix webcam access

If running in WSL and your webcam isn't detected (`cv2.VideoCapture(0)` fails), try index 1 or install USBIPD to forward the USB device:

```bash
# Test which camera index works
python3 -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"
python3 -c "import cv2; cap = cv2.VideoCapture(1); print(cap.isOpened())"
```

If both fail in WSL, run `main.py` from native Windows PowerShell instead.

## How to Run

Run all commands from inside `ar-glasses/` with the venv activated.

### Live Face Recognition

```bash
python3 main.py
```

Opens your webcam and runs the full pipeline: detect → embed → match → display.

- **Green box** = recognized person with name and confidence score
- **Red box** = unknown face
- Press **q** to quit

Use a different camera index (e.g. external webcam):
```bash
python3 main.py --camera 1
```

### Enroll a New Person

```bash
python3 main.py --mode enroll
```

1. Type the person's name when prompted.
2. Webcam opens — face detection boxes appear in real-time.
3. Press **c** to capture (3–5 captures from different angles recommended).
4. Press **q** when done — person is saved to `data/people.db`.

Next time you run `main.py`, enrolled people will be recognized automatically.

### Full Pipeline (retrieval + storage + HUD broadcast)

```bash
HUD_BROADCAST_ENABLED=true RETRIEVAL_ENABLED=True SAVE_TO_MEMORY=true \
    python pipeline/live.py test_videos/your_clip.mp4
```

- `RETRIEVAL_ENABLED` — queries Zep/Graphiti for context on recognized people
- `SAVE_TO_MEMORY` — flushes labeled conversations to the knowledge graph
- `HUD_BROADCAST_ENABLED` — exposes a WebSocket at `ws://0.0.0.0:8765` for the Unity glasses app (override with `HUD_BROADCAST_HOST` / `HUD_BROADCAST_PORT`)

## Testing

All test commands should be run from inside `ar-glasses/` with the venv active.

### 1. Smoke test — imports only, no webcam needed

```bash
python3 -c "
from models import BoundingBox, DetectedFace, FaceEmbedding, IdentityMatch, Person
from config import EDGEFACE_ROOT, EDGEFACE_CHECKPOINT, DB_PATH
print('Core imports: OK')
print(f'Checkpoint exists: {EDGEFACE_CHECKPOINT.exists()}')
"
```

Expected output:
```
Core imports: OK
Checkpoint exists: True
```

### 2. Face detection on a static image

```bash
python3 -c "
import cv2
from processing.face_detector import FaceDetector

detector = FaceDetector()
img = cv2.imread('path/to/image.jpg')
faces = detector.detect(img)
print(f'Detected {len(faces)} face(s)')
for f in faces:
    print(f'  bbox=({f.bbox.x1},{f.bbox.y1})-({f.bbox.x2},{f.bbox.y2}) conf={f.bbox.confidence:.2f}')
    print(f'  crop shape: {f.crop.shape}')  # expect (112, 112, 3)
detector.close()
"
```

### 3. Face embedding extraction

```bash
python3 -c "
import cv2
from processing.face_detector import FaceDetector
from processing.face_embedder import FaceEmbedder

detector = FaceDetector()
embedder = FaceEmbedder()

img = cv2.imread('path/to/image.jpg')
faces = detector.detect(img)
if faces:
    emb = embedder.embed(faces[0].crop)
    print(f'Embedding shape: {emb.vector.shape}')   # expect (512,)
    print(f'Embedding norm: {float(sum(emb.vector**2)**0.5):.4f}')
else:
    print('No face detected')
detector.close()
"
```

### 4. Compare two images (same/different person)

```bash
python3 -c "
import cv2, numpy as np
from processing.face_detector import FaceDetector
from processing.face_embedder import FaceEmbedder

detector = FaceDetector()
embedder = FaceEmbedder()

img1 = cv2.imread('path/to/person_a.jpg')
img2 = cv2.imread('path/to/person_b.jpg')

f1 = detector.detect(img1)
f2 = detector.detect(img2)

if f1 and f2:
    e1 = embedder.embed(f1[0].crop).vector
    e2 = embedder.embed(f2[0].crop).vector
    sim = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))
    print(f'Cosine similarity: {sim:.4f}')
    print(f'Same person: {\"YES\" if sim > 0.4 else \"NO\"} (threshold: 0.4)')
detector.close()
"
```

### 5. Database operations — no webcam needed

```bash
python3 -c "
from storage.database import Database
from models import FaceEmbedding
import numpy as np

db = Database(':memory:')  # in-memory, nothing written to disk

pid = db.add_person('Test User', notes='smoke test')
print(f'Created person ID: {pid}')

fake_emb = FaceEmbedding(vector=np.random.randn(512).astype(np.float32), model_name='test')
db.add_embedding(pid, fake_emb)

person = db.get_person(pid)
print(f'Name: {person.name}')
print(f'Embeddings: {len(person.embeddings)}, dim: {person.embeddings[0].vector.shape}')

db.close()
print('Database: OK')
"
```

### 6. End-to-end with webcam

```bash
python3 main.py --mode enroll   # enroll yourself
python3 main.py                 # run recognition
```

Your name should appear in a green box. If you see "Unknown" in red, try enrolling again with more captures (5+) from different angles and lighting.

## Phase 2 (not yet implemented)

The following modules are stubbed and raise `NotImplementedError` when called:

| Module | Requires |
|--------|---------|
| `input/microphone.py` | `pyaudio` + `portaudio19` system lib |
| `processing/audio_processor.py` | `whisperx` |
| `output/companion_app.py` | `flask` |

Install when ready:
```bash
sudo apt-get install portaudio19-dev   # system lib for PyAudio (Linux/WSL)
pip install pyaudio webrtcvad flask
pip install git+https://github.com/m-bain/whisperX.git
```
