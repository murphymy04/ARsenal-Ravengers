# 0001 — Face recognition pipeline (MediaPipe + EdgeFace)

**Status:** accepted
**Date:** 2026-02-15

## Context

We need to detect faces and recognize identities in a live 30 fps stream from the Inmo Air 3 (low-power Android glasses streaming to a laptop backend). Per-frame budget for the vision path is roughly 30 ms; we must leave headroom for VAD, transcription, and HUD broadcast on the same machine.

Constraints:

- CPU-only must be viable on a developer laptop; GPU is a bonus, not a requirement.
- Faces in 1-on-1 conversations are usually 1–3 m away, frontal-ish, and may be motion-blurred from H.264 compression.
- We need a fixed-dimension embedding to store in SQLite and match with cosine similarity. No retraining.

## Options considered

| Option | Pros | Cons |
|---|---|---|
| **MediaPipe (BlazeFace) + EdgeFace** | Both run CPU-only at real time. EdgeFace is purpose-built for edge devices, ships pretrained 512-d embeddings. MediaPipe is mature, batteries-included. | Two separate models to manage. EdgeFace less ubiquitous than ArcFace, fewer eyes on it. |
| **InsightFace (RetinaFace + ArcFace)** | Industry standard, strong accuracy, large community. | Heavier than EdgeFace, slower on CPU. ArcFace embeddings are 512-d but the inference path needs ONNX runtime and careful preprocessing. |
| **OpenCV Res10-SSD + dlib face_recognition** | Simplest install, dlib well-known. | dlib's 128-d encoding is older and less discriminative than ArcFace/EdgeFace; SSD detector is slower than BlazeFace. |
| **Pure cloud (AWS Rekognition / Google Vision)** | Zero local cost, very accurate. | Per-frame calls infeasible (latency + cost + bandwidth). Defeats the point of a local-first prototype. |

We additionally support **OpenCV Res10-SSD as the detector backend** (`FACE_DETECTOR_MODEL = "opencv"` in [ar-glasses/processing/config.py](../../ar-glasses/processing/config.py)) as a fallback because in our test footage it was slightly more robust to compression artifacts than MediaPipe at the same frame budget.

## Decision

Use **OpenCV Res10-SSD for detection + EdgeFace (`edgeface_base`, 512-d) for embeddings**, with cosine similarity ≥ `MATCH_THRESHOLD` (0.4) deciding "same person."

Cluster auto-discovery uses unsupervised accumulation: a face is promoted to a real cluster after `MIN_SIGHTINGS_TO_CLUSTER` (12) consistent observations with internal cosine similarity ≥ `PENDING_CLUSTER_SIMILARITY` (0.5). All thresholds live in [processing/config.py](../../ar-glasses/processing/config.py).

## Consequences

**Enables:**
- Real-time per-frame inference on a laptop CPU (~25–35 ms with embedding only on detected faces).
- Stride-based acceleration: `VISION_STRIDE` skips the embedding step on N-1 of every N frames, reusing the last identity. At stride=5 we get ~5× throughput on offline videos.
- Embeddings are 512 floats, so a person with 20 embeddings (the cap, `MAX_EMBEDDINGS_PER_PERSON`) is ~40 KB in SQLite — trivial.

**Costs:**
- Two model weight files to ship: the OpenCV face detector and the EdgeFace checkpoint. Both are bundled under `ar-glasses/edgeface/`.
- Threshold `MATCH_THRESHOLD = 0.4` is tuned for our test set; new mounting positions or cameras will likely require retuning.
- We rely on temporal smoothing (`TEMPORAL_SMOOTHING_FRAMES = 7`) and bbox-proximity tracking (`FACE_MAX_MOVE_PX = 100`) to absorb single-frame mismatches. This works for slow conversational motion; fast head turns can break the track and create a new pending cluster.

**Locks us out of:**
- Anything requiring 3-D facial geometry (mask detection, gaze estimation as a primary signal). MediaPipe FaceLandmarker can be bolted on (see `SPEAKING_BACKEND = "mediapipe"`) but is not on the recognition path.
- Cross-pose hard cases (extreme profile views) — the embedding space is tighter on frontal faces. We mitigate by accumulating up to 20 diverse embeddings per person (`EMBEDDING_DIVERSITY_THRESHOLD = 0.15`).
