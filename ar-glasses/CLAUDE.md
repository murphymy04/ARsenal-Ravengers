# AR Glasses — Diarization & Speaker Detection Pipeline

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
python debug_video.py test_videos/clip.mp4

# Debug overlay — accelerated, no audio sync, uses VISION_STRIDE
python debug_video.py --fast test_videos/clip.mp4
```

Tune `VISION_STRIDE` in `config.py`. Higher = faster but less temporal resolution on face detection.
For 1-on-1 conversations stride 3-5 is safe since faces move slowly.

## Key files

| File | Purpose |
|---|---|
| `config.py` | All tunable constants |
| `pipeline/live.py` | Main streaming pipeline (LivePipelineDriver) |
| `pipeline/driver.py` | Batch video pipeline (PipelineDriver) |
| `debug_video.py` | Visual debug overlay with face boxes |
| `processing/vad_speaker.py` | VAD + RMS speaker assignment |
| `processing/face_detector.py` | Face detection (3 backends) |
| `processing/face_tracker.py` | Temporal identity smoothing |
| `pipeline/identity.py` | NullIdentity / FullIdentity modules |

## Running

```bash
cd ar-glasses
# Live camera
python -m pipeline.live

# Video file (auto-caches results)
python -m pipeline.live test_videos/clip.mp4

# Debug overlay
python debug_video.py [--fast] test_videos/clip.mp4
```

## Code style

No inline comments in `pipeline/` and `processing/` — code speaks for itself.
