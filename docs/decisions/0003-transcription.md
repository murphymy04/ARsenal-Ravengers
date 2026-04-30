# 0003 — Transcription (Groq-hosted Whisper)

**Status:** accepted
**Date:** 2026-03-04

## Context

Each `LIVE_BUFFER_SECONDS` (7 s) window of audio needs to become a transcript with timestamps so downstream stages (RMS-based speaker assignment, knowledge-graph extraction) have something to chew on. We want to feel "live" — the dashboard captions should appear within a couple of seconds of speech ending.

Constraints:

- A 7-second audio window has to be transcribed in well under 7 seconds for the pipeline to keep up.
- The laptop is already running MediaPipe, EdgeFace, Silero VAD, and a Neo4j client. We do not have a spare GPU budget for a local Whisper-large.
- Cost has to be acceptable for student-project usage (single user, sporadic recording).

## Options considered

| Option | Pros | Cons |
|---|---|---|
| **Groq-hosted Whisper-large-v3** | Real-time-faster than real-time on Groq's hardware. `whisper-large-v3` quality. Trivial integration (`groq.Groq().audio.transcriptions.create`). Costs are low — sub-cent per 7 s clip. | Network round trip; offline operation is impossible. Audio leaves the device. |
| **Local `faster-whisper` (CTranslate2)** | Fully local. No data leaves the laptop. Good CPU performance with `small` or `medium`. | Quality drop vs `large-v3`. Eats CPU we'd rather spend on vision. Requires shipping a multi-GB model. |
| **Local `openai-whisper`** | Reference implementation. | Slowest of the three; no batching tricks. Not viable for live use without a GPU. |
| **OpenAI hosted Whisper** | Same `large-v3` quality. Simpler API key story (we already use OpenAI for the graph). | Slower than Groq on the same model. We're rate-limited differently than Groq. |

## Decision

Use **Groq-hosted `whisper-large-v3`** for transcription ([pipeline/transcription.py](../../ar-glasses/pipeline/transcription.py)). Each 7 s WAV is sent over HTTPS, response is `verbose_json` with per-segment timestamps which feed directly into our `TranscriptSegment` model.

We also use Groq for the HUD context formatter (`qwen/qwen3-32b`) since the API key and SDK are already wired up.

## Consequences

**Enables:**
- Real-time-ish captioning in the dashboard. Round-trip is typically well under a second for a 7 s clip.
- No local model weights to ship for transcription — `requirements.txt` stays slim.
- Easy to swap models: `model="whisper-large-v3"` is the only string to change.

**Costs:**
- Audio is uploaded to Groq's servers. This is a privacy footprint we accept for the prototype but it is documented in [docs/requirements.md](../requirements.md) §5.
- Without a network or without a `GROQ_API_KEY` the pipeline cannot transcribe — VAD and face detection still run, but the transcript and everything downstream is empty.
- We pay per request. For continuous all-day use this is non-trivial, but for demo and 1-hour test sessions the bill is in cents.

**Locks us out of:**
- Fully offline operation. Switching to local `faster-whisper` would be one file change in [pipeline/transcription.py](../../ar-glasses/pipeline/transcription.py); the rest of the pipeline is unaware of the backend.
- Diarization-aware transcription. Whisper itself does not separate speakers. We do speaker assignment downstream via per-track RMS in [processing/vad_speaker.py](../../ar-glasses/processing/vad_speaker.py) — see [docs/evaluation.md](../evaluation.md) for why that ceiling is ~25% segment error.
