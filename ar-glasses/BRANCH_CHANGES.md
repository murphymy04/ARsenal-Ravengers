# Branch Changes: `myles/retrieval` vs `main`

This document summarizes every meaningful change in this branch for a fresh agent coming in cold.

---

## 1. USB streaming via ADB/scrcpy — full rewrite of the input layer

**Files:** `input/glasses_net.py`, `input/glasses_adapter.py`, `input/config.py`

### What was removed
The old WiFi-based streaming protocol is gone:
- `DiscoveryService` — UDP broadcast on port 5002 for glasses discovery
- `VideoReceiver` — UDP listener on port 5000 for JPEG frames
- `AudioReceiver` — TCP listener on port 5003 for PCM audio

### What replaced it

**`input/glasses_net.py`** now contains two classes:

**`AdbWatcher`**
- Polls `adb devices` every 2 seconds
- On new device: runs `adb forward tcp:HUD_PORT tcp:HUD_PORT` so the glasses Unity app can reach the laptop's WebSocket server at `ws://localhost:8765` without WiFi
- Calls `on_device_connected(serial)` / `on_device_disconnected(serial)` callbacks

**`ScrcpyStream`**
- Called with a device serial; spawns scrcpy + two FFmpeg subprocesses over USB
- scrcpy flags: `--video-source=camera --camera-facing=back --camera-fps=10 --audio-source=mic-camcorder --no-playback --no-window --record=<named_pipe> --record-format=mkv --force-adb-forward`
- scrcpy writes mkv to a **named pipe** (Windows: `\\.\pipe\ar_glasses_XXXX` via `win32pipe`; POSIX: FIFO via `os.mkfifo`). A relay thread reads from the pipe and tees bytes into two `queue.Queue` objects — one per FFmpeg process
- **Video FFmpeg**: `-map 0:v -vf scale=1280:720 -r 10 -f rawvideo -pix_fmt bgr24 pipe:1` — outputs raw BGR frames to stdout
- **Audio FFmpeg**: `-map 0:a -f s16le -ar 16000 -ac 1 pipe:1` — outputs raw PCM to stdout
- Frames are read in fixed-size chunks (`1280 × 720 × 3` bytes) and stored in a bounded `deque[FrameData]`; audio in `AUDIO_CHUNK_SAMPLES=1600` sample chunks into `deque[AudioChunk]`
- Timestamps are synthetic (frame_seq × ms_per_frame, chunk_seq × ms_per_chunk) — not wall clock
- Exposes the same query interface as the old receivers: `get_frame_after(seq)`, `get_audio_range(ts_start, ts_end)`, `get_latest_audio_timestamp()`, `get_stats()`

**Critical implementation notes:**
- Named pipe is required (not `--record=-`): scrcpy's C runtime buffers stdout, so no data arrives until the process exits
- `-r 10` on the video FFmpeg is required: without it FFmpeg defaults to 25fps output and duplicates every 10fps frame 2.5×, flooding the deque with ~1600 duped frames per minute
- `--force-adb-forward` is required: INMO Air 3 and most test devices don't support `adb reverse`
- pywin32 (`win32pipe`, `win32file`, `pywintypes`) required on Windows; standard `os.mkfifo` on POSIX

**`input/glasses_adapter.py`** updated:
- `PairingLoop` now takes a single `stream: ScrcpyStream` (not separate `video_rx` + `audio_rx`)
- Removed `cv2.cvtColor(frame.data, cv2.COLOR_GRAY2BGR)` — scrcpy outputs BGR directly, `frame.data` is passed as-is
- `GlassesServer` now holds `self.stream = ScrcpyStream()` and `self.adb_watcher = AdbWatcher(...)` instead of discovery/video/audio receivers
- `_on_device_connected(serial)` → calls `recorder.start_session()`, resets pairing loop on reconnect, calls `stream.connect(serial)`
- `_on_device_disconnected(serial)` → calls `stream.disconnect()`

**`input/config.py`** additions:
```python
GLASSES_VIDEO_WIDTH = 1280
GLASSES_VIDEO_HEIGHT = 720
TRAINING_DATA = os.getenv("TRAINING_DATA", "false").lower() == "true"
TRAINING_DATA_DIR = DATA_DIR / "training_data"
```

---

## 2. Training data capture

**File:** `input/training_recorder.py` (new file)

When `TRAINING_DATA=true`, `GlassesServer` creates a `TrainingRecorder` and passes `recorder.write` as the `on_emit` callback to `PairingLoop`. Every emitted `(bgr, audio_float32, ts)` pair is written to disk:
- **MP4** via `cv2.VideoWriter` at 10fps nominal
- **WAV** (16-bit mono, 16 kHz) via Python `wave`

Files are named `YYYYMMDD_HHMM_NNN.mp4/.wav` under `TRAINING_DATA_DIR`. A new session (new filename pair) starts on each device connect event.

`PairingLoop` gained an `on_emit: Optional[Callable[[tuple], None]]` parameter — called synchronously in `_put_pair` before the pair is enqueued.

---

## 3. Flush pipeline refactor — single transcript, no segment list

**File:** `pipeline/recording_buffer.py`

`sanitize()` was a stub (pass-through). It is now real:
- Concatenates all buffered audio into one array, converts to WAV bytes
- Calls Groq Whisper (`whisper-large-v3`) for a single transcription of the whole window
- Counts `(person_id, name)` pairs across diarization segments to find the dominant speaker
- Returns a `SanitizedConversation` dataclass: `spoke_with`, `person_id`, `transcript` (plain string), `window_start`, `window_end`

The old code passed `list[dict]` segment lists everywhere. The entire flush path now passes a plain `transcript: str`:
- `_sanitize_and_flush` in `live.py` calls `sanitize()` → gets `SanitizedConversation` → calls `_store_interaction(person_id, transcript)` and `_save_conversation(conversation)`
- `_resolve_and_save` is gone; replaced by `_save_conversation(conversation: SanitizedConversation)` which resolves the labeled name and calls `save_to_memory(transcript, other_name=person.name)`
- `_store_interaction` now takes `transcript: str` directly (no more joining segments)
- `KnowledgeStore.save()` now takes `transcript: str` instead of `segments: list[dict]`

---

## 4. Knowledge store improvements

**File:** `pipeline/knowledge.py`

- `save()` and `save_to_memory()` signatures changed from `segments: list[dict]` to `transcript: str`
- Zep episode body is now the raw transcript string (not "Speaker: text\n..." formatted segments)
- `source_description` updated to note speaker attribution is unavailable and to treat statements as primarily from the other person
- `EXTRACTION_INSTRUCTIONS` rewritten: treat transcript as a single mixed stream, default attribution to the other person, skip filler/meta-discussion, focus on learning facts about the person the wearer is meeting
- `_extract_topic_hint` now takes `transcript: str` directly

---

## 5. Auto-enroll gate for identity

**File:** `processing/config.py`, `pipeline/identity.py`

New env var: `AUTO_ENROLL_ENABLED` (default `false`).

When `AUTO_ENROLL_ENABLED=false` (default):
- Known faces: only `update_last_seen()` is called — no new embeddings stored
- Unknown faces: immediately return `IdentityMatch(is_known=False)` — no pending cluster logic

When `AUTO_ENROLL_ENABLED=true`: original behavior (pending clusters, automatic promotion).

This prevents the identity module from auto-enrolling strangers in production use.

---

## 6. Retrieval improvements

**Files:** `pipeline/retrieval.py`, `pipeline/diarization.py`, `pipeline/config.py`

**Eviction fix (`retrieval.py`):**
Before: `_should_retrieve` used `self._last_retrieval.get(name, 0)`, so a person seen for the first time at `t=5` had `last=0` and `5 - 0 = 5 >= cooldown`, always passing. Now it checks `if last is not None` — first-time retrieval is always allowed, subsequent ones respect the cooldown.

**Better Graphiti query:** Changed from `"{name} projects work interests"` to `"What does {name} do? What are {name}'s projects, plans, and interests?"` with `num_results=3` (was unlimited).

**New config knobs:**
- `RETRIEVAL_MIN_TRACK_FRAMES` (default 1) — a face track must be seen for at least this many frames before retrieval fires, preventing spurious queries on momentary detections
- `RETRIEVAL_MAX_PENDING_FRAMES` (default 100) — evicts a pending track that never becomes known after this many frames

**Diarization pending tracker (`diarization.py`):** New `_pending_retrieval: dict[int, int]` counter tracks frames-seen per track_id. Retrieval only fires after `RETRIEVAL_MIN_TRACK_FRAMES`; unknown tracks are evicted after `RETRIEVAL_MAX_PENDING_FRAMES`.

---

## 7. Dashboard improvements

**Files:** `dashboard.py`, `templates/dashboard.html`

- **Conversations view**: `DebugState` gains `conversations: list[dict]` (one entry per flush: `spoke_with`, `person_id`, `transcript`, `start`, `end`). A flush interceptor (`install_flush_interceptor`) wraps `driver._sanitize_and_flush` to capture flushed conversations into the state — displayed as full conversation paragraphs with a toggle to fall back to per-segment diarization view
- **Recording indicator**: `DebugState.recording` and `DebugState.flushing` booleans; `state.recording = driver.recording_buffer.flag` updated each frame loop; surfaced in `/state` JSON response and shown in the UI
- **Glasses mode**: `--glasses` flag sets `hide_audio=True` in the template (same as `--fast`) — suppresses the `<audio>` playback element since live glasses mode has no playback track
- **Retrieval context**: `drain_new_retrieval` now passes `last_spoke`, `last_spoke_about`, `ask_about`, `raw_facts` from the context dict instead of a flat `facts` list
- **`/state` endpoint** now returns `recording`, `flushing`, `conversations` alongside existing fields

---

## 8. Config tuning

| Setting | Old | New |
|---|---|---|
| `LIVE_BUFFER_SECONDS` | 10 | 7 |
| `DIARIZATION_QUIET_CHUNKS_TO_FLUSH` | 2 | 1 |

Faster flush cadence — don't wait as long before processing a conversation window.

---

## Key invariants for new code

- **No WiFi anywhere**: all glasses communication is ADB over USB. The HUD WebSocket (`ws://localhost:8765`) is forwarded via `adb forward`.
- **Single stream object**: `ScrcpyStream` is passed as both video and audio source to `PairingLoop`.
- **Transcript not segments**: `sanitize()` returns `SanitizedConversation.transcript: str`. Nothing downstream receives `list[dict]` segment lists anymore.
- **`AUTO_ENROLL_ENABLED=false` by default**: identity module won't auto-create person records for new faces unless explicitly enabled.
- **`TRAINING_DATA=false` by default**: no disk writes during normal operation.
