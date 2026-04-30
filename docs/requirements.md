# Requirements & Scope

This document captures what ArgusEye is built to do, what it is **not** built to do, and the measurable targets we hold ourselves to. It is the contract between the project goals and the implementation.

## 1. Goals

ArgusEye is a smart-glasses platform that helps the wearer remember the people they meet by:

1. Recognizing previously-seen faces in real time on the HUD.
2. Recalling what was discussed with that person in prior conversations.
3. Suggesting a single concrete conversation opener tailored to that person.

Everything else — the dashboard, the companion mobile app, the knowledge-graph viewer — exists to support these three goals.

## 2. In scope

- **1-on-1, face-to-face conversations** between the wearer and one other person.
- **Continuous capture** of audio and video while the glasses are worn and the backend is running.
- **Per-person knowledge persistence** across sessions, days, and weeks.
- **Post-session labeling** via a companion mobile app: the wearer assigns real names to face clusters that the backend auto-discovers.
- **Local backend hosting**: the dashboard, People API, knowledge graph, and HUD broadcast all run on the wearer's laptop. The glasses are a thin client.

## 3. Out of scope (non-goals)

These were considered and explicitly excluded. They are not bugs.

- **Group conversations of 3+ speakers.** Robust real-time multi-party diarization is an open research problem. Speaker attribution degrades sharply past two speakers and corrupts the per-person knowledge graph. The system is validated only for 1-on-1.
- **On-device inference on the glasses.** The Inmo Air 3 cannot run Whisper, MediaPipe, EdgeFace, and a Neo4j client simultaneously at usable latency. All inference is offloaded to the laptop.
- **Cloud hosting / multi-tenant deployment.** ArgusEye is a single-user prototype. There is no auth on the People API, no encryption-at-rest beyond what SQLite/Neo4j ship with, and no notion of a second wearer.
- **Account creation, payments, or any third-party data sharing** beyond the OpenAI and Groq API calls the user explicitly opts into via API keys.
- **Recognition of people the wearer has never met.** No external face databases are consulted. A face is only known after it has been seen and labeled in this system.
- **Always-on background recording.** Capture runs only while the dashboard is running.

## 4. Functional requirements

| ID | Requirement |
|---|---|
| FR-1 | The system SHALL detect faces in the live video stream and produce a 512-d embedding for each. |
| FR-2 | The system SHALL match each face embedding against the local identity DB and return a `person_id` and confidence. |
| FR-3 | The system SHALL transcribe spoken audio to text with per-segment timestamps. |
| FR-4 | The system SHALL classify each speech segment as `Wearer` or `Other` based on per-track RMS amplitude. |
| FR-5 | The system SHALL extract entities, relationships, and facts from completed conversation transcripts and write them to a Neo4j knowledge graph. |
| FR-6 | The system SHALL, when a labeled person is recognized, retrieve facts about that person from the knowledge graph and synthesize a single conversation opener. |
| FR-7 | The system SHALL push the recognition payload to any connected glasses HUD over WebSocket within `RETRIEVAL_COOLDOWN_SECONDS` (default 10s) of recognition. |
| FR-8 | The companion mobile app SHALL be able to list unlabeled face clusters and assign a real name to each. |
| FR-9 | The system SHALL not write conversations to the knowledge graph for unlabeled face clusters; those interactions SHALL be deferred until the cluster is labeled. |
| FR-10 | The system SHALL operate in two mutually exclusive modes: `enroll` (write + retrieve) and `retrieval` (retrieve only). |

## 5. Non-functional requirements

### Performance

| Metric | Target | Measured (current) |
|---|---|---|
| Per-frame vision cost (detect + embed) | ≤ 30 ms at 30 fps | ~25–35 ms (see [docs/evaluation.md](evaluation.md)) |
| Speaker classification (segment-level) | ≤ 25% error in 1-on-1 audio | 25.5% best-case ([evaluation.md](evaluation.md)) |
| HUD context delivery (recognition → WebSocket send) | ≤ 2 s end-to-end | Bounded by `RETRIEVAL_COOLDOWN_SECONDS` |
| Backend cold start (`./run.sh` to dashboard ready) | ≤ 60 s | Dominated by Neo4j (~20–30 s) |

### Reliability

- The pipeline SHALL continue running if a single retrieval request fails (errors logged, not propagated).
- The People API SHALL remain responsive (< 50 ms steady state) when the knowledge-graph flush is slow; saves are fire-and-forget on a background loop.
- Pressing `Ctrl+C` on `run.sh` SHALL cleanly stop every service it started, including child processes that outlive their subshell.

### Hardware constraints

- Reference glasses: **Inmo Air 3** (Android 8). Any Android phone running the Unity APK is a valid stand-in.
- Reference laptop: any machine that can run Docker Desktop and Python 3.10+. NVIDIA GPU with CUDA 12.1 is recommended for vision throughput but not required.
- The wearer-side mic must be a near-field/bone-conducted mic distinct from the room mic for RMS-based speaker classification to work. See [docs/decisions/0001-face-pipeline.md](decisions/0001-face-pipeline.md) and [docs/evaluation.md](evaluation.md).

### Privacy & data handling

- All face embeddings, thumbnails, and conversation transcripts are stored **only locally** in `ar-glasses/data/people.db` (SQLite) and in the local Neo4j volume `neo4j_data`.
- The system makes outbound calls only to: Groq (Whisper transcription), OpenAI (entity extraction + context formatting). Audio bytes are sent to Groq, transcript text is sent to OpenAI. No data is sent without one of these two API calls.
- API keys live in `ar-glasses/.env` and are never committed.

## 6. Assumptions

- The wearer has working OpenAI and Groq API keys with sufficient quota.
- The wearer's laptop and the glasses are on the same network so the Unity app can reach `ws://<laptop-ip>:8765`.
- Conversations are in English (`WHISPER_LANGUAGE = "en"`).
- The wearer's mic gain is roughly stable within a session; large gain jumps invalidate the adaptive RMS boundary.

## 7. Acceptance — how we know we're done

The system is "done enough to demo" when:

1. A new person can walk into frame, be auto-clustered, be labeled in the mobile app, and be correctly recognized on the next encounter.
2. The HUD context card includes a `last_spoke_about` summary and an `ask_about` opener that reflects facts from the previous conversation.
3. `./run.sh --enroll` runs end-to-end on a fresh clone with only a `.env` populated and no manual setup beyond `pip install -r requirements.txt`.

Out-of-scope edge cases (3+ speakers, cross-room mic switching, identity drift after months of embedding accumulation) are tracked but not blocking.
