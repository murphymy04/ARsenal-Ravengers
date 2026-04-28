# ARsenal Ravengers — High-Level System Design

## Overview

Smart glasses that passively record conversations and video, build a persistent knowledge graph of every person the wearer has met, and surface relevant context (name, past interactions, facts) directly on the glasses display when a known face is recognized.

---

## Hardware

| Component | Role |
|---|---|
| Smart glasses | Capture continuous audio and video streams |
| Companion phone app | Post-session input: user identifies who they spoke with |

---

## Audio Pipeline

```
Audio Stream
  → Voice Activity Detection (VAD)
  → Whisper  [transcription via Groq API]
  → Transcript with speaker labels (assigned by VAD/RMS pipeline)
  → Zep Graphiti  [episode extraction]  [T]
  → Knowledge Graph (Neo4j)
```

1. **VAD** — filters silence and assigns each speech segment to wearer vs. other based on per-track RMS amplitude.
2. **Whisper** — transcribes speech via the Groq Whisper API.
3. **Transcript with speaker labels** — the central artifact that gets enriched and stored.
4. **Zep Graphiti** — parses the transcript into "episodes": entities, relationships, and facts are extracted and written into the Neo4j knowledge graph. *(T = triggered on transcript completion)*

---

## Video Pipeline

### Storage mode (building the identity DB)

```
Video Stream
  → Face & gesture detection (MediaPipe)
  → Storage
      ├── Blendshapes  [T]          — facial action / expression data per face
      └── Face Recognition (EdgeFace)
              → Store face as embedding
              → Facial Embeddings / Identities DB
```

- **MediaPipe** detects faces and body/hand gestures frame-by-frame.
- **Blendshapes** capture what each face is *doing* (expressions, micro-gestures) and are stored as timestamped events. *(T = triggered per detection)*
- **EdgeFace** converts each detected face into a vector embedding and persists it in the identity DB. This is the "enrollment" step — the first time a face is seen, it is stored without a name.

### Retrieval mode (recognizing known people)

```
Video Stream
  → Face & gesture detection (MediaPipe)
  → Retrieval
      → Face Recognition (EdgeFace)
          1. Fetch entity related to the face   ← Facial Embeddings / Identities DB
          2. Use identity as foreign key         → Knowledge Graph (Neo4j)
          3. Polish context                      → Context post-processing
          4. Display context on the glasses
```

When a face is recognized against the identity DB, the system:
1. Resolves the face embedding to an identity (person entity).
2. Queries the Neo4j knowledge graph using that identity as a key — retrieving everything known about this person (past conversations, facts, relationships).
3. Runs context post-processing to clean and format the retrieved knowledge.
4. Renders the final context on the glasses HUD.

---

## Companion App — Identity Linking

```
Phone app
  → Ask user: "Who did you talk to?"
      ├── Enrich transcript with name metadata  → Transcript with speaker labels
      │                                                   ↓
      │                                         Zep Graphiti (re-processes with real name)
      └── Store name in identity table          → Facial Embeddings / Identities DB
```

After each session the companion app prompts the user to name unidentified speakers. This closes two loops:

- **Knowledge graph loop** — the transcript is re-enriched with the real name, so Zep stores all facts under the correct identity.
- **Identity DB loop** — the name is written back to the face embedding record, so future face matches resolve to a named entity.

---

## Data Stores

| Store | Technology | Contents |
|---|---|---|
| Knowledge graph | Neo4j (via Zep Graphiti) | People, facts, relationships, conversation episodes |
| Identity / embeddings DB | Vector DB (e.g. pgvector / Qdrant) | Face embeddings keyed to person entities |

---

## End-to-End Flow Summary

1. Glasses capture audio + video continuously.
2. Audio → transcript → knowledge graph (Zep Graphiti → Neo4j).
3. Video → MediaPipe → faces enrolled as embeddings in identity DB.
4. Post-session: companion app links names to face embeddings and enriches the transcript.
5. On next encounter: glasses see a face → EdgeFace matches embedding → identity DB resolves name → Neo4j returns context → glasses display it.
