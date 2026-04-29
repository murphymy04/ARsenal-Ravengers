"""
FACE LABELING PIPELINE — How It Works

=== OVERVIEW ===

The labeling pipeline is how unknown faces (discovered during live video) transition
from "auto-labeled clusters" to "real people with names". It's a multi-stage workflow:

STAGE 1: DISCOVERY (video pipeline, ar-glasses/pipeline/live.py via FullIdentity)
  └─ Unknown face detected → Embedded → Accumulated in "pending buffer"
     After MIN_SIGHTINGS_TO_CLUSTER (8) consistent sightings
     └─ Promoted to auto-labeled cluster: Person_ID, name="Person {ID}", is_labeled=FALSE

STAGE 2: REVIEW (companion mobile app)
  └─ Queries GET /api/people/unlabeled
  └─ Shows thumbnails + stats (embeddings count, last_seen)
  └─ User types name or confirms existing person
  └─ POST /api/people/{person_id}/label

STAGE 3: LABELING (API backend)
  └─ If new name: Update person_id.name, set is_labeled=TRUE
  └─ If existing name: Merge two clusters (keep old, discard new)
     └─ Move all embeddings from new to old person
     └─ Mark old cluster as most recent

STAGE 4: USE (video pipeline recognition)
  └─ Live matching now includes the real person
  └─ Next time that face appears, displays actual name (not "Person {ID}")


=== DATA FLOW IN DETAIL ===

1. UNKNOWN FACE ACCUMULATION (FullIdentity in pipeline/identity.py)
   ─────────────────────────────────────────

   for each frame:
     detect faces (MediaPipe BlazeFace)
     for each face:
       compute embedding (EdgeFace)
       match against gallery (FaceMatcher.match)
       
       if known person:
         ✓ update embeddings (every EMBEDDING_UPDATE_INTERVAL frames)
       else:
         ✗ accumulate in pending buffer with ID=track_id
         if face seen MIN_SIGHTINGS_TO_CLUSTER times AND embeddings consistent:
           → db.add_auto_person() creates: person_id, name="Person {id}", is_labeled=False
           → pending buffer → confirmed cluster

2. MOBILE APP QUERIES API
   ──────────────────────

   GET /api/people/unlabeled
   
   Returns:
   {
     "people": [
       {
         "person_id": 42,
         "name": "Person 42",          # auto-generated
         "is_labeled": false,
         "embedding_count": 15,        # how many views captured
         "last_seen": "2026-03-23T14:30:00",
         "thumbnail_url": "/api/people/42/thumbnail"  # OR base64 inline
       },
       ...
     ]
   }

   Mobile app displays thumbnail + "Person 42" on card
   User taps → types name → hits "Label" button

3. MOBILE APP SENDS LABEL
   ──────────────────────

   POST /api/people/{person_id}/label
   {
     "name": "Will Chen"
   }

   API checks if "Will Chen" exists:
     - YES → merge cluster 42 into existing (keep existing, move embeddings)
     - NO → update person 42: name="Will Chen", is_labeled=True

4. DATABASE UPDATES
   ───────────────

   Before:
     people:
       42 | Person 42 | is_labeled=False | embeddings=15
       50 | Will Chen | is_labeled=True  | embeddings=8

   After (merge case):
     people:
       42 | Person 42 | is_labeled=False | DELETED
       50 | Will Chen | is_labeled=True  | embeddings=15+8=23

   (or if new name):
     people:
       42 | Will Chen | is_labeled=True  | embeddings=15


=== KEY INVARIANTS ===

• Only unlabeled clusters (is_labeled=False) appear in the mobile app
• Thumbnails must be stored + served for UI confidence
• Merging must update gallery in FaceMatcher (or rebuild on next match)
• Each person must have ≥1 embedding (enforced at DB creation)
• Cluster names auto-generated as "Person {person_id}" to avoid collisions


=== API ENDPOINTS (implemented below) ===

GET    /api/people                    — all people (with stats)
GET    /api/people/unlabeled          — only clusters awaiting labels
GET    /api/people/{person_id}        — one person + their embeddings metadata
GET    /api/people/{person_id}/thumbnail  — image as JPEG or base64
POST   /api/people/{person_id}/label  — assign name to cluster
POST   /api/people/merge              — merge two clusters (admin tool)
DELETE /api/people/{person_id}        — remove cluster + embeddings


=== EXAMPLE WORKFLOW ===

1. Glasses running: captures unknown face 20 times, promotes to Person 5
2. Mobile app opens: GET /api/people/unlabeled → shows thumbnail for "Person 5"
3. User recognizes: "That's my friend Sarah" → types "Sarah Chen"
4. User hits Label: POST /api/people/5/label {"name": "Sarah Chen"}
5. API checks: no existing "Sarah Chen" → db.update_person(5, name="Sarah Chen", is_labeled=True)
6. Glasses now recognize Sarah → display name on overlay

Alternative: User types "Sarah Chen", API finds existing person 12 with that name
7. API: merge(keep=12, discard=5) → move embeddings from 5→12, delete 5
8. Gallery now has 12 embeddings for Sarah instead of just the old 8
9. Glasses recognition improves over time as more views accumulate
"""
