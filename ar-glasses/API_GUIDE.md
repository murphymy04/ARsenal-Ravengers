"""
API USAGE GUIDE — Companion Mobile App Backend
===============================================

This guide explains how to use the REST API for labeling face clusters from a mobile app.


QUICK START
───────────

1. Start the API server alongside or separately from the video pipeline:

   # Option A: Run as its own process
   python main.py --mode api --api-host 0.0.0.0 --api-port 5000

   # Option B: Run as a module
   python -m api --db data/people.db --port 5000 --host 0.0.0.0

2. From your mobile app, query:

   GET http://<glasses-ip>:5000/api/people/unlabeled

   This returns all auto-discovered face clusters awaiting names.

3. For each cluster, display the thumbnail and let the user type a name:

   POST http://<glasses-ip>:5000/api/people/{person_id}/label
   {
     "name": "Will Chen"
   }


API ENDPOINTS
─────────────

All responses are JSON. Error responses follow HTTP status codes (4xx, 5xx).


PEOPLE MANAGEMENT
═════════════════

GET /api/people
  Returns all enrolled people (labeled and unlabeled).

  Response:
  {
    "people": [
      {
        "person_id": 1,
        "name": "Will Chen",
        "is_labeled": true,
        "embedding_count": 8,
        "notes": "Founder",
        "created_at": "2026-03-23T10:15:00",
        "last_seen": "2026-03-23T14:30:00",
        "thumbnail_url": "/api/people/1/thumbnail"
      },
      {
        "person_id": 2,
        "name": "Person 2",
        "is_labeled": false,
        "embedding_count": 15,
        "notes": "",
        "created_at": "2026-03-23T11:00:00",
        "last_seen": "2026-03-23T14:25:00",
        "thumbnail_url": "/api/people/2/thumbnail"
      }
    ]
  }


GET /api/people/{person_id}
  Returns details for a single person.

  Example: GET /api/people/1

  Response:
  {
    "person_id": 1,
    "name": "Will Chen",
    "is_labeled": true,
    "embedding_count": 8,
    "notes": "Founder",
    "created_at": "2026-03-23T10:15:00",
    "last_seen": "2026-03-23T14:30:00",
    "thumbnail_url": "/api/people/1/thumbnail"
  }


LABELING (Mobile App Focus)
════════════════════════════

GET /api/people/unlabeled
  Returns only clusters awaiting names from the mobile app.
  This is the main endpoint for companion app flow.

  Response:
  {
    "people": [
      {
        "person_id": 2,
        "name": "Person 2",
        "embedding_count": 15,
        "last_seen": "2026-03-23T14:25:00",
        "thumbnail_url": "/api/people/2/thumbnail"
      },
      {
        "person_id": 5,
        "name": "Person 5",
        "embedding_count": 8,
        "last_seen": "2026-03-23T14:20:00",
        "thumbnail_url": "/api/people/5/thumbnail"
      }
    ]
  }


POST /api/people/{person_id}/label
  Assign a name to an unlabeled cluster.

  If name matches existing person:
    → Merges clusters (keeps embeddings from both)
    → Returns merged person details

  If new name:
    → Updates cluster with name, marks as labeled
    → Returns updated person details

  Request body:
  {
    "name": "Will Chen"
  }

  Response:
  {
    "person_id": 2,
    "name": "Will Chen",
    "is_labeled": true,
    "action": "labeled",
    "details": "Cluster 2 labeled as 'Will Chen'"
  }

  Or (merge case):
  {
    "person_id": 1,
    "name": "Will Chen",
    "is_labeled": true,
    "action": "merged",
    "details": "Cluster 2 merged into existing person 1"
  }

  Error cases:
    - 404: Person not found
    - 400: Person already labeled (not awaiting label)
    - 400: Empty name


THUMBNAILS
══════════

GET /api/people/{person_id}/thumbnail
  Returns person's face thumbnail as a JPEG image stream.

  Query parameters:
    - format: "jpeg" (default) or "base64"
    - format=jpeg: Returns raw JPEG bytes (image/jpeg MIME type)
    - format=base64: Returns JSON with base64-encoded data URL

  Usage in mobile app:

    a) Stream JPEG directly (fast, binary):
       <img src="http://localhost:5000/api/people/2/thumbnail" />

    b) Fetch base64 (for embedding in JSON, self-contained):
       GET /api/people/2/thumbnail?format=base64
       {
         "person_id": 2,
         "name": "Person 2",
         "thumbnail_data_url": "data:image/jpeg;base64,/9j/4AAQ..."
       }

  Error cases:
    - 404: Person not found
    - 404: Person has no thumbnail


ADMIN TOOLS
═══════════

POST /api/people/merge
  Manually merge two clusters (keep embeddings from both).

  Request body:
  {
    "keep_person_id": 1,
    "discard_person_id": 2
  }

  Response:
  {
    "person_id": 1,
    "name": "Will Chen",
    "is_labeled": true,
    "action": "merged",
    "details": "Merged person 2 into 1"
  }

  Error cases:
    - 404: One or both people not found
    - 400: Cannot merge person with themselves


DELETE /api/people/{person_id}
  Permanently delete a person and all embeddings (irreversible).

  Response:
  {
    "person_id": 2,
    "name": "Person 2",
    "status": "deleted"
  }

  Error cases:
    - 404: Person not found


EXAMPLE MOBILE APP WORKFLOW
═════════════════════════════

1. App opens → queries unlabeled clusters:

   GET /api/people/unlabeled

   Response:
   {
     "people": [
       {
         "person_id": 2,
         "name": "Person 2",
         "embedding_count": 15,
         "thumbnail_url": "/api/people/2/thumbnail"
       }
     ]
   }

2. App displays thumbnail using:

   <img src="http://localhost:5000/api/people/2/thumbnail" alt="Person 2" />

3. User sees face, types "Sarah Chen", hits "Label" button

4. App sends:

   POST /api/people/2/label
   {
     "name": "Sarah Chen"
   }

5. Server responds:

   {
     "person_id": 2,
     "name": "Sarah Chen",
     "is_labeled": true,
     "action": "labeled"
   }

6. Next time glasses sees Sarah, display shows "Sarah Chen" (not "Person 2")


ERROR HANDLING
══════════════

All errors follow HTTP standards:

  400 Bad Request
    - Empty name in label request
    - Invalid JSON body
    - Cannot merge person with themselves

  404 Not Found
    - Person ID doesn't exist
    - Thumbnail not found for person
    - Invalid endpoint

  500 Internal Server Error
    - Database connection issues
    - File I/O errors

Example error response:

  HTTP 404 Not Found
  {
    "detail": "Person 123 not found"
  }


PERFORMANCE NOTES
═════════════════

• Thumbnails are lazy-loaded from database only when requested.
  No performance penalty for large person counts.

• Base64 format embeds image in JSON — useful for self-contained responses
  but larger payload (~10-15KB per thumbnail).

• JPEG stream format is more efficient for high-latency connections.

• All database operations use WAL (write-ahead logging) for safe concurrent access.
  Video pipeline and API can run simultaneously.


DEPLOYMENT
══════════

Development:
  python main.py --mode api --api-port 5000

Production (with reload disabled):
  python main.py --mode api --api-host 0.0.0.0 --api-port 5000

Behind reverse proxy (e.g., nginx):
  - Proxy /api/* to localhost:5000
  - Set X-Forwarded-For, X-Forwarded-Proto headers for proper HTTPS support

With Docker:
  See ../knowledge/docker-compose.yml for Neo4j example structure
  (similar setup for API but doesn't need Docker — runs as Python process)


ENVIRONMENT VARIABLES
══════════════════════

Default values in config.py:
  - FLASK_HOST = "0.0.0.0"
  - FLASK_PORT = 5000
  - DB_PATH = "ar-glasses/data/people.db"

Override via CLI flags:
  python main.py --mode api --api-host 127.0.0.1 --api-port 8080


TROUBLESHOOTING
═══════════════

"Connection refused" on mobile app:
  - Check API is running: http://localhost:5000/
  - Verify network connectivity between phone and glasses
  - Ensure firewall allows port 5000 inbound

"No unlabeled clusters":
  - Video pipeline hasn't discovered unknown faces yet
  - Run `python main.py run` to capture faces first
  - Wait for MIN_SIGHTINGS_TO_CLUSTER (8) sightings minimum

"Thumbnail returns 404":
  - Face was detected but thumbnail not stored
  - Should not happen in normal workflow — report bug if seen

API Docs in browser:
  - Interactive OpenAPI docs: http://localhost:5000/docs
  - ReDoc (alternative): http://localhost:5000/redoc
"""
