"""AR Glasses Prototype - Main Orchestrator.

Wires together the face recognition pipeline:
  Camera -> Detect -> Embed -> Match -> Display

Two threads + optional third:
  Thread 1 (main): Video loop - face recognition + display
  Thread 2: Audio loop (Phase 2 - stubbed)
  Thread 3: Companion app (Phase 2 - stubbed)
"""

import sys
import time
import argparse
from pathlib import Path

# Ensure the project root is on sys.path regardless of where the script is invoked from
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np

from config import (
    CAMERA_SOURCE, DB_PATH,
    EMBEDDING_UPDATE_INTERVAL, MAX_EMBEDDINGS_PER_PERSON,
    EMBEDDING_DIVERSITY_THRESHOLD, FACE_BLUR_THRESHOLD,
    MIN_SIGHTINGS_TO_CLUSTER, PENDING_CLUSTER_SIMILARITY, PENDING_EXPIRY_FRAMES,
    MERGE_SIMILARITY_THRESHOLD,
)
from models import FaceEmbedding, IdentityMatch, Person
from input.camera import Camera
from processing.face_detector import FaceDetector
from processing.face_embedder import FaceEmbedder
from processing.face_matcher import FaceMatcher
from processing.face_tracker import FaceTracker
from storage.database import Database
from storage.enrollment import Enrollment
from output.display import Display


def _is_diverse(
    new_emb: FaceEmbedding,
    existing: list[FaceEmbedding],
    min_distance: float,
) -> bool:
    """Return True if new_emb is at least min_distance (cosine) from every
    existing embedding.  Prevents storing near-duplicate embeddings (#8)."""
    if not existing:
        return True
    new_n = new_emb.vector / (np.linalg.norm(new_emb.vector) + 1e-8)
    for e in existing:
        e_n = e.vector / (np.linalg.norm(e.vector) + 1e-8)
        if float(np.dot(new_n, e_n)) >= 1.0 - min_distance:
            return False
    return True


def _cv2_input(image: np.ndarray, prompt: str) -> str:
    """Show image in a cv2 window and collect typed input. Returns the string on Enter, or
    'DELETE' / '' on special commands. Backspace supported. Esc = skip."""
    typed = ""
    win = "Labeling"
    while True:
        display = image.copy()
        cv2.putText(
            display, f"{prompt}: {typed}_",
            (10, display.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )
        cv2.imshow(win, display)
        key = cv2.waitKey(20) & 0xFF

        if key == 13:          # Enter — confirm
            break
        elif key == 27:        # Esc — skip
            typed = ""
            break
        elif key == 8 or key == 127:  # Backspace
            typed = typed[:-1]
        elif 32 <= key < 127:  # printable ASCII
            typed += chr(key)

    cv2.destroyWindow(win)
    return typed


def run_video_loop(
    camera: Camera,
    detector: FaceDetector,
    embedder: FaceEmbedder,
    matcher: FaceMatcher,
    display: Display,
    db: Database,
):
    """Main video loop: detect, embed, match, display.

    Unknown faces are held in a pending buffer until they have been seen
    MIN_SIGHTINGS_TO_CLUSTER times consistently (similar embeddings within
    PENDING_CLUSTER_SIMILARITY).  Only then is a real 'Person N' cluster
    created in the database.  This prevents frame-to-frame embedding noise
    from generating spurious clusters.

    Known people accumulate fresh embeddings every EMBEDDING_UPDATE_INTERVAL
    frames (capped at MAX_EMBEDDINGS_PER_PERSON) to stay robust over time.
    """
    print("Starting video loop... Press 'q' to quit.")

    people = db.get_all_people()
    matcher.update_gallery(people)
    labeled = sum(1 for p in people if p.is_labeled)
    auto = len(people) - labeled
    print(f"Loaded {len(people)} cluster(s) ({labeled} labeled, {auto} auto).")
    print(f"New faces need {MIN_SIGHTINGS_TO_CLUSTER} consistent sightings before a cluster is created.")

    frame_count = 0
    start_time = time.time()
    last_embedding_update: dict[int, int] = {}

    tracker = FaceTracker()

    # Pending observations for faces not yet promoted to real clusters.
    # Each entry: {'embeddings': [...], 'thumbnail': np.ndarray, 'last_frame': int}
    pending: list[dict] = []

    for frame in camera.frames():
        timestamp = time.time()
        faces = detector.detect(frame, timestamp=timestamp)

        raw_matches = []
        gallery_dirty = False

        # Expire stale pending entries
        pending = [p for p in pending if frame_count - p["last_frame"] < PENDING_EXPIRY_FRAMES]

        for face in faces:
            embedding = embedder.embed(face.crop)
            match = matcher.match(embedding)

            if match.is_known:
                db.update_last_seen(match.person_id)

                # Accumulate a new embedding if the frame is sharp and diverse (#8)
                frames_since = frame_count - last_embedding_update.get(match.person_id, 0)
                if frames_since >= EMBEDDING_UPDATE_INTERVAL:
                    person = db.get_person(match.person_id)
                    if (
                        person
                        and len(person.embeddings) < MAX_EMBEDDINGS_PER_PERSON
                        and face.blur_score >= FACE_BLUR_THRESHOLD
                        and _is_diverse(embedding, person.embeddings, EMBEDDING_DIVERSITY_THRESHOLD)
                    ):
                        db.add_embedding(match.person_id, embedding)
                        gallery_dirty = True
                    last_embedding_update[match.person_id] = frame_count
            else:
                # Find the best matching pending cluster for this embedding
                query = embedding.vector
                qnorm = np.linalg.norm(query)
                best_idx, best_score = -1, -1.0

                if qnorm > 0:
                    for i, pc in enumerate(pending):
                        vecs = np.array([e.vector for e in pc["embeddings"][-5:]])
                        mean_vec = vecs.mean(axis=0)
                        mnorm = np.linalg.norm(mean_vec)
                        if mnorm == 0:
                            continue
                        score = float(np.dot(query, mean_vec) / (qnorm * mnorm))
                        if score > best_score:
                            best_score, best_idx = score, i

                if best_idx >= 0 and best_score >= PENDING_CLUSTER_SIMILARITY:
                    # Accumulate into existing pending cluster
                    pending[best_idx]["embeddings"].append(embedding)
                    pending[best_idx]["last_frame"] = frame_count

                    if len(pending[best_idx]["embeddings"]) >= MIN_SIGHTINGS_TO_CLUSTER:
                        # Enough consistent sightings — promote to a real cluster
                        thumbnail = pending[best_idx]["thumbnail"]
                        person_id, auto_name = db.add_auto_person(thumbnail=thumbnail)
                        for emb in pending[best_idx]["embeddings"]:
                            db.add_embedding(person_id, emb)
                        last_embedding_update[person_id] = frame_count
                        pending.pop(best_idx)
                        gallery_dirty = True
                        print(f"\nDiscovered new face: {auto_name}")
                        match = IdentityMatch(
                            person_id=person_id, name=auto_name,
                            confidence=1.0, is_known=True,
                        )
                else:
                    # Start a new pending cluster for this face
                    pending.append({
                        "embeddings": [embedding],
                        "thumbnail": cv2.cvtColor(face.crop, cv2.COLOR_RGB2BGR),
                        "last_frame": frame_count,
                    })

            raw_matches.append(match)

        if gallery_dirty:
            matcher.update_gallery(db.get_all_people())

        # Smooth identity predictions across frames (#9)
        matches = tracker.update(faces, raw_matches, frame_count)

        display.draw(frame, faces, matches)
        if not display.show(frame):
            break

        frame_count += 1
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            print(f"\rFPS: {fps:.1f} | Faces: {len(faces)} | Pending: {len(pending)}", end="", flush=True)

    print("\nVideo loop ended.")


def enroll_mode(db: Database, detector: FaceDetector, embedder: FaceEmbedder):
    """Interactive enrollment from webcam."""
    import cv2

    enrollment = Enrollment(db, detector, embedder)

    name = input("Enter person's name: ").strip()
    if not name:
        print("Name cannot be empty.")
        return

    print("Opening camera... (if this hangs, run from Windows PowerShell instead)")
    camera = Camera()
    print("Enrollment mode. Press 'c' to capture, 'q' to quit.")

    captured_images = []
    for frame in camera.frames():
        faces = detector.detect(frame)
        # Draw detection boxes for preview
        for face in faces:
            b = face.bbox
            cv2.rectangle(frame, (b.x1, b.y1), (b.x2, b.y2), (0, 255, 0), 2)

        cv2.putText(
            frame, f"Enrolling: {name} | Captured: {len(captured_images)} | 'c'=capture 'q'=done",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )
        cv2.imshow("Enrollment", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("c") and faces:
            captured_images.append(frame.copy())
            print(f"  Captured image {len(captured_images)}")
        elif key == ord("q"):
            break

    camera.close()
    cv2.destroyAllWindows()

    if captured_images:
        person_id = enrollment.enroll_from_images(name, captured_images)
        if person_id:
            print(f"Enrolled '{name}' with ID {person_id} ({len(captured_images)} images)")
        else:
            print("Failed to detect a face in any captured image.")
    else:
        print("No images captured.")


def db_info_mode(db: Database):
    """Print a summary of everyone enrolled in the database."""
    people = db.get_all_people()
    if not people:
        print("Database is empty — no people enrolled yet.")
        return

    print(f"\n{'─' * 55}")
    print(f"  {'ID':<5} {'Name':<20} {'Embeddings':<12} {'Last Seen'}")
    print(f"{'─' * 55}")
    for p in people:
        last_seen = p.last_seen.strftime("%Y-%m-%d %H:%M") if p.last_seen else "never"
        print(f"  {p.person_id:<5} {p.name:<20} {len(p.embeddings):<12} {last_seen}")
    print(f"{'─' * 55}")
    print(f"  Total: {len(people)} person(s)\n")


def db_delete_mode(db: Database):
    """Interactively delete a person or wipe the entire database."""
    people = db.get_all_people()
    if not people:
        print("Database is empty.")
        return

    db_info_mode(db)
    print("Options:")
    print("  Enter a person ID to delete that person")
    print("  Enter 'all' to wipe the entire database")
    print("  Enter 'cancel' to exit")

    choice = input("\nChoice: ").strip().lower()

    if choice == "cancel":
        print("Cancelled.")
        return

    if choice == "all":
        confirm = input(f"Delete ALL {len(people)} people? This cannot be undone. (yes/no): ").strip().lower()
        if confirm == "yes":
            for p in people:
                db.delete_person(p.person_id)
            print("Database wiped.")
        else:
            print("Cancelled.")
        return

    try:
        person_id = int(choice)
    except ValueError:
        print(f"Invalid input: '{choice}'")
        return

    person = db.get_person(person_id)
    if not person:
        print(f"No person found with ID {person_id}.")
        return

    confirm = input(f"Delete '{person.name}' (ID {person_id})? (yes/no): ").strip().lower()
    if confirm == "yes":
        db.delete_person(person_id)
        print(f"Deleted '{person.name}'.")
    else:
        print("Cancelled.")


def _do_merge(db: Database, keep: Person, discard: Person):
    """Move all embeddings from discard into keep, then delete discard."""
    for emb in db.get_embeddings(discard.person_id):
        db.add_embedding(keep.person_id, emb)
    db.delete_person(discard.person_id)


def merge_clusters_mode(db: Database):
    """Find and interactively merge clusters that look like the same person."""
    people = [p for p in db.get_all_people() if p.embeddings]

    if len(people) < 2:
        print("Need at least 2 clusters with embeddings.")
        return

    # Compute per-person mean embedding
    def mean_vec(person: Person) -> np.ndarray:
        return np.mean([e.vector for e in person.embeddings], axis=0)

    means = {p.person_id: mean_vec(p) for p in people}

    # Find all pairs above the merge threshold
    suggestions = []
    for i in range(len(people)):
        va = means[people[i].person_id]
        na = np.linalg.norm(va)
        if na == 0:
            continue
        for j in range(i + 1, len(people)):
            vb = means[people[j].person_id]
            nb = np.linalg.norm(vb)
            if nb == 0:
                continue
            sim = float(np.dot(va, vb) / (na * nb))
            if sim >= MERGE_SIMILARITY_THRESHOLD:
                suggestions.append((sim, people[i], people[j]))

    suggestions.sort(reverse=True)

    if not suggestions:
        print(f"No cluster pairs found with similarity >= {MERGE_SIMILARITY_THRESHOLD}.")
        return

    print(f"Found {len(suggestions)} candidate merge(s). Higher = more likely the same person.\n")
    merged_ids: set[int] = set()

    for sim, pa, pb in suggestions:
        if pa.person_id in merged_ids or pb.person_id in merged_ids:
            continue

        last_a = pa.last_seen.strftime("%Y-%m-%d %H:%M") if pa.last_seen else "never"
        last_b = pb.last_seen.strftime("%Y-%m-%d %H:%M") if pb.last_seen else "never"
        print(
            f"[sim={sim:.2f}]  A: '{pa.name}' (ID {pa.person_id}, {len(pa.embeddings)} emb, seen {last_a})"
            f"\n         B: '{pb.name}' (ID {pb.person_id}, {len(pb.embeddings)} emb, seen {last_b})"
        )

        choice = input("  Merge? (a=keep A name, b=keep B name, n=skip): ").strip().lower()
        if choice == "a":
            _do_merge(db, keep=pa, discard=pb)
            merged_ids.add(pb.person_id)
            print(f"  Merged B into A → '{pa.name}'")
        elif choice == "b":
            _do_merge(db, keep=pb, discard=pa)
            merged_ids.add(pa.person_id)
            print(f"  Merged A into B → '{pb.name}'")
        else:
            print("  Skipped.")

    print("\nMerge complete.")


def label_mode(db: Database):
    """Interactively assign real names to auto-discovered clusters."""
    people = db.get_all_people()
    unlabeled = [p for p in people if not p.is_labeled]

    if not unlabeled:
        print("No unlabeled clusters found. Run in 'run' mode first to discover faces.")
        return

    print(f"\nFound {len(unlabeled)} unlabeled cluster(s).")
    print("Commands: type a name to label, press Enter to skip, 'delete' to remove.\n")

    for person in unlabeled:
        last_seen = person.last_seen.strftime("%Y-%m-%d %H:%M") if person.last_seen else "never"
        print(f"Cluster '{person.name}' | {len(person.embeddings)} embedding(s) | last seen: {last_seen}")
        print("  Type name + Enter | Esc to skip | type 'delete' + Enter to remove")

        if person.thumbnail is not None:
            name = _cv2_input(person.thumbnail, "Name")
        else:
            name = input("  Name: ").strip()

        if name.lower() == "delete":
            db.delete_person(person.person_id)
            print(f"  Deleted '{person.name}'.")
        elif name:
            existing = db.get_person_by_name(name)
            if existing:
                _do_merge(db, keep=existing, discard=person)
                print(f"  Merged into existing '{name}'")
            else:
                db.update_person(person.person_id, name=name, is_labeled=True)
                print(f"  '{person.name}' → '{name}'")
        else:
            print("  Skipped.")

    print("\nLabeling complete.")


def main():
    parser = argparse.ArgumentParser(description="AR Glasses Prototype")
    parser.add_argument(
        "--mode",
        choices=["run", "enroll", "label", "merge", "db-info", "db-delete"],
        default="run",
        help=(
            "'run' for live recognition (auto-clusters new faces) | "
            "'label' to assign names to auto-discovered clusters | "
            "'merge' to find and merge duplicate clusters | "
            "'enroll' to manually add a named person | "
            "'db-info' to list enrolled people | "
            "'db-delete' to remove a person or wipe the database"
        ),
    )
    parser.add_argument(
        "--camera", type=int, default=CAMERA_SOURCE,
        help="Camera source index (default: 0)",
    )
    args = parser.parse_args()

    # Modes that only need the database
    if args.mode in ("db-info", "db-delete", "label", "merge"):
        db = Database()
        try:
            if args.mode == "db-info":
                db_info_mode(db)
            elif args.mode == "db-delete":
                db_delete_mode(db)
            elif args.mode == "label":
                label_mode(db)
            else:
                merge_clusters_mode(db)
        finally:
            db.close()
        return

    # Modes that need the full pipeline
    print("Initializing database...")
    db = Database()
    print("Initializing face detector (MediaPipe)...")
    detector = FaceDetector()
    print("Loading EdgeFace model (this may take 10-20s)...")
    embedder = FaceEmbedder()
    print("Ready.")
    matcher = FaceMatcher()

    try:
        if args.mode == "enroll":
            enroll_mode(db, detector, embedder)
        else:
            camera = Camera(source=args.camera)
            display = Display()
            try:
                run_video_loop(camera, detector, embedder, matcher, display, db)
            finally:
                camera.close()
                display.close()
    finally:
        detector.close()
        db.close()


if __name__ == "__main__":
    main()
