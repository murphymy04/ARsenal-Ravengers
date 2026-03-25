"""AR Glasses — entry point.

Wires together the face recognition pipeline and dispatches CLI modes.

Modes
-----
run       Live recognition with auto face clustering (default)
label     Assign names to auto-discovered clusters
merge     Find and merge duplicate clusters
enroll    Manually enroll a named person via webcam
db-info   List everyone in the database
db-delete Remove a person or wipe the database
"""

import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np

from config import (
    CAMERA_SOURCE, DB_PATH,
    EMBEDDING_UPDATE_INTERVAL, MAX_EMBEDDINGS_PER_PERSON,
    EMBEDDING_DIVERSITY_THRESHOLD, FACE_BLUR_THRESHOLD,
    MIN_SIGHTINGS_TO_CLUSTER, PENDING_CLUSTER_SIMILARITY, PENDING_EXPIRY_FRAMES,
    SPEAKING_BACKEND,
    FLASK_HOST, FLASK_PORT,
)
from models import FaceEmbedding, IdentityMatch
from input.camera import Camera
from processing.face_detector import FaceDetector
from processing.face_embedder import FaceEmbedder
from processing.face_matcher import FaceMatcher
from processing.face_tracker import FaceTracker
from storage.database import Database
from storage.speaking_log import SpeakingLog
from output.display import Display


# ---------------------------------------------------------------------------
# Helpers used only by the video pipeline
# ---------------------------------------------------------------------------

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [-1, 1]. Returns 0 for zero vectors."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _is_diverse(
    new_emb: FaceEmbedding,
    existing: list[FaceEmbedding],
    min_distance: float,
) -> bool:
    """True if new_emb is at least min_distance (cosine) from all existing embeddings."""
    return all(
        _cosine_sim(new_emb.vector, e.vector) < 1.0 - min_distance
        for e in existing
    )


# ---------------------------------------------------------------------------
# Video pipeline
# ---------------------------------------------------------------------------

def run_video_loop(
    camera: Camera,
    detector: FaceDetector,
    embedder: FaceEmbedder,
    matcher: FaceMatcher,
    display: Display,
    db: Database,
):
    """Main video loop: detect → embed → match → display.

    Unknown faces accumulate in a pending buffer.  Once a face has been seen
    MIN_SIGHTINGS_TO_CLUSTER times with consistent embeddings it is promoted
    to a real cluster in the database.

    Known people receive fresh embeddings every EMBEDDING_UPDATE_INTERVAL
    frames (up to MAX_EMBEDDINGS_PER_PERSON) to stay robust over time.
    """
    people = db.get_all_people()
    matcher.update_gallery(people)
    labeled = sum(1 for p in people if p.is_labeled)
    print(f"Loaded {len(people)} cluster(s) ({labeled} labeled, {len(people) - labeled} auto).")
    print(f"New faces need {MIN_SIGHTINGS_TO_CLUSTER} consistent sightings to be clustered.")
    print("Press 'q' to quit.")

    tracker = FaceTracker()
    speaking_log = SpeakingLog()
    log_path = DB_PATH.parent / f"speaking_log_{int(time.time())}.json"

    speaking_det = _init_speaking_detector()

    # pending: list of {'embeddings', 'mean', 'thumbnail', 'last_frame'}
    pending: list[dict] = []
    last_embedding_update: dict[int, int] = {}
    frame_count = 0
    start_time = time.time()

    try:
        for frame in camera.frames():
            timestamp = time.time()
            faces = detector.detect(frame, timestamp=timestamp)

            pending = [
                p for p in pending
                if frame_count - p["last_frame"] < PENDING_EXPIRY_FRAMES
            ]

            raw_matches = []
            gallery_dirty = False

            for face in faces:
                embedding = embedder.embed(face.crop)
                match = matcher.match(embedding)

                if match.is_known:
                    gallery_dirty |= _maybe_store_embedding(
                        db, match.person_id, embedding, face,
                        last_embedding_update, frame_count,
                    )
                else:
                    match, promoted = _update_pending(
                        db, embedding, face, pending, last_embedding_update, frame_count,
                    )
                    gallery_dirty |= promoted

                raw_matches.append(match)

            if gallery_dirty:
                matcher.update_gallery(db.get_all_people())

            matches, track_ids = tracker.update(faces, raw_matches, frame_count)

            if speaking_det is not None:
                for face, tid in zip(faces, track_ids):
                    speaking_det.add_crop(tid, face.crop)
                speaking_det.run_inference(frame_count, active_track_ids=set(track_ids))

            ts = time.time()
            for face, match, tid in zip(faces, matches, track_ids):
                if speaking_det is not None:
                    face.is_speaking = speaking_det.get_speaking(tid)
                speaking_log.update(
                    track_id=tid,
                    person_id=match.person_id if match.is_known else None,
                    name=match.name,
                    is_speaking=face.is_speaking,
                    timestamp=ts,
                )

            display.draw(frame, faces, matches)
            if not display.show(frame):
                break

            frame_count += 1
            if frame_count % 30 == 0:
                fps = frame_count / (time.time() - start_time)
                print(
                    f"\rFPS: {fps:.1f} | Faces: {len(faces)} | Pending: {len(pending)}",
                    end="", flush=True,
                )

    finally:
        if speaking_det is not None:
            speaking_det.close()
        speaking_log.save(log_path)

    print("\nVideo loop ended.")


def _init_speaking_detector():
    """Load the speaking detector selected in config, or None on failure."""
    if SPEAKING_BACKEND != "light_asd":
        print("Speaking detection: MediaPipe jawOpen blendshape")
        return None
    try:
        from processing.speaking_detector import SpeakingDetector
        det = SpeakingDetector()
        print("Speaking detection: Light-ASD (audio-visual)")
        return det
    except Exception as e:
        print(f"Light-ASD init failed ({e}), falling back to MediaPipe.")
        return None


def _maybe_store_embedding(
    db: Database,
    person_id: int,
    embedding: FaceEmbedding,
    face,
    last_update: dict[int, int],
    frame_count: int,
) -> bool:
    """Store a new embedding for a recognized person if it is sharp and diverse.

    Returns True if the gallery was modified.
    """
    db.update_last_seen(person_id)
    if frame_count - last_update.get(person_id, 0) < EMBEDDING_UPDATE_INTERVAL:
        return False

    person = db.get_person(person_id)
    last_update[person_id] = frame_count

    if (
        person
        and len(person.embeddings) < MAX_EMBEDDINGS_PER_PERSON
        and face.blur_score >= FACE_BLUR_THRESHOLD
        and _is_diverse(embedding, person.embeddings, EMBEDDING_DIVERSITY_THRESHOLD)
    ):
        db.add_embedding(person_id, embedding)
        return True

    return False


def _update_pending(
    db: Database,
    embedding: FaceEmbedding,
    face,
    pending: list[dict],
    last_update: dict[int, int],
    frame_count: int,
) -> tuple[IdentityMatch, bool]:
    """Match an unknown embedding against pending clusters.

    Adds to an existing cluster or starts a new one.  Promotes a cluster to
    the database once it reaches MIN_SIGHTINGS_TO_CLUSTER.

    Returns (match, gallery_dirty).
    """
    best_idx, best_score = -1, -1.0
    for i, pc in enumerate(pending):
        score = _cosine_sim(embedding.vector, pc["mean"])
        if score > best_score:
            best_score, best_idx = score, i

    if best_idx >= 0 and best_score >= PENDING_CLUSTER_SIMILARITY:
        pc = pending[best_idx]
        pc["embeddings"].append(embedding)
        pc["last_frame"] = frame_count
        n = len(pc["embeddings"])
        pc["mean"] = pc["mean"] + (embedding.vector - pc["mean"]) / n

        if n >= MIN_SIGHTINGS_TO_CLUSTER:
            person_id, auto_name = db.add_auto_person(thumbnail=pc["thumbnail"])
            for emb in pc["embeddings"]:
                db.add_embedding(person_id, emb)
            last_update[person_id] = frame_count
            pending.pop(best_idx)
            print(f"\nDiscovered new face: {auto_name}")
            return IdentityMatch(
                person_id=person_id, name=auto_name, confidence=1.0, is_known=True,
            ), True

        return IdentityMatch(person_id=None, name="Unknown", confidence=0.0, is_known=False), False

    # No match — start a new pending cluster
    pending.append({
        "embeddings": [embedding],
        "mean": embedding.vector.copy(),
        "thumbnail": cv2.cvtColor(face.crop, cv2.COLOR_RGB2BGR),
        "last_frame": frame_count,
    })
    return IdentityMatch(person_id=None, name="Unknown", confidence=0.0, is_known=False), False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AR Glasses Prototype")
    parser.add_argument(
        "--mode",
        choices=["run", "live", "enroll", "label", "merge", "db-info", "db-delete", "api"],
        default="run",
        help=(
            "run: live recognition (default) | "
            "live: streaming diarization + transcription | "
            "label: name auto-discovered clusters | "
            "merge: consolidate duplicate clusters | "
            "enroll: manually add a named person | "
            "db-info: list database contents | "
            "db-delete: remove a person or wipe the database | "
            "api: REST API server for companion mobile app"
        ),
    )
    def _camera_source(val):
        if val == "android":
            return "android"
        return int(val)

    parser.add_argument(
        "--camera", type=_camera_source, default=CAMERA_SOURCE,
        help="Camera source index or 'android' for ADB/scrcpy (default: 0)",
    )
    parser.add_argument(
        "--api-host", type=str, default=FLASK_HOST,
        help="API server bind address (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--api-port", type=int, default=FLASK_PORT,
        help="API server port (default: 5000)",
    )
    args = parser.parse_args()

    # API-only mode — no camera or embedder needed
    if args.mode == "api":
        from api.api import PeopleAPI
        db = Database()
        try:
            api = PeopleAPI(db)
            api.run(host=args.api_host, port=args.api_port)
        finally:
            db.close()
        return

    # DB-only modes — no camera or embedder needed
    if args.mode in ("db-info", "db-delete", "label", "merge"):
        from commands.db import db_info_mode, db_delete_mode
        from commands.label import label_mode
        from commands.merge import merge_clusters_mode

        db = Database()
        try:
            {"db-info": db_info_mode, "db-delete": db_delete_mode,
             "label": label_mode, "merge": merge_clusters_mode}[args.mode](db)
        finally:
            db.close()
        return

    # Full-pipeline modes
    print("Initializing...")
    db = Database()
    detector = FaceDetector()
    embedder = FaceEmbedder()
    matcher = FaceMatcher()

    try:
        if args.mode == "enroll":
            from commands.enroll import enroll_mode
            enroll_mode(db, detector, embedder)
        elif args.mode == "live":
            from pipeline.live import LivePipelineDriver
            from pipeline.identity import FullIdentity
            from pipeline.transcription import TranscriptionPipeline

            if args.camera == "android":
                from input.android_camera import AndroidCamera
                camera = AndroidCamera()
            else:
                camera = Camera(source=args.camera)
            try:
                identity = FullIdentity(embedder, matcher, db)
                driver = LivePipelineDriver(identity, TranscriptionPipeline())
                driver.run(camera)
            finally:
                camera.close()
        else:
            if args.camera == "android":
                from input.android_camera import AndroidCamera
                camera = AndroidCamera()
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
