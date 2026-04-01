"""E2E test for the retrieval pipeline.

Requires Neo4j running (docker compose up -d from knowledge/ directory).

Phase 1 — Seed: processes two videos through the live pipeline with
SAVE_TO_MEMORY=True to populate the knowledge graph with conversation facts.

Phase 2 — Retrieve: processes a third video with retrieval enabled.
When known faces are recognized, the retrieval worker queries the knowledge
graph and prints the results.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "ravengers")

import pipeline.live as live_module
from config import VISION_STRIDE
from input.camera import Camera
from input.microphone import SimulatedMic
from pipeline.identity import FullIdentity
from pipeline.live import LivePipelineDriver, extract_audio_pcm, get_video_fps
from pipeline.retrieval import drain_results
from pipeline.transcription import TranscriptionPipeline
from processing.face_embedder import FaceEmbedder
from processing.face_matcher import FaceMatcher
from storage.database import Database

TEST_VIDEOS = Path(__file__).resolve().parent / "test_videos"
SEED_VIDEOS = [
    TEST_VIDEOS / "timur_myles.mp4",
    TEST_VIDEOS / "timur_will.mp4",
]
RETRIEVAL_VIDEO = TEST_VIDEOS / "myles_and_will.mp4"


def create_driver():
    db = Database()
    identity = FullIdentity(FaceEmbedder(), FaceMatcher(), db)
    transcription = TranscriptionPipeline()
    return LivePipelineDriver(identity, transcription)


def run_video(driver: LivePipelineDriver, video_path: Path):
    fps = get_video_fps(video_path)
    audio = extract_audio_pcm(video_path)
    sim_mic = SimulatedMic(audio, fps, gain=1.5, denoise=False)
    camera = Camera(source=str(video_path))

    return driver.run(
        camera,
        mic=sim_mic,
        clock_fn=lambda fi, f=fps: fi / f,
        fps=fps,
        vision_stride=VISION_STRIDE,
    )


def seed_knowledge():
    print("\n" + "=" * 60)
    print("PHASE 1: Seeding knowledge graph")
    print("=" * 60)

    driver = create_driver()
    driver.save_to_memory = True

    for video_path in SEED_VIDEOS:
        print(f"\nProcessing: {video_path.name}")
        run_video(driver, video_path)

    print("\nSeeding complete.")


def test_retrieval():
    print("\n" + "=" * 60)
    print("PHASE 2: Testing retrieval")
    print("=" * 60)

    # Enable retrieval for this run (driver checks this in run())
    live_module.RETRIEVAL_ENABLED = True

    driver = create_driver()

    print(f"\nProcessing: {RETRIEVAL_VIDEO.name}")
    combined, _diarization_segs = run_video(driver, RETRIEVAL_VIDEO)

    all_results = drain_results(driver._retrieval_result_queue)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    if all_results:
        for person_name, person_id, facts in all_results:
            print(f"\n  {person_name} (id={person_id}): {len(facts)} facts")
            for fact in facts:
                print(f"    - {fact}")
    else:
        print("  No retrieval results collected at end (may have been printed inline)")

    print(f"\n  Combined segments: {len(combined)}")

    live_module.RETRIEVAL_ENABLED = False


if __name__ == "__main__":
    seed_knowledge()
    test_retrieval()
    print("\nDone.")
