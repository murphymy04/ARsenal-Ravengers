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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "ravengers")

import pipeline.live as live_module
from config import DATA_DIR, VISION_STRIDE
from input.camera import Camera
from input.microphone import SimulatedMic
from pipeline.identity import FullIdentity
from pipeline.live import LivePipelineDriver, extract_audio_pcm, get_video_fps
from pipeline.transcription import TranscriptionPipeline
from processing.face_embedder import FaceEmbedder
from processing.face_matcher import FaceMatcher
from storage.database import Database

TEST_VIDEOS = Path(__file__).resolve().parent.parent / "test_videos"
SEED_VIDEOS = [
    TEST_VIDEOS / "timur_myles_2.mp4",
    TEST_VIDEOS / "timur_will.mp4",
]
RETRIEVAL_VIDEO = TEST_VIDEOS / "myles_and_will.mp4"

RMS_BOUNDARIES = {
    "timur_myles_2.mp4": 0.035,
    "timur_will.mp4": 0.1,
    "myles_and_will.mp4": 0.2,
}

LOG_DIR = DATA_DIR / "e2e_logs"
CONVERSATIONS_LOG = LOG_DIR / "conversations.txt"
RETRIEVAL_LOG = LOG_DIR / "retrieval.txt"


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
    boundary = RMS_BOUNDARIES.get(video_path.name)

    return driver.run(
        camera,
        mic=sim_mic,
        clock_fn=lambda fi, f=fps: fi / f,
        fps=fps,
        vision_stride=VISION_STRIDE,
        static_boundary=boundary,
    )


def write_conversations(video_name: str, combined: list[dict], diarization: list[dict]):
    with open(CONVERSATIONS_LOG, "a") as f:
        f.write(f"{'=' * 60}\n")
        f.write(f"{video_name}\n")
        f.write(f"{'=' * 60}\n\n")

        for seg in combined:
            f.write(
                f"  [{seg['start']:7.2f} - {seg['end']:7.2f}] "
                f"{seg['speaker']}: {seg['text']}\n"
            )

        f.write(f"\n  ({len(combined)} combined, {len(diarization)} diarization)\n\n")


def write_retrieval(results: list[tuple]):
    with open(RETRIEVAL_LOG, "a") as f:
        for person_name, person_id, facts in results:
            f.write(f"{person_name} (id={person_id}): {len(facts)} facts\n")
            for fact in facts:
                f.write(f"  - {fact}\n")
            f.write("\n")


def seed_knowledge():
    print("\n" + "=" * 60)
    print("PHASE 1: Seeding knowledge graph")
    print("=" * 60)

    driver = create_driver()
    driver.save_to_memory = True

    for video_path in SEED_VIDEOS:
        print(f"\nProcessing: {video_path.name}")
        combined, diarization = run_video(driver, video_path)
        write_conversations(video_path.name, combined, diarization)

    print("\nSeeding complete.")


def test_retrieval():
    print("\n" + "=" * 60)
    print("PHASE 2: Testing retrieval")
    print("=" * 60)

    live_module.RETRIEVAL_ENABLED = True

    driver = create_driver()

    print(f"\nProcessing: {RETRIEVAL_VIDEO.name}")
    combined, diarization = run_video(driver, RETRIEVAL_VIDEO)
    write_conversations(RETRIEVAL_VIDEO.name, combined, diarization)
    write_retrieval(driver.retrieval_results)

    print(f"\n  Combined segments: {len(combined)}")
    print(f"  Retrieval results: {len(driver.retrieval_results)}")

    live_module.RETRIEVAL_ENABLED = False


if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CONVERSATIONS_LOG.write_text("")
    RETRIEVAL_LOG.write_text("")

    seed_knowledge()
    test_retrieval()

    print("\nLogs written to:")
    print(f"  {CONVERSATIONS_LOG}")
    print(f"  {RETRIEVAL_LOG}")
    print("\nDone.")
