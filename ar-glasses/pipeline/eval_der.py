import argparse
import subprocess
import sys
import tempfile
from itertools import permutations
from pathlib import Path

from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate

_AR_ROOT = Path(__file__).resolve().parent.parent
if str(_AR_ROOT) not in sys.path:
    sys.path.insert(0, str(_AR_ROOT))

from pipeline.diarization import DiarizationPipeline
from pipeline.driver import PipelineDriver
from pipeline.identity import FullIdentity
from pipeline.egocom import (
    parse_ground_truth,
    collapse_to_segments,
    video_path_for,
)
from processing.face_embedder import FaceEmbedder
from processing.face_matcher import FaceMatcher
from storage.database import Database

_EVAL_DB = str(_AR_ROOT / "data" / "eval_people.db")


def clip_video(video_path: Path, duration: int) -> Path:
    tmp = Path(tempfile.mktemp(suffix=".mp4"))
    subprocess.run(
        ["ffmpeg", "-i", str(video_path), "-t", str(duration),
         "-c", "copy", str(tmp), "-y", "-loglevel", "error"],
        check=True,
    )
    return tmp


def run_diarization(video_path: Path) -> list[dict]:
    db = Database(db_path=_EVAL_DB)
    identity = FullIdentity(FaceEmbedder(), FaceMatcher(), db)

    driver = PipelineDriver(
        diarization=DiarizationPipeline(identity=identity),
    )
    diarization_segments, _ = driver.run(video_path)
    return diarization_segments


def ground_truth_to_annotation(gt_segments, wearer_id: int = 1) -> Annotation:
    annotation = Annotation()
    for seg in gt_segments:
        if seg.speaker_id == wearer_id:
            label = "wearer"
        else:
            label = f"speaker_{seg.speaker_id}"
        annotation[Segment(seg.start, seg.end)] = label
    return annotation


def prediction_to_annotation(diarization_segments: list[dict]) -> tuple[Annotation, list[str]]:
    track_names = sorted({seg["name"] for seg in diarization_segments})
    annotation = Annotation()
    for seg in diarization_segments:
        annotation[Segment(seg["start"], seg["end"])] = seg["name"]
    return annotation, track_names


def remap_annotation(hypothesis: Annotation, mapping: dict) -> Annotation:
    remapped = Annotation()
    for seg, _, label in hypothesis.itertracks(yield_label=True):
        remapped[seg] = mapping.get(label, label)
    return remapped


def find_best_mapping(reference: Annotation, hypothesis: Annotation, track_names: list[str]) -> tuple[float, dict]:
    gt_visible = sorted({label for label in reference.labels() if label != "wearer"})

    if not track_names:
        metric = DiarizationErrorRate()
        return metric(reference, hypothesis), {}

    best_der = float("inf")
    best_mapping = {}

    for perm in permutations(gt_visible):
        mapping = {track_names[i]: perm[i] for i in range(min(len(track_names), len(perm)))}
        remapped = remap_annotation(hypothesis, mapping)

        metric = DiarizationErrorRate()
        der = metric(reference, remapped)

        if der < best_der:
            best_der = der
            best_mapping = mapping

    return best_der, best_mapping


def evaluate(conversation_id: str, clip_duration: int | None = None):
    conversations = parse_ground_truth()

    if conversation_id not in conversations:
        print(f"Conversation '{conversation_id}' not found in ground truth")
        print(f"Available: {sorted(conversations.keys())[:10]}...")
        sys.exit(1)

    video = video_path_for(conversation_id)
    if not video:
        print(f"Video not found for {conversation_id}")
        sys.exit(1)

    print(f"Conversation: {conversation_id}")
    print(f"Video: {video.name}")

    if clip_duration:
        print(f"Clipping to {clip_duration}s")
        video = clip_video(video, clip_duration)

    try:
        pred_segments = run_diarization(video)
    finally:
        if clip_duration:
            video.unlink(missing_ok=True)

    gt_words = conversations[conversation_id]
    gt_segments = collapse_to_segments(gt_words)

    if clip_duration:
        gt_segments = [s for s in gt_segments if s.start < clip_duration]
        for s in gt_segments:
            s.end = min(s.end, float(clip_duration))

    reference = ground_truth_to_annotation(gt_segments)
    hypothesis, track_names = prediction_to_annotation(pred_segments)

    print(f"\nGround truth: {len(gt_segments)} segments, speakers: {sorted({s.speaker_id for s in gt_segments})}")
    print(f"Prediction:   {len(pred_segments)} segments, tracks: {track_names}")

    der, mapping = find_best_mapping(reference, hypothesis, track_names)
    remapped = remap_annotation(hypothesis, mapping)

    print(f"\nBest speaker mapping: {mapping}")
    print(f"DER: {der:.1%}")

    metric = DiarizationErrorRate()
    detail = metric(reference, remapped, detailed=True)
    print(f"  missed speech:     {detail['missed detection']:.1%}")
    print(f"  false alarm:       {detail['false alarm']:.1%}")
    print(f"  speaker confusion: {detail['confusion']:.1%}")

    return der


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("conversation_id", help="e.g. day_1__con_1__part1")
    parser.add_argument("--clip", type=int, default=None, help="clip video to N seconds")
    args = parser.parse_args()

    evaluate(args.conversation_id, clip_duration=args.clip)
