"""Face recognition evaluation script.

Runs the full production pipeline (detect → align → embed → match) on a
captured dataset and reports:

  - Detection rate          how many images yielded a usable face crop
  - Rank-1 accuracy         correct identity is the top match
  - False Reject Rate       known person returned as Unknown
  - False Accept Rate       cross-person impostor score exceeds threshold
  - Equal Error Rate        threshold where FAR == FRR (lower is better)
  - Per-person accuracy     per-identity breakdown
  - Confidence statistics   mean score for correct vs. wrong matches

Dataset layout (created by capture.py)
---------------------------------------
    eval/dataset/
        Alice/
            0001.jpg
            0002.jpg
            ...
        Bob/
            0001.jpg
            ...

Usage
-----
    python eval/evaluate.py
    python eval/evaluate.py --dataset eval/dataset --enroll 5 --threshold 0.4
"""

import sys
import argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from config import MATCH_THRESHOLD, UNKNOWN_LABEL
from models import Person, FaceEmbedding
from processing.face_detector import FaceDetector
from processing.face_embedder import FaceEmbedder
from processing.face_matcher import FaceMatcher


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(dataset_dir: Path) -> dict[str, list[Path]]:
    """Return {person_name: [sorted image paths]} for each sub-folder."""
    dataset = {}
    for folder in sorted(dataset_dir.iterdir()):
        if not folder.is_dir():
            continue
        images = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png"))
        if images:
            dataset[folder.name] = images
    return dataset


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def embed_images(
    paths: list[Path],
    detector: FaceDetector,
    embedder: FaceEmbedder,
) -> list[tuple[Path, FaceEmbedding | None]]:
    """Detect and embed each image. Returns (path, embedding_or_None)."""
    results = []
    for path in paths:
        frame = cv2.imread(str(path))
        if frame is None:
            results.append((path, None))
            continue
        faces = detector.detect(frame)
        if not faces:
            results.append((path, None))
            continue
        # Use the largest detected face
        face = max(faces, key=lambda f: f.bbox.width * f.bbox.height)
        emb = embedder.embed(face.crop)
        results.append((path, emb))
    return results


def build_gallery(
    enrollment: dict[str, list[FaceEmbedding]],
    matcher: FaceMatcher,
):
    """Load enrollment embeddings into the matcher."""
    people = []
    for pid, (name, embs) in enumerate(enrollment.items()):
        people.append(Person(person_id=pid, name=name, embeddings=embs))
    matcher.update_gallery(people)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_eer(genuine_scores: np.ndarray, impostor_scores: np.ndarray) -> tuple[float, float]:
    """Return (EER, EER_threshold) where FAR ≈ FRR."""
    all_scores = np.concatenate([genuine_scores, impostor_scores])
    thresholds = np.unique(all_scores)

    best_diff = float("inf")
    eer = 0.0
    eer_threshold = 0.0

    for t in thresholds:
        far = float(np.mean(impostor_scores >= t))
        frr = float(np.mean(genuine_scores < t))
        diff = abs(far - frr)
        if diff < best_diff:
            best_diff = diff
            eer = (far + frr) / 2.0
            eer_threshold = t

    return eer, eer_threshold


def score_against_all(
    query_emb: FaceEmbedding,
    gallery_means: dict[str, np.ndarray],
) -> dict[str, float]:
    """Cosine similarity of query against every gallery mean embedding."""
    q = query_emb.vector
    qn = np.linalg.norm(q)
    if qn == 0:
        return {name: 0.0 for name in gallery_means}
    q_norm = q / qn
    return {
        name: float(np.dot(q_norm, mean_n))
        for name, mean_n in gallery_means.items()
    }


def compute_gallery_means(enrollment: dict[str, list[FaceEmbedding]]) -> dict[str, np.ndarray]:
    """Pre-compute normalised mean embedding for each enrolled person."""
    means = {}
    for name, embs in enrollment.items():
        vecs = np.array([e.vector for e in embs], dtype=np.float32)
        mean = vecs.mean(axis=0)
        norm = np.linalg.norm(mean)
        if norm > 0:
            means[name] = mean / norm
    return means


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _bar(value: float, width: int = 30) -> str:
    filled = int(round(value * width))
    return "█" * filled + "░" * (width - filled)


def print_report(
    results: list[dict],
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    threshold: float,
    n_enrolled: int,
    n_no_detection: int,
):
    total = len(results)
    detected = total  # results only contain detected faces

    # --- Rank-1 accuracy ---
    correct = sum(1 for r in results if r["pred"] == r["true"])
    rank1 = correct / total if total else 0.0

    # --- FRR: known person predicted as Unknown ---
    rejected = sum(1 for r in results if r["pred"] == UNKNOWN_LABEL)
    frr = rejected / total if total else 0.0

    # --- FAR: highest impostor score exceeds threshold ---
    far = float(np.mean(impostor_scores >= threshold)) if len(impostor_scores) else float("nan")

    # --- EER ---
    if len(genuine_scores) and len(impostor_scores):
        eer, eer_threshold = compute_eer(genuine_scores, impostor_scores)
    else:
        eer, eer_threshold = float("nan"), float("nan")

    # --- Per-person ---
    per_person: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        per_person[r["true"]]["total"] += 1
        if r["pred"] == r["true"]:
            per_person[r["true"]]["correct"] += 1

    # --- Confidence stats ---
    correct_confs = [r["confidence"] for r in results if r["pred"] == r["true"]]
    wrong_confs   = [r["confidence"] for r in results if r["pred"] != r["true"]]

    # -----------------------------------------------------------------------
    W = 60
    print("\n" + "═" * W)
    print("  FACE RECOGNITION EVALUATION REPORT")
    print("═" * W)
    print(f"  People in gallery : {n_enrolled}")
    print(f"  Query images      : {total + n_no_detection}")
    print(f"    └─ face detected: {detected}  (no detection: {n_no_detection})")
    print(f"  Threshold         : {threshold:.2f}")
    print()

    print("  ── Core Metrics ─────────────────────────────")
    print(f"  Rank-1 Accuracy   : {rank1*100:5.1f}%  {_bar(rank1)}")
    print(f"  False Reject Rate : {frr*100:5.1f}%  {_bar(frr)}")
    print(f"  False Accept Rate : {far*100:5.1f}%  {_bar(far)}")
    print(f"  Equal Error Rate  : {eer*100:5.1f}%  (threshold ≈ {eer_threshold:.2f})")
    print()

    print("  ── Per-Person Accuracy ──────────────────────")
    for name, stats in sorted(per_person.items()):
        acc = stats["correct"] / stats["total"] if stats["total"] else 0.0
        print(f"  {name:<18} {acc*100:5.1f}%  ({stats['correct']}/{stats['total']})  {_bar(acc, 20)}")
    print()

    print("  ── Confidence Distributions ─────────────────")
    if correct_confs:
        print(f"  Correct matches   : mean={np.mean(correct_confs):.3f}  "
              f"min={np.min(correct_confs):.3f}  max={np.max(correct_confs):.3f}")
    if wrong_confs:
        print(f"  Wrong   matches   : mean={np.mean(wrong_confs):.3f}  "
              f"min={np.min(wrong_confs):.3f}  max={np.max(wrong_confs):.3f}")
    print()

    print("  ── Score Distributions ──────────────────────")
    if len(genuine_scores):
        print(f"  Genuine  scores   : mean={genuine_scores.mean():.3f}  std={genuine_scores.std():.3f}")
    if len(impostor_scores):
        print(f"  Impostor scores   : mean={impostor_scores.mean():.3f}  std={impostor_scores.std():.3f}")
    print()

    # Confusion matrix (only for small galleries)
    if len(per_person) <= 8:
        names = sorted(per_person.keys())
        confusion: dict[tuple, int] = defaultdict(int)
        for r in results:
            confusion[(r["true"], r["pred"])] += 1

        all_preds = sorted({r["pred"] for r in results})
        col_w = max(len(n) for n in names + all_preds) + 2

        print("  ── Confusion Matrix ─────────────────────────")
        header = "  " + " " * 18 + "".join(p[:col_w].ljust(col_w) for p in all_preds)
        print(header)
        for true_name in names:
            row = f"  {true_name:<18}"
            for pred_name in all_preds:
                count = confusion.get((true_name, pred_name), 0)
                cell = str(count).center(col_w)
                row += cell
            print(row)
        print()

    print("═" * W)

    # Optional: ROC plot
    _try_plot_roc(genuine_scores, impostor_scores, threshold)


def _try_plot_roc(genuine: np.ndarray, impostor: np.ndarray, threshold: float):
    """Plot ROC curve if matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if not len(genuine) or not len(impostor):
        return

    all_scores = np.concatenate([genuine, impostor])
    thresholds = np.sort(np.unique(all_scores))[::-1]

    fars, tars = [], []
    for t in thresholds:
        fars.append(float(np.mean(impostor >= t)))
        tars.append(float(np.mean(genuine >= t)))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fars, tars, lw=2, label="ROC curve")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    # Mark current operating point
    op_far = float(np.mean(impostor >= threshold))
    op_tar = float(np.mean(genuine >= threshold))
    ax.scatter([op_far], [op_tar], color="red", zorder=5, label=f"threshold={threshold:.2f}")
    ax.set_xlabel("False Accept Rate (FAR)")
    ax.set_ylabel("True Accept Rate (TAR = 1 - FRR)")
    ax.set_title("ROC Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("eval_roc.png", dpi=150)
    print("  ROC curve saved → eval_roc.png")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate(dataset_dir: Path, enroll_count: int, threshold: float):
    dataset = load_dataset(dataset_dir)

    if not dataset:
        print(f"No dataset found in {dataset_dir}. Run eval/capture.py first.")
        return

    people = list(dataset.keys())
    print(f"\nDataset: {dataset_dir}")
    print(f"People : {', '.join(people)}")
    for name, paths in dataset.items():
        print(f"  {name}: {len(paths)} images")
    print()

    print("Loading face detector and embedder…")
    detector = FaceDetector()
    embedder = FaceEmbedder()
    matcher  = FaceMatcher(threshold=threshold)

    # --- Embed all images ---
    print("Embedding all images…")
    all_embeddings: dict[str, list[tuple[Path, FaceEmbedding | None]]] = {}
    for name, paths in dataset.items():
        print(f"  {name}…", end="", flush=True)
        embs = embed_images(paths, detector, embedder)
        all_embeddings[name] = embs
        detected = sum(1 for _, e in embs if e is not None)
        print(f" {detected}/{len(paths)} detected")

    # --- Split: enrollment / query ---
    enrollment: dict[str, list[FaceEmbedding]] = {}
    queries:    list[tuple[str, Path, FaceEmbedding]] = []  # (true_name, path, embedding)
    n_no_detection = 0

    for name, emb_list in all_embeddings.items():
        detected = [(p, e) for p, e in emb_list if e is not None]
        not_detected = [(p, e) for p, e in emb_list if e is None]
        n_no_detection += len(not_detected)

        enroll_embs = [e for _, e in detected[:enroll_count]]
        query_items = detected[enroll_count:]

        if not enroll_embs:
            print(f"WARNING: no enrollment embeddings for {name} — skipping.")
            continue

        enrollment[name] = enroll_embs
        for path, emb in query_items:
            queries.append((name, path, emb))

    if not queries:
        print("No query images remaining after enrollment split. Capture more images or reduce --enroll.")
        detector.close()
        return

    print(f"\nEnrolled {len(enrollment)} people ({enroll_count} images each).")
    print(f"Query set: {len(queries)} images.")

    # --- Build gallery ---
    build_gallery(enrollment, matcher)
    gallery_means = compute_gallery_means(enrollment)

    # --- Run queries ---
    results = []
    genuine_scores  = []
    impostor_scores = []

    for true_name, path, query_emb in queries:
        match = matcher.match(query_emb)
        results.append({
            "true": true_name,
            "pred": match.name,
            "confidence": match.confidence,
        })

        # Collect genuine and impostor scores for EER / ROC
        all_scores = score_against_all(query_emb, gallery_means)
        for gallery_name, score in all_scores.items():
            if gallery_name == true_name:
                genuine_scores.append(score)
            else:
                impostor_scores.append(score)

    genuine_arr  = np.array(genuine_scores,  dtype=np.float32)
    impostor_arr = np.array(impostor_scores, dtype=np.float32)

    detector.close()

    print_report(results, genuine_arr, impostor_arr, threshold, len(enrollment), n_no_detection)


def main():
    root = Path(__file__).parent.parent

    parser = argparse.ArgumentParser(description="Evaluate face recognition accuracy.")
    parser.add_argument(
        "--dataset", default="eval/dataset",
        help="Path to dataset directory (default: eval/dataset).",
    )
    parser.add_argument(
        "--enroll", type=int, default=5,
        help="Images per person used for enrollment (default: 5). Rest become queries.",
    )
    parser.add_argument(
        "--threshold", type=float, default=MATCH_THRESHOLD,
        help=f"Cosine similarity threshold (default: {MATCH_THRESHOLD}).",
    )
    args = parser.parse_args()

    dataset_dir = (root / args.dataset).resolve()
    evaluate(dataset_dir, args.enroll, args.threshold)


if __name__ == "__main__":
    main()
