"""Download and prepare the LFW dataset for evaluation.

Labeled Faces in the Wild (LFW) — http://vis-www.cs.umass.edu/lfw/
  13,233 color JPEG images of 5,749 people, 250×250 px.
  Directly downloadable, no authentication required.

This script:
  1. Downloads lfw.tgz (~170 MB) if not already present.
  2. Extracts it.
  3. Filters to people with at least --min-images images.
  4. Copies up to --max-images images per person into eval/dataset/
     using the same layout expected by evaluate.py.

Usage
-----
    python eval/prepare_lfw.py
    python eval/prepare_lfw.py --min-images 10 --max-images 20 --people 20

Notes
-----
  Domain mismatch: LFW images come from the web (varied cameras, lighting,
  compression).  Results on LFW reflect embedding model quality rather than
  end-to-end webcam system performance.  Use capture.py for that.
"""

import sys
import argparse
import shutil
import tarfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

LFW_URLS = [
    "https://vis-www.cs.umass.edu/lfw/lfw.tgz",
    "http://vis-www.cs.umass.edu/lfw/lfw.tgz",
]
LFW_TGZ  = "lfw.tgz"
LFW_DIR  = "lfw"


def _download(dest: Path):
    def _progress(block, block_size, total):
        if total > 0:
            pct = min(block * block_size / total * 100, 100)
            bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
            print(f"\r  [{bar}] {pct:.1f}%", end="", flush=True)

    for url in LFW_URLS:
        try:
            print(f"Trying {url} …")
            urllib.request.urlretrieve(url, dest, reporthook=_progress)
            print()
            return
        except Exception as e:
            print(f"\n  Failed: {e}")

    print("\nAutomatic download failed. Please download manually:")
    print("  1. Go to https://www.kaggle.com/datasets/jessicali9530/lfw-dataset")
    print("     or search 'LFW dataset lfw.tgz' and download the archive.")
    print(f"  2. Place the file at: {dest}")
    print(f"  3. Re-run: python eval/prepare_lfw.py --file \"{dest}\"")
    sys.exit(1)


def prepare(
    work_dir: Path,
    output_dir: Path,
    min_images: int,
    max_images: int,
    max_people: int,
    local_file: Path | None = None,
):
    work_dir.mkdir(parents=True, exist_ok=True)

    tgz_path = local_file if local_file else work_dir / LFW_TGZ
    lfw_path  = work_dir / LFW_DIR

    # --- Download ---
    if not tgz_path.exists():
        _download(tgz_path)
    else:
        print(f"Archive already present: {tgz_path}")

    # --- Extract ---
    if not lfw_path.exists():
        print(f"Extracting {tgz_path} …")
        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall(work_dir)
        print("Extracted.")
    else:
        print(f"Already extracted: {lfw_path}")

    # --- Filter and copy ---
    all_people = sorted(p for p in lfw_path.iterdir() if p.is_dir())

    eligible = [
        p for p in all_people
        if len(list(p.glob("*.jpg"))) >= min_images
    ]

    print(f"\nTotal people in LFW          : {len(all_people)}")
    print(f"People with ≥{min_images} images        : {len(eligible)}")

    if max_people:
        eligible = eligible[:max_people]
    print(f"People to copy               : {len(eligible)}")
    print(f"Max images per person        : {max_images}")
    print(f"Output directory             : {output_dir}\n")

    output_dir.mkdir(parents=True, exist_ok=True)
    total_copied = 0

    for person_dir in eligible:
        name = person_dir.name
        images = sorted(person_dir.glob("*.jpg"))[:max_images]
        dest = output_dir / name
        dest.mkdir(exist_ok=True)

        for i, src in enumerate(images, 1):
            shutil.copy2(src, dest / f"{i:04d}.jpg")

        total_copied += len(images)
        print(f"  {name:<40} {len(images)} images")

    print(f"\nDone — {len(eligible)} people, {total_copied} images → {output_dir}")
    print("\nYou can now run:")
    print("  python eval/evaluate.py --enroll 5")


def main():
    root = Path(__file__).parent.parent

    parser = argparse.ArgumentParser(description="Prepare LFW subset for evaluate.py.")
    parser.add_argument(
        "--min-images", type=int, default=10,
        help="Minimum images per person to include (default: 10).",
    )
    parser.add_argument(
        "--max-images", type=int, default=20,
        help="Maximum images per person to copy (default: 20).",
    )
    parser.add_argument(
        "--people", type=int, default=30,
        help="Max number of people to include (default: 30, 0 = all eligible).",
    )
    parser.add_argument(
        "--work-dir", default="eval/lfw_raw",
        help="Directory to download and extract LFW into (default: eval/lfw_raw).",
    )
    parser.add_argument(
        "--output", default="eval/dataset",
        help="Output dataset directory (default: eval/dataset).",
    )
    parser.add_argument(
        "--file", default=None,
        help="Path to a manually downloaded lfw.tgz (skips download).",
    )
    args = parser.parse_args()

    prepare(
        work_dir   = (root / args.work_dir).resolve(),
        output_dir = (root / args.output).resolve(),
        min_images = args.min_images,
        max_images = args.max_images,
        max_people = args.people,
        local_file = Path(args.file).resolve() if args.file else None,
    )


if __name__ == "__main__":
    main()
