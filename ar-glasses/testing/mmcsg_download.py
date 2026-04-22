"""Partial download of MMCSG eval recordings via HTTP range requests.

Streams individual files out of the 17 GB MMCSG_eval.zip without ever storing
the full archive on disk. Usage:

    python testing/mmcsg_download.py --list
    python testing/mmcsg_download.py --count 5
    python testing/mmcsg_download.py --ids rec001 rec002
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from remotezip import RemoteZip

EVAL_URL = (
    "https://scontent.xx.fbcdn.net/m1/v/t6/"
    "An9Nt9_z9TF77-UP41kcl3QGbRaT2M2qFbE5uhDkUlrgMy1z0I-0HNHik5cr7barpuCztR00ezemhR2JzcFvhUUyxT0.zip"
    "?_nc_gid&ccb=10-5&oh=00_Af073nAMoCta_BWbdRjuY4__wpcsmz15NSCCDfCV_WhHtg&oe=69DCB65D&_nc_sid=e28c19"
)

OUT_DIR = Path(__file__).parent.parent / "test_videos" / "mmcsg"


def group_by_recording(names):
    groups = defaultdict(list)
    for name in names:
        parts = Path(name).parts
        if len(parts) < 3:
            continue
        recording_id = Path(parts[-1]).stem
        groups[recording_id].append(name)
    return groups


def format_size(n_bytes):
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def open_zip():
    return RemoteZip(EVAL_URL, support_suffix_range=False)


def list_recordings():
    with open_zip() as z:
        names = z.namelist()
        sizes = {n: z.getinfo(n).file_size for n in names}

    groups = group_by_recording(names)
    print(f"Total files in archive: {len(names)}")
    print(f"Unique recordings: {len(groups)}\n")
    print(f"{'recording_id':<40} {'total_size':>12}  n_files")
    print("-" * 70)
    for rid, files in sorted(groups.items()):
        total = sum(sizes[f] for f in files)
        print(f"{rid:<40} {format_size(total):>12}  {len(files)}")


def download_recordings(recording_ids):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open_zip() as z:
        names = z.namelist()
        groups = group_by_recording(names)

        missing = [rid for rid in recording_ids if rid not in groups]
        if missing:
            sys.exit(f"Unknown recording ids: {missing}")

        total_bytes = sum(
            z.getinfo(f).file_size for rid in recording_ids for f in groups[rid]
        )
        print(
            f"Downloading {len(recording_ids)} recordings ({format_size(total_bytes)})"
        )

        for rid in recording_ids:
            for member in groups[rid]:
                size = z.getinfo(member).file_size
                print(f"  {member}  ({format_size(size)})")
                z.extract(member, OUT_DIR)

    print(f"\nExtracted to {OUT_DIR}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--list", action="store_true", help="List recording ids and sizes"
    )
    parser.add_argument(
        "--count", type=int, help="Download first N recordings alphabetically"
    )
    parser.add_argument("--ids", nargs="+", help="Download specific recording ids")
    args = parser.parse_args()

    if args.list:
        list_recordings()
        return

    if args.count:
        with open_zip() as z:
            groups = group_by_recording(z.namelist())
        ids = sorted(groups.keys())[: args.count]
        download_recordings(ids)
        return

    if args.ids:
        download_recordings(args.ids)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
