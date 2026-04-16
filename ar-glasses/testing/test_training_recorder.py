"""Verifies the TRAINING_DATA capture path end-to-end.

Strategy: instead of standing up a fake UDP/TCP glasses sender, drive the
real integration points directly — `_on_audio_connect` opens a session and
`pairing._put_pair` is the exact hook the emit loop calls. We feed pairs
built from a real test video + synthetic audio, then assert that the
expected MP4 + WAV files exist and are non-empty.
"""

import os
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["TRAINING_DATA"] = "true"


def extract_audio_pcm(video_path: Path, sample_rate: int) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-acodec",
            "pcm_s16le",
            wav_path,
        ],
        check=True,
        capture_output=True,
    )
    with wave.open(wav_path, "rb") as w:
        raw = w.readframes(w.getnframes())
    os.unlink(wav_path)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def feed_video_through_recorder(video_path: Path, server) -> int:
    cap = cv2.VideoCapture(str(video_path))
    sample_rate = server.recorder.sample_rate
    samples_per_frame = sample_rate // 10
    audio_pcm = extract_audio_pcm(video_path, sample_rate)
    pairs_emitted = 0
    while True:
        ok, frame = cap.read()
        if not ok or pairs_emitted >= 30:
            break
        start = pairs_emitted * samples_per_frame
        audio = audio_pcm[start : start + samples_per_frame]
        if audio.size == 0:
            audio = np.zeros(samples_per_frame, dtype=np.float32)
        server.pairing._put_pair((frame, audio, pairs_emitted / 10.0))
        pairs_emitted += 1
    cap.release()
    return pairs_emitted


def assert_outputs_valid(output_dir: Path, expected_pairs: int, sample_rate: int):
    mp4_files = sorted(output_dir.glob("*.mp4"))
    wav_files = sorted(output_dir.glob("*.wav"))
    assert len(mp4_files) == 2, f"expected 2 mp4 files, got {len(mp4_files)}"
    assert len(wav_files) == 2, f"expected 2 wav files, got {len(wav_files)}"

    for mp4, wav in zip(mp4_files, wav_files):
        assert mp4.stem == wav.stem, f"prefix mismatch: {mp4.stem} vs {wav.stem}"
        assert mp4.stat().st_size > 0, f"{mp4.name} is empty"
        with wave.open(str(wav), "rb") as w:
            assert w.getframerate() == sample_rate
            assert w.getnchannels() == 1
            assert w.getnframes() > 0


def main():
    project_root = Path(__file__).resolve().parent.parent
    videos = [
        project_root / "test_videos" / "timur_myles.mp4",
        project_root / "test_videos" / "myles_and_will.mp4",
    ]
    for v in videos:
        assert v.exists(), f"missing test video: {v}"

    keep = "--keep" in sys.argv
    if keep:
        out_dir = project_root / "data" / "training_data"
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in out_dir.glob("*.mp4"):
            f.unlink()
        for f in out_dir.glob("*.wav"):
            f.unlink()
        run_test(videos, out_dir)
        print(f"\nPASS — files kept at {out_dir}")
        return

    with tempfile.TemporaryDirectory() as tmp:
        run_test(videos, Path(tmp))
        print(f"\nPASS — wrote and verified files in {tmp}")


def run_test(videos: list[Path], out_dir: Path):
    import input.glasses_adapter as ga

    ga.TRAINING_DATA_DIR = out_dir
    server = ga.GlassesServer(sample_rate=16000)
    server.recorder.output_dir = out_dir
    assert server.recorder is not None, "recorder should be enabled"

    for v in videos:
        server._on_audio_connect()
        n = feed_video_through_recorder(v, server)
        print(f"  pushed {n} pairs from {v.name}")

    server.recorder.close()
    assert_outputs_valid(out_dir, expected_pairs=30, sample_rate=16000)
    for f in sorted(out_dir.iterdir()):
        if f.suffix in (".mp4", ".wav"):
            print(f"  {f.name}  ({f.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
