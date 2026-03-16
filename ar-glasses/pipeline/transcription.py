"""Transcription pipeline: Groq Whisper API.

Extracts audio from a video file and sends it to Groq's hosted Whisper model
for transcription. Returns a list of TranscriptSegment with timestamps.

The output of this step will be combined with the diarization step to get the full conversation.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

_AR_ROOT = Path(__file__).resolve().parent.parent
if str(_AR_ROOT) not in sys.path:
    sys.path.insert(0, str(_AR_ROOT))

load_dotenv(_AR_ROOT / ".env")

from models import TranscriptSegment

class TranscriptionPipeline:
    def __init__(self):
        self._client = Groq()

    def run(self, video_path: Path) -> list[TranscriptSegment]:
        audio_path = _extract_audio(video_path)
        try:
            with open(audio_path, "rb") as f:
                response = self._client.audio.transcriptions.create(
                    file=(audio_path.name, f),
                    model="whisper-large-v3",
                    response_format="verbose_json",
                )
            return [
                TranscriptSegment(
                    text=seg["text"].strip(),
                    start_time=seg["start"],
                    end_time=seg["end"],
                )
                for seg in response.segments
            ]
        finally:
            audio_path.unlink(missing_ok=True)

def _extract_audio(video_path: Path) -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000",
        "-y", "-loglevel", "error",
        tmp.name,
    ]
    subprocess.run(cmd, check=True)
    return Path(tmp.name)


if __name__ == "__main__":
    VIDEO_PATH = _AR_ROOT.parent / "test_movie.mp4"

    if not VIDEO_PATH.exists():
        print(f"Test video not found: {VIDEO_PATH}")
        sys.exit(1)

    pipeline = TranscriptionPipeline()
    segments = pipeline.run(VIDEO_PATH)

    print(f"\n{'='*60}")
    print(f"Transcript: {len(segments)} segments")
    print(f"{'='*60}")

    for seg in segments:
        print(f"  [{seg.start_time:7.2f}s - {seg.end_time:7.2f}s]  {seg.text}")
