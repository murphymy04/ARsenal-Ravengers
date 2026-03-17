"""Transcription pipeline: Groq Whisper API.

Takes raw wav bytes and sends them to Groq's hosted Whisper model for
transcription. Returns a list of TranscriptSegment with timestamps.

Stateless with respect to I/O: the driver is responsible for extracting
audio from the source.
"""

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

    def run(self, audio: bytes) -> list[TranscriptSegment]:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = Path(tmp.name)
        try:
            tmp.write(audio)
            tmp.close()
            with open(tmp_path, "rb") as f:
                response = self._client.audio.transcriptions.create(
                    file=(tmp_path.name, f),
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
            tmp_path.unlink(missing_ok=True)

    @staticmethod
    def test(segments: list[TranscriptSegment]):
        print(f"\n{'='*60}")
        print(f"Transcript: {len(segments)} segments")
        print(f"{'='*60}")

        for seg in segments:
            print(f"  [{seg.start_time:7.2f}s - {seg.end_time:7.2f}s]  {seg.text}")
