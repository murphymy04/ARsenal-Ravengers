"""Audio processing using WhisperX.

Transcribes speech with speaker diarization.
Stubbed for Phase 2 implementation.
"""

import numpy as np
from models import TranscriptSegment

from config import WHISPER_LANGUAGE, WHISPER_MODEL


class AudioProcessor:
    """WhisperX transcription with speaker diarization (stub - Phase 2)."""

    def __init__(
        self,
        model_name: str = WHISPER_MODEL,
        language: str = WHISPER_LANGUAGE,
    ):
        self.model_name = model_name
        self.language = language

    def transcribe(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> list[TranscriptSegment]:
        """Transcribe audio and return segments with speaker labels.

        Args:
            audio: float32 audio array.
            sample_rate: audio sample rate.

        Returns:
            List of TranscriptSegment with speaker labels.
        """
        raise NotImplementedError("Audio processing is Phase 2 - requires WhisperX")
