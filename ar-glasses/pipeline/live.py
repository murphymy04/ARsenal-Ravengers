"""Live streaming pipeline driver.

Captures camera frames and mic audio continuously, running diarization
frame-by-frame and transcription in buffered windows.
"""

import io
import struct
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from config import CAMERA_FPS, LIVE_BUFFER_SECONDS, SAMPLE_RATE, SAVE_TO_MEMORY
from input.camera import Camera
from input.microphone import Microphone
from models import IdentityModule
from pipeline.driver import combine_segments
from pipeline.transcription import TranscriptionPipeline
from processing.face_detector import FaceDetector
from processing.face_tracker import FaceTracker
from processing.speaking_detector import SpeakingDetector
from storage.speaking_log import SpeakingLog

def pcm_to_wav(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    pcm16 = (samples * 32767).clip(-32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()

class LivePipelineDriver:
    def __init__(self, identity: IdentityModule, transcription: TranscriptionPipeline):
        self._identity = identity
        self._transcription = transcription
        self._save_to_memory = SAVE_TO_MEMORY
        self._conversation_buffer: list[dict] = []

    def run(self, camera: Camera) -> None:
        detector = FaceDetector()
        tracker = FaceTracker()
        speaker = SpeakingDetector(use_mic=True)
        log = SpeakingLog()
        mic = Microphone()

        mic.open()
        stream_start = time.time()
        window_start = 0.0
        frame_idx = 0

        try:
            for frame in camera.frames():
                timestamp = time.time() - stream_start

                faces = detector.detect(frame, timestamp=timestamp)
                raw_matches = [self._identity.identify(face, frame_idx) for face in faces]
                smoothed, track_ids = tracker.update(faces, raw_matches, frame_idx)

                for face, tid in zip(faces, track_ids):
                    speaker.add_crop(tid, face.crop)
                speaker.run_inference(frame_idx, active_track_ids=set(track_ids))

                for face, match, tid in zip(faces, smoothed, track_ids):
                    is_speaking = speaker.get_speaking(tid)
                    log.update(
                        track_id=tid,
                        person_id=match.person_id if match.is_known else None,
                        name=match.name if match.is_known else f"track_{tid}",
                        is_speaking=is_speaking,
                        timestamp=timestamp,
                    )

                frame_idx += 1

                if timestamp - window_start >= LIVE_BUFFER_SECONDS:
                    self._flush_window(log, mic, window_start, timestamp)
                    log = SpeakingLog()
                    window_start = timestamp

        finally:
            if self._save_to_memory and self._conversation_buffer:
                from pipeline.knowledge import save_to_memory
                save_to_memory(self._conversation_buffer)
            speaker.close()
            detector.close()
            mic.close()

    def _flush_window(
        self,
        log: SpeakingLog,
        mic: Microphone,
        window_start: float,
        window_end: float,
    ):
        log.close(timestamp=window_end)
        diarization_segments = log.get_segments()

        audio = mic.get_buffer_and_clear()
        if len(audio) == 0:
            print(f"\n[{window_start:.1f}s - {window_end:.1f}s] No audio captured")
            return

        wav_bytes = pcm_to_wav(audio)
        transcript_segments = self._transcription.run(wav_bytes)

        for seg in transcript_segments:
            seg.start_time += window_start
            seg.end_time += window_start

        combined = combine_segments(diarization_segments, transcript_segments)

        if self._save_to_memory and combined:
            self._conversation_buffer.extend(combined)
            from pipeline.conversation_end import is_conversation_end
            if is_conversation_end(combined):
                from pipeline.knowledge import save_to_memory
                save_to_memory(self._conversation_buffer)
                self._conversation_buffer = []

        print(f"\n{'='*60}")
        print(f"[{window_start:.1f}s - {window_end:.1f}s] {len(combined)} segments")
        print(f"{'='*60}")
        for seg in combined:
            print(f"  [{seg['start']:7.2f} - {seg['end']:7.2f}] {seg['speaker']}: {seg['text']}")
