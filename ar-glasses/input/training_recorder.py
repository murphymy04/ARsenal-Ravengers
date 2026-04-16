"""Per-connection recorder: writes emitted glasses pairs to MP4 + WAV.

Split (not muxed) because variable-framerate PTS muxing via ffmpeg is heavy
for what is essentially raw capture. WAV is sample-accurate truth for sync.
"""

import time
import wave
from pathlib import Path

import cv2
import numpy as np


NOMINAL_FPS = 10.0


class TrainingRecorder:
    def __init__(self, output_dir: Path, sample_rate: int):
        self.output_dir = output_dir
        self.sample_rate = sample_rate
        self.video_writer: cv2.VideoWriter | None = None
        self.wave_writer: wave.Wave_write | None = None
        self.basename: str | None = None

    def start_session(self):
        self.close()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        prefix = time.strftime("%Y%m%d_%H%M")
        seq = len(list(self.output_dir.glob(f"{prefix}_*.mp4"))) + 1
        self.basename = f"{prefix}_{seq:03d}"
        print(f"[TrainingRecorder] new session: {self.basename}")

    def write(self, pair):
        bgr, audio_float32, _ = pair
        if self.basename is None:
            return
        if self.video_writer is None:
            self._open_writers(bgr.shape[:2])
        self.video_writer.write(bgr)
        if audio_float32.size:
            int16 = (np.clip(audio_float32, -1.0, 1.0) * 32767).astype(np.int16)
            self.wave_writer.writeframes(int16.tobytes())

    def close(self):
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        if self.wave_writer is not None:
            self.wave_writer.close()
            self.wave_writer = None

    def _open_writers(self, frame_shape: tuple[int, int]):
        height, width = frame_shape
        mp4_path = self.output_dir / f"{self.basename}.mp4"
        wav_path = self.output_dir / f"{self.basename}.wav"
        self.video_writer = cv2.VideoWriter(
            str(mp4_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            NOMINAL_FPS,
            (width, height),
        )
        self.wave_writer = wave.open(str(wav_path), "wb")
        self.wave_writer.setnchannels(1)
        self.wave_writer.setsampwidth(2)
        self.wave_writer.setframerate(self.sample_rate)
        print(f"[TrainingRecorder] writing {mp4_path.name} + {wav_path.name}")
