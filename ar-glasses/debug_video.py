"""Debug overlay: plays video with face boxes and ASD speaking probabilities."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
import sounddevice as sd
import time

from config import CAMERA_FPS, SAMPLE_RATE, SIMULATION_AUDIO_GAIN
from input.microphone import SimulatedMic
from pipeline.live import extract_audio_pcm, get_video_fps
from pipeline.identity import NullIdentity
from processing.face_detector import FaceDetector
from processing.face_tracker import FaceTracker
from processing.speaking_detector import SpeakingDetector


def main(video_path: Path):
    fps = get_video_fps(video_path)
    audio = extract_audio_pcm(video_path)
    mic = SimulatedMic(audio, fps, gain=SIMULATION_AUDIO_GAIN, denoise=True)

    identity = NullIdentity()
    detector = FaceDetector()
    tracker = FaceTracker()
    speaker = SpeakingDetector(fps=fps)

    # Start audio playback (non-blocking) — uses the same denoised+boosted audio
    sd.play(mic.audio, samplerate=SAMPLE_RATE)

    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    paused = False
    playback_start = time.time()
    frame_delay = 1.0 / fps

    try:
        while cap.isOpened():
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    break

                timestamp = frame_idx / fps
                chunk = mic.advance_frame()
                speaker.drip_audio(chunk)

                faces = detector.detect(frame, timestamp=timestamp)
                raw_matches = [identity.identify(face, frame_idx) for face in faces]
                _, track_ids = tracker.update(faces, raw_matches, frame_idx)

                for face, tid in zip(faces, track_ids):
                    speaker.add_crop(tid, face.crop)
                speaker.run_inference(frame_idx, active_track_ids=set(track_ids))

                display = frame.copy()
                for face, tid in zip(faces, track_ids):
                    prob = speaker._speaking.get(tid, None)
                    is_speaking = speaker.get_speaking(tid)
                    b = face.bbox

                    color = (0, 255, 0) if is_speaking else (0, 0, 255)
                    cv2.rectangle(display, (b.x1, b.y1), (b.x2, b.y2), color, 2)

                    label = f"t{tid}"
                    if prob is not None:
                        label += f" {'SPK' if is_speaking else '   '}"

                    cv2.putText(display, label, (b.x1, b.y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                ts_label = f"{timestamp:.2f}s  frame {frame_idx}"
                cv2.putText(display, ts_label, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                frame_idx += 1

                # Sync video to real-time so audio stays aligned
                target_time = playback_start + frame_idx * frame_delay
                sleep_time = target_time - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)

            h, w = display.shape[:2]
            scale = min(1280 / w, 720 / h, 1.0)
            if scale < 1.0:
                display_resized = cv2.resize(display, (int(w * scale), int(h * scale)))
            else:
                display_resized = display

            cv2.imshow("debug", display_resized)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
    finally:
        sd.stop()
        speaker.close()
        detector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_video.py <video_path>")
        sys.exit(1)
    main(Path(sys.argv[1]))
