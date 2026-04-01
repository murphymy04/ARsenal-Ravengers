"""Debug overlay: plays video with face boxes and ASD speaking probabilities."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import sounddevice as sd
import time

from config import SAMPLE_RATE, SIMULATION_AUDIO_GAIN, SPEAKING_BACKEND, VISION_STRIDE
from input.microphone import SimulatedMic
from pipeline.live import extract_audio_pcm, get_video_fps
from pipeline.identity import NullIdentity
from processing.face_detector import FaceDetector
from processing.face_tracker import FaceTracker


def _create_speaker(fps: float):
    if SPEAKING_BACKEND == "vad_rms":
        from processing.vad_speaker import VadSpeaker
        return VadSpeaker(fps=fps)
    from processing.speaking_detector import SpeakingDetector
    return SpeakingDetector(fps=fps)


def main(video_path: Path, fast: bool = False):
    fps = get_video_fps(video_path)
    audio = extract_audio_pcm(video_path)
    mic = SimulatedMic(audio, fps, gain=SIMULATION_AUDIO_GAIN, denoise=False)

    identity = NullIdentity()
    detector = FaceDetector()
    tracker = FaceTracker()
    speaker = _create_speaker(fps)

    stride = VISION_STRIDE if fast else 1

    if not fast:
        sd.play(mic.audio, samplerate=SAMPLE_RATE)
    else:
        print(f"[fast] stride={stride}, audio playback disabled")

    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    paused = False
    playback_start = time.time()
    frame_delay = 1.0 / fps

    last_faces = []
    last_track_ids = []

    try:
        while cap.isOpened():
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    break

                timestamp = frame_idx / fps
                chunk = mic.advance_frame()
                speaker.drip_audio(chunk)

                if frame_idx % stride == 0:
                    faces = detector.detect(frame, timestamp=timestamp)
                    raw_matches = [identity.identify(face, frame_idx) for face in faces]
                    _, track_ids = tracker.update(faces, raw_matches, frame_idx)

                    for face, tid in zip(faces, track_ids):
                        speaker.add_crop(tid, face.crop)

                    last_faces = faces
                    last_track_ids = track_ids
                else:
                    faces = last_faces
                    track_ids = last_track_ids

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

                if not fast:
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
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    fast = "--fast" in sys.argv

    if not args:
        print("Usage: python debug_video.py [--fast] <video_path>")
        sys.exit(1)
    main(Path(args[0]), fast=fast)
