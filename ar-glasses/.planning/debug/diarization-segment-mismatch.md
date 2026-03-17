---
status: awaiting_human_verify
trigger: "diarization-segment-mismatch"
created: 2026-03-16T00:00:00Z
updated: 2026-03-16T00:01:00Z
---

## Current Focus

hypothesis: SpeakingDetector hard-codes self._fps = CAMERA_FPS (30) at construction time and never accepts the actual video FPS. For offline video files the real FPS may differ, causing the MFCC winstep and audio window sizing to be computed against the wrong FPS, desynchronising audio from video.
test: Compare self._fps value used in MFCC winstep / audio window calculation vs the fps passed to DiarizationPipeline.run()
expecting: self._fps is always CAMERA_FPS=30, but actual video FPS may differ (e.g. 25, 29.97, 60). This means every audio window is the wrong length.
next_action: Fix SpeakingDetector to accept fps as a constructor parameter and update DiarizationPipeline to pass fps through. Also fix the mfcc_winstep recalculation.

## Symptoms

expected: Diarization module should correctly detect when someone is speaking, producing segments that roughly match WhisperX's speech segments
actual: Segments from diarization.py don't match even remotely what WhisperX transcription produces
errors: No errors — runs cleanly but output is wrong
reproduction: Process an offline video file through the pipeline
started: Never worked on offline video. Works correctly on live streams.

## Eliminated

- hypothesis: audio is not fed at all / _mic_ok flag is False
  evidence: feed_audio() sets _mic_ok = True, and run_inference() checks _mic_ok before proceeding
  timestamp: 2026-03-16T00:00:00Z

- hypothesis: timestamp-based audio window selection is broken
  evidence: The offline branch (timestamp + _full_audio) is actually correct — it slices [audio_end - needed : audio_end] from the full PCM array
  timestamp: 2026-03-16T00:00:00Z

## Evidence

- timestamp: 2026-03-16T00:00:00Z
  checked: SpeakingDetector.__init__ in processing/speaking_detector.py lines 61-65
  found: self._fps = CAMERA_FPS (hard-coded from config, always 30). mfcc_winstep = 1/(4*self._fps) = 1/120 s. audio_sec = T / self._fps.
  implication: If the video has a different FPS (e.g. 25 fps), every audio window will be (T/30) seconds long instead of (T/25) seconds. At 25 fps a 30-frame buffer covers 1.2 s of video, but the code slices only 1.0 s of audio — a 17% underread. At 60 fps the code slices 2.0 s of audio for a 1-second video window — doubling the audio.

- timestamp: 2026-03-16T00:00:00Z
  checked: DiarizationPipeline.run() in pipeline/diarization.py line 41
  found: speaker = SpeakingDetector(use_mic=False) — no fps argument passed
  implication: Even though the driver extracts the real fps from the video file and passes it to DiarizationPipeline.run(fps=...), that fps is never forwarded to SpeakingDetector.

- timestamp: 2026-03-16T00:00:00Z
  checked: DiarizationPipeline.run() line 48
  found: timestamp = frame_idx / fps — uses the real fps for timestamps
  implication: The timestamps given to run_inference() are correct. The audio window end-point (audio_end = int(timestamp * sample_rate)) is therefore correct. But audio_sec = T / self._fps uses the wrong fps, so the window *length* (needed = int(audio_sec * sample_rate)) is wrong. The window end is right but the window start is wrong.

- timestamp: 2026-03-16T00:00:00Z
  checked: mfcc_winstep calculation (line 65)
  found: mfcc_winstep = 1/(4 * CAMERA_FPS) = 1/120 — this is used to generate MFCC frames at 4x camera FPS. If actual FPS differs, the MFCC frame rate no longer matches the 4:1 ratio the model expects.
  implication: Dual error — both window length AND MFCC temporal resolution are computed against wrong FPS.

- timestamp: 2026-03-16T00:00:00Z
  checked: Live stream path
  found: Live stream always runs at CAMERA_FPS (it's reading from the webcam at that FPS). So the bug is invisible there — CAMERA_FPS happens to equal the actual FPS.
  implication: Confirms why it works live but breaks offline.

## Resolution

root_cause: SpeakingDetector hard-codes self._fps = CAMERA_FPS instead of accepting the actual video fps. This means: (1) the audio window length (audio_sec = T / self._fps) is wrong for offline videos with a different FPS, making the audio/video window misaligned; (2) the MFCC winstep (1/(4*fps)) is computed at the wrong rate, breaking the 4:1 audio/visual frame ratio the model expects.
fix: Added fps parameter to SpeakingDetector.__init__ (defaults to CAMERA_FPS for backwards compatibility). mfcc_winstep and mfcc_winlen are now derived from the passed fps. DiarizationPipeline.run() passes its fps argument through as SpeakingDetector(use_mic=False, fps=fps).
verification: awaiting human confirmation on offline video
files_changed:
  - processing/speaking_detector.py  (line 56: fps param; line 61: self._fps = fps)
  - pipeline/diarization.py          (line 41: fps=fps passed to SpeakingDetector)
