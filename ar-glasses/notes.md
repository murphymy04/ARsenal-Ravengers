## ASD / Speaker Detection Improvements

1. **Temporal smoothing on ASD output** — Instead of thresholding each inference independently, smooth the probability over a sliding window (e.g., if 3 out of 5 recent inferences are above 0.35, mark as speaking). Handles the "bouncing around threshold" case without lowering the hard threshold.

2. **Bandpass filter on audio** — Isolate the speech band (300Hz–3kHz) before MFCC extraction to remove background noise that weakens the audio-visual correlation.

3. **Audio normalization** — Normalize audio to peak amplitude before feeding to the pipeline. Different mic qualities (glasses mic vs laptop mic) produce very different signal levels, which affects MFCC features and ASD scores. Currently trying a simple 1.5x gain boost as a first step.
