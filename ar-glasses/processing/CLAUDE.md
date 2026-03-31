## ML Code Conventions

### Pipeline Structure
- Every pipeline step is its own function: load → preprocess → infer → postprocess → output. No monolithic `process_video()` functions that do everything.
- Entry points read like a recipe. Someone should be able to read `main()` and understand the full pipeline without diving into any function.

### Configuration & Magic Numbers
- No magic numbers anywhere. Every threshold, resolution, frame rate, model parameter comes from the config.py file.
  - BAD: `if confidence > 0.73:`
  - GOOD: `if confidence > config.face_match_threshold:`

### Tensor & Array Handling
- Always comment the shape after non-obvious tensor/array operations — this is the one place comments ARE required.
  - `features = encoder(frame)  # (batch, 512)`
  - `mel = extract_mel(audio)  # (n_frames, 80)`
- Name dimensions in variable names when ambiguous: `frame_bgr`, `frame_rgb`, `embedding_512d`, `batch_features`.
- Never silently convert between BGR/RGB. Every conversion is explicit and the variable name reflects the current format.
- Keep numpy and torch operations separate. Don't mix `.numpy()` and `torch.tensor()` calls back and forth in the same function.

### Model Loading & Inference
- Model loading is always a separate function that returns the model. Never load a model inside a processing loop or inside a function that also does inference.
- Use a single `models` dict or dataclass to hold all loaded models. Pass it around explicitly — no globals.
  - `models = load_models(config)` then `result = run_diarization(frame, audio, models)`
- Wrap inference calls in `torch.no_grad()` (or equivalent). Every time.
- Log model load times and device placement at startup.

### OpenCV Specifics
- Always use named constants for OpenCV flags, never raw ints.
  - BAD: `cap.get(3)`
  - GOOD: `cap.get(cv2.CAP_PROP_FRAME_WIDTH)`
- Always release resources: `cap.release()`, `cv2.destroyAllWindows()`. Prefer context managers or try/finally.
- Frames are always `frame_bgr` until explicitly converted. If a model expects RGB, convert and rename: `frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)`

### File Paths & I/O
- All paths go through `pathlib.Path`, never string concatenation.
- Model weights paths are config values, never hardcoded.
- Data I/O (reading video, audio, writing results) lives in its own module separate from model logic.

### Error Handling in Pipelines
- Fail loud and early. If a model file doesn't exist, raise immediately with the expected path — don't return None and let it crash three functions later.
- No silent fallbacks. If face detection returns nothing, log it and return an empty result — don't substitute a default or skip silently.

### Naming
- `frame` = single image, `frames` = list/batch, `clip` = sequence of frames
- `audio_segment` not `audio_chunk` or `audio_piece` — pick one term and use it everywhere
- `embedding` or `feature_vector`, not `vec`, `feat`, `repr`, `enc`
- `prediction` or `result`, not `output`, `out`, `res`, `ret`
- Model names are explicit: `face_detector`, `speaker_encoder`, `vad_model` — not `model1`, `net`, `m`