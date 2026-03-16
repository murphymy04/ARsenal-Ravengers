import whisperx
import gc 

# 1. Configuration
device = "cpu" # Use "cuda" for GPU, or "cpu" if you don't have one
audio_file = "output.wav" # Path to your audio file
batch_size = 16 # Reduce this if your GPU runs out of memory
compute_type = "float32" # Use "int8" if you have low GPU memory, or "float32" for CPU
huggingface_token = "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"

print("Loading audio...")
audio = whisperx.load_audio(audio_file)

# 2. Transcription (The Audio to Text part)
print("Loading Whisper model...")
# large-v3 is the most accurate, but requires more VRAM. You can use "base" or "small" for testing.
model = whisperx.load_model("large-v3", device, compute_type=compute_type)

print("Transcribing...")
result = model.transcribe(audio, batch_size=batch_size)
print("Transcription complete. Freeing memory...")
# Free up GPU memory for the next steps
del model
gc.collect()

# 3. Alignment (The Word-Level Precision part)
print("Loading alignment model...")
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)

print("Aligning transcript with audio...")
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
del model_a
gc.collect()

# 4. Diarization (The "Who is speaking" part)
print("Loading diarization model...")

# Import the pipeline from the new location
from whisperx.diarize import DiarizationPipeline

# Initialize the model (Note: some newer versions prefer 'token=' instead of 'use_auth_token=')
diarize_model = DiarizationPipeline(token=huggingface_token, device=device)

print("Diarizing audio...")
diarize_segments = diarize_model(audio, min_speakers=1, max_speakers=5)
# 5. Combining Text and Speakers
print("Assigning speakers to words...")
result = whisperx.assign_word_speakers(diarize_segments, result)

# 6. Print the Final Output
print("\n--- FINAL DIARIZED TRANSCRIPT ---\n")
for segment in result["segments"]:
    # WhisperX assigns labels like "SPEAKER_00", "SPEAKER_01"
    speaker = segment.get("speaker", "UNKNOWN")
    text = segment.get("text", "").strip()
    start_time = round(segment.get("start", 0), 2)
    
    print(f"[{start_time}s] {speaker}: {text}")
