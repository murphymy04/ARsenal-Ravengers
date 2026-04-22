import os

import torch

import numpy as np

import sounddevice as sd

from scipy.io.wavfile import write

from pyannote.audio import Model, Inference

from scipy.spatial.distance import cdist

from dotenv import load_dotenv


import soundfile as sf


def load_audio_dict(path):
    """Load a .wav file as a dict that pyannote accepts natively (no torchcodec needed)."""
    data, sr = sf.read(path, dtype='float32')
    # soundfile gives (samples, channels) or (samples,) for mono
    if data.ndim == 1:
        data = data[np.newaxis, :]   # -> (1, samples)
    else:
        data = data.T                 # -> (channels, samples)
    return {"waveform": torch.from_numpy(data), "sample_rate": sr}

# --- CONFIGURATION ---

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

HF_TOKEN = os.environ.get("andrii_hf_token", "")

SAMPLE_RATE = 16000 # 16kHz is the standard for Pyannote models



def load_inference_model():

    """Loads the pretrained model from Hugging Face."""

    print("Loading pyannote/embedding model...")

    model = Model.from_pretrained("pyannote/embedding", use_auth_token=HF_TOKEN)

    return Inference(model, window="whole")



def record_audio(duration, save_path):

    """Records audio from the default microphone and saves it as a .wav file."""

    print(f"\n�� RECORDING STARTING for {duration} seconds. Speak now...")



    # Record audio (mono channel)

    recording = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')

    sd.wait() # Wait until recording is finished



    # Save as WAV file

    write(save_path, SAMPLE_RATE, recording)

    print(f"�� Recording finished. Saved to {save_path}\n")

    return save_path



def enroll_speaker(audio_path=None, save_path="my_voice_fingerprint.npy"):

    """Reads audio, extracts the embedding, and saves it to disk."""

    if not audio_path:

        # If no path provided, prompt to record

        audio_path = "temp_enrollment.wav"

        record_audio(duration=30, save_path=audio_path)



    inference = load_inference_model()

    print(f"Analyzing {audio_path} to learn your voice...")



    embedding = inference(load_audio_dict(audio_path))

    np.save(save_path, embedding)

    print(f"Success! Voice embedding saved to '{save_path}'.")



def verify_speaker(reference_npy, test_audio_path=None, threshold=0.5):

    """Loads saved embedding, compares to new audio, and determines match."""

    if not os.path.exists(reference_npy):

        print(f"Error: Could not find {reference_npy}. You must enroll first!")

        return



    if not test_audio_path:

        # If no path provided, prompt to record

        test_audio_path = "temp_verification.wav"

        record_audio(duration=3, save_path=test_audio_path)



    inference = load_inference_model()

    reference_embedding = np.load(reference_npy)



    print(f"Extracting embedding from test file: {test_audio_path}...")

    test_embedding = inference(load_audio_dict(test_audio_path))



    distance = cdist(

        reference_embedding.reshape(1, -1),

        test_embedding.reshape(1, -1),

        metric="cosine"

    )[0, 0]



    print("\n" + "="*40)

    print(f"Cosine Distance Score: {distance:.4f}")



    if distance < threshold:

        print("Result: ✅ MATCH - This is recognized as YOU.")

    else:

        print("Result: ❌ NO MATCH - This is recognized as SOMEONE ELSE.")

    print("="*40 + "\n")





# --- EXECUTION FLOW ---

if __name__ == "__main__":

    print("Welcome to Voice Embedder")

    print("1. Enroll (Learn my voice)")

    print("2. Verify (Test a voice)")



    choice = input("Enter 1 or 2: ")



    if choice == '1':

        print("\nYou need about 30 seconds of clean audio to enroll.")

        mode = input("Press 'R' to Record now, or 'F' to provide a File path: ").upper()

        if mode == 'R':

            enroll_speaker(save_path="my_voice.npy")

        elif mode == 'F':

            filepath = input("Enter path to your .wav file: ")

            enroll_speaker(audio_path=filepath, save_path="my_voice.npy")



    elif choice == '2':

        print("\nYou need about 5-10 seconds of audio to verify.")

        mode = input("Press 'R' to Record now, or 'F' to provide a File path: ").upper()

        if mode == 'R':
            while True:
                verify_speaker(reference_npy="my_voice.npy", threshold=0.5)
	
        elif mode == 'F':

            filepath = input("Enter path to your .wav file: ")

            verify_speaker(reference_npy="my_voice.npy", test_audio_path=filepath, threshold=0.5)
