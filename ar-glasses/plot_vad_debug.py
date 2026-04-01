"""Plot VAD debug CSV + audio waveform — run after debug_video.py or pipeline/live.py."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "data" / "vad_debug.csv"
df = pd.read_csv(csv_path)

# Try to load the denoised+gained audio for the waveform panel
audio_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
audio = None
if audio_path and audio_path.exists():
    import subprocess
    from config import SAMPLE_RATE, SIMULATION_AUDIO_GAIN
    cmd = [
        "ffmpeg", "-i", str(audio_path),
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-f", "f32le", "-loglevel", "error", "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True, check=True)
    raw = np.frombuffer(result.stdout, dtype=np.float32)
    audio = np.clip(raw * SIMULATION_AUDIO_GAIN, -1.0, 1.0)

n_panels = 5 if audio is not None else 4
fig, axes = plt.subplots(n_panels, 1, figsize=(14, 2.5 * n_panels), sharex=True)

panel = 0

# 0. Combined VAD + RMS
ax = axes[panel]; panel += 1
vad_active = df["vad_prob"] >= 0.5
ax.fill_between(df["timestamp"], 0, 0.05, where=vad_active, color="darkorange", alpha=0.8, label="VAD active")
rms_norm = df["rms"] / df["rms"].max() if df["rms"].max() > 0 else df["rms"]
ax.plot(df["timestamp"], rms_norm, color="steelblue", linewidth=0.7, alpha=0.7, label="RMS (normalized)")
if "boundary" in df.columns:
    boundary_norm = df["boundary"] / df["rms"].max() if df["rms"].max() > 0 else df["boundary"] * 0
    ax.plot(df["timestamp"], boundary_norm, color="red", linestyle="--", linewidth=1, label="adaptive boundary (norm'd)")
ax.set_ylabel("norm amplitude")
ax.set_title("VAD + RMS combined")
ax.legend(loc="upper right", fontsize=8)
ax.set_ylim(0, 1.05)

# 1. Audio waveform (if available)
if audio is not None:
    ax = axes[panel]; panel += 1
    from config import SAMPLE_RATE
    t = np.arange(len(audio)) / SAMPLE_RATE
    ax.plot(t, audio, color="steelblue", linewidth=0.3, alpha=0.7)
    ax.set_ylabel("amplitude")
    ax.set_title("Audio waveform (gained)")
    ax.set_ylim(-0.5, 0.5)

# 2. Raw RMS + boundary line
ax = axes[panel]; panel += 1
ax.plot(df["timestamp"], df["rms"], alpha=0.6, linewidth=0.8, label="raw RMS")
if "boundary" in df.columns:
    ax.plot(df["timestamp"], df["boundary"], color="red", linestyle="--", linewidth=1.5, label="adaptive boundary")
elif "rms_ewma" in df.columns:
    ax.plot(df["timestamp"], df["rms_ewma"], color="red", linewidth=1.5, label="EWMA")
ax.set_ylabel("amplitude")
ax.set_title("RMS + wearer/other boundary")
ax.legend(loc="upper right")

# 3. VAD probability
ax = axes[panel]; panel += 1
ax.plot(df["timestamp"], df["vad_prob"], color="orange", linewidth=0.8, label="VAD prob")
ax.axhline(y=0.5, color="red", linestyle="--", linewidth=0.8, label="threshold (0.5)")
ax.set_ylabel("probability")
ax.set_title("Silero VAD")
ax.legend(loc="upper right")

# 4. Classification
ax = axes[panel]; panel += 1
colors = {"silence": "gray", "wearer": "blue", "other": "green"}
for label, color in colors.items():
    mask = df["classification"] == label
    ax.fill_between(df["timestamp"], 0, 1, where=mask, alpha=0.4, color=color, label=label)
ax.set_ylabel("speaker")
ax.set_xlabel("time (s)")
ax.set_title("Classification")
ax.legend(loc="upper right")
ax.set_ylim(0, 1)

plt.tight_layout()
out_path = csv_path.with_suffix(".png")
plt.savefig(out_path, dpi=150)
print(f"Saved {out_path}")
plt.show()
