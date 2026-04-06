"""Plot VAD debug CSV and an optional audio waveform."""

import argparse
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import SAMPLE_RATE, SIMULATION_AUDIO_GAIN

DEFAULT_CSV_PATH = Path(__file__).parent / "data" / "vad_debug.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the VAD debug CSV.")
    parser.add_argument(
        "video_path",
        nargs="?",
        type=Path,
        help="Optional video file used to render an audio waveform panel.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help=f"Path to the VAD CSV. Defaults to {DEFAULT_CSV_PATH}.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the PNG without opening an interactive plot window.",
    )
    return parser.parse_args()


def load_debug_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"VAD debug CSV not found: {csv_path}")
    return pd.read_csv(csv_path)


def load_audio(video_path: Path | None) -> np.ndarray | None:
    if video_path is None:
        return None
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cmd = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "f32le",
        "-loglevel",
        "error",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True, check=True)
    raw_audio = np.frombuffer(result.stdout, dtype=np.float32)
    return np.clip(raw_audio * SIMULATION_AUDIO_GAIN, -1.0, 1.0)


def plot_debug(df: pd.DataFrame, audio: np.ndarray | None, out_path: Path) -> None:
    num_panels = 5 if audio is not None else 4
    figure, axes = plt.subplots(
        num_panels,
        1,
        figsize=(14, 2.5 * num_panels),
        sharex=True,
    )
    panel_index = 0

    axis = axes[panel_index]
    panel_index += 1
    vad_active = df["vad_prob"] >= 0.5
    rms_min = df["rms"].min()
    rms_max = df["rms"].max()
    rms_range = rms_max - rms_min if rms_max != rms_min else 1.0

    def _normalize(series: "pd.Series") -> "pd.Series":
        return (series - rms_min) / rms_range

    rms_norm = _normalize(df["rms"])
    axis.fill_between(
        df["timestamp"],
        0,
        0.05,
        where=vad_active,
        color="darkorange",
        alpha=0.8,
        label="VAD active",
    )
    axis.plot(
        df["timestamp"],
        rms_norm,
        color="steelblue",
        linewidth=0.7,
        alpha=0.7,
        label="RMS (normalized)",
    )
    if "boundary" in df.columns:
        axis.plot(
            df["timestamp"],
            _normalize(df["boundary"]),
            color="red",
            linestyle="--",
            linewidth=1,
            label="adaptive boundary (normalized)",
        )
    if "noise_floor" in df.columns:
        axis.plot(
            df["timestamp"],
            _normalize(df["noise_floor"]),
            color="purple",
            linestyle=":",
            linewidth=1,
            label="noise floor (normalized)",
        )
    if {"noise_floor", "other_excess", "wearer_excess"} <= set(df.columns):
        axis.plot(
            df["timestamp"],
            _normalize(df["noise_floor"] + df["other_excess"]),
            color="teal",
            linestyle="-.",
            linewidth=1,
            label="other speaker mean (normalized)",
        )
        axis.plot(
            df["timestamp"],
            _normalize(df["noise_floor"] + df["wearer_excess"]),
            color="brown",
            linestyle="-.",
            linewidth=1,
            label="wearer mean (normalized)",
        )
    axis.set_ylabel("norm amplitude")
    axis.set_title("VAD + RMS combined")
    axis.legend(loc="upper right", fontsize=8)
    axis.set_ylim(0, 1.05)

    if audio is not None:
        axis = axes[panel_index]
        panel_index += 1
        time_axis = np.arange(len(audio)) / SAMPLE_RATE
        axis.plot(time_axis, audio, color="steelblue", linewidth=0.3, alpha=0.7)
        axis.set_ylabel("amplitude")
        axis.set_title("Audio waveform (gained)")
        axis.set_ylim(-0.5, 0.5)

    axis = axes[panel_index]
    panel_index += 1
    axis.plot(df["timestamp"], df["rms"], alpha=0.6, linewidth=0.8, label="raw RMS")
    if "boundary" in df.columns:
        axis.plot(
            df["timestamp"],
            df["boundary"],
            color="red",
            linestyle="--",
            linewidth=1.5,
            label="adaptive boundary",
        )
    elif "rms_ewma" in df.columns:
        axis.plot(
            df["timestamp"],
            df["rms_ewma"],
            color="red",
            linewidth=1.5,
            label="EWMA",
        )
    if "noise_floor" in df.columns:
        axis.plot(
            df["timestamp"],
            df["noise_floor"],
            color="purple",
            linestyle=":",
            linewidth=1.5,
            label="noise floor",
        )
    if {"noise_floor", "other_excess"} <= set(df.columns):
        axis.plot(
            df["timestamp"],
            df["noise_floor"] + df["other_excess"],
            color="teal",
            linestyle="-.",
            linewidth=1.2,
            label="other speaker mean",
        )
    if {"noise_floor", "wearer_excess"} <= set(df.columns):
        axis.plot(
            df["timestamp"],
            df["noise_floor"] + df["wearer_excess"],
            color="brown",
            linestyle="-.",
            linewidth=1.2,
            label="wearer mean (anchored)",
        )
    axis.set_ylabel("amplitude")
    axis.set_title("RMS + wearer/other boundary")
    axis.legend(loc="upper right")

    axis = axes[panel_index]
    panel_index += 1
    axis.plot(
        df["timestamp"],
        df["vad_prob"],
        color="orange",
        linewidth=0.8,
        label="VAD prob",
    )
    axis.axhline(
        y=0.5,
        color="red",
        linestyle="--",
        linewidth=0.8,
        label="threshold (0.5)",
    )
    axis.set_ylabel("probability")
    axis.set_title("Silero VAD")
    axis.legend(loc="upper right")

    axis = axes[panel_index]
    colors = {"silence": "gray", "wearer": "blue", "other": "green"}
    for label, color in colors.items():
        label_mask = df["classification"] == label
        axis.fill_between(
            df["timestamp"],
            0,
            1,
            where=label_mask,
            alpha=0.4,
            color=color,
            label=label,
        )
    axis.set_ylabel("speaker")
    axis.set_xlabel("time (s)")
    axis.set_title("Classification")
    axis.legend(loc="upper right")
    axis.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def main() -> None:
    args = parse_args()
    debug_df = load_debug_csv(args.csv)
    audio = load_audio(args.video_path)
    output_path = args.csv.with_suffix(".png")
    plot_debug(debug_df, audio, output_path)
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
