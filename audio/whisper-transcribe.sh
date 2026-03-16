#!/bin/bash
# Transcribes audio/video files using whisper.cpp with CoreML acceleration
# Automatically detects and skips silent portions at the start (common in Zoom recordings)
# Usage: whisper-transcribe <input-file> [output-name]
#   output-name defaults to input filename without extension

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WHISPER_CLI="$SCRIPT_DIR/build/bin/whisper-cli"
MODEL="$SCRIPT_DIR/models/ggml-small.en.bin"

if [ -z "$1" ]; then
    echo "Usage: whisper-transcribe <input-file> [output-name]"
    exit 1
fi

INPUT="$1"
OUTPUT="${2:-${INPUT%.*}}"

# Detect silence at start and find where speech begins
# Check volume at 0s, 5min, 10min, 15min, 20min, 25min, 30min
echo "Detecting speech start position..."
START_TIME=0
for t in 0 300 600 900 1200 1500 1800; do
    VOLUME=$(ffmpeg -i "$INPUT" -ss $t -t 10 -af volumedetect -f null /dev/null 2>&1 | grep "mean_volume" | awk '{print $5}')
    # Extract numeric part (remove "dB")
    VOLUME_NUM=${VOLUME% dB}
    # If mean volume > -85 dB, consider it speech start
    if (( $(echo "$VOLUME_NUM > -85" | bc -l) )); then
        START_TIME=$t
        echo "Speech detected at ${t}s (${VOLUME})"
        break
    fi
done

if [ $START_TIME -gt 0 ]; then
    echo "Skipping first ${START_TIME}s of silence..."
fi

# Convert to 16kHz mono WAV, starting from detected speech position
TMPWAV=$(mktemp /tmp/whisper_XXXX.wav)
if [[ "$INPUT" != *.wav ]] || [ $START_TIME -gt 0 ]; then
    ffmpeg -i "$INPUT" -ss $START_TIME -ar 16000 -ac 1 -c:a pcm_s16le "$TMPWAV" -y -loglevel error
    "$WHISPER_CLI" -m "$MODEL" -f "$TMPWAV" -otxt -of "$OUTPUT"
    rm -f "$TMPWAV"
else
    "$WHISPER_CLI" -m "$MODEL" -f "$INPUT" -otxt -of "$OUTPUT"
fi

echo "Transcript saved to ${OUTPUT}.txt"
