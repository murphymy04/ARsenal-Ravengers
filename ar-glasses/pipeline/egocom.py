import csv
from dataclasses import dataclass
from pathlib import Path


EGOCOM_ROOT = Path(__file__).resolve().parent.parent.parent / "egocom"
GROUND_TRUTH_CSV = EGOCOM_ROOT.parent / "ground_truth_transcriptions.csv"
VIDEOS_DIR = EGOCOM_ROOT / "EGOCOM" / "480p" / "5min_parts"

MAX_WORD_GAP = 1.5


@dataclass
class Segment:
    speaker_id: int
    start: float
    end: float


@dataclass
class Word:
    speaker_id: int
    start: float
    end: float
    text: str


def parse_ground_truth(csv_path: Path = GROUND_TRUTH_CSV) -> dict[str, list[Word]]:
    conversations: dict[str, list[Word]] = {}

    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if not row["startTime"] or not row["endTime"]:
                continue

            word = row["word"].strip()
            if not word or not any(c.isalnum() for c in word):
                continue

            conv_id = row["conversation_id"]
            if conv_id not in conversations:
                conversations[conv_id] = []

            conversations[conv_id].append(Word(
                speaker_id=int(row["speaker_id"]),
                start=float(row["startTime"]),
                end=float(row["endTime"]),
                text=word,
            ))

    for words in conversations.values():
        words.sort(key=lambda w: w.start)

    return conversations


def collapse_to_segments(words: list[Word], max_gap: float = MAX_WORD_GAP) -> list[Segment]:
    if not words:
        return []

    segments = []
    current = Segment(
        speaker_id=words[0].speaker_id,
        start=words[0].start,
        end=words[0].end,
    )

    for word in words[1:]:
        same_speaker = word.speaker_id == current.speaker_id
        small_gap = (word.start - current.end) <= max_gap

        if same_speaker and small_gap:
            current.end = max(current.end, word.end)
        else:
            segments.append(current)
            current = Segment(
                speaker_id=word.speaker_id,
                start=word.start,
                end=word.end,
            )

    segments.append(current)
    return segments


def video_path_for(conversation_id: str, person: int = 1) -> Path | None:
    parts = conversation_id.replace("__", "_").split("_")
    day = parts[1]
    con = parts[3]
    part = parts[4] if len(parts) > 4 else "part1"

    pattern = f"*__day_{day}__con_{con}__person_{person}_{part}.MP4"
    matches = list(VIDEOS_DIR.glob(pattern))
    return matches[0] if matches else None


def print_summary(conversations: dict[str, list[Word]]):
    conv_ids = sorted(conversations.keys())
    print(f"Parsed {len(conv_ids)} conversations, {sum(len(w) for w in conversations.values())} words total\n")

    for conv_id in conv_ids[:3]:
        words = conversations[conv_id]
        segments = collapse_to_segments(words)
        speakers = {s.speaker_id for s in segments}
        total_duration = max(s.end for s in segments) - min(s.start for s in segments)

        video = video_path_for(conv_id)
        video_status = f"found: {video.name}" if video else "NOT FOUND"

        print(f"{conv_id}")
        print(f"  words: {len(words)}, segments: {len(segments)}, speakers: {speakers}")
        print(f"  duration: {total_duration:.1f}s, video: {video_status}")

        for seg in segments[:8]:
            print(f"    [{seg.start:7.2f} - {seg.end:7.2f}] speaker_{seg.speaker_id}")

        if len(segments) > 8:
            print(f"    ... ({len(segments) - 8} more segments)")
        print()


if __name__ == "__main__":
    conversations = parse_ground_truth()
    print_summary(conversations)
