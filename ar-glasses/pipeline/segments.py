"""Shared segment utilities used by both diarization and live pipelines."""


def merge_close_segments(segments: list[dict], max_gap: float = 2.0) -> list[dict]:
    if not segments:
        return segments

    by_track: dict[int, list[dict]] = {}
    for seg in segments:
        by_track.setdefault(seg["track_id"], []).append(seg)

    merged: list[dict] = []
    for segs in by_track.values():
        segs.sort(key=lambda s: s["start"])
        current = dict(segs[0])
        for nxt in segs[1:]:
            if nxt["start"] - current["end"] < max_gap:
                current["end"] = nxt["end"]
                if nxt.get("person_id") is not None:
                    current["person_id"] = nxt["person_id"]
                    current["name"] = nxt["name"]
            else:
                merged.append(current)
                current = dict(nxt)
        merged.append(current)

    merged.sort(key=lambda s: s["start"])
    return merged
