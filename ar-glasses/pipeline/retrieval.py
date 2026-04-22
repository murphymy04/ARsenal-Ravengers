"""Retrieval pipeline: queries knowledge graph when a known face appears.

Runs as an independent side-channel. The diarization pipeline pushes
track events to an input queue; this module applies per-person cooldown,
queries Graphiti for relevant facts plus the most recent conversation,
formats them into HUD-ready context via an LLM, and pushes the result
to an output queue.
"""

import asyncio
import json
import queue
import threading
from collections.abc import Callable
from datetime import UTC, datetime

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodicNode
from groq import Groq
from models import IdentityMatch

from config import (
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    RETRIEVAL_COOLDOWN_SECONDS,
    RETRIEVAL_MIN_TRACK_FRAMES,
)

FORMATTER_MODEL = "qwen/qwen3-32b"

FORMATTER_SYSTEM = """\
You format knowledge-graph context about someone the glasses wearer just \
met face-to-face. You will be given the person's name, known facts about \
them from prior conversations, and the transcript of the most recent \
conversation.

Return a JSON object with exactly these two string fields and no others:

{
  "last_spoke_about": "...",
  "ask_about": "..."
}

Rules:
- last_spoke_about: a concise sentence summarising what they talked about \
last time. No names prefixed, no quotes, under 15 words.
- ask_about: pick ONE fact most likely to have had an update since last \
time. Favour future plans, ongoing projects, recurring activities, \
upcoming events, deadlines. Phrase it as a natural short follow-up the \
wearer could say aloud, under 10 words, no trailing punctuation.
"""

FORMATTER_RESPONSE_FORMAT = {"type": "json_object"}


_groq = Groq()


class RetrievalDispatcher:
    """Per-track gating between vision and the retrieval worker.

    Waits MIN_TRACK_FRAMES of continuous sightings before firing, then
    uses per-track cooldown so the same tid can re-fire after
    RETRIEVAL_COOLDOWN_SECONDS. Dead tracks are evicted every dispatch
    via the active_set filter, so unknown faces don't need a separate
    frame-count ceiling.
    """

    def __init__(
        self,
        enqueue: Callable[[tuple], None],
        min_track_frames: int = RETRIEVAL_MIN_TRACK_FRAMES,
        cooldown_seconds: float = RETRIEVAL_COOLDOWN_SECONDS,
    ):
        self._enqueue = enqueue
        self._min_track_frames = min_track_frames
        self._cooldown_seconds = cooldown_seconds
        self._frames_seen: dict[int, int] = {}
        self._last_queued_ts: dict[int, float] = {}

    def dispatch(
        self,
        smoothed: list[IdentityMatch],
        track_ids: list[int],
        new_track_ids: set[int],
        timestamp: float,
    ):
        for tid in new_track_ids:
            self._frames_seen[tid] = 0

        active_set = set(track_ids)
        self._frames_seen = {
            tid: n for tid, n in self._frames_seen.items() if tid in active_set
        }
        self._last_queued_ts = {
            tid: t for tid, t in self._last_queued_ts.items() if tid in active_set
        }

        for tid in list(self._frames_seen):
            self._frames_seen[tid] += 1
            if self._frames_seen[tid] < self._min_track_frames:
                continue
            match = smoothed[track_ids.index(tid)]
            if not match.is_known:
                continue
            last = self._last_queued_ts.get(tid)
            if last is not None and timestamp - last < self._cooldown_seconds:
                continue
            self._enqueue((tid, match, timestamp))
            self._last_queued_ts[tid] = timestamp
            print(f"[retrieval] QUEUED {match.name} (track={tid}, t={timestamp:.1f}s)")

    def reset(self):
        self._frames_seen.clear()
        self._last_queued_ts.clear()


def _humanize_delta(past: datetime, now: datetime) -> str:
    seconds = int((now - past).total_seconds())
    if seconds < 60:
        return "just now"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    days = hours // 24
    if days < 7:
        return f"{days} day{'s' if days != 1 else ''} ago"
    if days < 30:
        weeks = days // 7
        return f"{weeks} week{'s' if weeks != 1 else ''} ago"
    if days < 365:
        months = days // 30
        return f"{months} month{'s' if months != 1 else ''} ago"
    years = days // 365
    return f"{years} year{'s' if years != 1 else ''} ago"


def _format_context(name: str, facts: list[str], last_episode: EpisodicNode) -> dict:
    user_prompt = (
        f"Person: {name}\n\n"
        f"Known facts:\n" + "\n".join(f"- {f}" for f in facts) + "\n\n"
        f"Last conversation (on {last_episode.valid_at.date().isoformat()}):\n"
        f"{last_episode.content}"
    )
    response = _groq.chat.completions.create(
        model=FORMATTER_MODEL,
        messages=[
            {"role": "system", "content": FORMATTER_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        response_format=FORMATTER_RESPONSE_FORMAT,
        temperature=0.2,
        reasoning_effort="none",
    )
    return json.loads(response.choices[0].message.content)


class RetrievalWorker:
    def __init__(
        self,
        event_queue: queue.Queue,
        result_queue: queue.Queue,
        cooldown_seconds: float = RETRIEVAL_COOLDOWN_SECONDS,
    ):
        self._event_queue = event_queue
        self._result_queue = result_queue
        self._cooldown_seconds = cooldown_seconds
        self._last_retrieval: dict[str, float] = {}
        self._stop = threading.Event()

        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._worker_thread = threading.Thread(target=self._run, daemon=True)

        self._client: Graphiti | None = None
        self._ready = False
        self._ready_lock = threading.Lock()

    def start(self):
        self._loop_thread.start()
        threading.Thread(target=self._warmup, daemon=True).start()
        self._worker_thread.start()
        print("[retrieval] worker started")

    def _warmup(self):
        try:
            self._ensure_client()
            print("[retrieval] graphiti client warm")
        except Exception as exc:
            print(f"[retrieval] warmup failed: {exc}")

    def stop(self):
        self._stop.set()
        self._worker_thread.join(timeout=5)
        if self._client:
            self._run_async(self._client.close())
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._loop_thread.join(timeout=5)

    def _run(self):
        while not self._stop.is_set():
            try:
                track_id, match, timestamp = self._event_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if not match.is_known or match.name.startswith("Person "):
                print(
                    f"[retrieval] skip track={track_id} name={match.name!r} "
                    f"(unlabeled or auto-cluster)"
                )
                continue

            if not self._should_retrieve(match.name, timestamp):
                print(f"[retrieval] skip {match.name} (cooldown)")
                continue

            print(f"[retrieval] FIRING for {match.name} (track={track_id})")
            context = self._retrieve(match.name)
            print(
                f"[retrieval] SENT {match.name} -> queue | "
                f"last_spoke={context.get('last_spoke')!r} "
                f"ask_about={context.get('ask_about')!r} "
                f"facts={len(context.get('raw_facts') or [])}"
            )
            self._result_queue.put_nowait((match.name, match.person_id, context))

    def _should_retrieve(self, name: str, timestamp: float) -> bool:
        last = self._last_retrieval.get(name)
        if last is not None and timestamp - last < self._cooldown_seconds:
            return False
        self._last_retrieval[name] = timestamp
        return True

    def _retrieve(self, query_name: str) -> dict:
        self._ensure_client()
        facts: list[str] = []
        last_episode: EpisodicNode | None = None
        try:
            facts, last_episode = self._run_async(self._fetch_all(query_name))
            return self._build_context(query_name, facts, last_episode)
        except Exception as exc:
            print(f"[retrieval] error formatting context for {query_name}: {exc}")
            return {
                "name": query_name,
                "last_spoke": (
                    _humanize_delta(last_episode.valid_at, datetime.now(UTC))
                    if last_episode
                    else None
                ),
                "last_spoke_about": None,
                "ask_about": None,
                "raw_facts": facts,
            }

    async def _fetch_all(self, name: str) -> tuple[list[str], EpisodicNode | None]:
        search_coro = self._client.search(
            f"What does {name} do? What are {name}'s projects, plans, and interests?",
            num_results=3,
        )
        episodes_coro = self._client.retrieve_episodes(datetime.now(UTC), last_n=50)
        search_results, episodes = await asyncio.gather(search_coro, episodes_coro)
        facts = [r.fact for r in search_results]
        needle = name.lower()
        last_episode = next(
            (
                ep
                for ep in episodes
                if needle in ep.name.lower() or needle in ep.content.lower()
            ),
            None,
        )
        return facts, last_episode

    def _build_context(
        self, name: str, facts: list[str], last_episode: EpisodicNode | None
    ) -> dict:
        if not last_episode:
            return {
                "name": name,
                "last_spoke": None,
                "last_spoke_about": None,
                "ask_about": None,
                "raw_facts": facts,
            }

        formatted = _format_context(name, facts, last_episode)
        return {
            "name": name,
            "last_spoke": _humanize_delta(last_episode.valid_at, datetime.now(UTC)),
            "last_spoke_about": formatted["last_spoke_about"],
            "ask_about": formatted["ask_about"],
            "raw_facts": facts,
        }

    def _ensure_client(self):
        with self._ready_lock:
            if self._ready:
                return

            async def _init():
                self._client = Graphiti(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
                await self._client.build_indices_and_constraints()

            self._run_async(_init())
            self._ready = True

    def _run_async(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()


def drain_results(result_queue: queue.Queue) -> list[tuple]:
    results = []
    while True:
        try:
            results.append(result_queue.get_nowait())
        except queue.Empty:
            break
    return results
