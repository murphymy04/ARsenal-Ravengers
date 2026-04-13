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
from datetime import UTC, datetime

from config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER, RETRIEVAL_COOLDOWN_SECONDS
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodicNode
from groq import Groq

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

    def start(self):
        self._loop_thread.start()
        self._worker_thread.start()
        print("[retrieval] worker started")

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
                continue

            if not self._should_retrieve(match.name, timestamp):
                continue

            context = self._retrieve(match.name)
            self._result_queue.put_nowait((match.name, match.person_id, context))

    def _should_retrieve(self, name: str, timestamp: float) -> bool:
        last = self._last_retrieval.get(name, 0)
        if timestamp - last < self._cooldown_seconds:
            return False
        self._last_retrieval[name] = timestamp
        return True

    def _retrieve(self, query_name: str) -> dict:
        self._ensure_client()
        facts: list[str] = []
        last_episode: EpisodicNode | None = None
        try:
            facts = self._fetch_facts(query_name)
            last_episode = self._fetch_last_episode(query_name)
            return self._build_context(query_name, facts, last_episode)
        except Exception as exc:
            print(f"[retrieval] error formatting context for {query_name}: {exc}")
            return {
                "last_spoke": (
                    _humanize_delta(last_episode.valid_at, datetime.now(UTC))
                    if last_episode
                    else None
                ),
                "last_spoke_about": None,
                "ask_about": None,
                "raw_facts": facts,
            }

    def _fetch_facts(self, name: str) -> list[str]:
        results = self._run_async(
            self._client.search(f"{name} projects work interests")
        )
        return [r.fact for r in results]

    def _fetch_last_episode(self, name: str) -> EpisodicNode | None:
        episodes = self._run_async(
            self._client.retrieve_episodes(datetime.now(UTC), last_n=50)
        )
        needle = name.lower()
        for episode in episodes:
            if needle in episode.name.lower() or needle in episode.content.lower():
                return episode
        return None

    def _build_context(
        self, name: str, facts: list[str], last_episode: EpisodicNode | None
    ) -> dict:
        if not last_episode:
            return {
                "last_spoke": None,
                "last_spoke_about": None,
                "ask_about": None,
                "raw_facts": facts,
            }

        formatted = _format_context(name, facts, last_episode)
        return {
            "last_spoke": _humanize_delta(last_episode.valid_at, datetime.now(UTC)),
            "last_spoke_about": formatted["last_spoke_about"],
            "ask_about": formatted["ask_about"],
            "raw_facts": facts,
        }

    def _ensure_client(self):
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
