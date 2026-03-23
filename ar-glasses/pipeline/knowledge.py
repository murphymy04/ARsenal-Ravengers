import asyncio
import threading
from datetime import datetime, timezone

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType

from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

class KnowledgeStore:
    def __init__(self):
        self._client: Graphiti | None = None
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._ready = False

    def _run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def _ensure_client(self):
        if not self._ready:
            async def _init():
                self._client = Graphiti(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
                await self._client.build_indices_and_constraints()
            self._run(_init())
            self._ready = True

    def save(self, segments: list[dict], reference_time: datetime | None = None):
        self._ensure_client()

        if reference_time is None:
            reference_time = datetime.now(timezone.utc)
        elif reference_time.tzinfo is None:
            reference_time = reference_time.replace(tzinfo=timezone.utc)

        speakers = ", ".join(dict.fromkeys(seg["speaker"] for seg in segments))
        episode_name = f"Conversation on {reference_time.strftime('%Y-%m-%d')} — {speakers}"
        episode_body = "\n".join(f"{seg['speaker']}: {seg['text']}" for seg in segments)

        async def _add():
            await self._client.add_episode(
                name=episode_name,
                episode_body=episode_body,
                source=EpisodeType.text,
                source_description="smart glasses conversation",
                reference_time=reference_time,
            )

        future = asyncio.run_coroutine_threadsafe(_add(), self._loop)
        future.add_done_callback(
            lambda f: print(
                f"Saved to knowledge graph: {episode_name}"
                if f.exception() is None
                else f"Knowledge graph error: {f.exception()}"
            )
        )

    def close(self):
        if self._client:
            self._run(self._client.close())
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)

_store: KnowledgeStore | None = None

def save_to_memory(segments: list[dict], reference_time: datetime | None = None):
    global _store
    if _store is None:
        _store = KnowledgeStore()
    _store.save(segments, reference_time)
