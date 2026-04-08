"""Retrieval pipeline: queries knowledge graph when a known face appears.

Runs as an independent side-channel. The diarization pipeline pushes
track events to an input queue; this module applies per-person cooldown,
queries Graphiti for relevant facts, and pushes results to an output queue.
"""

import asyncio
import queue
import threading

from config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER, RETRIEVAL_COOLDOWN_SECONDS
from graphiti_core import Graphiti


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

            facts = self._retrieve(match.name)
            self._result_queue.put_nowait((match.name, match.person_id, facts))

    def _should_retrieve(self, name: str, timestamp: float) -> bool:
        last = self._last_retrieval.get(name, 0)
        if timestamp - last < self._cooldown_seconds:
            return False
        self._last_retrieval[name] = timestamp
        return True

    def _retrieve(self, query_name: str) -> list[str]:
        self._ensure_client()
        try:
            results = self._run_async(
                self._client.search(f"{query_name} projects work interests")
            )
            return [r.fact for r in results]
        except Exception as exc:
            print(f"[retrieval] error querying for {query_name}: {exc}")
            return []

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
