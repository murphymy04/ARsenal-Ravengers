import asyncio
import atexit
import logging
import threading
from datetime import UTC, datetime

from config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from pydantic import BaseModel, Field

logging.getLogger("neo4j.debug").setLevel(logging.CRITICAL)
logging.getLogger("neo4j.io").setLevel(logging.CRITICAL)
logging.getLogger("neo4j").setLevel(logging.CRITICAL)

class Person(BaseModel):
    """A person involved in or mentioned during a conversation.
    This includes the wearer and anyone they speak with."""

    role: str = Field(
        default="other",
        description="'wearer' if this is the glasses user, otherwise 'other'",
    )


class Product(BaseModel):
    """A product, device, technology, or tool mentioned in the conversation."""

    category: str = Field(
        default="",
        description="Category such as 'smart glasses', 'app', 'framework', etc.",
    )


class Topic(BaseModel):
    """An abstract topic, project, event, or idea discussed in the conversation."""


ENTITY_TYPES: dict[str, type[BaseModel]] = {
    "Person": Person,
    "Product": Product,
    "Topic": Topic,
}

EDGE_TYPES: dict[str, type[BaseModel]] = {}

EDGE_TYPE_MAP: dict[tuple[str, str], list[str]] = {
    ("Person", "Person"): ["SPOKE_WITH", "KNOWS", "MENTIONED"],
    ("Person", "Product"): ["DISCUSSED", "USES", "OWNS", "REVIEWED"],
    ("Person", "Topic"): ["DISCUSSED", "INTERESTED_IN", "WORKS_ON"],
    ("Product", "Topic"): ["RELATES_TO"],
    ("Product", "Product"): ["COMPARED_TO", "ALTERNATIVE_TO", "COMPONENT_OF"],
}

EXTRACTION_INSTRUCTIONS = """\
This is a transcript from smart glasses worn by a person (the "wearer").
The wearer is recording a face-to-face conversation with another person.

Key rules:
- Create a Person entity for the wearer with role="wearer".
- Always create a SPOKE_WITH edge between the wearer and the other person.
- Extract concrete facts as edges: opinions stated, products discussed, \
plans mentioned, preferences expressed.
- Capture BOTH speakers' contributions. The wearer's questions and statements \
matter too — "wearer asked about X", "wearer confirmed Y".
- For products/tech, extract specific details (features, comparisons, opinions).
- Prefer specific facts over vague summaries. "X said Y is better than Z \
because of W" is better than "X discussed Y".
"""


class KnowledgeStore:
    def __init__(self):
        self._client: Graphiti | None = None
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._ready = False
        self._pending: list[asyncio.Future] = []

    def _run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def _ensure_client(self):
        if not self._ready:

            async def _init():
                self._client = Graphiti(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
                await self._client.build_indices_and_constraints()

            self._run(_init())
            self._ready = True

    def save(
        self,
        segments: list[dict],
        wearer_name: str = "Wearer",
        other_name: str | None = None,
        reference_time: datetime | None = None,
    ):
        self._ensure_client()

        if reference_time is None:
            reference_time = datetime.now(UTC)
        elif reference_time.tzinfo is None:
            reference_time = reference_time.replace(tzinfo=UTC)

        speakers = list(dict.fromkeys(seg["speaker"] for seg in segments))
        other = other_name or next((s for s in speakers if s != wearer_name), None)

        date_str = reference_time.strftime("%Y-%m-%d")
        episode_name = f"{wearer_name} spoke with {other or 'someone'} on {date_str}"
        episode_body = "\n".join(f"{seg['speaker']}: {seg['text']}" for seg in segments)

        topics = _extract_topic_hint(segments)
        source_desc = (
            f"Face-to-face conversation recorded by {wearer_name}'s smart glasses. "
            f"Participants: {', '.join(speakers)}."
        )
        if topics:
            source_desc += f" Topics discussed: {topics}."

        async def _add():
            await self._client.add_episode(
                name=episode_name,
                episode_body=episode_body,
                source=EpisodeType.text,
                source_description=source_desc,
                reference_time=reference_time,
                entity_types=ENTITY_TYPES,
                edge_types=EDGE_TYPES,
                edge_type_map=EDGE_TYPE_MAP,
                custom_extraction_instructions=EXTRACTION_INSTRUCTIONS,
            )

        future = asyncio.run_coroutine_threadsafe(_add(), self._loop)
        self._pending.append(future)
        future.add_done_callback(
            lambda f: print(
                f"Saved to knowledge graph: {episode_name}"
                if f.exception() is None
                else f"Knowledge graph error: {f.exception()}"
            )
        )

    def flush(self, timeout: float = 120):
        for future in self._pending:
            future.result(timeout=timeout)
        self._pending.clear()

    def close(self):
        self.flush()
        if self._client:
            self._run(self._client.close())
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)


def _extract_topic_hint(segments: list[dict]) -> str:
    all_text = " ".join(seg["text"] for seg in segments).lower()
    if len(all_text) < 50:
        return ""
    words = all_text.split()
    unique = list(dict.fromkeys(w for w in words if len(w) > 5))
    return ", ".join(unique[:10])


_store: KnowledgeStore | None = None


def _cleanup():
    if _store is not None:
        print("[knowledge] waiting for pending saves...")
        _store.close()
        print("[knowledge] done.")


atexit.register(_cleanup)


def save_to_memory(
    segments: list[dict],
    wearer_name: str = "Wearer",
    other_name: str | None = None,
    reference_time: datetime | None = None,
):
    global _store
    if _store is None:
        _store = KnowledgeStore()
    _store.save(segments, wearer_name, other_name, reference_time)


def flush_memory(timeout: float = 120):
    if _store is not None:
        _store.flush(timeout)
