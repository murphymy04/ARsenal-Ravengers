import asyncio
import atexit
import logging
import os
import threading
from datetime import UTC, datetime

from graphiti_core import Graphiti
from graphiti_core.llm_client import LLMConfig, OpenAIClient
from graphiti_core.nodes import EpisodeType
from pydantic import BaseModel, Field

from config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER

GRAPHITI_LLM_MODEL = os.getenv("GRAPHITI_LLM_MODEL", "gpt-4.1")
GRAPHITI_SMALL_LLM_MODEL = os.getenv("GRAPHITI_SMALL_LLM_MODEL", "gpt-4.1-mini")

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


class Commitment(BaseModel):
    """A promise or unresolved action item between two people,
    e.g. to send something, make an intro, follow up on a request."""

    description: str = Field(
        description="What was promised, in a few words",
    )
    status: str = Field(
        default="open",
        description="'open' if not yet fulfilled, 'fulfilled' if completed",
    )


ENTITY_TYPES: dict[str, type[BaseModel]] = {
    "Person": Person,
    "Product": Product,
    "Topic": Topic,
    "Commitment": Commitment,
}

EDGE_TYPES: dict[str, type[BaseModel]] = {}

EDGE_TYPE_MAP: dict[tuple[str, str], list[str]] = {
    ("Person", "Person"): ["SPOKE_WITH", "KNOWS", "MENTIONED"],
    ("Person", "Product"): ["DISCUSSED", "USES", "OWNS", "REVIEWED"],
    ("Person", "Topic"): ["DISCUSSED", "INTERESTED_IN", "WORKS_ON"],
    ("Person", "Commitment"): ["PROMISED", "OWED"],
    ("Commitment", "Person"): ["PROMISED_TO"],
    ("Product", "Topic"): ["RELATES_TO"],
    ("Product", "Product"): ["COMPARED_TO", "ALTERNATIVE_TO", "COMPONENT_OF"],
}


def build_extraction_instructions(wearer_name: str, other_name: str | None) -> str:
    other = other_name or "the other person"
    forbid = (
        f'Do NOT create generic entities named "other person", "the other '
        f'person", "someone", or similar. Every reference to the wearer\'s '
        f'conversation partner MUST use the exact name "{other}" as the '
        f"entity name, even when that name is never spoken inside this "
        f"transcript chunk."
        if other_name
        else ""
    )
    return f"""\
This is a transcript from smart glasses worn by {wearer_name} (the "wearer").
{wearer_name} is having a face-to-face conversation with {other}.
Speaker attribution in the transcript is NOT reliable — treat it as a \
single mixed stream and infer attribution from content.

Entity naming:
- Create a Person entity named "{wearer_name}" with role="wearer".
- Create a Person entity named "{other}" with role="other".
- {forbid}
- All facts about the wearer's conversation partner belong on the \
"{other}" entity.

Key rules:
- Always create a SPOKE_WITH edge between "{wearer_name}" and "{other}".
- Default assumption: statements describe or come from {other}, not \
{wearer_name}. Attribute facts to {other} unless the content is \
unmistakably self-referential by {wearer_name} (e.g. "me too", "same \
here, I also...", or reciprocal disclosures after {other} has shared \
something).
- Focus on learning about {other}: their work, projects, opinions, \
plans, interests, and background.
- Extract concrete facts as edges: opinions stated, products discussed, \
plans mentioned, preferences expressed.
- For products/tech, extract specific details (features, comparisons, \
opinions).
- Prefer specific facts over vague summaries. "X works on Y because of Z" \
is better than "X discussed Y".
- Skip filler, back-channel responses, and meta-discussion about the \
recording setup or glasses themselves.

Commitments (promises, intros, follow-ups):
- Look for explicit commitment language: "I'll send...", "I'll introduce \
you to...", "let me share...", "I can connect you with...", "yeah, I'd \
love that", "sure, send it over".
- Infer direction from the verb and context, not speaker tags:
  - "I'll send you X" → speaker promises, addressee receives
  - "can you send me X" / "would love an intro to X" → addressee \
  promises, speaker receives
- Use topical context to disambiguate when unclear. If the commitment \
concerns {wearer_name}'s known work or network, {wearer_name} is likely \
the one promising; if it concerns {other}'s domain, they are.
- Create a Commitment entity with status="open" and connect it with \
PROMISED (from the promiser) and PROMISED_TO (to the recipient) edges.
- Only create commitments for explicit promises. Do not infer \
commitments from vague interest ("that sounds cool") or hypotheticals.
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
                llm_client = OpenAIClient(
                    config=LLMConfig(
                        model=GRAPHITI_LLM_MODEL,
                        small_model=GRAPHITI_SMALL_LLM_MODEL,
                    )
                )
                self._client = Graphiti(
                    NEO4J_URI,
                    NEO4J_USER,
                    NEO4J_PASSWORD,
                    llm_client=llm_client,
                )
                await self._client.build_indices_and_constraints()

            self._run(_init())
            self._ready = True

    def save(
        self,
        transcript: str,
        wearer_name: str = "Wearer",
        other_name: str | None = None,
        reference_time: datetime | None = None,
    ):
        self._ensure_client()

        if reference_time is None:
            reference_time = datetime.now(UTC)
        elif reference_time.tzinfo is None:
            reference_time = reference_time.replace(tzinfo=UTC)

        other = other_name or "someone"
        date_str = reference_time.strftime("%Y-%m-%d")
        episode_name = f"{wearer_name} spoke with {other} on {date_str}"

        topics = _extract_topic_hint(transcript)
        source_desc = (
            f"Transcript of face-to-face conversation between {wearer_name} "
            f"and {other}. Speaker attribution is unavailable; "
            f"treat statements as primarily describing {other}."
        )
        if topics:
            source_desc += f" Topics discussed: {topics}."

        extraction_instructions = build_extraction_instructions(wearer_name, other_name)

        async def _add():
            await self._client.add_episode(
                name=episode_name,
                episode_body=transcript,
                source=EpisodeType.text,
                source_description=source_desc,
                reference_time=reference_time,
                entity_types=ENTITY_TYPES,
                edge_types=EDGE_TYPES,
                edge_type_map=EDGE_TYPE_MAP,
                custom_extraction_instructions=extraction_instructions,
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


def _extract_topic_hint(transcript: str) -> str:
    if len(transcript) < 50:
        return ""
    words = transcript.lower().split()
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
    transcript: str,
    wearer_name: str = "Wearer",
    other_name: str | None = None,
    reference_time: datetime | None = None,
):
    global _store
    if _store is None:
        _store = KnowledgeStore()
    _store.save(transcript, wearer_name, other_name, reference_time)


def flush_memory(timeout: float = 120):
    if _store is not None:
        _store.flush(timeout)
