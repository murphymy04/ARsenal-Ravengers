"""
Retrieval pipeline PoC — simulates recognising Peter Nguyen's face and
fetching relevant context from the knowledge graph to prompt Will.

Usage:
    python retrieve.py

Requires Neo4j running and seed.py to have been run first.
"""

import asyncio
import os

from dotenv import load_dotenv
from graphiti_core import Graphiti
from openai import AsyncOpenAI

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "ravengers")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

WEARER = "Will"
RECOGNIZED_PERSON = "Peter Nguyen"

# Topics we expect the graph to know about Peter — used for assertion checks
EXPECTED_TOPICS = ["game", "steam", "chillenium", "card", "platformer", "launch", "jam"]


async def retrieve_context(person_name: str) -> tuple[list[str], str]:
    """
    Simulate a face-recognition trigger for `person_name`.

    Returns:
        facts   — raw fact strings pulled from the knowledge graph
        prompt  — post-processed, display-ready conversation starter for the wearer
    """
    graph = Graphiti(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    oai = AsyncOpenAI(api_key=OPENAI_API_KEY)

    print(f"\n[face recognised] → {person_name}")
    print("Querying knowledge graph...")

    results = await graph.search(f"{person_name} projects work interests")
    await graph.close()

    facts = [r.fact for r in results]

    if not facts:
        return [], f"No context found for {person_name}."

    facts_block = "\n".join(f"- {f}" for f in facts)
    print(f"\nRaw facts ({len(facts)}):")
    print(facts_block)

    # Post-process: turn raw facts into a single friendly conversation starter
    response = await oai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    f"You are an assistant helping {WEARER} have a great conversation. "
                    f"{WEARER} is wearing smart glasses that just recognised {person_name}'s face. "
                    "Based on what they last talked about, write a single, natural, "
                    "friendly conversation-starter sentence for the wearer to use. "
                    "Be specific — reference something concrete from their last conversation. "
                    "No preamble, no labels, just the sentence."
                ),
            },
            {
                "role": "user",
                "content": f"Facts about {person_name} from last conversation:\n{facts_block}",
            },
        ],
        max_tokens=120,
    )

    prompt = response.choices[0].message.content.strip()
    return facts, prompt


async def main() -> None:
    facts, prompt = await retrieve_context(RECOGNIZED_PERSON)

    print(f"\n{'─' * 50}")
    print(f"[glasses display] → {prompt}")
    print(f"{'─' * 50}\n")

    # ── Assertions ────────────────────────────────────────────────────────────
    assert facts, "Knowledge graph returned no facts for Peter Nguyen"
    assert len(facts) >= 2, f"Expected at least 2 facts, got {len(facts)}"

    assert prompt, "Generated prompt is empty"
    assert len(prompt) > 15, f"Prompt is suspiciously short: {prompt!r}"

    matched = [kw for kw in EXPECTED_TOPICS if kw.lower() in prompt.lower()]
    assert matched, (
        f"Prompt doesn't reference any expected topics {EXPECTED_TOPICS}\n"
        f"Prompt was: {prompt!r}"
    )

    print(f"✓ {len(facts)} facts retrieved from graph")
    print(f"✓ Prompt references known topics: {matched}")
    print("✓ All assertions passed")


if __name__ == "__main__":
    asyncio.run(main())
