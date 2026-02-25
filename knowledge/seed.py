"""
Seed script — adds Will's conference conversation with Peter Nguyen into graphiti.
Run once after Neo4j is healthy.

Usage:
    docker compose up -d
    # wait ~15s for Neo4j to be ready
    python seed.py
"""

import asyncio
import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "ravengers")

# Rich back-and-forth between Will (glasses wearer) and Peter Nguyen.
# Starts on AI coding tools, naturally drifts into Peter's game from Chillenium.
CONVERSATION = """\
Will: Hey, great panel! Loved the bit on context windows. I'm Will by the way.
Peter: Peter Nguyen — good to meet you. Yeah that section was wild. The jump from \
4k to a million tokens and people still complain it forgets stuff by the end.
Will: Ha, right. Are you building anything that actually uses long context, or more \
the agentic side of things?
Peter: Bit of both honestly. I've been deep in AI coding tools for a few months now — \
Cursor mostly, Claude under the hood. It's kind of changed how I think about writing \
code entirely.
Will: Same. I went from treating it like autocomplete to basically describing intent \
and reviewing output. Took a while to trust it though.
Peter: Exactly, the trust calibration is the real skill. I caught it confidently \
generating a database schema that was internally consistent but completely wrong for \
what I needed. Looked totally reasonable.
Will: The confident wrongness is the thing. I've started treating it like a fast junior \
dev — great for scaffolding, needs supervision on anything that touches prod.
Peter: That's exactly how I use it. Actually the project where it clicked for me was \
this game I built at a jam last month — Chillenium.
Will: Oh nice, what's Chillenium?
Peter: It's a 48-hour game jam. You show up, get a theme prompt, build something from \
scratch. I used Cursor basically the entire time for the engine scaffolding and \
it was the first jam where I didn't spend half my time fighting boilerplate.
Will: What did you end up making?
Peter: A 2D platformer, but the whole hook is the boss fights. Instead of just learning \
attack patterns, you draft a hand of cards at the start of each run — the cards \
define your moves and abilities. So every fight plays completely differently \
depending on what you drafted.
Will: Oh that's a sick mechanic. So it's roguelite-ish with the card layer on top?
Peter: Yeah, lightweight roguelite. The card system also means you can counter specific \
bosses if you know what you're going for, but if you're playing blind it's way \
harder. It came together better than anything I'd built in a jam before.
Will: So what happens to it now — does it just sit on your hard drive after the jam?
Peter: No, I actually want to ship it. I've been polishing it since — tightening the \
card balance, adding a couple more boss patterns, doing proper UI. Goal is to get \
it on Steam by summer.
Will: That's awesome. Do you have a name for it yet?
Peter: Still workshopping it. Something that hints at the card-combat angle without \
being too on the nose. I've got a few ideas but nothing locked in.
"""


async def main() -> None:
    client = Graphiti(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    print("Building graph indices...")
    await client.build_indices_and_constraints()

    print("Seeding conversation episode...")
    await client.add_episode(
        name="Will meets Peter Nguyen at tech conference",
        episode_body=CONVERSATION,
        source=EpisodeType.text,
        source_description=(
            "In-person conversation between Will and Peter Nguyen "
            "at a tech conference networking session, January 2025"
        ),
        reference_time=datetime(2025, 1, 15, 18, 30, tzinfo=timezone.utc),
    )

    print("Done — Peter Nguyen's episode is in the graph.")
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
