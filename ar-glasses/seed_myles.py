"""Seed script — clears Neo4j and adds Timur/Myles conversation about Argus.

Usage:
    docker compose -f ../knowledge/docker-compose.yml up -d
    python seed_myles.py
"""

import asyncio
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "ravengers")

from config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from neo4j import AsyncGraphDatabase

CONVERSATION = """\
Timur: Hey Myles, so I've been thinking about the companion app for Argus — \
the mobile piece that pairs with the glasses.
Myles Murphy: Yeah, what's the current thinking on the stack? Are we going native or cross-platform?
Timur: I'm leaning React Native. We need iOS and Android, and the app is mostly \
UI for reviewing sessions and linking identities — nothing that needs raw native perf.
Myles Murphy: Makes sense. What about the backend? Are we running everything through the \
glasses or does the phone need its own API layer?
Timur: The phone talks to a FastAPI backend. The glasses stream raw data — video frames, \
audio chunks — and the backend handles the heavy processing. Face detection, embeddings, \
transcription, all server-side. The phone app is mainly for post-session review.
Myles Murphy: So the identity linking flow — that's the part where the wearer says "this is \
Person X" after the glasses auto-cluster a face?
Timur: Exactly. The glasses detect and cluster faces automatically, assign temporary \
IDs like "Person 1", "Person 2". Then in the companion app you open the session, \
see the face thumbnails, and tap to assign a real name. That label propagates back \
to the embeddings database.
Myles Murphy: What about offline? If the glasses lose connection mid-conversation, do we \
buffer locally and sync later?
Timur: That's the plan. Local SQLite on the glasses caches everything — face embeddings, \
audio segments, diarization results. When connectivity comes back, it syncs to the \
backend and the knowledge graph. The tricky part is conflict resolution if the same \
person gets labeled differently across sessions.
Myles Murphy: Have you thought about the session management piece? Like how do you define \
when a conversation starts and ends?
Timur: Right now it's silence-based. If there's no speech detected for 30 seconds, \
we consider the conversation ended. That triggers the save-to-memory flow where \
the transcript gets pushed to Graphiti as an episode. But I want to add manual \
session boundaries too — a gesture or voice command to explicitly mark start and end.
Myles Murphy: The Graphiti integration — is that working end-to-end now?
Timur: Yeah, we have it running. Conversations get saved as episodes, Graphiti extracts \
entities and relationships, and when you see someone again the retrieval pipeline \
queries the graph for relevant facts. The query is just the person's name plus some \
context keywords. Works surprisingly well for surfacing things like "last time you \
talked about X" or "this person works on Y".
Myles Murphy: That's sick. What about the latency on retrieval? If I'm walking up to someone \
and the glasses recognize them, how fast do I get the context overlay?
Timur: Under two seconds from face match to facts on the HUD. The bottleneck is the \
Graphiti search, not the face recognition. We have a cooldown so it doesn't re-query \
for the same person every frame — currently 30 seconds between retrievals per person.
"""


async def main():
    client = Graphiti(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    print("Building graph indices...")
    await client.build_indices_and_constraints()

    print("Clearing existing graph data...")
    driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    async with driver.session() as session:
        await session.run("MATCH (n) DETACH DELETE n")
    await driver.close()
    await client.build_indices_and_constraints()

    print("Seeding Timur/Myles conversation...")
    await client.add_episode(
        name="Timur and Myles discuss Argus companion app architecture",
        episode_body=CONVERSATION,
        source=EpisodeType.text,
        source_description=(
            "In-person conversation between Timur (wearer) and Myles "
            "about the Argus mobile companion app — React Native, FastAPI, "
            "identity linking, offline caching, session management"
        ),
        reference_time=datetime(2025, 3, 20, 14, 0, tzinfo=timezone.utc),
    )

    print("Verifying — searching for 'Person 1'...")
    results = await client.search("Person 1 Myles Argus mobile app")
    print(f"  Got {len(results)} facts:")
    for r in results:
        print(f"    - {r.fact}")

    await client.close()
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
