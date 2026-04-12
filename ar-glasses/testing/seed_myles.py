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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "ravengers")

from config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from neo4j import AsyncGraphDatabase

CONVERSATION_1 = """\
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
backend and the knowledge graph.
Myles Murphy: The Graphiti integration — is that working end-to-end now?
Timur: Yeah, we have it running. Conversations get saved as episodes, Graphiti extracts \
entities and relationships, and when you see someone again the retrieval pipeline \
queries the graph for relevant facts.
"""

CONVERSATION_2 = """\
Timur: Myles, I keep meaning to ask — what did you end up doing for your senior design project?
Myles Murphy: We built a mobile fitness app with computer vision. It uses the phone camera to \
track your form during exercises — squats, deadlifts, that kind of thing. Gives you \
real-time feedback if your knees are caving in or your back is rounding.
Timur: That's cool. What did you use for the pose estimation?
Myles Murphy: MediaPipe Pose. It's fast enough to run on-device and the landmark quality is \
decent for fitness use cases. We had to add our own angle-based heuristics on top \
for the actual form scoring.
Timur: Did you present at the expo?
Myles Murphy: Yeah, we got second place at the engineering expo. The judges really liked the \
real-time feedback aspect. We're thinking about publishing the form scoring algorithm \
as a short paper — our advisor Professor Kim said it could work for a workshop paper.
Timur: Nice. By the way, are you still playing volleyball on Tuesdays?
Myles Murphy: Yeah, every Tuesday at the rec center, 7pm. You should come out sometime. \
We need more people — usually only get like eight showing up.
Timur: I might. What about after graduation — are you staying in town or heading out?
Myles Murphy: I got a return offer from Capital One in McLean, Virginia. Software engineering \
on their fraud detection team. I interned there last summer and really liked the team. \
Starting in August.
Timur: Congrats. That's a solid gig. Fraud detection sounds interesting too — \
lots of ML involved?
Myles Murphy: Yeah, they use a mix of gradient boosted trees and some neural nets for \
real-time transaction scoring. My intern project was on feature engineering for \
the card-not-present fraud model. Basically extracting behavioral patterns from \
transaction sequences.
"""

CONVERSATION_3 = """\
Timur: Hey Myles, quick question — have you looked at the INMO Go glasses at all?
Myles Murphy: Yeah actually I was just reading about them. They're interesting — they have \
a built-in display and apparently decent audio, but no camera at all.
Timur: Right, that's the dealbreaker for us. We need the camera for face detection.
Myles Murphy: True. But their form factor is way better than the Meta Ray-Bans. They actually \
look like normal glasses. I think for a pure notification and audio use case they're \
probably the best option right now.
Timur: Speaking of hardware — did you ever get that Raspberry Pi cluster set up? \
You were talking about it a few weeks ago.
Myles Murphy: Oh yeah, I have four Pi 5s running in a mini rack now. Using it as a \
Kubernetes learning cluster. I set up K3s on it and I'm deploying small services \
to practice for the Capital One infrastructure. They run a lot of internal Kubernetes.
Timur: That's smart, getting a head start. What services are you running on it?
Myles Murphy: Right now a simple Flask API, a Redis cache, and Prometheus for monitoring. \
Nothing fancy but it's the operational side I wanted to learn — rolling deployments, \
health checks, resource limits. The kind of stuff you don't learn from tutorials.
"""


EPISODES = [
    {
        "name": "Timur and Myles discuss Argus companion app architecture",
        "body": CONVERSATION_1,
        "desc": (
            "In-person conversation between Timur (wearer) and Myles Murphy "
            "about the Argus mobile companion app — React Native, FastAPI, "
            "identity linking, offline caching, Graphiti integration"
        ),
        "time": datetime(2025, 3, 10, 14, 0, tzinfo=timezone.utc),
    },
    {
        "name": "Timur and Myles catch up — senior design, volleyball, Capital One",
        "body": CONVERSATION_2,
        "desc": (
            "In-person conversation between Timur (wearer) and Myles Murphy "
            "about Myles's senior design project (fitness CV app), volleyball, "
            "and his post-graduation job at Capital One"
        ),
        "time": datetime(2025, 3, 17, 15, 30, tzinfo=timezone.utc),
    },
    {
        "name": "Timur and Myles discuss INMO Go glasses and Raspberry Pi cluster",
        "body": CONVERSATION_3,
        "desc": (
            "In-person conversation between Timur (wearer) and Myles Murphy "
            "about INMO Go smart glasses, Raspberry Pi K3s cluster, "
            "and preparing for Capital One infrastructure work"
        ),
        "time": datetime(2025, 3, 24, 11, 0, tzinfo=timezone.utc),
    },
]


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

    for ep in EPISODES:
        print(f"Seeding: {ep['name']}...")
        await client.add_episode(
            name=ep["name"],
            episode_body=ep["body"],
            source=EpisodeType.text,
            source_description=ep["desc"],
            reference_time=ep["time"],
        )

    print("\nVerifying — searching for Myles Murphy...")
    results = await client.search("Myles Murphy projects work interests")
    print(f"  Got {len(results)} facts:")
    for r in results:
        print(f"    - {r.fact}")

    await client.close()
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
