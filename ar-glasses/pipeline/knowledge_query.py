"""Read-only Zep/Graphiti queries for the dashboard Knowledge tab.

fetch_graph() returns a node/link payload for 3d-force-graph.
ask() wraps Graphiti hybrid search + an LLM summariser with citations.
"""

import asyncio
import threading
from datetime import datetime

from graphiti_core import Graphiti
from groq import Groq
from neo4j import GraphDatabase

from config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER

SUMMARIZER_MODEL = "qwen/qwen3-32b"

SUMMARIZER_SYSTEM = """\
Answer the user's question using ONLY the numbered facts below.
Cite facts inline as [1], [2], etc. when you use them.
Keep the reply to one short paragraph, max three sentences.
If the facts don't answer the question, say so directly.
"""


_groq = Groq()

_lock = threading.Lock()
_loop: asyncio.AbstractEventLoop | None = None
_client: Graphiti | None = None


def _ensure_client() -> Graphiti:
    global _loop, _client
    with _lock:
        if _client is not None:
            return _client

        _loop = asyncio.new_event_loop()
        threading.Thread(target=_loop.run_forever, daemon=True).start()

        async def _init() -> Graphiti:
            client = Graphiti(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
            await client.build_indices_and_constraints()
            return client

        _client = asyncio.run_coroutine_threadsafe(_init(), _loop).result()
        return _client


def _run(coro):
    return asyncio.run_coroutine_threadsafe(coro, _loop).result()


def _iso(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def fetch_graph(node_limit: int = 500) -> dict:
    with (
        GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver,
        driver.session() as session,
    ):
        node_rows = session.run(
            """
                MATCH (n:Entity)
                RETURN n.uuid AS id,
                       n.name AS name,
                       n.summary AS summary,
                       labels(n) AS labels,
                       n.created_at AS created_at
                LIMIT $limit
                """,
            limit=node_limit,
        ).data()

        edge_rows = session.run(
            """
                MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
                RETURN r.uuid AS id,
                       a.uuid AS source,
                       b.uuid AS target,
                       r.name AS name,
                       r.fact AS fact,
                       r.valid_at AS valid_at,
                       r.invalid_at AS invalid_at,
                       r.created_at AS created_at
                """
        ).data()

    nodes = [
        {
            "id": row["id"],
            "name": row["name"] or "(unnamed)",
            "summary": row["summary"] or "",
            "labels": [label for label in row["labels"] if label != "Entity"],
            "created_at": _iso(row["created_at"]),
        }
        for row in node_rows
        if row["id"]
    ]

    links = [
        {
            "id": row["id"],
            "source": row["source"],
            "target": row["target"],
            "name": row["name"] or "",
            "fact": row["fact"] or "",
            "valid_at": _iso(row["valid_at"]),
            "invalid_at": _iso(row["invalid_at"]),
            "created_at": _iso(row["created_at"]),
        }
        for row in edge_rows
        if row["source"] and row["target"]
    ]

    return {"nodes": nodes, "links": links}


def ask(question: str, num_results: int = 10) -> dict:
    client = _ensure_client()
    edges = _run(client.search(question, num_results=num_results))

    facts = [
        {
            "id": edge.uuid,
            "fact": edge.fact,
            "source": edge.source_node_uuid,
            "target": edge.target_node_uuid,
            "valid_at": _iso(edge.valid_at),
            "invalid_at": _iso(edge.invalid_at),
        }
        for edge in edges
    ]

    answer = _summarize(question, facts)
    return {"answer": answer, "facts": facts}


def _summarize(question: str, facts: list[dict]) -> str:
    if not facts:
        return "Nothing in the knowledge graph matches that question yet."

    numbered = "\n".join(f"[{i}] {f['fact']}" for i, f in enumerate(facts, 1))
    user = f"Question: {question}\n\nFacts:\n{numbered}"

    response = _groq.chat.completions.create(
        model=SUMMARIZER_MODEL,
        messages=[
            {"role": "system", "content": SUMMARIZER_SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()
