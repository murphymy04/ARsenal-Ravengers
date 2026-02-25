"""
ARsenal Knowledge API — FastAPI application.

Endpoints:
  POST /conversations   — ingest a structured conversation into the knowledge graph
  GET  /retrieve/{name} — fetch graph context + LLM prompt for a recognised person

Usage:
  uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "ravengers")
WEARER_NAME = os.getenv("WEARER_NAME", "Will")

oai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

graphiti_client: Graphiti | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global graphiti_client
    graphiti_client = Graphiti(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    await graphiti_client.build_indices_and_constraints()
    yield
    await graphiti_client.close()


app = FastAPI(title="ARsenal Knowledge API", lifespan=lifespan)


# ── Pydantic models ────────────────────────────────────────────────────────────


class Message(BaseModel):
    speaker: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)


class ConversationRequest(BaseModel):
    date: datetime
    messages: list[Message] = Field(..., min_length=1)


class ConversationResponse(BaseModel):
    status: str
    episode_name: str
    message_count: int


class RetrieveResponse(BaseModel):
    person: str
    facts: list[str]
    prompt: str


# ── Helpers ────────────────────────────────────────────────────────────────────


def format_episode_body(messages: list[Message]) -> str:
    return "\n".join(f"{msg.speaker}: {msg.text}" for msg in messages)


# ── Routes ────────────────────────────────────────────────────────────────────


@app.post("/conversations", response_model=ConversationResponse, status_code=201)
async def store_conversation(body: ConversationRequest):
    ref_time = body.date
    if ref_time.tzinfo is None:
        ref_time = ref_time.replace(tzinfo=timezone.utc)

    speakers = ", ".join(dict.fromkeys(m.speaker for m in body.messages))
    episode_name = f"Conversation on {ref_time.strftime('%Y-%m-%d')} — {speakers}"
    episode_body = format_episode_body(body.messages)

    await graphiti_client.add_episode(
        name=episode_name,
        episode_body=episode_body,
        source=EpisodeType.text,
        source_description="smart glasses conversation",
        reference_time=ref_time,
    )

    return ConversationResponse(
        status="ok",
        episode_name=episode_name,
        message_count=len(body.messages),
    )


@app.get("/retrieve/{name}", response_model=RetrieveResponse)
async def retrieve_context(name: str):
    results = await graphiti_client.search(f"{name} projects work interests")
    facts = [r.fact for r in results]

    if not facts:
        return RetrieveResponse(
            person=name,
            facts=[],
            prompt=f"No context found for {name}.",
        )

    facts_block = "\n".join(f"- {f}" for f in facts)
    response = await oai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    f"You are an assistant helping {WEARER_NAME} have a great conversation. "
                    f"{WEARER_NAME} is wearing smart glasses that just recognised {name}'s face. "
                    "Based on what they last talked about, write a single, natural, "
                    "friendly conversation-starter sentence for the wearer to use. "
                    "Be specific — reference something concrete from their last conversation. "
                    "No preamble, no labels, just the sentence."
                ),
            },
            {
                "role": "user",
                "content": f"Facts about {name} from last conversation:\n{facts_block}",
            },
        ],
        max_tokens=120,
    )

    prompt = response.choices[0].message.content.strip()
    return RetrieveResponse(person=name, facts=facts, prompt=prompt)
