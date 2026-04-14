import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

_AR_ROOT = Path(__file__).resolve().parent.parent
if str(_AR_ROOT) not in sys.path:
    sys.path.insert(0, str(_AR_ROOT))

load_dotenv(_AR_ROOT / ".env")

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq()
    return _client

_SYSTEM_PROMPT = (
    "You are analyzing a transcript snippet from a live conversation. "
    "Determine if the conversation has ENDED in this snippet. "
    "Signs of ending: goodbyes, farewells, 'nice meeting you', 'see you later', "
    "'take care', wrapping-up language, or clear disengagement. "
    "If the conversation is still ongoing or mid-topic, it has NOT ended."
)

_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "conversation_end",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"ended": {"type": "boolean"}},
            "required": ["ended"],
            "additionalProperties": False,
        },
    },
}


def is_conversation_end(segments: list[dict]) -> bool:
    if not segments:
        return False

    transcript = "\n".join(f"{s['speaker']}: {s['text']}" for s in segments)

    response = _get_client().chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
        ],
        response_format=_RESPONSE_FORMAT,
        temperature=0,
    )

    print(
        "is_conversation_end responded with ",
        json.loads(response.choices[0].message.content)["ended"],
    )

    return json.loads(response.choices[0].message.content)["ended"]


if __name__ == "__main__":
    cases = [
        (
            "mid-conversation",
            False,
            [
                {
                    "speaker": "Person 1",
                    "text": "So what are you working on these days?",
                },
                {
                    "speaker": "Person 2",
                    "text": "I am building a smart glasses app that "
                    "acts as a personal secretary.",
                },
                {
                    "speaker": "Person 1",
                    "text": "Oh thats really cool, what kind of glasses are you using?",
                },
            ],
        ),
        (
            "ending",
            True,
            [
                {
                    "speaker": "Person 1",
                    "text": "Well it was great meeting you, "
                    "good luck with the project!",
                },
                {"speaker": "Person 2", "text": "Thanks, you too! See you around."},
                {"speaker": "Person 1", "text": "Bye!"},
            ],
        ),
        (
            "short fragment",
            False,
            [
                {"speaker": "wearer", "text": "Oh."},
            ],
        ),
        ("empty", False, []),
    ]

    for name, expected, segments in cases:
        result = is_conversation_end(segments)
        status = "PASS" if result == expected else "FAIL"
        print(f"[{status}] {name}: got={result} expected={expected}")
