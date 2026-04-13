## Code style

### Naming & Readability
- Names are the documentation. Use descriptive, intention-revealing names so the code reads without comments.
  - BAD: `d = get_data(src)` 
  - GOOD: `user_records = fetch_active_users(database)`
- No comments except for genuinely non-obvious "why" explanations. Never comment "what" the code does. If you need a comment to explain what code does, rename things until you don't.
- No commented-out code. Ever. That's what git is for.

### Structure & Simplicity
- Functions do one thing. If you're tempted to name it `process_and_save` or `validate_and_transform`, split it.
- Keep functions short — if it doesn't fit on one screen (~30 lines), break it up.
- Flat is better than nested. If you have 3+ levels of indentation, refactor with early returns, guard clauses, or extraction.
- Early returns are great, prefer them over deep nesting
- Use Pythonic idioms: list comprehensions, f-strings, `pathlib`, unpacking, `enumerate`, `zip`. But still make it skimmable

### No Over-Engineering
- No defensive code for situations that won't happen. Don't check types at runtime, don't add try/except around things that shouldn't fail, don't validate inputs that you control.
- No abstractions until there's a clear second use case. No base classes with one subclass. No factory functions that return one thing.
- Prefer plain functions over classes. Only use a class when you need to manage state across multiple methods.
- That being said, if there are two modules that do similar things, use a base class for a consistent interface

### Visual Cleanliness
- Group related code into visual blocks with one blank line between them. Use two blank lines between top-level functions/classes.
- Imports at the top, organized: stdlib → third-party → local. Let ruff handle ordering.
- No deeply nested data transformations on one line. If a list comprehension has a condition and a nested loop, break it into a regular loop.

## Code Quality
- Run `ruff check --fix .` and `ruff format .` after modifying Python files
- All code must pass `ruff check` with zero violations
- Follow the ruff config in pyproject.toml

## HUD broadcast (Unity glasses app)

`pipeline/hud_broadcast.py` runs a WebSocket server in a daemon thread. When retrieval fires, `LivePipelineDriver.flush_window` publishes the formatted context to every connected client. The Unity app just connects and reads JSON messages — no polling, no framing to handle.

### Running

```bash
HUD_BROADCAST_ENABLED=true RETRIEVAL_ENABLED=true SAVE_TO_MEMORY=true \
  python pipeline/live.py test_videos/timur_myles_2.mp4
```

Server listens on `ws://0.0.0.0:8765` by default. Override with `HUD_BROADCAST_HOST` / `HUD_BROADCAST_PORT`.

### Connecting from Unity

```
ws://<laptop-ip>:8765
```

Use `NativeWebSocket` (or any WS client). Each `OnMessage` delivers one JSON object:

```json
{
  "type": "person_context",
  "name": "Myles Murphy",
  "person_id": 1,
  "context": {
    "last_spoke": "1 year ago",
    "last_spoke_about": "...",
    "ask_about": "...",
    "raw_facts": ["...", "..."]
  }
}
```

One message per retrieval event (bounded by `RETRIEVAL_COOLDOWN_SECONDS` per person). Clients can connect/disconnect at any time; disconnected clients are dropped silently on next broadcast.

### Testing without Unity

```bash
python -m websockets ws://localhost:8765
```

Or use a minimal Python client to print received payloads.