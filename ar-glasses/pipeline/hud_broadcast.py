"""WebSocket broadcaster for the Unity glasses HUD.

Runs a small asyncio websockets server in a daemon thread so the rest of
the pipeline can stay synchronous. Any client that connects receives
every JSON payload passed to ``publish`` until it disconnects.
"""

import asyncio
import json
import threading

import websockets
from websockets.asyncio.server import ServerConnection, serve


class HudBroadcastServer:
    def __init__(self, host: str, port: int):
        self._host = host
        self._port = port
        self._clients: set[ServerConnection] = set()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._ready = threading.Event()
        self._server: websockets.Server | None = None

    def start(self):
        self._thread.start()
        self._ready.wait(timeout=5)
        print(f"[hud] websocket server listening on ws://{self._host}:{self._port}")

    def stop(self):
        if self._server:
            self._loop.call_soon_threadsafe(self._server.close)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)

    def publish(self, payload: dict):
        message = json.dumps(payload)
        asyncio.run_coroutine_threadsafe(self._broadcast(message), self._loop)

    async def _broadcast(self, message: str):
        if not self._clients:
            return
        dead: list[ServerConnection] = []
        for client in self._clients:
            try:
                await client.send(message)
            except websockets.ConnectionClosed:
                dead.append(client)
        for client in dead:
            self._clients.discard(client)

    async def _handler(self, connection: ServerConnection):
        self._clients.add(connection)
        print(f"[hud] client connected ({len(self._clients)} total)")
        try:
            await connection.wait_closed()
        finally:
            self._clients.discard(connection)
            print(f"[hud] client disconnected ({len(self._clients)} total)")

    def _serve(self):
        asyncio.set_event_loop(self._loop)

        async def _start():
            self._server = await serve(self._handler, self._host, self._port)
            self._ready.set()

        self._loop.run_until_complete(_start())
        self._loop.run_forever()
