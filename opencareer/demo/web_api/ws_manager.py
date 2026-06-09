"""
WebSocket connection manager.

Tracks active WebSocket connections per session and provides
send_personal() for sending messages to a specific connection.
"""

import logging
from typing import Any, Dict, Set

from fastapi import WebSocket

logger = logging.getLogger("web_api.ws_manager")


class WsManager:
    """Manages WebSocket connections grouped by session_id."""

    def __init__(self):
        # session_id -> set of WebSocket connections
        self._connections: Dict[str, Set[WebSocket]] = {}
        # session_id -> AgentPipeline instance
        self._handlers: Dict[str, Any] = {}

    async def connect(self, session_id: str, ws: WebSocket) -> None:
        """Accept a new WebSocket connection and register it."""
        await ws.accept()
        if session_id not in self._connections:
            self._connections[session_id] = set()
        self._connections[session_id].add(ws)
        logger.info(f"WebSocket connected: session={session_id}, total={len(self._connections[session_id])}")

    async def disconnect(self, session_id: str, ws: WebSocket) -> None:
        """Remove a disconnected WebSocket."""
        if session_id in self._connections:
            self._connections[session_id].discard(ws)
            if not self._connections[session_id]:
                del self._connections[session_id]
        logger.info(f"WebSocket disconnected: session={session_id}")

    async def send_personal(self, session_id: str, data: dict) -> None:
        """Send a JSON message to all connections for a session."""
        if session_id not in self._connections:
            return
        dead: list[WebSocket] = []
        for ws in self._connections[session_id]:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            await self.disconnect(session_id, ws)

    async def broadcast(self, data: dict) -> None:
        """Send a message to all connected sessions."""
        for session_id in list(self._connections.keys()):
            await self.send_personal(session_id, data)

    @property
    def active_sessions(self) -> list[str]:
        return list(self._connections.keys())

    def set_handler(self, session_id: str, handler) -> None:
        """Store streaming handler for a session."""
        self._handlers[session_id] = handler

    def get_handler(self, session_id: str):
        """Get streaming handler for a session."""
        return self._handlers.get(session_id)

    def remove_handler(self, session_id: str) -> None:
        self._handlers.pop(session_id, None)


# Singleton
ws_manager = WsManager()
