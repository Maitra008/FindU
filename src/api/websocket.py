"""
WebSocket Connection Manager.
Manages active WebSocket client connections, safe broadcasts, and dead-connection cleanups.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Set
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class WebSocketConnectionManager:
    """
    Central connection manager for alert broadcast streams.
    Thread-safe and resilient to abrupt client disconnects.
    """

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Set or update the active asyncio event loop for thread-safe cross-thread broadcasts."""
        self._loop = loop

    async def connect(self, websocket: WebSocket) -> None:
        """Accept a new WebSocket connection and add to active client pool."""
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info("WebSocket client connected. Active clients: %d", len(self.active_connections))

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove disconnected WebSocket from active pool."""
        self.active_connections.discard(websocket)
        logger.info("WebSocket client disconnected. Active clients: %d", len(self.active_connections))

    async def broadcast(self, message: Dict[str, Any]) -> None:
        """
        Broadcast JSON message to all active WebSocket clients.
        Automatically catches and cleans up broken connections.
        """
        if not self.active_connections:
            return

        payload = json.dumps(message)
        dead_connections: List[WebSocket] = []

        for connection in list(self.active_connections):
            try:
                await connection.send_text(payload)
            except (WebSocketDisconnect, RuntimeError, Exception) as e:
                logger.debug("Failed to send to WebSocket client (%s); scheduling disconnect.", e)
                dead_connections.append(connection)

        for dead_conn in dead_connections:
            self.active_connections.discard(dead_conn)

    def broadcast_sync(self, message: Dict[str, Any]) -> None:
        """
        Thread-safe synchronous bridge for background worker threads to broadcast messages.
        """
        if not self.active_connections:
            return

        try:
            loop = self._loop or asyncio.get_event_loop()
            if loop.is_running():
                asyncio.run_coroutine_threadsafe(self.broadcast(message), loop)
            else:
                loop.run_until_complete(self.broadcast(message))
        except Exception as e:
            logger.debug("Could not schedule sync WebSocket broadcast: %s", e)


# Global WebSocket Manager singleton instance
ws_manager = WebSocketConnectionManager()

