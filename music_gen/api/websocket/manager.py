"""
WebSocket connection manager with Redis pub/sub for scaling.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

import redis.asyncio as redis
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class Connection(BaseModel):
    """WebSocket connection information."""

    connection_id: str
    websocket: WebSocket
    user_id: Optional[str] = None
    authenticated: bool = False
    rooms: Set[str] = set()
    metadata: Dict[str, Any] = {}
    connected_at: datetime = datetime.utcnow()
    last_ping: datetime = datetime.utcnow()

    class Config:
        arbitrary_types_allowed = True


class WebSocketManager:
    """
    Manages WebSocket connections with Redis pub/sub for horizontal scaling.

    Features:
    - Connection pooling and management
    - Room-based messaging
    - Redis pub/sub for multi-server support
    - Automatic reconnection handling
    - Health monitoring with ping/pong
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        ping_interval: int = 30,
        ping_timeout: int = 10,
    ):
        self.connections: Dict[str, Connection] = {}
        self.rooms: Dict[str, Set[str]] = {}
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.pubsub: Optional[redis.client.PubSub] = None
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        self._tasks: List[asyncio.Task] = []
        self._running = False

    async def startup(self):
        """Initialize manager and start background tasks."""
        self.redis_client = await redis.from_url(self.redis_url, decode_responses=True)
        self.pubsub = self.redis_client.pubsub()

        # Subscribe to WebSocket channels
        await self.pubsub.subscribe(
            "websocket:broadcast",
            "websocket:room:*",
            "websocket:user:*",
        )

        # Start background tasks
        self._running = True
        self._tasks.extend(
            [
                asyncio.create_task(self._redis_listener()),
                asyncio.create_task(self._ping_monitor()),
                asyncio.create_task(self._cleanup_monitor()),
            ]
        )

        logger.info("WebSocket manager started")

    async def shutdown(self):
        """Shutdown manager and cleanup resources."""
        self._running = False

        # Cancel background tasks
        for task in self._tasks:
            task.cancel()

        await asyncio.gather(*self._tasks, return_exceptions=True)

        # Close all connections
        for connection in list(self.connections.values()):
            await self.disconnect(connection.connection_id)

        # Close Redis connections
        if self.pubsub:
            await self.pubsub.unsubscribe()
            await self.pubsub.close()

        if self.redis_client:
            await self.redis_client.close()

        logger.info("WebSocket manager shutdown")

    async def connect(
        self,
        websocket: WebSocket,
        user_id: Optional[str] = None,
    ) -> Connection:
        """Accept and register a new WebSocket connection."""
        await websocket.accept()

        connection_id = str(uuid4())
        connection = Connection(
            connection_id=connection_id,
            websocket=websocket,
            user_id=user_id,
            authenticated=user_id is not None,
        )

        self.connections[connection_id] = connection

        # Notify connection event
        await self._publish_event(
            "connection",
            {
                "connection_id": connection_id,
                "user_id": user_id,
                "event": "connected",
            },
        )

        logger.info(f"WebSocket connected: {connection_id} (user: {user_id})")
        return connection

    async def disconnect(self, connection_id: str):
        """Disconnect and cleanup a WebSocket connection."""
        connection = self.connections.get(connection_id)
        if not connection:
            return

        # Leave all rooms
        for room in list(connection.rooms):
            await self.leave_room(connection_id, room)

        # Close WebSocket
        try:
            await connection.websocket.close()
        except Exception:
            pass

        # Remove connection
        del self.connections[connection_id]

        # Notify disconnection event
        await self._publish_event(
            "connection",
            {
                "connection_id": connection_id,
                "user_id": connection.user_id,
                "event": "disconnected",
            },
        )

        logger.info(f"WebSocket disconnected: {connection_id}")

    async def authenticate(
        self,
        connection_id: str,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Authenticate a WebSocket connection."""
        connection = self.connections.get(connection_id)
        if not connection:
            return False

        connection.user_id = user_id
        connection.authenticated = True

        if metadata:
            connection.metadata.update(metadata)

        # Subscribe to user-specific channel
        if self.pubsub:
            await self.pubsub.subscribe(f"websocket:user:{user_id}")

        logger.info(f"WebSocket authenticated: {connection_id} as {user_id}")
        return True

    async def join_room(self, connection_id: str, room: str):
        """Join a connection to a room."""
        connection = self.connections.get(connection_id)
        if not connection:
            return

        # Add to local room tracking
        if room not in self.rooms:
            self.rooms[room] = set()

        self.rooms[room].add(connection_id)
        connection.rooms.add(room)

        # Subscribe to room channel
        if self.pubsub:
            await self.pubsub.subscribe(f"websocket:room:{room}")

        # Notify room join
        await self._publish_event(
            "room",
            {
                "room": room,
                "connection_id": connection_id,
                "user_id": connection.user_id,
                "event": "joined",
            },
        )

        logger.info(f"Connection {connection_id} joined room: {room}")

    async def leave_room(self, connection_id: str, room: str):
        """Remove a connection from a room."""
        connection = self.connections.get(connection_id)
        if not connection or room not in connection.rooms:
            return

        # Remove from local room tracking
        connection.rooms.discard(room)

        if room in self.rooms:
            self.rooms[room].discard(connection_id)

            # Unsubscribe if no more connections in room
            if not self.rooms[room]:
                del self.rooms[room]
                if self.pubsub:
                    await self.pubsub.unsubscribe(f"websocket:room:{room}")

        # Notify room leave
        await self._publish_event(
            "room",
            {
                "room": room,
                "connection_id": connection_id,
                "user_id": connection.user_id,
                "event": "left",
            },
        )

        logger.info(f"Connection {connection_id} left room: {room}")

    async def send_to_connection(
        self,
        connection_id: str,
        message: Dict[str, Any],
    ):
        """Send a message to a specific connection."""
        connection = self.connections.get(connection_id)
        if not connection:
            return

        try:
            await connection.websocket.send_json(message)
        except WebSocketDisconnect:
            await self.disconnect(connection_id)
        except Exception as e:
            logger.error(f"Error sending to connection {connection_id}: {e}")
            await self.disconnect(connection_id)

    async def send_to_user(
        self,
        user_id: str,
        message: Dict[str, Any],
        exclude_connection: Optional[str] = None,
    ):
        """Send a message to all connections of a user."""
        # Use Redis pub/sub for multi-server support
        await self._publish_to_channel(
            f"websocket:user:{user_id}",
            {
                "message": message,
                "exclude_connection": exclude_connection,
            },
        )

    async def send_to_room(
        self,
        room: str,
        message: Dict[str, Any],
        exclude_connection: Optional[str] = None,
    ):
        """Send a message to all connections in a room."""
        # Use Redis pub/sub for multi-server support
        await self._publish_to_channel(
            f"websocket:room:{room}",
            {
                "message": message,
                "exclude_connection": exclude_connection,
            },
        )

    async def broadcast(
        self,
        message: Dict[str, Any],
        exclude_connection: Optional[str] = None,
    ):
        """Broadcast a message to all connections."""
        # Use Redis pub/sub for multi-server support
        await self._publish_to_channel(
            "websocket:broadcast",
            {
                "message": message,
                "exclude_connection": exclude_connection,
            },
        )

    async def send_generation_update(
        self,
        user_id: str,
        task_id: str,
        status: str,
        progress: float,
        stage: str,
        data: Optional[Dict[str, Any]] = None,
    ):
        """Send generation progress update to user."""
        message = {
            "type": "generation_update",
            "data": {
                "id": task_id,
                "status": status,
                "progress": progress,
                "stage": stage,
                **(data or {}),
            },
            "timestamp": datetime.utcnow().isoformat(),
        }

        await self.send_to_user(user_id, message)

    async def send_collaboration_update(
        self,
        room: str,
        user_id: str,
        action: str,
        data: Dict[str, Any],
    ):
        """Send collaboration update to a room."""
        message = {
            "type": "collaboration_update",
            "data": {
                "user_id": user_id,
                "action": action,
                **data,
            },
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Get user info for the connection
        user_connection = next((c for c in self.connections.values() if c.user_id == user_id), None)

        await self.send_to_room(
            room,
            message,
            exclude_connection=user_connection.connection_id if user_connection else None,
        )

    def get_room_users(self, room: str) -> List[Dict[str, Any]]:
        """Get all users in a room."""
        users = []

        if room in self.rooms:
            for connection_id in self.rooms[room]:
                connection = self.connections.get(connection_id)
                if connection and connection.user_id:
                    users.append(
                        {
                            "user_id": connection.user_id,
                            "connection_id": connection_id,
                            "metadata": connection.metadata,
                            "connected_at": connection.connected_at.isoformat(),
                        }
                    )

        return users

    async def ping_connection(self, connection_id: str) -> bool:
        """Send ping to a connection and wait for pong."""
        connection = self.connections.get(connection_id)
        if not connection:
            return False

        try:
            pong_waiter = await connection.websocket.ping()
            await asyncio.wait_for(pong_waiter, timeout=self.ping_timeout)
            connection.last_ping = datetime.utcnow()
            return True
        except (WebSocketDisconnect, asyncio.TimeoutError):
            return False
        except Exception as e:
            logger.error(f"Ping error for {connection_id}: {e}")
            return False

    async def _redis_listener(self):
        """Listen for Redis pub/sub messages."""
        if not self.pubsub:
            return

        while self._running:
            try:
                message = await self.pubsub.get_message(timeout=1.0)
                if message and message["type"] == "message":
                    await self._handle_redis_message(
                        message["channel"],
                        message["data"],
                    )
            except Exception as e:
                logger.error(f"Redis listener error: {e}")
                await asyncio.sleep(1)

    async def _handle_redis_message(self, channel: str, data: str):
        """Handle incoming Redis pub/sub message."""
        try:
            payload = json.loads(data)
            message = payload.get("message", {})
            exclude = payload.get("exclude_connection")

            if channel == "websocket:broadcast":
                # Broadcast to all local connections
                for connection in self.connections.values():
                    if connection.connection_id != exclude:
                        await self.send_to_connection(
                            connection.connection_id,
                            message,
                        )

            elif channel.startswith("websocket:room:"):
                # Send to room members
                room = channel.replace("websocket:room:", "")
                if room in self.rooms:
                    for connection_id in self.rooms[room]:
                        if connection_id != exclude:
                            await self.send_to_connection(connection_id, message)

            elif channel.startswith("websocket:user:"):
                # Send to user connections
                user_id = channel.replace("websocket:user:", "")
                for connection in self.connections.values():
                    if connection.user_id == user_id and connection.connection_id != exclude:
                        await self.send_to_connection(
                            connection.connection_id,
                            message,
                        )

        except Exception as e:
            logger.error(f"Error handling Redis message: {e}")

    async def _ping_monitor(self):
        """Monitor connection health with periodic pings."""
        while self._running:
            try:
                # Ping all connections
                tasks = []
                for connection_id in list(self.connections.keys()):
                    tasks.append(self._check_connection_health(connection_id))

                if tasks:
                    await asyncio.gather(*tasks)

                await asyncio.sleep(self.ping_interval)

            except Exception as e:
                logger.error(f"Ping monitor error: {e}")
                await asyncio.sleep(1)

    async def _check_connection_health(self, connection_id: str):
        """Check health of a single connection."""
        if not await self.ping_connection(connection_id):
            logger.warning(f"Connection {connection_id} failed ping check")
            await self.disconnect(connection_id)

    async def _cleanup_monitor(self):
        """Clean up stale connections and rooms."""
        while self._running:
            try:
                # Clean up empty rooms
                empty_rooms = [room for room, connections in self.rooms.items() if not connections]

                for room in empty_rooms:
                    del self.rooms[room]
                    if self.pubsub:
                        await self.pubsub.unsubscribe(f"websocket:room:{room}")

                await asyncio.sleep(60)  # Run every minute

            except Exception as e:
                logger.error(f"Cleanup monitor error: {e}")
                await asyncio.sleep(60)

    async def _publish_to_channel(self, channel: str, data: Dict[str, Any]):
        """Publish message to Redis channel."""
        if self.redis_client:
            try:
                await self.redis_client.publish(
                    channel,
                    json.dumps(data),
                )
            except Exception as e:
                logger.error(f"Redis publish error: {e}")

    async def _publish_event(self, event_type: str, data: Dict[str, Any]):
        """Publish internal event for monitoring."""
        await self._publish_to_channel(
            f"websocket:events:{event_type}",
            {
                "server_id": id(self),
                "timestamp": datetime.utcnow().isoformat(),
                **data,
            },
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get current WebSocket statistics."""
        return {
            "total_connections": len(self.connections),
            "authenticated_connections": sum(
                1 for c in self.connections.values() if c.authenticated
            ),
            "total_rooms": len(self.rooms),
            "connections_by_room": {
                room: len(connections) for room, connections in self.rooms.items()
            },
            "users_online": len(set(c.user_id for c in self.connections.values() if c.user_id)),
        }
