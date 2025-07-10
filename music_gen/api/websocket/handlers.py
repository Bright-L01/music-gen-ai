"""
WebSocket message handlers for different event types.
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from .manager import Connection, WebSocketManager

logger = logging.getLogger(__name__)


class WebSocketHandler:
    """Handle different types of WebSocket messages."""

    def __init__(self, manager: WebSocketManager):
        self.manager = manager

        # Message type handlers
        self.handlers = {
            "ping": self.handle_ping,
            "join_room": self.handle_join_room,
            "leave_room": self.handle_leave_room,
            "generation_start": self.handle_generation_start,
            "generation_cancel": self.handle_generation_cancel,
            "collaboration_action": self.handle_collaboration_action,
            "chat_message": self.handle_chat_message,
            "user_presence": self.handle_user_presence,
            "subscribe": self.handle_subscribe,
            "unsubscribe": self.handle_unsubscribe,
        }

    async def handle_message(
        self,
        connection: Connection,
        message: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Route message to appropriate handler."""
        msg_type = message.get("type")

        if not msg_type:
            return {
                "type": "error",
                "error": "Message type required",
            }

        handler = self.handlers.get(msg_type)

        if not handler:
            return {
                "type": "error",
                "error": f"Unknown message type: {msg_type}",
            }

        try:
            return await handler(connection, message.get("data", {}))
        except Exception as e:
            logger.error(f"Handler error for {msg_type}: {e}")
            return {
                "type": "error",
                "error": f"Handler error: {str(e)}",
            }

    async def handle_ping(
        self,
        connection: Connection,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Handle ping message."""
        return {
            "type": "pong",
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def handle_join_room(
        self,
        connection: Connection,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Handle room join request."""
        room_id = data.get("roomId") or data.get("room_id")

        if not room_id:
            return {
                "type": "error",
                "error": "Room ID required",
            }

        # Join room
        await self.manager.join_room(connection.connection_id, room_id)

        # Get current room users
        users = self.manager.get_room_users(room_id)

        # Notify others in room
        await self.manager.send_collaboration_update(
            room_id,
            connection.user_id or "anonymous",
            "user_joined",
            {
                "user_id": connection.user_id,
                "metadata": connection.metadata,
            },
        )

        return {
            "type": "room_joined",
            "data": {
                "room_id": room_id,
                "users": users,
            },
        }

    async def handle_leave_room(
        self,
        connection: Connection,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Handle room leave request."""
        room_id = data.get("roomId") or data.get("room_id")

        if not room_id:
            return {
                "type": "error",
                "error": "Room ID required",
            }

        # Leave room
        await self.manager.leave_room(connection.connection_id, room_id)

        # Notify others in room
        await self.manager.send_collaboration_update(
            room_id,
            connection.user_id or "anonymous",
            "user_left",
            {
                "user_id": connection.user_id,
            },
        )

        return {
            "type": "room_left",
            "data": {
                "room_id": room_id,
            },
        }

    async def handle_generation_start(
        self,
        connection: Connection,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Handle generation start notification."""
        if not connection.authenticated:
            return {
                "type": "error",
                "error": "Authentication required",
            }

        task_id = data.get("task_id")
        prompt = data.get("prompt")

        if not task_id or not prompt:
            return {
                "type": "error",
                "error": "Task ID and prompt required",
            }

        # Subscribe to generation updates
        await self.manager.join_room(
            connection.connection_id,
            f"generation:{task_id}",
        )

        # TODO: Start actual generation task

        return {
            "type": "generation_started",
            "data": {
                "task_id": task_id,
                "status": "pending",
            },
        }

    async def handle_generation_cancel(
        self,
        connection: Connection,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Handle generation cancellation."""
        if not connection.authenticated:
            return {
                "type": "error",
                "error": "Authentication required",
            }

        task_id = data.get("task_id")

        if not task_id:
            return {
                "type": "error",
                "error": "Task ID required",
            }

        # TODO: Cancel actual generation task

        # Leave generation room
        await self.manager.leave_room(
            connection.connection_id,
            f"generation:{task_id}",
        )

        return {
            "type": "generation_cancelled",
            "data": {
                "task_id": task_id,
            },
        }

    async def handle_collaboration_action(
        self,
        connection: Connection,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Handle collaboration action."""
        if not connection.authenticated:
            return {
                "type": "error",
                "error": "Authentication required",
            }

        room_id = data.get("room_id")
        action = data.get("action")
        action_data = data.get("data", {})

        if not room_id or not action:
            return {
                "type": "error",
                "error": "Room ID and action required",
            }

        # Broadcast to room
        await self.manager.send_collaboration_update(
            room_id,
            connection.user_id,
            action,
            action_data,
        )

        return {
            "type": "collaboration_action_sent",
            "data": {
                "room_id": room_id,
                "action": action,
            },
        }

    async def handle_chat_message(
        self,
        connection: Connection,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Handle chat message."""
        if not connection.authenticated:
            return {
                "type": "error",
                "error": "Authentication required",
            }

        room_id = data.get("room_id")
        message = data.get("message")

        if not room_id or not message:
            return {
                "type": "error",
                "error": "Room ID and message required",
            }

        # Broadcast chat message
        chat_data = {
            "id": str(datetime.utcnow().timestamp()),
            "user_id": connection.user_id,
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": connection.metadata,
        }

        await self.manager.send_to_room(
            room_id,
            {
                "type": "chat_message",
                "data": chat_data,
            },
        )

        return {
            "type": "chat_message_sent",
            "data": {
                "room_id": room_id,
                "message_id": chat_data["id"],
            },
        }

    async def handle_user_presence(
        self,
        connection: Connection,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Handle user presence update."""
        status = data.get("status", "online")

        # Update connection metadata
        connection.metadata["presence"] = {
            "status": status,
            "updated_at": datetime.utcnow().isoformat(),
        }

        # Broadcast to user's rooms
        for room in connection.rooms:
            await self.manager.send_to_room(
                room,
                {
                    "type": "user_presence",
                    "data": {
                        "user_id": connection.user_id,
                        "status": status,
                        "metadata": connection.metadata,
                    },
                },
                exclude_connection=connection.connection_id,
            )

        return {
            "type": "presence_updated",
            "data": {
                "status": status,
            },
        }

    async def handle_subscribe(
        self,
        connection: Connection,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Handle subscription request."""
        channels = data.get("channels", [])

        if not channels:
            return {
                "type": "error",
                "error": "Channels required",
            }

        subscribed = []

        for channel in channels:
            # Validate channel access
            if channel.startswith("user:") and connection.user_id:
                # Can only subscribe to own user channel
                if channel == f"user:{connection.user_id}":
                    await self.manager.join_room(connection.connection_id, channel)
                    subscribed.append(channel)
            elif channel.startswith("public:"):
                # Anyone can subscribe to public channels
                await self.manager.join_room(connection.connection_id, channel)
                subscribed.append(channel)
            elif connection.authenticated:
                # Authenticated users can subscribe to other channels
                await self.manager.join_room(connection.connection_id, channel)
                subscribed.append(channel)

        return {
            "type": "subscribed",
            "data": {
                "channels": subscribed,
            },
        }

    async def handle_unsubscribe(
        self,
        connection: Connection,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Handle unsubscription request."""
        channels = data.get("channels", [])

        if not channels:
            return {
                "type": "error",
                "error": "Channels required",
            }

        unsubscribed = []

        for channel in channels:
            if channel in connection.rooms:
                await self.manager.leave_room(connection.connection_id, channel)
                unsubscribed.append(channel)

        return {
            "type": "unsubscribed",
            "data": {
                "channels": unsubscribed,
            },
        }
