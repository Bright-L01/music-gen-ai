"""
WebSocket routes for the API.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from .auth import WebSocketAuth
from .handlers import WebSocketHandler
from .manager import WebSocketManager

logger = logging.getLogger(__name__)

router = APIRouter()

# Global WebSocket manager instance
ws_manager: Optional[WebSocketManager] = None
ws_handler: Optional[WebSocketHandler] = None
ws_auth = WebSocketAuth()


async def get_ws_manager() -> WebSocketManager:
    """Get WebSocket manager instance."""
    global ws_manager
    if ws_manager is None:
        raise RuntimeError("WebSocket manager not initialized")
    return ws_manager


@router.on_event("startup")
async def startup_websocket():
    """Initialize WebSocket manager on startup."""
    global ws_manager, ws_handler

    # Initialize manager
    ws_manager = WebSocketManager()
    await ws_manager.startup()

    # Initialize handler
    ws_handler = WebSocketHandler(ws_manager)

    logger.info("WebSocket service started")


@router.on_event("shutdown")
async def shutdown_websocket():
    """Shutdown WebSocket manager."""
    global ws_manager

    if ws_manager:
        await ws_manager.shutdown()
        ws_manager = None

    logger.info("WebSocket service stopped")


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    token: Optional[str] = Query(None),
):
    """
    Main WebSocket endpoint.

    Supports authentication via:
    - Query parameter: ws://localhost:8000/ws?token=<token>
    - Authorization header
    - Cookie
    """
    # Authenticate connection
    user_data = await ws_auth.authenticate_websocket(websocket, token)

    # Connect to manager
    connection = await ws_manager.connect(
        websocket,
        user_id=user_data.get("user_id") if user_data else None,
    )

    # Set authentication status
    if user_data:
        await ws_manager.authenticate(
            connection.connection_id,
            user_data["user_id"],
            metadata=user_data,
        )

    try:
        # Send initial connection message
        await websocket.send_json(
            {
                "type": "connected",
                "data": {
                    "connection_id": connection.connection_id,
                    "authenticated": connection.authenticated,
                    "user_id": connection.user_id,
                },
            }
        )

        # Handle messages
        while True:
            # Receive message
            data = await websocket.receive_json()

            # Process message
            response = await ws_handler.handle_message(connection, data)

            # Send response if any
            if response:
                await websocket.send_json(response)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {connection.connection_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Cleanup connection
        await ws_manager.disconnect(connection.connection_id)


@router.get("/ws/stats")
async def websocket_stats(
    manager: WebSocketManager = Depends(get_ws_manager),
):
    """Get WebSocket service statistics."""
    return JSONResponse(content=manager.get_stats())


@router.get("/ws/rooms/{room_id}/users")
async def get_room_users(
    room_id: str,
    manager: WebSocketManager = Depends(get_ws_manager),
):
    """Get users in a specific room."""
    users = manager.get_room_users(room_id)
    return JSONResponse(
        content={
            "room_id": room_id,
            "users": users,
            "count": len(users),
        }
    )


@router.post("/ws/broadcast")
async def broadcast_message(
    message: dict,
    manager: WebSocketManager = Depends(get_ws_manager),
):
    """Broadcast a message to all connected clients."""
    await manager.broadcast(message)
    return JSONResponse(content={"status": "broadcasted"})


@router.post("/ws/rooms/{room_id}/send")
async def send_to_room(
    room_id: str,
    message: dict,
    manager: WebSocketManager = Depends(get_ws_manager),
):
    """Send a message to all clients in a room."""
    await manager.send_to_room(room_id, message)
    return JSONResponse(
        content={
            "status": "sent",
            "room_id": room_id,
        }
    )


@router.post("/ws/users/{user_id}/send")
async def send_to_user(
    user_id: str,
    message: dict,
    manager: WebSocketManager = Depends(get_ws_manager),
):
    """Send a message to all connections of a user."""
    await manager.send_to_user(user_id, message)
    return JSONResponse(
        content={
            "status": "sent",
            "user_id": user_id,
        }
    )
