"""
WebSocket module for real-time features.
"""

from .auth import WebSocketAuth
from .handlers import WebSocketHandler
from .manager import WebSocketManager

__all__ = ["WebSocketManager", "WebSocketHandler", "WebSocketAuth"]
