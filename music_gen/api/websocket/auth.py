from datetime import timedelta

"""
WebSocket authentication and authorization.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import jwt
from fastapi import Query, WebSocket

from ...core.config import get_settings

logger = logging.getLogger(__name__)


class WebSocketAuth:
    """Handle WebSocket authentication and authorization."""

    def __init__(self):
        self.settings = get_settings()

    async def authenticate_websocket(
        self,
        websocket: WebSocket,
        token: Optional[str] = Query(None),
    ) -> Optional[Dict[str, Any]]:
        """
        Authenticate WebSocket connection.

        Supports multiple authentication methods:
        1. Query parameter token
        2. Authorization header
        3. Cookie-based auth
        """
        user_data = None

        # Try query parameter token
        if token:
            user_data = await self._verify_token(token)

        # Try Authorization header
        if not user_data:
            auth_header = websocket.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]
                user_data = await self._verify_token(token)

        # Try cookie-based auth
        if not user_data:
            session_cookie = websocket.cookies.get("session")
            if session_cookie:
                user_data = await self._verify_session_cookie(session_cookie)

        if user_data:
            logger.info(f"WebSocket authenticated for user: {user_data.get('user_id')}")
        else:
            logger.warning("WebSocket connection without authentication")

        return user_data

    async def _verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token and extract user data."""
        try:
            payload = jwt.decode(
                token,
                self.settings.JWT_SECRET_KEY,
                algorithms=[self.settings.JWT_ALGORITHM],
            )

            # Check token expiration
            exp = payload.get("exp")
            if exp and datetime.fromtimestamp(exp) < datetime.utcnow():
                logger.warning("Expired token")
                return None

            return {
                "user_id": payload.get("sub"),
                "email": payload.get("email"),
                "roles": payload.get("roles", []),
                "authenticated_at": datetime.utcnow().isoformat(),
            }

        except jwt.InvalidTokenError as e:
            logger.error(f"Invalid token: {e}")
            return None
        except Exception as e:
            logger.error(f"Token verification error: {e}")
            return None

    async def _verify_session_cookie(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Verify session cookie and extract user data."""
        # TODO: Implement session validation with Redis
        # For now, we'll decode it as a JWT
        return await self._verify_token(session_id)

    def generate_connection_token(
        self,
        user_id: str,
        connection_id: str,
        expires_in: timedelta = timedelta(hours=24),
    ) -> str:
        """Generate a connection-specific token."""
        payload = {
            "sub": user_id,
            "connection_id": connection_id,
            "exp": datetime.utcnow() + expires_in,
            "iat": datetime.utcnow(),
            "type": "websocket",
        }

        return jwt.encode(
            payload,
            self.settings.JWT_SECRET_KEY,
            algorithm=self.settings.JWT_ALGORITHM,
        )

    def check_permissions(
        self,
        user_data: Dict[str, Any],
        resource: str,
        action: str,
    ) -> bool:
        """Check if user has permission for resource/action."""
        # Simple role-based check
        roles = user_data.get("roles", [])

        # Admin has all permissions
        if "admin" in roles:
            return True

        # Check specific permissions
        permission_map = {
            "generation": {
                "create": ["user", "premium"],
                "view": ["user", "premium"],
                "cancel": ["user", "premium"],
            },
            "collaboration": {
                "join": ["user", "premium"],
                "edit": ["user", "premium"],
                "admin": ["premium"],
            },
            "chat": {
                "send": ["user", "premium"],
                "receive": ["user", "premium"],
            },
        }

        required_roles = permission_map.get(resource, {}).get(action, [])
        return any(role in roles for role in required_roles)

    async def authorize_room_access(
        self,
        user_data: Dict[str, Any],
        room: str,
        action: str = "join",
    ) -> bool:
        """Check if user can access a specific room."""
        user_id = user_data.get("user_id")

        # Personal rooms (format: user:<user_id>)
        if room.startswith("user:"):
            room_user_id = room.split(":", 1)[1]
            return user_id == room_user_id

        # Project rooms (format: project:<project_id>)
        if room.startswith("project:"):
            # TODO: Check project membership
            return True

        # Public rooms
        if room.startswith("public:"):
            return True

        # Default: check permissions
        return self.check_permissions(user_data, "collaboration", action)
