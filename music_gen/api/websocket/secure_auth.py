"""
Hardened WebSocket authentication for Music Gen AI.

Implements secure WebSocket authentication with JWT validation,
periodic re-authentication, and comprehensive security measures.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set
from urllib.parse import parse_qs, urlparse

import jwt
from fastapi import WebSocket, WebSocketDisconnect
from jwt import PyJWTError

from ...core.config import get_config

logger = logging.getLogger(__name__)


@dataclass
class AuthenticatedConnection:
    """Represents an authenticated WebSocket connection."""

    websocket: WebSocket
    user_id: str
    token: str
    authenticated_at: float
    last_validation: float
    ip_address: str
    user_agent: str
    channels: Set[str]
    message_count: int = 0
    last_activity: float = None

    def __post_init__(self):
        if self.last_activity is None:
            self.last_activity = time.time()


class WebSocketSecurityConfig:
    """Configuration for WebSocket security."""

    def __init__(self):
        config = get_config()

        # Token validation
        self.jwt_secret = getattr(config, "jwt_secret", "your-secret-key")
        self.jwt_algorithm = getattr(config, "jwt_algorithm", "HS256")
        self.token_validation_interval = 300  # 5 minutes
        self.max_token_age = 3600  # 1 hour

        # Connection limits
        self.max_connections_per_ip = 10
        self.max_connections_per_user = 5
        self.max_message_rate = 100  # messages per minute
        self.max_channels_per_connection = 10

        # Security timeouts
        self.auth_timeout = 30  # 30 seconds to authenticate
        self.idle_timeout = 1800  # 30 minutes idle timeout
        self.max_message_size = 1024 * 10  # 10KB per message

        # Allowed origins for CORS
        self.allowed_origins = getattr(
            config,
            "websocket_allowed_origins",
            ["http://localhost:3000", "https://app.musicgenai.com"],
        )


class WebSocketSecurityError(Exception):
    """WebSocket security-related error."""

    pass


class WebSocketAuthenticator:
    """Handles secure WebSocket authentication and connection management."""

    def __init__(self, config: Optional[WebSocketSecurityConfig] = None):
        self.config = config or WebSocketSecurityConfig()

        # Connection tracking
        self.connections: Dict[str, AuthenticatedConnection] = {}
        self.connections_by_ip: Dict[str, List[str]] = {}
        self.connections_by_user: Dict[str, List[str]] = {}

        # Rate limiting
        self.message_rates: Dict[str, List[float]] = {}

        # Security monitoring
        self.failed_auth_attempts: Dict[str, List[float]] = {}
        self.blocked_ips: Set[str] = set()

        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._validation_task: Optional[asyncio.Task] = None

    async def start_background_tasks(self):
        """Start background security tasks."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._validation_task = asyncio.create_task(self._validation_loop())
        logger.info("WebSocket security background tasks started")

    async def stop_background_tasks(self):
        """Stop background security tasks."""
        for task in [self._cleanup_task, self._validation_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        logger.info("WebSocket security background tasks stopped")

    def _get_client_ip(self, websocket: WebSocket) -> str:
        """Extract client IP address with proxy support."""
        # Check for forwarded headers (common with reverse proxies)
        forwarded_for = websocket.headers.get("x-forwarded-for")
        if forwarded_for:
            # Take the first IP in the chain
            return forwarded_for.split(",")[0].strip()

        real_ip = websocket.headers.get("x-real-ip")
        if real_ip:
            return real_ip

        # Fallback to client IP
        client_ip = websocket.client.host if websocket.client else "unknown"
        return client_ip

    def _validate_origin(self, websocket: WebSocket) -> bool:
        """Validate WebSocket origin for CSRF protection."""
        origin = websocket.headers.get("origin")

        if not origin:
            logger.warning("WebSocket connection without Origin header")
            return False

        if origin not in self.config.allowed_origins:
            logger.warning(f"WebSocket connection from unauthorized origin: {origin}")
            return False

        return True

    def _decode_jwt_token(self, token: str) -> Dict[str, Any]:
        """Decode and validate JWT token."""
        try:
            payload = jwt.decode(
                token,
                self.config.jwt_secret,
                algorithms=[self.config.jwt_algorithm],
                options={
                    "require_exp": True,
                    "verify_exp": True,
                    "require_iat": True,
                    "verify_iat": True,
                },
            )

            # Check token age
            issued_at = payload.get("iat", 0)
            if time.time() - issued_at > self.config.max_token_age:
                raise WebSocketSecurityError("Token too old")

            return payload

        except jwt.ExpiredSignatureError:
            raise WebSocketSecurityError("Token expired")
        except jwt.InvalidTokenError as e:
            raise WebSocketSecurityError(f"Invalid token: {e}")

    def _check_rate_limit(self, connection_id: str) -> bool:
        """Check if connection is within rate limits."""
        now = time.time()

        # Clean old entries
        if connection_id in self.message_rates:
            self.message_rates[connection_id] = [
                ts for ts in self.message_rates[connection_id] if now - ts < 60  # Keep last minute
            ]
        else:
            self.message_rates[connection_id] = []

        # Check current rate
        message_count = len(self.message_rates[connection_id])
        if message_count >= self.config.max_message_rate:
            return False

        # Record this message
        self.message_rates[connection_id].append(now)
        return True

    def _check_connection_limits(self, ip_address: str, user_id: str) -> bool:
        """Check connection limits per IP and user."""
        # Check IP limit
        ip_connections = len(self.connections_by_ip.get(ip_address, []))
        if ip_connections >= self.config.max_connections_per_ip:
            logger.warning(f"IP {ip_address} exceeded connection limit")
            return False

        # Check user limit
        user_connections = len(self.connections_by_user.get(user_id, []))
        if user_connections >= self.config.max_connections_per_user:
            logger.warning(f"User {user_id} exceeded connection limit")
            return False

        return True

    def _record_failed_auth(self, ip_address: str):
        """Record failed authentication attempt."""
        now = time.time()

        if ip_address not in self.failed_auth_attempts:
            self.failed_auth_attempts[ip_address] = []

        # Clean old attempts (last hour)
        self.failed_auth_attempts[ip_address] = [
            ts for ts in self.failed_auth_attempts[ip_address] if now - ts < 3600
        ]

        # Record new attempt
        self.failed_auth_attempts[ip_address].append(now)

        # Block IP if too many failures
        if len(self.failed_auth_attempts[ip_address]) >= 10:
            self.blocked_ips.add(ip_address)
            logger.warning(f"Blocked IP {ip_address} due to repeated auth failures")

    async def authenticate_connection(
        self, websocket: WebSocket, connection_id: str
    ) -> AuthenticatedConnection:
        """Authenticate WebSocket connection with comprehensive security checks."""

        client_ip = self._get_client_ip(websocket)
        user_agent = websocket.headers.get("user-agent", "unknown")

        # Check if IP is blocked
        if client_ip in self.blocked_ips:
            logger.warning(f"Blocked IP {client_ip} attempted connection")
            raise WebSocketSecurityError("IP address blocked")

        # Validate origin for CSRF protection
        if not self._validate_origin(websocket):
            self._record_failed_auth(client_ip)
            raise WebSocketSecurityError("Invalid origin")

        # Wait for authentication message with timeout
        try:
            auth_message = await asyncio.wait_for(
                websocket.receive_text(), timeout=self.config.auth_timeout
            )
        except asyncio.TimeoutError:
            self._record_failed_auth(client_ip)
            raise WebSocketSecurityError("Authentication timeout")

        # Parse authentication message
        try:
            auth_data = json.loads(auth_message)
            if auth_data.get("type") != "authenticate":
                raise ValueError("Invalid message type")

            token = auth_data.get("token")
            if not token:
                raise ValueError("Missing token")

        except (json.JSONDecodeError, ValueError) as e:
            self._record_failed_auth(client_ip)
            raise WebSocketSecurityError(f"Invalid authentication message: {e}")

        # Validate JWT token
        try:
            payload = self._decode_jwt_token(token)
            user_id = payload.get("user_id") or payload.get("sub")

            if not user_id:
                raise WebSocketSecurityError("Token missing user ID")

        except WebSocketSecurityError:
            self._record_failed_auth(client_ip)
            raise

        # Check connection limits
        if not self._check_connection_limits(client_ip, user_id):
            self._record_failed_auth(client_ip)
            raise WebSocketSecurityError("Connection limit exceeded")

        # Create authenticated connection
        now = time.time()
        connection = AuthenticatedConnection(
            websocket=websocket,
            user_id=user_id,
            token=token,
            authenticated_at=now,
            last_validation=now,
            ip_address=client_ip,
            user_agent=user_agent,
            channels=set(),
        )

        # Track connection
        self.connections[connection_id] = connection

        # Update tracking dictionaries
        if client_ip not in self.connections_by_ip:
            self.connections_by_ip[client_ip] = []
        self.connections_by_ip[client_ip].append(connection_id)

        if user_id not in self.connections_by_user:
            self.connections_by_user[user_id] = []
        self.connections_by_user[user_id].append(connection_id)

        # Send authentication success
        await websocket.send_text(
            json.dumps({"type": "auth_success", "message": "Authentication successful"})
        )

        logger.info(f"WebSocket authenticated: user={user_id}, ip={client_ip}")
        return connection

    async def validate_message(self, connection_id: str, message: str) -> bool:
        """Validate incoming message from authenticated connection."""

        if connection_id not in self.connections:
            raise WebSocketSecurityError("Connection not found")

        connection = self.connections[connection_id]

        # Check message size
        if len(message) > self.config.max_message_size:
            logger.warning(f"Oversized message from {connection.user_id}: {len(message)} bytes")
            raise WebSocketSecurityError("Message too large")

        # Check rate limit
        if not self._check_rate_limit(connection_id):
            logger.warning(f"Rate limit exceeded for {connection.user_id}")
            raise WebSocketSecurityError("Rate limit exceeded")

        # Update activity
        connection.last_activity = time.time()
        connection.message_count += 1

        return True

    async def validate_token_periodically(self, connection_id: str) -> bool:
        """Perform periodic token validation."""

        if connection_id not in self.connections:
            return False

        connection = self.connections[connection_id]
        now = time.time()

        # Check if validation is due
        if now - connection.last_validation < self.config.token_validation_interval:
            return True

        # Validate token
        try:
            payload = self._decode_jwt_token(connection.token)
            connection.last_validation = now
            return True

        except WebSocketSecurityError as e:
            logger.warning(f"Token validation failed for {connection.user_id}: {e}")
            await self.disconnect_connection(connection_id, f"Token validation failed: {e}")
            return False

    async def disconnect_connection(self, connection_id: str, reason: str = "Unknown"):
        """Safely disconnect and clean up connection."""

        if connection_id not in self.connections:
            return

        connection = self.connections[connection_id]

        try:
            # Send disconnection message
            await connection.websocket.send_text(
                json.dumps({"type": "disconnect", "reason": reason})
            )

            # Close WebSocket
            await connection.websocket.close()

        except Exception as e:
            logger.warning(f"Error during disconnect: {e}")

        # Clean up tracking
        self._cleanup_connection(connection_id)

        logger.info(f"WebSocket disconnected: user={connection.user_id}, reason={reason}")

    def _cleanup_connection(self, connection_id: str):
        """Clean up connection tracking data."""

        if connection_id not in self.connections:
            return

        connection = self.connections[connection_id]

        # Remove from main tracking
        del self.connections[connection_id]

        # Remove from IP tracking
        if connection.ip_address in self.connections_by_ip:
            if connection_id in self.connections_by_ip[connection.ip_address]:
                self.connections_by_ip[connection.ip_address].remove(connection_id)
            if not self.connections_by_ip[connection.ip_address]:
                del self.connections_by_ip[connection.ip_address]

        # Remove from user tracking
        if connection.user_id in self.connections_by_user:
            if connection_id in self.connections_by_user[connection.user_id]:
                self.connections_by_user[connection.user_id].remove(connection_id)
            if not self.connections_by_user[connection.user_id]:
                del self.connections_by_user[connection.user_id]

        # Clean up rate limiting
        if connection_id in self.message_rates:
            del self.message_rates[connection_id]

    async def _cleanup_loop(self):
        """Background task to clean up stale connections."""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute

                now = time.time()
                stale_connections = []

                for connection_id, connection in self.connections.items():
                    # Check for idle timeout
                    if now - connection.last_activity > self.config.idle_timeout:
                        stale_connections.append((connection_id, "Idle timeout"))

                # Disconnect stale connections
                for connection_id, reason in stale_connections:
                    await self.disconnect_connection(connection_id, reason)

                # Clean up old failed auth attempts
                cutoff = now - 3600  # 1 hour
                for ip in list(self.failed_auth_attempts.keys()):
                    self.failed_auth_attempts[ip] = [
                        ts for ts in self.failed_auth_attempts[ip] if ts > cutoff
                    ]
                    if not self.failed_auth_attempts[ip]:
                        del self.failed_auth_attempts[ip]

                # Unblock IPs after 24 hours
                if len(self.blocked_ips) > 0:
                    logger.info(f"Currently blocked IPs: {len(self.blocked_ips)}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")

    async def _validation_loop(self):
        """Background task for periodic token validation."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute

                # Validate tokens for all connections
                validation_tasks = []
                for connection_id in list(self.connections.keys()):
                    task = asyncio.create_task(self.validate_token_periodically(connection_id))
                    validation_tasks.append(task)

                if validation_tasks:
                    await asyncio.gather(*validation_tasks, return_exceptions=True)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Validation loop error: {e}")

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get WebSocket connection statistics."""
        return {
            "total_connections": len(self.connections),
            "connections_by_ip": {
                ip: len(connections) for ip, connections in self.connections_by_ip.items()
            },
            "connections_by_user": {
                user: len(connections) for user, connections in self.connections_by_user.items()
            },
            "blocked_ips": len(self.blocked_ips),
            "failed_auth_attempts": len(self.failed_auth_attempts),
        }


# Global authenticator instance
_ws_authenticator: Optional[WebSocketAuthenticator] = None


async def get_websocket_authenticator() -> WebSocketAuthenticator:
    """Get or create global WebSocket authenticator."""
    global _ws_authenticator

    if _ws_authenticator is None:
        _ws_authenticator = WebSocketAuthenticator()
        await _ws_authenticator.start_background_tasks()

    return _ws_authenticator


async def cleanup_websocket_authenticator():
    """Clean up global WebSocket authenticator."""
    global _ws_authenticator

    if _ws_authenticator:
        await _ws_authenticator.stop_background_tasks()
        _ws_authenticator = None
