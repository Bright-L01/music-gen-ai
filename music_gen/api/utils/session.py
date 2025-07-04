"""
Session Management for Music Gen AI API.
Provides server-side session storage and management using Redis.
"""

import json
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any

import redis
from pydantic import BaseModel

from music_gen.core.config import get_config
from music_gen.utils.logging import get_logger

logger = get_logger(__name__)
config = get_config()

# Redis connection for session storage
try:
    redis_client = redis.Redis(
        host=config.redis_host or "localhost",
        port=config.redis_port or 6379,
        db=config.redis_db or 1,  # Use different DB for sessions
        decode_responses=True
    )
    # Test connection
    redis_client.ping()
    logger.info("Redis connected for session management")
except (redis.ConnectionError, AttributeError):
    logger.warning("Redis not available. Session storage will be disabled.")
    redis_client = None


class SessionData(BaseModel):
    """Session data model"""
    session_id: str
    user_id: str
    created_at: datetime
    last_accessed: datetime
    expires_at: datetime
    ip_address: str
    user_agent: Optional[str] = None
    data: Dict[str, Any] = {}


class SessionManager:
    """
    Manages server-side sessions with Redis backend.
    
    Features:
    - Secure session ID generation
    - Automatic session expiration
    - Session data storage
    - Activity tracking
    - Multi-device session support
    """
    
    SESSION_PREFIX = "session:"
    USER_SESSIONS_PREFIX = "user_sessions:"
    DEFAULT_SESSION_DURATION = 3600  # 1 hour
    EXTENDED_SESSION_DURATION = 2592000  # 30 days for "remember me"
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis = redis_client or globals()['redis_client']
        self.enabled = self.redis is not None
    
    def create_session(
        self,
        user_id: str,
        ip_address: str,
        user_agent: Optional[str] = None,
        remember_me: bool = False,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> Optional[SessionData]:
        """Create a new session for user."""
        if not self.enabled:
            logger.warning("Session storage disabled - Redis not available")
            return None
        
        try:
            # Generate secure session ID
            session_id = secrets.token_urlsafe(32)
            
            # Set expiration based on remember_me
            duration = self.EXTENDED_SESSION_DURATION if remember_me else self.DEFAULT_SESSION_DURATION
            now = datetime.now(timezone.utc)
            expires_at = now + timedelta(seconds=duration)
            
            # Create session data
            session_data = SessionData(
                session_id=session_id,
                user_id=user_id,
                created_at=now,
                last_accessed=now,
                expires_at=expires_at,
                ip_address=ip_address,
                user_agent=user_agent,
                data=additional_data or {}
            )
            
            # Store session in Redis
            session_key = f"{self.SESSION_PREFIX}{session_id}"
            self.redis.setex(
                session_key,
                duration,
                session_data.json()
            )
            
            # Track session for user (for multi-device support)
            user_sessions_key = f"{self.USER_SESSIONS_PREFIX}{user_id}"
            self.redis.sadd(user_sessions_key, session_id)
            self.redis.expire(user_sessions_key, duration)
            
            logger.info(f"Created session for user {user_id}")
            return session_data
            
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            return None
    
    def get_session(self, session_id: str) -> Optional[SessionData]:
        """Retrieve session data by ID."""
        if not self.enabled:
            return None
        
        try:
            session_key = f"{self.SESSION_PREFIX}{session_id}"
            session_json = self.redis.get(session_key)
            
            if not session_json:
                return None
            
            session_data = SessionData.parse_raw(session_json)
            
            # Update last accessed time
            session_data.last_accessed = datetime.now(timezone.utc)
            self.redis.setex(
                session_key,
                int((session_data.expires_at - datetime.now(timezone.utc)).total_seconds()),
                session_data.json()
            )
            
            return session_data
            
        except Exception as e:
            logger.error(f"Failed to get session: {e}")
            return None
    
    def update_session(
        self,
        session_id: str,
        data: Dict[str, Any],
        extend_expiration: bool = False
    ) -> bool:
        """Update session data."""
        if not self.enabled:
            return False
        
        try:
            session = self.get_session(session_id)
            if not session:
                return False
            
            # Update data
            session.data.update(data)
            session.last_accessed = datetime.now(timezone.utc)
            
            # Optionally extend expiration
            if extend_expiration:
                duration = self.DEFAULT_SESSION_DURATION
                session.expires_at = session.last_accessed + timedelta(seconds=duration)
            
            # Save updated session
            session_key = f"{self.SESSION_PREFIX}{session_id}"
            ttl = int((session.expires_at - datetime.now(timezone.utc)).total_seconds())
            
            if ttl > 0:
                self.redis.setex(session_key, ttl, session.json())
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to update session: {e}")
            return False
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if not self.enabled:
            return False
        
        try:
            # Get session to find user ID
            session = self.get_session(session_id)
            if not session:
                return True  # Already deleted
            
            # Delete session
            session_key = f"{self.SESSION_PREFIX}{session_id}"
            self.redis.delete(session_key)
            
            # Remove from user's session list
            user_sessions_key = f"{self.USER_SESSIONS_PREFIX}{session.user_id}"
            self.redis.srem(user_sessions_key, session_id)
            
            logger.info(f"Deleted session {session_id} for user {session.user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete session: {e}")
            return False
    
    def get_user_sessions(self, user_id: str) -> list[SessionData]:
        """Get all active sessions for a user."""
        if not self.enabled:
            return []
        
        try:
            user_sessions_key = f"{self.USER_SESSIONS_PREFIX}{user_id}"
            session_ids = self.redis.smembers(user_sessions_key)
            
            sessions = []
            for session_id in session_ids:
                session = self.get_session(session_id)
                if session:
                    sessions.append(session)
                else:
                    # Clean up invalid session reference
                    self.redis.srem(user_sessions_key, session_id)
            
            return sessions
            
        except Exception as e:
            logger.error(f"Failed to get user sessions: {e}")
            return []
    
    def delete_user_sessions(self, user_id: str, except_session: Optional[str] = None) -> int:
        """
        Delete all sessions for a user, optionally keeping one session.
        
        Args:
            user_id: User ID
            except_session: Session ID to keep (for "logout other devices")
            
        Returns:
            Number of sessions deleted
        """
        if not self.enabled:
            return 0
        
        try:
            sessions = self.get_user_sessions(user_id)
            deleted_count = 0
            
            for session in sessions:
                if session.session_id != except_session:
                    if self.delete_session(session.session_id):
                        deleted_count += 1
            
            logger.info(f"Deleted {deleted_count} sessions for user {user_id}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to delete user sessions: {e}")
            return 0
    
    def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions. Should be run periodically.
        
        Returns:
            Number of sessions cleaned up
        """
        if not self.enabled:
            return 0
        
        try:
            # Redis automatically expires keys, but we can clean up references
            cleaned = 0
            
            # Get all user session sets
            cursor = 0
            while True:
                cursor, keys = self.redis.scan(
                    cursor,
                    match=f"{self.USER_SESSIONS_PREFIX}*",
                    count=100
                )
                
                for key in keys:
                    # Check each session in the set
                    session_ids = self.redis.smembers(key)
                    for session_id in session_ids:
                        session_key = f"{self.SESSION_PREFIX}{session_id}"
                        if not self.redis.exists(session_key):
                            # Remove expired session reference
                            self.redis.srem(key, session_id)
                            cleaned += 1
                
                if cursor == 0:
                    break
            
            if cleaned > 0:
                logger.info(f"Cleaned up {cleaned} expired session references")
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Failed to cleanup sessions: {e}")
            return 0
    
    def get_session_stats(self, user_id: str) -> Dict[str, Any]:
        """Get session statistics for a user."""
        if not self.enabled:
            return {"enabled": False}
        
        try:
            sessions = self.get_user_sessions(user_id)
            
            return {
                "enabled": True,
                "active_sessions": len(sessions),
                "devices": [
                    {
                        "session_id": s.session_id[:8] + "...",  # Partial ID for security
                        "created_at": s.created_at.isoformat(),
                        "last_accessed": s.last_accessed.isoformat(),
                        "ip_address": s.ip_address,
                        "user_agent": s.user_agent,
                        "expires_at": s.expires_at.isoformat()
                    }
                    for s in sessions
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to get session stats: {e}")
            return {"enabled": True, "error": str(e)}


# Global session manager instance
session_manager = SessionManager()


# Utility functions
def create_user_session(
    user_id: str,
    ip_address: str,
    user_agent: Optional[str] = None,
    remember_me: bool = False,
    **kwargs
) -> Optional[str]:
    """
    Create a new session for user and return session ID.
    
    Args:
        user_id: User ID
        ip_address: Client IP address
        user_agent: Client user agent
        remember_me: Extended session duration
        **kwargs: Additional session data
        
    Returns:
        Session ID if successful, None otherwise
    """
    session = session_manager.create_session(
        user_id=user_id,
        ip_address=ip_address,
        user_agent=user_agent,
        remember_me=remember_me,
        additional_data=kwargs
    )
    
    return session.session_id if session else None


def get_session_data(session_id: str) -> Optional[Dict[str, Any]]:
    """Get session data by ID."""
    session = session_manager.get_session(session_id)
    return session.dict() if session else None


def update_session_data(session_id: str, **kwargs) -> bool:
    """Update session data."""
    return session_manager.update_session(session_id, kwargs)


def delete_session(session_id: str) -> bool:
    """Delete a session."""
    return session_manager.delete_session(session_id)


def logout_all_devices(user_id: str, current_session: Optional[str] = None) -> int:
    """Logout user from all devices except current."""
    return session_manager.delete_user_sessions(user_id, current_session)


# Export public interface
__all__ = [
    "SessionManager",
    "SessionData",
    "session_manager",
    "create_user_session",
    "get_session_data",
    "update_session_data",
    "delete_session",
    "logout_all_devices"
]