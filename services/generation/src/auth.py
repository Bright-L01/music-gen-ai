"""
Authentication helpers for microservices

Provides JWT validation and user context extraction for inter-service communication.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

import jwt
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel


logger = logging.getLogger(__name__)


# Configuration
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRATION_HOURS = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))

# Service-to-service auth
SERVICE_API_KEY = os.getenv("SERVICE_API_KEY", "internal-service-key-change-in-production")


class TokenData(BaseModel):
    """JWT token payload"""
    user_id: str
    email: str
    username: str
    tier: str = "free"  # free, premium, enterprise
    permissions: List[str] = []
    exp: Optional[datetime] = None
    iat: Optional[datetime] = None


class ServiceAuth(BaseModel):
    """Service-to-service authentication"""
    service_name: str
    api_key: str
    timestamp: datetime


# Security scheme
security = HTTPBearer()


async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> TokenData:
    """
    Verify JWT token and extract user data
    
    Args:
        credentials: Bearer token from Authorization header
        
    Returns:
        TokenData with user information
        
    Raises:
        HTTPException: If token is invalid or expired
    """
    token = credentials.credentials
    
    try:
        # Decode JWT
        payload = jwt.decode(
            token,
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM]
        )
        
        # Extract user data
        token_data = TokenData(
            user_id=payload.get("user_id"),
            email=payload.get("email"),
            username=payload.get("username"),
            tier=payload.get("tier", "free"),
            permissions=payload.get("permissions", []),
            exp=datetime.fromtimestamp(payload.get("exp")) if payload.get("exp") else None,
            iat=datetime.fromtimestamp(payload.get("iat")) if payload.get("iat") else None
        )
        
        # Validate required fields
        if not token_data.user_id or not token_data.email:
            raise HTTPException(
                status_code=401,
                detail="Invalid token data"
            )
            
        return token_data
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=401,
            detail="Token has expired"
        )
    except jwt.InvalidTokenError as e:
        logger.error(f"Invalid token: {e}")
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token"
        )
    except Exception as e:
        logger.error(f"Token verification error: {e}")
        raise HTTPException(
            status_code=401,
            detail="Could not validate credentials"
        )


async def get_current_user(token_data: TokenData = Depends(verify_token)) -> Dict[str, Any]:
    """
    Get current user information from token
    
    Args:
        token_data: Verified token data
        
    Returns:
        User dictionary with relevant information
    """
    return {
        "id": token_data.user_id,
        "email": token_data.email,
        "username": token_data.username,
        "tier": token_data.tier,
        "permissions": token_data.permissions
    }


async def require_permission(
    permission: str,
    token_data: TokenData = Depends(verify_token)
) -> TokenData:
    """
    Require specific permission for endpoint
    
    Args:
        permission: Required permission
        token_data: User token data
        
    Returns:
        Token data if permission exists
        
    Raises:
        HTTPException: If permission is missing
    """
    if permission not in token_data.permissions:
        raise HTTPException(
            status_code=403,
            detail=f"Permission '{permission}' required"
        )
    return token_data


async def require_tier(
    min_tier: str,
    token_data: TokenData = Depends(verify_token)
) -> TokenData:
    """
    Require minimum subscription tier
    
    Args:
        min_tier: Minimum required tier (free, premium, enterprise)
        token_data: User token data
        
    Returns:
        Token data if tier is sufficient
        
    Raises:
        HTTPException: If tier is insufficient
    """
    tier_levels = {
        "free": 0,
        "premium": 1,
        "enterprise": 2
    }
    
    user_level = tier_levels.get(token_data.tier, 0)
    required_level = tier_levels.get(min_tier, 0)
    
    if user_level < required_level:
        raise HTTPException(
            status_code=403,
            detail=f"Subscription tier '{min_tier}' or higher required"
        )
    return token_data


def create_service_token(service_name: str) -> str:
    """
    Create JWT token for service-to-service communication
    
    Args:
        service_name: Name of the calling service
        
    Returns:
        JWT token string
    """
    payload = {
        "service": service_name,
        "type": "service",
        "iat": datetime.utcnow(),
        "exp": datetime.utcnow() + timedelta(hours=1)  # Short expiry for services
    }
    
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


async def verify_service_token(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> Dict[str, Any]:
    """
    Verify service-to-service authentication token
    
    Args:
        credentials: Bearer token from Authorization header
        
    Returns:
        Service information
        
    Raises:
        HTTPException: If token is invalid
    """
    token = credentials.credentials
    
    try:
        # Try to decode as service token
        payload = jwt.decode(
            token,
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM]
        )
        
        if payload.get("type") != "service":
            raise HTTPException(
                status_code=401,
                detail="Invalid service token"
            )
            
        return {
            "service": payload.get("service"),
            "type": "service"
        }
        
    except jwt.InvalidTokenError:
        # Try API key authentication as fallback
        if token == SERVICE_API_KEY:
            return {
                "service": "internal",
                "type": "api_key"
            }
        raise HTTPException(
            status_code=401,
            detail="Invalid service authentication"
        )


class RateLimiter:
    """
    Simple rate limiter for API protection
    """
    
    def __init__(self, redis_client):
        self.redis = redis_client
        
    async def check_rate_limit(
        self,
        user_id: str,
        tier: str,
        action: str = "generate"
    ) -> bool:
        """
        Check if user has exceeded rate limit
        
        Args:
            user_id: User identifier
            tier: User subscription tier
            action: Action being performed
            
        Returns:
            True if within limits, False if exceeded
        """
        # Define limits by tier
        limits = {
            "free": {"minute": 2, "hour": 10, "day": 50},
            "premium": {"minute": 10, "hour": 100, "day": 1000},
            "enterprise": {"minute": 50, "hour": 500, "day": 10000}
        }
        
        tier_limits = limits.get(tier, limits["free"])
        
        # Check each time window
        for window, limit in tier_limits.items():
            key = f"ratelimit:{user_id}:{action}:{window}"
            
            # Get current count
            count = await self.redis.get(key)
            count = int(count) if count else 0
            
            if count >= limit:
                logger.warning(f"Rate limit exceeded for user {user_id} ({window})")
                return False
                
        return True
        
    async def increment_usage(
        self,
        user_id: str,
        action: str = "generate"
    ):
        """
        Increment usage counters
        
        Args:
            user_id: User identifier
            action: Action being performed
        """
        # Time windows in seconds
        windows = {
            "minute": 60,
            "hour": 3600,
            "day": 86400
        }
        
        for window, ttl in windows.items():
            key = f"ratelimit:{user_id}:{action}:{window}"
            
            # Increment with expiry
            await self.redis.incr(key)
            await self.redis.expire(key, ttl)


# Dependency for rate limiting
async def check_rate_limit_dependency(
    current_user: Dict = Depends(get_current_user)
) -> Dict:
    """
    FastAPI dependency to check rate limits
    
    Args:
        current_user: Current user information
        
    Returns:
        User information if within limits
        
    Raises:
        HTTPException: If rate limit exceeded
    """
    # This would be connected to actual Redis in production
    # For now, just return the user
    return current_user


# Utility functions
def decode_token_unsafe(token: str) -> Optional[Dict[str, Any]]:
    """
    Decode JWT token without verification (for debugging only)
    
    Args:
        token: JWT token string
        
    Returns:
        Decoded payload or None
    """
    try:
        return jwt.decode(token, options={"verify_signature": False})
    except Exception:
        return None