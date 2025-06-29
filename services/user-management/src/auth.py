"""
Authentication helpers for User Management Service

Handles JWT token creation and validation.
"""

import os
from datetime import datetime, timedelta
from typing import Optional, Dict

import jwt
from fastapi import HTTPException, Security, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .database import database


# Configuration
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRATION_HOURS = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))

# Security scheme
security = HTTPBearer()


def create_access_token(data: Dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create JWT access token
    
    Args:
        data: Token payload data
        expires_delta: Custom expiration time
        
    Returns:
        JWT token string
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
        
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access"
    })
    
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)


async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> Dict:
    """
    Verify JWT token and extract user data
    
    Args:
        credentials: Bearer token from Authorization header
        
    Returns:
        Token payload data
        
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
        
        # Validate token type
        if payload.get("type") != "access":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type"
            )
            
        # Extract user ID
        user_id = payload.get("user_id")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload"
            )
            
        return payload
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(token_data: Dict = Depends(verify_token)) -> Dict:
    """
    Get current user information from token
    
    Args:
        token_data: Verified token data
        
    Returns:
        User information
        
    Raises:
        HTTPException: If user not found or inactive
    """
    user_id = token_data.get("user_id")
    
    # Get user from database to verify they still exist and are active
    user_row = await database.fetch_one("""
        SELECT id, username, email, tier, status, is_verified
        FROM users 
        WHERE id = ? AND status = 'active'
    """, [user_id])
    
    if not user_row:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive"
        )
    
    return {
        "id": user_row["id"],
        "username": user_row["username"],
        "email": user_row["email"],
        "tier": user_row["tier"],
        "status": user_row["status"],
        "is_verified": user_row["is_verified"]
    }


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


def decode_token_unsafe(token: str) -> Optional[Dict]:
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