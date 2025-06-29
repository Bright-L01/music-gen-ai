"""
Authentication helpers for API Gateway

Handles JWT token validation and user context extraction.
"""

import os
from typing import Dict, Optional

import jwt
from fastapi import HTTPException, Security, Depends, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials


# Configuration
JWT_SECRET = os.getenv("JWT_SECRET", "music-gen-secret-key-change-in-production")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")

# Security scheme
security = HTTPBearer()


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
        
        # Extract user ID
        user_id = payload.get("user_id")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload"
            )
            
        # Add token to payload for forwarding to services
        payload["token"] = token
        
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
        User information for use in endpoints
    """
    return {
        "id": token_data.get("user_id"),
        "email": token_data.get("email"),
        "username": token_data.get("username"),
        "tier": token_data.get("tier", "free"),
        "token": token_data.get("token")  # For forwarding to services
    }


async def get_optional_user(request: Request) -> Optional[Dict]:
    """
    Get user information if token is provided (optional auth)
    
    Args:
        request: FastAPI request object
        
    Returns:
        User information or None if no token
    """
    auth_header = request.headers.get("Authorization")
    
    if not auth_header or not auth_header.startswith("Bearer "):
        return None
        
    try:
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials=auth_header.split(" ")[1]
        )
        token_data = await verify_token(credentials)
        return await get_current_user(token_data)
    except HTTPException:
        return None


def require_admin(current_user: Dict = Depends(get_current_user)) -> Dict:
    """
    Require admin role for endpoint access
    
    Args:
        current_user: Current user data
        
    Returns:
        User data if admin
        
    Raises:
        HTTPException: If user is not admin
    """
    # Check if user has admin role (this would be in the JWT payload)
    user_roles = current_user.get("roles", [])
    
    if "admin" not in user_roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
        
    return current_user


def require_tier(min_tier: str):
    """
    Create dependency that requires minimum subscription tier
    
    Args:
        min_tier: Minimum required tier (free, premium, enterprise)
        
    Returns:
        Dependency function
    """
    def check_tier(current_user: Dict = Depends(get_current_user)) -> Dict:
        tier_levels = {
            "free": 0,
            "premium": 1,
            "enterprise": 2
        }
        
        user_level = tier_levels.get(current_user.get("tier", "free"), 0)
        required_level = tier_levels.get(min_tier, 0)
        
        if user_level < required_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Subscription tier '{min_tier}' or higher required"
            )
            
        return current_user
    
    return check_tier