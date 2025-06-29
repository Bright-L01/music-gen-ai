"""
Authentication helpers for Audio Processing Service

Provides JWT validation and service authentication.
"""

import os
from typing import Dict

from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt


# Security scheme
security = HTTPBearer()

# Configuration
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
SERVICE_API_KEY = os.getenv("SERVICE_API_KEY", "internal-service-key-change-in-production")


async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> Dict:
    """Verify JWT token and extract user data"""
    token = credentials.credentials
    
    try:
        payload = jwt.decode(
            token,
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM]
        )
        
        return {
            "user_id": payload.get("user_id"),
            "email": payload.get("email"),
            "username": payload.get("username"),
            "tier": payload.get("tier", "free")
        }
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


async def get_current_user(token_data: Dict = Depends(verify_token)) -> Dict:
    """Get current user information"""
    if not token_data.get("user_id"):
        raise HTTPException(status_code=401, detail="Invalid user data")
    
    return {
        "id": token_data["user_id"],
        "email": token_data["email"],
        "username": token_data["username"],
        "tier": token_data["tier"]
    }


async def verify_service_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> Dict:
    """Verify service-to-service authentication"""
    token = credentials.credentials
    
    # Check API key
    if token == SERVICE_API_KEY:
        return {"service": "internal", "type": "api_key"}
    
    # Try JWT
    try:
        payload = jwt.decode(
            token,
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM]
        )
        
        if payload.get("type") == "service":
            return {"service": payload.get("service"), "type": "service"}
            
    except jwt.InvalidTokenError:
        pass
    
    raise HTTPException(status_code=401, detail="Invalid service authentication")