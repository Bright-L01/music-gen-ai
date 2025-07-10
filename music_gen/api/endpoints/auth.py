"""
Authentication endpoints for Music Gen AI API.
Implements secure cookie-based authentication with CSRF protection.
"""

import secrets
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, EmailStr, validator

from music_gen.api.middleware.auth import (
    UserClaims,
    UserRole,
    auth_middleware,
    auth_rate_limiter,
    get_client_ip,
    require_auth,
    security_audit,
)
from music_gen.api.middleware.csrf import get_csrf_token
from music_gen.api.utils.cookies import (
    SecureCookieManager,
    clear_auth_cookies,
    get_auth_token,
    get_refresh_token,
    set_auth_cookies,
)
from music_gen.core.exceptions import AuthenticationError, ValidationError
from music_gen.utils.logging import get_logger

logger = get_logger(__name__)

# Create router
router = APIRouter(prefix="/api/auth", tags=["authentication"])


# Request/Response models
class LoginRequest(BaseModel):
    """Login request model"""

    email: EmailStr
    password: str
    remember_me: bool = False


class RegisterRequest(BaseModel):
    """Registration request model"""

    email: EmailStr
    username: str
    password: str

    @validator("username")
    def validate_username(cls, v):
        if len(v) < 3:
            raise ValueError("Username must be at least 3 characters long")
        if not v.isalnum() and "_" not in v:
            raise ValueError("Username must be alphanumeric (underscores allowed)")
        return v

    @validator("password")
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        return v


class TokenResponse(BaseModel):
    """Token response model"""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: dict


class UserResponse(BaseModel):
    """User information response"""

    id: str
    email: str
    username: str
    roles: list[str]
    tier: str
    is_verified: bool
    created_at: datetime
    last_login: Optional[datetime] = None


class MigrateTokenRequest(BaseModel):
    """Token migration request for localStorage to cookie transition"""

    access_token: str
    refresh_token: str


# Endpoints
@router.post("/login")
async def login(
    request: Request,
    response: Response,
    login_data: LoginRequest,
) -> Dict[str, Any]:
    """
    Authenticate user and set secure httpOnly cookies.

    This endpoint:
    1. Validates user credentials
    2. Creates JWT tokens
    3. Sets secure httpOnly cookies
    4. Returns user information
    """
    client_ip = get_client_ip(request)

    # Check rate limiting
    if not auth_rate_limiter.check_auth_limit(client_ip):
        security_audit.log_authentication(
            event_type="login_rate_limited",
            user_id=None,
            client_ip=client_ip,
            success=False,
            details={"email": login_data.email},
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many login attempts. Please try again later.",
            headers={"Retry-After": "900"},
        )

    try:
        # TODO: Validate credentials against database
        # For now, using mock validation
        if login_data.email == "user@example.com" and login_data.password == "password123":
            user_id = "user123"
            username = "testuser"
            roles = [UserRole.USER]
            tier = "free"
            is_verified = True
        else:
            # Record failed attempt
            auth_rate_limiter.record_failed_attempt(client_ip)
            security_audit.log_authentication(
                event_type="login_failed",
                user_id=None,
                client_ip=client_ip,
                success=False,
                details={"email": login_data.email, "reason": "invalid_credentials"},
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password"
            )

        # Create tokens
        access_token_expires = 900 if not login_data.remember_me else 3600  # 15 min or 1 hour
        refresh_token_expires = 604800 if not login_data.remember_me else 2592000  # 7 or 30 days

        access_token = auth_middleware.create_access_token(
            user_id=user_id,
            email=login_data.email,
            username=username,
            roles=roles,
            tier=tier,
            is_verified=is_verified,
            expires_delta=access_token_expires // 60,  # Convert to minutes
        )

        refresh_token = auth_middleware.create_refresh_token(
            user_id=user_id, expires_delta=refresh_token_expires // 60
        )

        # Set secure cookies
        set_auth_cookies(
            response=response,
            access_token=access_token,
            refresh_token=refresh_token,
            access_expires=access_token_expires,
            refresh_expires=refresh_token_expires,
        )

        # Generate and set CSRF token
        csrf_token = secrets.token_urlsafe(32)
        SecureCookieManager.set_csrf_cookie(response, csrf_token)

        # Log successful login
        security_audit.log_authentication(
            event_type="login_success",
            user_id=user_id,
            client_ip=client_ip,
            success=True,
            details={"remember_me": login_data.remember_me},
        )

        # Return user info (no tokens in response body)
        return {
            "success": True,
            "user": {
                "id": user_id,
                "email": login_data.email,
                "username": username,
                "roles": [role.value for role in roles],
                "tier": tier,
                "is_verified": is_verified,
            },
            "csrfToken": csrf_token,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service unavailable",
        )


@router.post("/logout")
async def logout(
    request: Request, response: Response, user_claims: Optional[UserClaims] = Depends(require_auth)
) -> Dict[str, str]:
    """
    Logout user by clearing cookies and blacklisting tokens.
    """
    try:
        # Blacklist current token
        if user_claims and user_claims.jti:
            auth_middleware.blacklist_token(user_claims.jti, user_claims.expires_at)

        # Clear all auth cookies
        clear_auth_cookies(response)

        # Log logout
        if user_claims:
            security_audit.log_authentication(
                event_type="logout",
                user_id=user_claims.user_id,
                client_ip=get_client_ip(request),
                success=True,
            )

        return {"message": "Logged out successfully"}

    except Exception as e:
        logger.error(f"Logout error: {e}")
        # Still clear cookies even if blacklisting fails
        clear_auth_cookies(response)
        return {"message": "Logged out successfully"}


@router.post("/register")
async def register(
    request: Request, response: Response, register_data: RegisterRequest
) -> Dict[str, Any]:
    """
    Register a new user account.
    """
    client_ip = get_client_ip(request)

    # Check rate limiting
    if not auth_rate_limiter.check_auth_limit(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many registration attempts. Please try again later.",
        )

    try:
        # TODO: Create user in database
        # For now, simulate registration
        user_id = f"user_{secrets.token_hex(8)}"

        # Create tokens for auto-login after registration
        access_token = auth_middleware.create_access_token(
            user_id=user_id,
            email=register_data.email,
            username=register_data.username,
            roles=[UserRole.USER],
            tier="free",
            is_verified=False,  # Email verification required
        )

        refresh_token = auth_middleware.create_refresh_token(user_id)

        # Set cookies
        set_auth_cookies(response, access_token, refresh_token)

        # Generate CSRF token
        csrf_token = secrets.token_urlsafe(32)
        SecureCookieManager.set_csrf_cookie(response, csrf_token)

        # Log registration
        security_audit.log_authentication(
            event_type="registration",
            user_id=user_id,
            client_ip=client_ip,
            success=True,
            details={"email": register_data.email, "username": register_data.username},
        )

        return {
            "success": True,
            "user": {
                "id": user_id,
                "email": register_data.email,
                "username": register_data.username,
                "roles": ["user"],
                "tier": "free",
                "is_verified": False,
            },
            "csrfToken": csrf_token,
            "message": "Registration successful. Please verify your email.",
        }

    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration service unavailable",
        )


@router.post("/refresh")
async def refresh_tokens(request: Request, response: Response) -> Dict[str, Any]:
    """
    Refresh access token using refresh token from cookie.
    """
    refresh_token = get_refresh_token(request)

    if not refresh_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token not found"
        )

    try:
        # Verify and refresh tokens
        claims = auth_middleware.verify_token(refresh_token)

        # Create new tokens
        new_access_token, new_refresh_token = auth_middleware.refresh_access_token(refresh_token)

        # Set new cookies
        set_auth_cookies(response, new_access_token, new_refresh_token)

        # Log token refresh
        security_audit.log_authentication(
            event_type="token_refresh",
            user_id=claims.user_id,
            client_ip=get_client_ip(request),
            success=True,
        )

        return {"success": True, "message": "Tokens refreshed successfully"}

    except AuthenticationError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token"
        )
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Token refresh failed"
        )


@router.get("/session")
async def check_session(
    request: Request,
) -> Dict[str, Any]:
    """
    Check current session status and return user info if authenticated.
    """
    # Get auth token from cookies
    auth_token = get_auth_token(request)

    if not auth_token:
        return {"authenticated": False}

    try:
        # Verify token
        user_claims = auth_middleware.verify_token(auth_token)
        csrf_token = get_csrf_token(request)

        return {
            "authenticated": True,
            "user": {
                "id": user_claims.user_id,
                "email": user_claims.email,
                "username": user_claims.username,
                "roles": [role.value for role in user_claims.roles],
                "tier": user_claims.tier,
                "is_verified": user_claims.is_verified,
            },
            "csrfToken": csrf_token,
        }
    except AuthenticationError:
        return {"authenticated": False}


@router.get("/csrf-token")
async def get_csrf_token_endpoint(request: Request, response: Response) -> Dict[str, str]:
    """
    Get or generate CSRF token for the current session.
    """
    csrf_token = get_csrf_token(request)

    if not csrf_token:
        # Generate new token
        csrf_token = secrets.token_urlsafe(32)
        SecureCookieManager.set_csrf_cookie(response, csrf_token)

    return {"csrfToken": csrf_token}


@router.post("/migrate")
async def migrate_tokens(
    request: Request, response: Response, migration_data: MigrateTokenRequest
) -> Dict[str, Any]:
    """
    Migrate localStorage tokens to secure httpOnly cookies.
    This endpoint is used during the transition period.
    """
    try:
        # Verify the access token
        claims = auth_middleware.verify_token(migration_data.access_token)

        # Verify the refresh token
        auth_middleware.verify_token(migration_data.refresh_token)

        # Set cookies with the existing tokens
        set_auth_cookies(
            response=response,
            access_token=migration_data.access_token,
            refresh_token=migration_data.refresh_token,
        )

        # Generate CSRF token
        csrf_token = secrets.token_urlsafe(32)
        SecureCookieManager.set_csrf_cookie(response, csrf_token)

        # Log migration
        security_audit.log_authentication(
            event_type="token_migration",
            user_id=claims.user_id,
            client_ip=get_client_ip(request),
            success=True,
        )

        return {
            "success": True,
            "message": "Tokens migrated successfully",
            "user": {
                "id": claims.user_id,
                "email": claims.email,
                "username": claims.username,
                "roles": [role.value for role in claims.roles],
                "tier": claims.tier,
                "is_verified": claims.is_verified,
            },
            "csrfToken": csrf_token,
        }

    except AuthenticationError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid tokens provided"
        )
    except Exception as e:
        logger.error(f"Token migration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Migration failed"
        )


@router.post("/verify-email/{token}")
async def verify_email(token: str, request: Request) -> Dict[str, str]:
    """
    Verify user email with verification token.
    """
    # TODO: Implement email verification
    # This would verify the token and update user's is_verified status

    return {"success": True, "message": "Email verified successfully"}


@router.post("/forgot-password")
async def forgot_password(
    email: EmailStr = Body(..., embed=True), request: Request = None
) -> Dict[str, str]:
    """
    Initiate password reset process.
    """
    # TODO: Implement password reset
    # This would send a password reset email

    # Log password reset request
    security_audit.log_authentication(
        event_type="password_reset_requested",
        user_id=None,
        client_ip=get_client_ip(request),
        success=True,
        details={"email": email},
    )

    return {"success": True, "message": "If the email exists, a password reset link has been sent"}


@router.post("/reset-password")
async def reset_password(
    token: str = Body(...), new_password: str = Body(...), request: Request = None
) -> Dict[str, str]:
    """
    Reset password with reset token.
    """
    # TODO: Implement password reset
    # This would verify the reset token and update the password

    return {"success": True, "message": "Password reset successfully"}


# Export router
__all__ = ["router"]
