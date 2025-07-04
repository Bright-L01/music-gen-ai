"""
CSRF Protection Middleware for Music Gen AI API.
Implements Double Submit Cookie pattern for CSRF protection.
"""

import secrets
import logging
from typing import Optional, Set
from datetime import datetime, timezone

from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from music_gen.utils.logging import get_logger

logger = get_logger(__name__)


class CSRFProtectionMiddleware(BaseHTTPMiddleware):
    """
    CSRF Protection using Double Submit Cookie pattern.
    
    This middleware:
    1. Generates CSRF tokens for sessions
    2. Validates CSRF tokens on state-changing requests
    3. Manages CSRF cookie lifecycle
    """
    
    # Methods that require CSRF protection
    PROTECTED_METHODS: Set[str] = {"POST", "PUT", "PATCH", "DELETE"}
    
    # Paths that are exempt from CSRF protection
    EXEMPT_PATHS: Set[str] = {
        "/api/auth/csrf-token",  # CSRF token endpoint itself
        "/api/auth/login",       # Login needs to work without existing CSRF
        "/api/auth/register",    # Registration needs to work without CSRF
        "/api/auth/refresh",     # Token refresh endpoint
        "/docs",                 # API documentation
        "/openapi.json",         # OpenAPI schema
        "/health",               # Health checks
    }
    
    def __init__(
        self,
        app,
        cookie_name: str = "csrf_token",
        header_name: str = "X-CSRF-Token",
        cookie_path: str = "/",
        cookie_domain: Optional[str] = None,
        cookie_secure: bool = True,
        cookie_httponly: bool = False,  # CSRF cookie needs to be readable by JS
        cookie_samesite: str = "lax",
        token_length: int = 32,
    ):
        super().__init__(app)
        self.cookie_name = cookie_name
        self.header_name = header_name
        self.cookie_path = cookie_path
        self.cookie_domain = cookie_domain
        self.cookie_secure = cookie_secure
        self.cookie_httponly = cookie_httponly
        self.cookie_samesite = cookie_samesite
        self.token_length = token_length
    
    async def dispatch(self, request: Request, call_next):
        """Process request with CSRF protection."""
        # Skip CSRF check for exempt paths
        if self._is_exempt_path(request.url.path):
            return await call_next(request)
        
        # Skip CSRF check for safe methods
        if request.method not in self.PROTECTED_METHODS:
            response = await call_next(request)
            # Ensure CSRF cookie is set for future requests
            self._ensure_csrf_cookie(request, response)
            return response
        
        # Validate CSRF token for protected methods
        if not await self._validate_csrf_token(request):
            logger.warning(
                f"CSRF validation failed for {request.method} {request.url.path} "
                f"from {request.client.host if request.client else 'unknown'}"
            )
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={
                    "error": "CSRF validation failed",
                    "code": "CSRF_TOKEN_INVALID",
                    "message": "Invalid or missing CSRF token"
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Ensure CSRF cookie is set
        self._ensure_csrf_cookie(request, response)
        
        return response
    
    def _is_exempt_path(self, path: str) -> bool:
        """Check if path is exempt from CSRF protection."""
        # Exact match
        if path in self.EXEMPT_PATHS:
            return True
        
        # Prefix match for certain paths
        exempt_prefixes = ["/static/", "/favicon", "/_next/"]
        return any(path.startswith(prefix) for prefix in exempt_prefixes)
    
    async def _validate_csrf_token(self, request: Request) -> bool:
        """Validate CSRF token from cookie matches header/form."""
        # Get token from cookie
        cookie_token = request.cookies.get(self.cookie_name)
        if not cookie_token:
            return False
        
        # Get token from header
        header_token = request.headers.get(self.header_name)
        
        # If no header token, try to get from form data (for form submissions)
        if not header_token and request.method == "POST":
            try:
                form = await request.form()
                header_token = form.get("csrf_token")
            except Exception:
                # Not form data, that's ok
                pass
        
        if not header_token:
            return False
        
        # Constant-time comparison
        return secrets.compare_digest(cookie_token, header_token)
    
    def _ensure_csrf_cookie(self, request: Request, response: Response) -> None:
        """Ensure CSRF cookie is set on response."""
        # Check if cookie already exists
        existing_token = request.cookies.get(self.cookie_name)
        
        # Only set new cookie if none exists
        if not existing_token:
            new_token = self._generate_csrf_token()
            self._set_csrf_cookie(response, new_token)
            logger.debug("Generated new CSRF token")
    
    def _generate_csrf_token(self) -> str:
        """Generate a new CSRF token."""
        return secrets.token_urlsafe(self.token_length)
    
    def _set_csrf_cookie(self, response: Response, token: str) -> None:
        """Set CSRF cookie on response."""
        response.set_cookie(
            key=self.cookie_name,
            value=token,
            path=self.cookie_path,
            domain=self.cookie_domain,
            secure=self.cookie_secure,
            httponly=self.cookie_httponly,
            samesite=self.cookie_samesite,
            max_age=86400,  # 24 hours
        )


def get_csrf_token(request: Request) -> Optional[str]:
    """Get CSRF token from request cookies."""
    return request.cookies.get("csrf_token")


async def csrf_token_endpoint(request: Request) -> dict:
    """Endpoint to get current CSRF token."""
    token = get_csrf_token(request)
    if not token:
        # Generate new token if none exists
        token = secrets.token_urlsafe(32)
        # Token will be set by middleware
        request.state.new_csrf_token = token
    
    return {"csrfToken": token}


# CSRF exempt decorator for specific endpoints
def csrf_exempt(func):
    """Decorator to exempt an endpoint from CSRF protection."""
    func._csrf_exempt = True
    return func