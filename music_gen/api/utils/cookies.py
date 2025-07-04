"""
Secure Cookie Management Utilities for Music Gen AI API.
Handles cookie setting, parsing, and security configuration.
"""

import os
from typing import Optional, Dict, Any
from datetime import datetime, timedelta, timezone

from fastapi import Response, Request
from fastapi.security.utils import get_authorization_scheme_param

from music_gen.core.config import get_config
from music_gen.utils.logging import get_logger

logger = get_logger(__name__)
config = get_config()


class CookieConfig:
    """Cookie configuration based on environment."""
    
    def __init__(self):
        self.environment = os.getenv("ENVIRONMENT", "development").lower()
        self.domain = os.getenv("COOKIE_DOMAIN", None)
        self.secure = self.environment == "production"
        self.samesite = "lax"  # Good balance between security and usability
        self.path = "/"
        
        # Development overrides
        if self.environment == "development":
            self.secure = False  # Allow HTTP in development
            self.domain = None   # No domain restriction in dev
        
        # Staging configuration
        elif self.environment == "staging":
            self.secure = True   # Always use HTTPS in staging
            if not self.domain:
                self.domain = os.getenv("STAGING_DOMAIN", None)
    
    def get_cookie_settings(self, max_age: Optional[int] = None) -> Dict[str, Any]:
        """Get cookie settings for set_cookie."""
        settings = {
            "path": self.path,
            "secure": self.secure,
            "httponly": True,  # Default to httpOnly for security
            "samesite": self.samesite,
        }
        
        if self.domain:
            settings["domain"] = self.domain
        
        if max_age is not None:
            settings["max_age"] = max_age
        
        return settings


# Global cookie configuration
cookie_config = CookieConfig()


class SecureCookieManager:
    """Manages secure cookie operations."""
    
    # Cookie names
    AUTH_TOKEN_COOKIE = "auth_token"
    REFRESH_TOKEN_COOKIE = "refresh_token"
    CSRF_TOKEN_COOKIE = "csrf_token"
    SESSION_ID_COOKIE = "session_id"
    
    @staticmethod
    def set_auth_cookies(
        response: Response,
        access_token: str,
        refresh_token: str,
        access_token_expires: int = 900,  # 15 minutes
        refresh_token_expires: int = 604800,  # 7 days
    ) -> None:
        """Set authentication cookies on response."""
        # Access token cookie
        access_settings = cookie_config.get_cookie_settings(access_token_expires)
        access_settings["httponly"] = True  # Always httpOnly for auth tokens
        
        response.set_cookie(
            key=SecureCookieManager.AUTH_TOKEN_COOKIE,
            value=access_token,
            **access_settings
        )
        
        # Refresh token cookie
        refresh_settings = cookie_config.get_cookie_settings(refresh_token_expires)
        refresh_settings["httponly"] = True
        refresh_settings["path"] = "/api/auth/refresh"  # Restrict to refresh endpoint
        
        response.set_cookie(
            key=SecureCookieManager.REFRESH_TOKEN_COOKIE,
            value=refresh_token,
            **refresh_settings
        )
        
        logger.debug(
            f"Set auth cookies for environment: {cookie_config.environment}, "
            f"secure: {cookie_config.secure}, domain: {cookie_config.domain}"
        )
    
    @staticmethod
    def set_csrf_cookie(response: Response, csrf_token: str) -> None:
        """Set CSRF token cookie."""
        settings = cookie_config.get_cookie_settings(86400)  # 24 hours
        settings["httponly"] = False  # CSRF cookie must be readable by JavaScript
        
        response.set_cookie(
            key=SecureCookieManager.CSRF_TOKEN_COOKIE,
            value=csrf_token,
            **settings
        )
    
    @staticmethod
    def set_session_cookie(
        response: Response,
        session_id: str,
        expires_in: int = 3600  # 1 hour
    ) -> None:
        """Set session ID cookie."""
        settings = cookie_config.get_cookie_settings(expires_in)
        settings["httponly"] = True
        
        response.set_cookie(
            key=SecureCookieManager.SESSION_ID_COOKIE,
            value=session_id,
            **settings
        )
    
    @staticmethod
    def clear_auth_cookies(response: Response) -> None:
        """Clear all authentication-related cookies."""
        # Get base settings with max_age=0 to delete cookies
        settings = cookie_config.get_cookie_settings(0)
        
        # Clear access token
        response.set_cookie(
            key=SecureCookieManager.AUTH_TOKEN_COOKIE,
            value="",
            **settings
        )
        
        # Clear refresh token (with its restricted path)
        refresh_settings = settings.copy()
        refresh_settings["path"] = "/api/auth/refresh"
        response.set_cookie(
            key=SecureCookieManager.REFRESH_TOKEN_COOKIE,
            value="",
            **refresh_settings
        )
        
        # Clear session cookie
        response.set_cookie(
            key=SecureCookieManager.SESSION_ID_COOKIE,
            value="",
            **settings
        )
        
        # Clear CSRF cookie
        response.set_cookie(
            key=SecureCookieManager.CSRF_TOKEN_COOKIE,
            value="",
            **settings
        )
        
        logger.debug("Cleared all auth cookies")
    
    @staticmethod
    def get_auth_token_from_request(request: Request) -> Optional[str]:
        """
        Extract auth token from request.
        Checks both cookie and Authorization header for compatibility.
        """
        # First, try to get from cookie
        token = request.cookies.get(SecureCookieManager.AUTH_TOKEN_COOKIE)
        if token:
            return token
        
        # Fallback to Authorization header for API clients
        authorization = request.headers.get("Authorization")
        if authorization:
            scheme, token = get_authorization_scheme_param(authorization)
            if scheme.lower() == "bearer":
                return token
        
        return None
    
    @staticmethod
    def get_refresh_token_from_request(request: Request) -> Optional[str]:
        """Extract refresh token from request cookies."""
        return request.cookies.get(SecureCookieManager.REFRESH_TOKEN_COOKIE)
    
    @staticmethod
    def get_session_id_from_request(request: Request) -> Optional[str]:
        """Extract session ID from request cookies."""
        return request.cookies.get(SecureCookieManager.SESSION_ID_COOKIE)
    
    @staticmethod
    def create_secure_cookie_header(
        name: str,
        value: str,
        max_age: Optional[int] = None,
        **kwargs
    ) -> str:
        """Create a Set-Cookie header value with security settings."""
        settings = cookie_config.get_cookie_settings(max_age)
        settings.update(kwargs)
        
        # Build cookie string
        cookie_parts = [f"{name}={value}"]
        
        if settings.get("domain"):
            cookie_parts.append(f"Domain={settings['domain']}")
        
        if settings.get("path"):
            cookie_parts.append(f"Path={settings['path']}")
        
        if settings.get("max_age") is not None:
            cookie_parts.append(f"Max-Age={settings['max_age']}")
            # Also set Expires for older browsers
            expires = datetime.now(timezone.utc) + timedelta(seconds=settings['max_age'])
            cookie_parts.append(f"Expires={expires.strftime('%a, %d %b %Y %H:%M:%S GMT')}")
        
        if settings.get("secure"):
            cookie_parts.append("Secure")
        
        if settings.get("httponly"):
            cookie_parts.append("HttpOnly")
        
        if settings.get("samesite"):
            cookie_parts.append(f"SameSite={settings['samesite']}")
        
        return "; ".join(cookie_parts)


# Utility functions for easy access
def set_auth_cookies(
    response: Response,
    access_token: str,
    refresh_token: str,
    access_expires: Optional[int] = None,
    refresh_expires: Optional[int] = None
) -> None:
    """Convenience function to set auth cookies."""
    SecureCookieManager.set_auth_cookies(
        response,
        access_token,
        refresh_token,
        access_expires or 900,
        refresh_expires or 604800
    )


def clear_auth_cookies(response: Response) -> None:
    """Convenience function to clear auth cookies."""
    SecureCookieManager.clear_auth_cookies(response)


def get_auth_token(request: Request) -> Optional[str]:
    """Convenience function to get auth token."""
    return SecureCookieManager.get_auth_token_from_request(request)


def get_refresh_token(request: Request) -> Optional[str]:
    """Convenience function to get refresh token."""
    return SecureCookieManager.get_refresh_token_from_request(request)