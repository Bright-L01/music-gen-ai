"""
Security headers middleware for Music Gen AI.

Implements comprehensive security headers based on 2024 OWASP recommendations
including CSP, HSTS, and other essential security headers.
"""

import os
from typing import Callable, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from ...core.config import get_config


class SecurityHeadersConfig:
    """Configuration for security headers."""

    def __init__(self):
        config = get_config()
        self.environment = getattr(config, "environment", "development")
        self.domain = getattr(config, "domain", "localhost")
        self.enable_hsts = self.environment == "production"
        self.enable_preload = getattr(config, "hsts_preload", False)

        # CSP configuration
        self.csp_report_uri = getattr(config, "csp_report_uri", None)
        self.allowed_origins = getattr(config, "cors_origins", ["http://localhost:3000"])

    def get_content_security_policy(self) -> str:
        """Generate Content Security Policy header value."""
        # Base CSP policy - very restrictive
        policies = [
            "default-src 'none'",
            "script-src 'self'",
            "style-src 'self' 'unsafe-inline'",  # 'unsafe-inline' for styled-components
            "img-src 'self' data: https:",  # Allow data URLs and HTTPS images
            "font-src 'self' https:",
            "connect-src 'self'",
            "media-src 'self' blob:",  # Allow blob URLs for audio playback
            "object-src 'none'",
            "base-uri 'self'",
            "form-action 'self'",
            "frame-ancestors 'none'",  # Equivalent to X-Frame-Options: DENY
            "upgrade-insecure-requests",  # Upgrade HTTP to HTTPS
        ]

        # Add WebSocket support
        ws_origins = []
        for origin in self.allowed_origins:
            ws_origin = origin.replace("http://", "ws://").replace("https://", "wss://")
            ws_origins.append(ws_origin)

        if ws_origins:
            connect_src = f"connect-src 'self' {' '.join(self.allowed_origins + ws_origins)}"
            policies[4] = connect_src  # Replace connect-src

        # Add CSP report URI if configured
        if self.csp_report_uri:
            policies.append(f"report-uri {self.csp_report_uri}")

        return "; ".join(policies)

    def get_hsts_header(self) -> Optional[str]:
        """Generate HSTS header value."""
        if not self.enable_hsts:
            return None

        # 1 year max-age with subdomains
        hsts = "max-age=31536000; includeSubDomains"

        # Add preload directive if enabled
        if self.enable_preload:
            hsts += "; preload"

        return hsts

    def get_permissions_policy(self) -> str:
        """Generate Permissions Policy header."""
        # Restrictive permissions policy
        policies = [
            "accelerometer=()",
            "ambient-light-sensor=()",
            "autoplay=(self)",  # Allow autoplay for audio
            "battery=()",
            "camera=()",
            "cross-origin-isolated=()",
            "display-capture=()",
            "document-domain=()",
            "encrypted-media=()",
            "execution-while-not-rendered=()",
            "execution-while-out-of-viewport=()",
            "fullscreen=(self)",  # Allow fullscreen for media
            "geolocation=()",
            "gyroscope=()",
            "magnetometer=()",
            "microphone=()",  # May need microphone for recording features
            "midi=()",
            "navigation-override=()",
            "payment=()",
            "picture-in-picture=()",
            "publickey-credentials-get=()",
            "screen-wake-lock=()",
            "sync-xhr=()",
            "usb=()",
            "web-share=()",
            "xr-spatial-tracking=()",
        ]

        return ", ".join(policies)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware that adds comprehensive security headers to all responses."""

    def __init__(self, app: ASGIApp, config: Optional[SecurityHeadersConfig] = None):
        super().__init__(app)
        self.config = config or SecurityHeadersConfig()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add security headers to the response."""
        response = await call_next(request)

        # Content Security Policy (CSP)
        response.headers["Content-Security-Policy"] = self.config.get_content_security_policy()

        # HTTP Strict Transport Security (HSTS) - only for HTTPS
        if self.config.enable_hsts and request.url.scheme == "https":
            hsts_header = self.config.get_hsts_header()
            if hsts_header:
                response.headers["Strict-Transport-Security"] = hsts_header

        # X-Content-Type-Options
        response.headers["X-Content-Type-Options"] = "nosniff"

        # Referrer Policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Permissions Policy (formerly Feature Policy)
        response.headers["Permissions-Policy"] = self.config.get_permissions_policy()

        # Cross-Origin Embedder Policy
        response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"

        # Cross-Origin Opener Policy
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"

        # Cross-Origin Resource Policy
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"

        # Remove server header that might expose server information
        response.headers.pop("Server", None)

        # Cache-Control for security-sensitive responses
        if request.url.path.startswith("/api/auth"):
            response.headers[
                "Cache-Control"
            ] = "no-store, no-cache, must-revalidate, proxy-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"

        return response


class DevelopmentSecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Relaxed security headers for development environment."""

    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add relaxed security headers for development."""
        response = await call_next(request)

        # Relaxed CSP for development
        csp_policies = [
            "default-src 'self' 'unsafe-inline'",
            "script-src 'self' 'unsafe-inline' http://localhost:* ws://localhost:*",
            "style-src 'self' 'unsafe-inline'",
            "img-src 'self' data: blob: http://localhost:*",
            "font-src 'self' data:",
            "connect-src 'self' http://localhost:* ws://localhost:*",
            "media-src 'self' blob:",
            "object-src 'none'",
            "base-uri 'self'",
            "form-action 'self'",
        ]
        response.headers["Content-Security-Policy"] = "; ".join(csp_policies)

        # Basic security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer-when-downgrade"

        # Remove server header
        response.headers.pop("Server", None)

        return response


def create_security_headers_middleware(app: ASGIApp) -> ASGIApp:
    """Create appropriate security headers middleware based on environment."""
    config = SecurityHeadersConfig()

    if config.environment == "development":
        return DevelopmentSecurityHeadersMiddleware(app)
    else:
        return SecurityHeadersMiddleware(app, config)


# Content Security Policy violation reporting endpoint
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import json
import logging

csp_router = APIRouter()
logger = logging.getLogger(__name__)


class CSPReport(BaseModel):
    """CSP violation report structure."""

    csp_report: Dict[str, Any]


@csp_router.post("/csp-report")
async def handle_csp_report(report: CSPReport):
    """Handle CSP violation reports."""
    try:
        violation = report.csp_report

        # Log the violation
        logger.warning(
            f"CSP Violation: "
            f"document-uri={violation.get('document-uri')}, "
            f"violated-directive={violation.get('violated-directive')}, "
            f"blocked-uri={violation.get('blocked-uri')}, "
            f"source-file={violation.get('source-file')}, "
            f"line-number={violation.get('line-number')}"
        )

        # In production, you might want to:
        # - Store violations in database
        # - Send alerts for critical violations
        # - Track violation patterns

        return {"status": "received"}

    except Exception as e:
        logger.error(f"Failed to process CSP report: {e}")
        raise HTTPException(status_code=400, detail="Invalid CSP report")


# Security headers validation utility
class SecurityHeadersValidator:
    """Utility to validate security headers implementation."""

    @staticmethod
    def validate_csp(csp_header: str) -> Dict[str, Any]:
        """Validate CSP header and return analysis."""
        directives = {}
        issues = []

        for directive in csp_header.split(";"):
            directive = directive.strip()
            if not directive:
                continue

            parts = directive.split()
            if parts:
                directive_name = parts[0]
                sources = parts[1:] if len(parts) > 1 else []
                directives[directive_name] = sources

        # Check for common issues
        if "default-src" not in directives:
            issues.append("Missing default-src directive")

        if "'unsafe-eval'" in str(directives):
            issues.append("'unsafe-eval' allows arbitrary code execution")

        if "'unsafe-inline'" in directives.get("script-src", []):
            issues.append("'unsafe-inline' in script-src reduces XSS protection")

        if "frame-ancestors" not in directives:
            issues.append("Missing frame-ancestors directive (clickjacking protection)")

        return {
            "directives": directives,
            "issues": issues,
            "score": max(0, 100 - len(issues) * 20),  # Simple scoring
        }

    @staticmethod
    def check_headers_presence(headers: Dict[str, str]) -> Dict[str, Any]:
        """Check presence of important security headers."""
        required_headers = {
            "Content-Security-Policy": "Critical",
            "X-Content-Type-Options": "High",
            "Referrer-Policy": "Medium",
            "Permissions-Policy": "Medium",
            "Strict-Transport-Security": "High",  # For HTTPS only
        }

        present = {}
        missing = {}

        for header, importance in required_headers.items():
            if header in headers:
                present[header] = {"value": headers[header], "importance": importance}
            else:
                missing[header] = importance

        return {
            "present": present,
            "missing": missing,
            "score": len(present) / len(required_headers) * 100,
        }
