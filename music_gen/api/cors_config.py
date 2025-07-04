"""
CORS Configuration for Music Gen AI API.
Implements secure cross-origin resource sharing with environment-based whitelisting.
"""

import os
from typing import List, Set
from urllib.parse import urlparse
import logging

from music_gen.utils.logging import get_logger

logger = get_logger(__name__)


class CORSConfig:
    """Centralized CORS configuration with origin validation."""
    
    # Default allowed origins for different environments
    DEVELOPMENT_ORIGINS = [
        "http://localhost:3000",      # Next.js dev server
        "http://localhost:3001",      # Alternative Next.js port
        "http://localhost:8000",      # API documentation
        "http://localhost:8080",      # Alternative API port
        "http://127.0.0.1:3000",      # IP-based localhost
        "http://127.0.0.1:3001",      
        "http://127.0.0.1:8000",
        "http://127.0.0.1:8080",
        "http://[::1]:3000",         # IPv6 localhost
        "http://[::1]:8000",
    ]
    
    STAGING_ORIGINS = [
        "https://staging.musicgen.ai",
        "https://staging-api.musicgen.ai",
        "https://preview.musicgen.ai",
        "https://beta.musicgen.ai",
    ]
    
    PRODUCTION_ORIGINS = [
        "https://musicgen.ai",
        "https://www.musicgen.ai",
        "https://app.musicgen.ai",
        "https://api.musicgen.ai",
    ]
    
    # Allowed methods for CORS
    ALLOWED_METHODS = ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"]
    
    # Allowed headers
    ALLOWED_HEADERS = [
        "Accept",
        "Accept-Language",
        "Content-Language",
        "Content-Type",
        "Authorization",
        "X-Requested-With",
        "X-Request-ID",
        "X-CSRF-Token",
        "Cache-Control",
        "Pragma",
    ]
    
    # Exposed headers (accessible to client)
    EXPOSE_HEADERS = [
        "X-Request-ID",
        "X-RateLimit-Limit",
        "X-RateLimit-Remaining",
        "X-RateLimit-Reset",
        "Content-Disposition",
    ]
    
    def __init__(self):
        """Initialize CORS configuration from environment."""
        self.environment = os.getenv("ENVIRONMENT", "development").lower()
        self.allowed_origins = self._load_allowed_origins()
        self.allow_credentials = True
        self.max_age = 86400  # 24 hours for preflight cache
        
        logger.info(
            f"CORS initialized for {self.environment} environment with "
            f"{len(self.allowed_origins)} allowed origins"
        )
    
    def _load_allowed_origins(self) -> Set[str]:
        """Load allowed origins based on environment."""
        origins = set()
        
        # Load environment-specific defaults
        if self.environment == "development":
            origins.update(self.DEVELOPMENT_ORIGINS)
        elif self.environment == "staging":
            origins.update(self.STAGING_ORIGINS)
            # In staging, also allow specific development origins for testing
            staging_dev_origins = os.getenv("STAGING_DEV_ORIGINS", "").split(",")
            origins.update(filter(None, [o.strip() for o in staging_dev_origins]))
        elif self.environment == "production":
            origins.update(self.PRODUCTION_ORIGINS)
        else:
            # Unknown environment - be restrictive
            logger.warning(f"Unknown environment: {self.environment}")
        
        # Load additional origins from environment variable
        env_origins = os.getenv("ALLOWED_ORIGINS", "").split(",")
        env_origins = filter(None, [o.strip() for o in env_origins])
        origins.update(env_origins)
        
        # Load domain-based origins if configured
        allowed_domains = os.getenv("ALLOWED_DOMAINS", "").split(",")
        allowed_domains = filter(None, [d.strip() for d in allowed_domains])
        for domain in allowed_domains:
            origins.add(f"https://{domain}")
            origins.add(f"https://www.{domain}")
            if self.environment == "development":
                origins.add(f"http://{domain}")
                origins.add(f"http://www.{domain}")
        
        # Remove empty strings and validate origins
        validated_origins = set()
        for origin in origins:
            if origin and self._validate_origin(origin):
                validated_origins.add(origin)
            elif origin:
                logger.warning(f"Invalid origin format: {origin}")
        
        return validated_origins
    
    def _validate_origin(self, origin: str) -> bool:
        """Validate origin format."""
        try:
            parsed = urlparse(origin)
            # Must have scheme and netloc
            if not parsed.scheme or not parsed.netloc:
                return False
            # No path, query, or fragment allowed in origin
            if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
                return False
            # Scheme must be http or https
            if parsed.scheme not in ("http", "https"):
                return False
            # In production, only allow HTTPS
            if self.environment == "production" and parsed.scheme != "https":
                logger.warning(f"HTTP origin not allowed in production: {origin}")
                return False
            return True
        except Exception:
            return False
    
    def is_origin_allowed(self, origin: str) -> bool:
        """Check if an origin is allowed."""
        if not origin:
            return False
        
        # Exact match
        if origin in self.allowed_origins:
            return True
        
        # Check wildcard subdomains if configured
        if os.getenv("ALLOW_SUBDOMAIN_WILDCARDS", "false").lower() == "true":
            for allowed in self.allowed_origins:
                if self._match_wildcard_origin(origin, allowed):
                    return True
        
        return False
    
    def _match_wildcard_origin(self, origin: str, pattern: str) -> bool:
        """Match origin against wildcard pattern (e.g., https://*.example.com)."""
        if "*" not in pattern:
            return origin == pattern
        
        # Parse both URLs
        try:
            origin_parsed = urlparse(origin)
            pattern_parsed = urlparse(pattern)
            
            # Schemes must match
            if origin_parsed.scheme != pattern_parsed.scheme:
                return False
            
            # Handle wildcard in hostname
            pattern_host = pattern_parsed.hostname
            origin_host = origin_parsed.hostname
            
            if pattern_host and origin_host and pattern_host.startswith("*."):
                # Extract base domain from pattern
                base_domain = pattern_host[2:]  # Remove "*."
                # Check if origin ends with base domain
                return origin_host == base_domain or origin_host.endswith(f".{base_domain}")
            
            return False
        except Exception:
            return False
    
    def get_cors_options(self) -> dict:
        """Get CORS middleware options for FastAPI."""
        return {
            "allow_origins": list(self.allowed_origins),
            "allow_credentials": self.allow_credentials,
            "allow_methods": self.ALLOWED_METHODS,
            "allow_headers": self.ALLOWED_HEADERS,
            "expose_headers": self.EXPOSE_HEADERS,
            "max_age": self.max_age,
        }
    
    def validate_origin_header(self, origin: str) -> bool:
        """
        Validate Origin header for manual CORS handling.
        
        Args:
            origin: The Origin header value
            
        Returns:
            True if origin is allowed, False otherwise
        """
        if not origin:
            logger.debug("No Origin header provided")
            return False
        
        allowed = self.is_origin_allowed(origin)
        
        if not allowed:
            logger.warning(
                f"CORS request from unauthorized origin: {origin} "
                f"(environment: {self.environment})"
            )
        
        return allowed
    
    def get_preflight_headers(self, origin: str, request_method: str = None, 
                            request_headers: str = None) -> dict:
        """
        Get headers for CORS preflight response.
        
        Args:
            origin: The Origin header value
            request_method: The Access-Control-Request-Method header
            request_headers: The Access-Control-Request-Headers header
            
        Returns:
            Dictionary of CORS headers
        """
        if not self.validate_origin_header(origin):
            return {}
        
        headers = {
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Max-Age": str(self.max_age),
            "Vary": "Origin",
        }
        
        # Validate requested method
        if request_method:
            if request_method in self.ALLOWED_METHODS:
                headers["Access-Control-Allow-Methods"] = ", ".join(self.ALLOWED_METHODS)
            else:
                logger.warning(f"CORS preflight requested disallowed method: {request_method}")
                return {}
        
        # Validate requested headers
        if request_headers:
            requested = [h.strip() for h in request_headers.split(",")]
            # Check if all requested headers are allowed
            allowed_lower = [h.lower() for h in self.ALLOWED_HEADERS]
            for header in requested:
                if header.lower() not in allowed_lower:
                    logger.warning(f"CORS preflight requested disallowed header: {header}")
                    return {}
            headers["Access-Control-Allow-Headers"] = ", ".join(self.ALLOWED_HEADERS)
        
        return headers
    
    def get_response_headers(self, origin: str) -> dict:
        """
        Get CORS headers for regular responses.
        
        Args:
            origin: The Origin header value
            
        Returns:
            Dictionary of CORS headers
        """
        if not self.validate_origin_header(origin):
            return {}
        
        return {
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Expose-Headers": ", ".join(self.EXPOSE_HEADERS),
            "Vary": "Origin",
        }


# Global CORS configuration instance
cors_config = CORSConfig()


# Convenience function for FastAPI
def get_cors_config() -> dict:
    """Get CORS configuration for FastAPI CORSMiddleware."""
    return cors_config.get_cors_options()