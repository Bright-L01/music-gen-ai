"""
Production-ready Rate Limiting Middleware for Music Gen AI API.
Implements proxy-aware rate limiting with Redis backend and comprehensive security features.
"""

import os
import time
import ipaddress
from collections import defaultdict
from typing import Optional, Dict, Tuple, List, Set, Union, Callable
from datetime import datetime, timezone
from enum import Enum

import redis
from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, validator

from music_gen.core.config import get_config
from music_gen.utils.logging import get_logger

logger = get_logger(__name__)
config = get_config()


class RateLimitTier(str, Enum):
    """Rate limit tiers for different user types"""
    FREE = "free"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"
    INTERNAL = "internal"
    ADMIN = "admin"


class RateLimitConfig(BaseModel):
    """Rate limit configuration"""
    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    burst_size: int = 10  # Allow short bursts
    
    @validator('burst_size')
    def validate_burst_size(cls, v, values):
        if 'requests_per_minute' in values and v > values['requests_per_minute']:
            raise ValueError('Burst size cannot exceed requests per minute')
        return v


# Default rate limits per tier
RATE_LIMIT_TIERS: Dict[RateLimitTier, RateLimitConfig] = {
    RateLimitTier.FREE: RateLimitConfig(
        requests_per_minute=30,
        requests_per_hour=500,
        requests_per_day=5000,
        burst_size=5
    ),
    RateLimitTier.PREMIUM: RateLimitConfig(
        requests_per_minute=60,
        requests_per_hour=2000,
        requests_per_day=20000,
        burst_size=10
    ),
    RateLimitTier.ENTERPRISE: RateLimitConfig(
        requests_per_minute=300,
        requests_per_hour=10000,
        requests_per_day=100000,
        burst_size=50
    ),
    RateLimitTier.INTERNAL: RateLimitConfig(
        requests_per_minute=1000,
        requests_per_hour=50000,
        requests_per_day=1000000,
        burst_size=100
    ),
    RateLimitTier.ADMIN: RateLimitConfig(
        requests_per_minute=1000,
        requests_per_hour=50000,
        requests_per_day=1000000,
        burst_size=100
    ),
}


class IPExtractor:
    """
    Extracts client IP address from request with proxy support.
    Implements security best practices for handling proxy headers.
    """
    
    def __init__(
        self,
        trusted_proxies: Optional[List[str]] = None,
        trusted_proxy_headers: List[str] = None,
        trust_all_proxies: bool = False
    ):
        """
        Args:
            trusted_proxies: List of trusted proxy IPs/CIDRs
            trusted_proxy_headers: Headers to check for client IP
            trust_all_proxies: Trust all proxies (DANGEROUS - only for testing)
        """
        self.trust_all_proxies = trust_all_proxies
        self.trusted_proxy_headers = trusted_proxy_headers or [
            "X-Forwarded-For",
            "X-Real-IP",
            "CF-Connecting-IP",  # Cloudflare
            "X-Client-IP",
            "X-Forwarded",
            "Forwarded-For",
            "Forwarded"
        ]
        
        # Parse trusted proxies
        self.trusted_networks: List[Union[ipaddress.IPv4Network, ipaddress.IPv6Network]] = []
        if trusted_proxies:
            for proxy in trusted_proxies:
                try:
                    # Support both individual IPs and CIDR notation
                    network = ipaddress.ip_network(proxy, strict=False)
                    self.trusted_networks.append(network)
                except ValueError:
                    logger.warning(f"Invalid trusted proxy: {proxy}")
    
    def extract_ip(self, request: Request) -> str:
        """
        Extract client IP from request, handling proxy headers securely.
        
        Returns:
            Client IP address
        """
        # Get direct connection IP
        direct_ip = request.client.host if request.client else None
        
        if not direct_ip:
            return "unknown"
        
        # If not behind a proxy, return direct IP
        if not self._is_trusted_proxy(direct_ip):
            return direct_ip
        
        # Check proxy headers in order of preference
        for header in self.trusted_proxy_headers:
            header_value = request.headers.get(header)
            if header_value:
                # Handle different header formats
                client_ip = self._parse_proxy_header(header, header_value)
                if client_ip and self._is_valid_ip(client_ip):
                    logger.debug(f"Extracted IP {client_ip} from {header} header")
                    return client_ip
        
        # Fallback to direct connection if no valid proxy header
        return direct_ip
    
    def _is_trusted_proxy(self, ip: str) -> bool:
        """Check if IP is from a trusted proxy."""
        if self.trust_all_proxies:
            return True
        
        if not self.trusted_networks:
            return False
        
        try:
            ip_addr = ipaddress.ip_address(ip)
            return any(ip_addr in network for network in self.trusted_networks)
        except ValueError:
            return False
    
    def _parse_proxy_header(self, header_name: str, header_value: str) -> Optional[str]:
        """Parse proxy header to extract client IP."""
        if not header_value:
            return None
        
        # X-Forwarded-For can contain multiple IPs
        if header_name in ["X-Forwarded-For", "X-Forwarded", "Forwarded-For"]:
            # Take the first IP (original client)
            ips = [ip.strip() for ip in header_value.split(",")]
            return ips[0] if ips else None
        
        # Forwarded header (RFC 7239)
        elif header_name == "Forwarded":
            # Parse "for=192.0.2.123" format
            for part in header_value.split(";"):
                if part.strip().startswith("for="):
                    ip = part.split("=", 1)[1].strip('"')
                    # Remove port if present
                    if "[" in ip and "]" in ip:
                        # IPv6 with port
                        ip = ip[ip.find("[")+1:ip.find("]")]
                    elif ":" in ip and ip.count(":") == 1:
                        # IPv4 with port
                        ip = ip.split(":")[0]
                    return ip
        
        # Single IP headers
        else:
            return header_value.strip()
        
        return None
    
    def _is_valid_ip(self, ip: str) -> bool:
        """Validate IP address."""
        try:
            ipaddress.ip_address(ip)
            return True
        except ValueError:
            return False


class RateLimitStorage:
    """Abstract storage backend for rate limit data."""
    
    def increment(
        self,
        key: str,
        window: int,
        limit: int
    ) -> Tuple[int, int]:
        """
        Increment counter and return current count and TTL.
        
        Args:
            key: Storage key
            window: Time window in seconds
            limit: Maximum allowed requests
            
        Returns:
            Tuple of (current_count, ttl_seconds)
        """
        raise NotImplementedError
    
    def get_count(self, key: str) -> int:
        """Get current count for key."""
        raise NotImplementedError
    
    def reset(self, key: str):
        """Reset counter for key."""
        raise NotImplementedError


class RedisRateLimitStorage(RateLimitStorage):
    """Redis-based rate limit storage."""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
    
    def increment(
        self,
        key: str,
        window: int,
        limit: int
    ) -> Tuple[int, int]:
        """Increment using Redis with atomic operations."""
        try:
            pipe = self.redis.pipeline()
            pipe.incr(key)
            pipe.ttl(key)
            pipe.expire(key, window)
            results = pipe.execute()
            
            count = results[0]
            ttl = results[1] if results[1] > 0 else window
            
            return count, ttl
        except Exception as e:
            logger.error(f"Redis rate limit error: {e}")
            # Fail open - don't block on Redis errors
            return 0, window
    
    def get_count(self, key: str) -> int:
        """Get current count."""
        try:
            count = self.redis.get(key)
            return int(count) if count else 0
        except Exception:
            return 0
    
    def reset(self, key: str):
        """Reset counter."""
        try:
            self.redis.delete(key)
        except Exception as e:
            logger.error(f"Redis reset error: {e}")


class MemoryRateLimitStorage(RateLimitStorage):
    """In-memory rate limit storage (for development/testing)."""
    
    def __init__(self):
        # Storage: key -> (count, expiry_time)
        self.storage: Dict[str, Tuple[int, float]] = {}
        self._cleanup_interval = 300  # 5 minutes
        self._last_cleanup = time.time()
    
    def increment(
        self,
        key: str,
        window: int,
        limit: int
    ) -> Tuple[int, int]:
        """Increment counter in memory."""
        current_time = time.time()
        
        # Periodic cleanup
        if current_time - self._last_cleanup > self._cleanup_interval:
            self._cleanup_expired()
        
        if key in self.storage:
            count, expiry = self.storage[key]
            if current_time < expiry:
                # Increment existing counter
                count += 1
                ttl = int(expiry - current_time)
            else:
                # Reset expired counter
                count = 1
                expiry = current_time + window
                ttl = window
        else:
            # New counter
            count = 1
            expiry = current_time + window
            ttl = window
        
        self.storage[key] = (count, expiry)
        return count, ttl
    
    def get_count(self, key: str) -> int:
        """Get current count."""
        if key not in self.storage:
            return 0
        
        count, expiry = self.storage[key]
        if time.time() < expiry:
            return count
        return 0
    
    def reset(self, key: str):
        """Reset counter."""
        if key in self.storage:
            del self.storage[key]
    
    def _cleanup_expired(self):
        """Remove expired entries."""
        current_time = time.time()
        expired_keys = [
            key for key, (_, expiry) in self.storage.items()
            if current_time >= expiry
        ]
        for key in expired_keys:
            del self.storage[key]
        self._last_cleanup = current_time


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Advanced rate limiting middleware with proxy support and tiered limits.
    
    Features:
    - Proxy-aware IP extraction
    - Tiered rate limits based on user type
    - Redis backend with fallback to memory
    - Bypass for internal services
    - Comprehensive security logging
    - Burst protection
    """
    
    # Paths exempt from rate limiting
    EXEMPT_PATHS = {
        "/health",
        "/health/",
        "/metrics",
        "/docs",
        "/redoc",
        "/openapi.json",
    }
    
    # Internal service identifiers
    INTERNAL_SERVICE_HEADERS = {
        "X-Internal-Service",
        "X-Service-Name",
        "X-API-Key",
    }
    
    def __init__(
        self,
        app,
        redis_client: Optional[redis.Redis] = None,
        trusted_proxies: Optional[List[str]] = None,
        enable_proxy_headers: bool = True,
        internal_service_keys: Optional[Set[str]] = None,
        default_tier: RateLimitTier = RateLimitTier.FREE
    ):
        """
        Args:
            app: FastAPI application
            redis_client: Redis client for storage
            trusted_proxies: List of trusted proxy IPs/CIDRs
            enable_proxy_headers: Enable proxy header parsing
            internal_service_keys: Valid internal service API keys
            default_tier: Default rate limit tier
        """
        super().__init__(app)
        
        # Initialize storage backend
        if redis_client:
            self.storage = RedisRateLimitStorage(redis_client)
            logger.info("Using Redis for rate limit storage")
        else:
            self.storage = MemoryRateLimitStorage()
            logger.warning("Using in-memory rate limit storage (not suitable for production)")
        
        # Initialize IP extractor
        self.ip_extractor = IPExtractor(
            trusted_proxies=trusted_proxies or self._get_default_trusted_proxies(),
            trust_all_proxies=os.getenv("TRUST_ALL_PROXIES", "false").lower() == "true"
        ) if enable_proxy_headers else None
        
        # Internal service keys
        self.internal_service_keys = internal_service_keys or set()
        
        # Default tier
        self.default_tier = default_tier
        
        # Statistics
        self.stats = defaultdict(int)
    
    def _get_default_trusted_proxies(self) -> List[str]:
        """Get default trusted proxies from environment."""
        default_proxies = [
            "127.0.0.1",
            "::1",
            "10.0.0.0/8",
            "172.16.0.0/12",
            "192.168.0.0/16",
            "fc00::/7",  # IPv6 private
        ]
        
        # Add proxies from environment
        env_proxies = os.getenv("TRUSTED_PROXIES", "").split(",")
        for proxy in env_proxies:
            proxy = proxy.strip()
            if proxy:
                default_proxies.append(proxy)
        
        return default_proxies
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with rate limiting."""
        # Skip exempt paths
        if any(request.url.path.startswith(path) for path in self.EXEMPT_PATHS):
            return await call_next(request)
        
        # Extract client identifier
        client_id = self._get_client_identifier(request)
        
        # Determine rate limit tier
        tier = await self._get_rate_limit_tier(request)
        
        # Skip rate limiting for internal services
        if tier == RateLimitTier.INTERNAL:
            logger.debug(f"Bypassing rate limit for internal service: {client_id}")
            self.stats["internal_requests"] += 1
            return await call_next(request)
        
        # Get rate limit config
        config = RATE_LIMIT_TIERS[tier]
        
        # Check rate limits
        allowed, retry_after, limit_info = self._check_rate_limits(client_id, config)
        
        if not allowed:
            self.stats["rate_limited"] += 1
            logger.warning(
                f"Rate limit exceeded for {client_id} "
                f"(tier: {tier}, path: {request.url.path})"
            )
            
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "rate_limit_exceeded",
                    "message": "Too many requests. Please try again later.",
                    "retry_after": retry_after,
                    "limits": limit_info
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Tier": tier.value,
                    **self._get_rate_limit_headers(limit_info)
                }
            )
        
        # Process request
        self.stats["allowed_requests"] += 1
        response = await call_next(request)
        
        # Add rate limit headers to response
        for header, value in self._get_rate_limit_headers(limit_info).items():
            response.headers[header] = value
        response.headers["X-RateLimit-Tier"] = tier.value
        
        return response
    
    def _get_client_identifier(self, request: Request) -> str:
        """Get unique client identifier."""
        # Extract IP address
        if self.ip_extractor:
            ip = self.ip_extractor.extract_ip(request)
        else:
            ip = request.client.host if request.client else "unknown"
        
        # Optional: Include user ID if authenticated
        user_id = getattr(request.state, "user_id", None)
        if user_id:
            return f"user:{user_id}"
        
        return f"ip:{ip}"
    
    async def _get_rate_limit_tier(self, request: Request) -> RateLimitTier:
        """Determine rate limit tier for request."""
        # Check for internal service headers
        for header in self.INTERNAL_SERVICE_HEADERS:
            service_key = request.headers.get(header)
            if service_key and service_key in self.internal_service_keys:
                return RateLimitTier.INTERNAL
        
        # Check user tier from authentication
        user_claims = getattr(request.state, "user_claims", None)
        if user_claims:
            # Check admin role
            if any(role == "admin" for role in user_claims.roles):
                return RateLimitTier.ADMIN
            
            # Check subscription tier
            tier_map = {
                "free": RateLimitTier.FREE,
                "premium": RateLimitTier.PREMIUM,
                "pro": RateLimitTier.PREMIUM,
                "enterprise": RateLimitTier.ENTERPRISE,
            }
            user_tier = getattr(user_claims, "tier", "free")
            return tier_map.get(user_tier, self.default_tier)
        
        return self.default_tier
    
    def _check_rate_limits(
        self,
        client_id: str,
        config: RateLimitConfig
    ) -> Tuple[bool, int, Dict[str, Any]]:
        """
        Check all rate limits for client.
        
        Returns:
            Tuple of (allowed, retry_after_seconds, limit_info)
        """
        current_time = time.time()
        limit_info = {}
        
        # Check minute limit
        minute_key = f"rl:min:{client_id}"
        minute_count, minute_ttl = self.storage.increment(minute_key, 60, config.requests_per_minute)
        limit_info["minute"] = {
            "limit": config.requests_per_minute,
            "remaining": max(0, config.requests_per_minute - minute_count),
            "reset": int(current_time + minute_ttl)
        }
        
        if minute_count > config.requests_per_minute:
            return False, minute_ttl, limit_info
        
        # Check burst limit (sliding window)
        if minute_count > config.burst_size:
            # Check if requests are too fast
            burst_key = f"rl:burst:{client_id}"
            burst_count = self.storage.get_count(burst_key)
            if burst_count > config.burst_size:
                return False, 1, limit_info  # 1 second penalty for burst
            self.storage.increment(burst_key, 1, config.burst_size)
        
        # Check hour limit
        hour_key = f"rl:hour:{client_id}"
        hour_count, hour_ttl = self.storage.increment(hour_key, 3600, config.requests_per_hour)
        limit_info["hour"] = {
            "limit": config.requests_per_hour,
            "remaining": max(0, config.requests_per_hour - hour_count),
            "reset": int(current_time + hour_ttl)
        }
        
        if hour_count > config.requests_per_hour:
            return False, hour_ttl, limit_info
        
        # Check day limit
        day_key = f"rl:day:{client_id}"
        day_count, day_ttl = self.storage.increment(day_key, 86400, config.requests_per_day)
        limit_info["day"] = {
            "limit": config.requests_per_day,
            "remaining": max(0, config.requests_per_day - day_count),
            "reset": int(current_time + day_ttl)
        }
        
        if day_count > config.requests_per_day:
            return False, day_ttl, limit_info
        
        return True, 0, limit_info
    
    def _get_rate_limit_headers(self, limit_info: Dict[str, Any]) -> Dict[str, str]:
        """Generate rate limit headers."""
        headers = {}
        
        # Use the most restrictive limit for headers
        for window in ["minute", "hour", "day"]:
            if window in limit_info:
                info = limit_info[window]
                prefix = f"X-RateLimit-{window.capitalize()}"
                headers[f"{prefix}-Limit"] = str(info["limit"])
                headers[f"{prefix}-Remaining"] = str(info["remaining"])
                headers[f"{prefix}-Reset"] = str(info["reset"])
        
        return headers
    
    def reset_limits(self, client_id: str):
        """Reset all limits for a client (admin function)."""
        for window in ["min", "hour", "day", "burst"]:
            key = f"rl:{window}:{client_id}"
            self.storage.reset(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiting statistics."""
        return {
            "allowed_requests": self.stats["allowed_requests"],
            "rate_limited_requests": self.stats["rate_limited"],
            "internal_requests": self.stats["internal_requests"],
            "storage_type": type(self.storage).__name__
        }


# Initialize Redis for rate limiting
try:
    redis_client = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        db=int(os.getenv("REDIS_RATELIMIT_DB", "2")),
        decode_responses=True
    )
    redis_client.ping()
    logger.info("Redis connected for rate limiting")
except (redis.ConnectionError, ValueError):
    logger.warning("Redis not available for rate limiting, using in-memory storage")
    redis_client = None


# Export configured middleware
def create_rate_limit_middleware(
    app,
    trusted_proxies: Optional[List[str]] = None,
    internal_service_keys: Optional[Set[str]] = None
) -> RateLimitMiddleware:
    """
    Create rate limit middleware with configuration.
    
    Args:
        app: FastAPI application
        trusted_proxies: List of trusted proxy IPs
        internal_service_keys: Valid internal service API keys
        
    Returns:
        Configured RateLimitMiddleware instance
    """
    # Get configuration from environment
    trusted_proxies = trusted_proxies or os.getenv("TRUSTED_PROXIES", "").split(",")
    internal_keys = internal_service_keys or set(
        key.strip() for key in os.getenv("INTERNAL_API_KEYS", "").split(",")
        if key.strip()
    )
    
    return RateLimitMiddleware(
        app=app,
        redis_client=redis_client,
        trusted_proxies=[p.strip() for p in trusted_proxies if p.strip()],
        enable_proxy_headers=os.getenv("ENABLE_PROXY_HEADERS", "true").lower() == "true",
        internal_service_keys=internal_keys,
        default_tier=RateLimitTier(os.getenv("DEFAULT_RATE_LIMIT_TIER", "free"))
    )


# Export public interface
__all__ = [
    "RateLimitMiddleware",
    "RateLimitTier",
    "RateLimitConfig",
    "RATE_LIMIT_TIERS",
    "IPExtractor",
    "create_rate_limit_middleware",
    "redis_client"
]
