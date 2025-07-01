"""
Rate limiting middleware for Music Gen AI API.
"""

import time
from collections import defaultdict
from typing import Callable, Dict, Tuple

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple rate limiting middleware using in-memory storage.
    In production, use Redis or similar.
    """
    
    def __init__(
        self,
        app,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        
        # Storage: IP -> (minute_count, minute_start, hour_count, hour_start)
        self.rate_limits: Dict[str, Tuple[int, float, int, float]] = defaultdict(
            lambda: (0, time.time(), 0, time.time())
        )
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Check rate limits and process request.
        """
        # Skip rate limiting for health checks
        if request.url.path.startswith("/health"):
            return await call_next(request)
        
        # Get client IP
        client_ip = request.client.host
        
        # Check rate limits
        if not self._check_rate_limit(client_ip):
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "message": "Too many requests. Please try again later.",
                },
                headers={
                    "Retry-After": "60",
                    "X-RateLimit-Limit-Minute": str(self.requests_per_minute),
                    "X-RateLimit-Limit-Hour": str(self.requests_per_hour),
                },
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        minute_count, _, hour_count, _ = self.rate_limits[client_ip]
        response.headers["X-RateLimit-Limit-Minute"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining-Minute"] = str(
            max(0, self.requests_per_minute - minute_count)
        )
        response.headers["X-RateLimit-Limit-Hour"] = str(self.requests_per_hour)
        response.headers["X-RateLimit-Remaining-Hour"] = str(
            max(0, self.requests_per_hour - hour_count)
        )
        
        return response
    
    def _check_rate_limit(self, client_ip: str) -> bool:
        """
        Check if client has exceeded rate limits.
        """
        current_time = time.time()
        minute_count, minute_start, hour_count, hour_start = self.rate_limits[client_ip]
        
        # Reset minute counter if needed
        if current_time - minute_start > 60:
            minute_count = 0
            minute_start = current_time
        
        # Reset hour counter if needed
        if current_time - hour_start > 3600:
            hour_count = 0
            hour_start = current_time
        
        # Check limits
        if minute_count >= self.requests_per_minute:
            return False
        
        if hour_count >= self.requests_per_hour:
            return False
        
        # Increment counters
        minute_count += 1
        hour_count += 1
        
        # Update storage
        self.rate_limits[client_ip] = (minute_count, minute_start, hour_count, hour_start)
        
        return True
    
    def reset_limits(self, client_ip: str = None):
        """
        Reset rate limits for a specific IP or all IPs.
        """
        if client_ip:
            if client_ip in self.rate_limits:
                del self.rate_limits[client_ip]
        else:
            self.rate_limits.clear()