"""
Middleware for API Gateway

Rate limiting, logging, and request processing middleware.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any

from fastapi import Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware


logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple in-memory rate limiting middleware
    In production, this would use Redis for distributed rate limiting
    """
    
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.client_requests = {}  # In production: use Redis
        
    async def dispatch(self, request: Request, call_next):
        # Get client identifier (IP address)
        client_ip = request.client.host
        
        # Check rate limit
        current_time = time.time()
        
        # Clean old entries
        if client_ip in self.client_requests:
            self.client_requests[client_ip] = [
                req_time for req_time in self.client_requests[client_ip]
                if current_time - req_time < 60  # 1 minute window
            ]
        else:
            self.client_requests[client_ip] = []
            
        # Check if limit exceeded
        if len(self.client_requests[client_ip]) >= self.requests_per_minute:
            return Response(
                content=json.dumps({
                    "error": "Rate limit exceeded",
                    "detail": f"Maximum {self.requests_per_minute} requests per minute"
                }),
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                headers={"Content-Type": "application/json"}
            )
            
        # Record request
        self.client_requests[client_ip].append(current_time)
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = max(0, self.requests_per_minute - len(self.client_requests[client_ip]))
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(current_time + 60))
        
        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    """Request/response logging middleware"""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Log request
        logger.info(
            f"Request: {request.method} {request.url.path} "
            f"from {request.client.host}"
        )
        
        # Process request
        try:
            response = await call_next(request)
            
            # Log response
            process_time = time.time() - start_time
            logger.info(
                f"Response: {response.status_code} "
                f"in {process_time:.3f}s"
            )
            
            # Add timing header
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"Error: {str(e)} "
                f"after {process_time:.3f}s"
            )
            raise


class AuthMiddleware(BaseHTTPMiddleware):
    """Authentication middleware for protected routes"""
    
    def __init__(self, app):
        super().__init__(app)
        
        # Routes that don't require authentication
        self.public_routes = {
            "/health",
            "/metrics",
            "/auth/register",
            "/auth/login",
            "/docs",
            "/openapi.json",
            "/redoc"
        }
        
    async def dispatch(self, request: Request, call_next):
        # Skip auth for public routes
        if request.url.path in self.public_routes:
            return await call_next(request)
            
        # Check for authentication header
        auth_header = request.headers.get("Authorization")
        
        if not auth_header or not auth_header.startswith("Bearer "):
            return Response(
                content=json.dumps({
                    "error": "Authentication required",
                    "detail": "Missing or invalid Authorization header"
                }),
                status_code=status.HTTP_401_UNAUTHORIZED,
                headers={"Content-Type": "application/json"}
            )
            
        # Extract token
        token = auth_header.split(" ")[1] if len(auth_header.split(" ")) > 1 else ""
        
        if not token:
            return Response(
                content=json.dumps({
                    "error": "Authentication required",
                    "detail": "Missing token"
                }),
                status_code=status.HTTP_401_UNAUTHORIZED,
                headers={"Content-Type": "application/json"}
            )
            
        # Add token to request state for use in route handlers
        request.state.token = token
        
        return await call_next(request)


class CacheMiddleware(BaseHTTPMiddleware):
    """Simple response caching middleware"""
    
    def __init__(self, app, cache_ttl: int = 300):
        super().__init__(app)
        self.cache_ttl = cache_ttl
        self.cache = {}  # In production: use Redis
        
        # Routes to cache
        self.cacheable_routes = {
            "/health/services",
            "/admin/services/stats"
        }
        
    async def dispatch(self, request: Request, call_next):
        # Only cache GET requests for specific routes
        if (request.method != "GET" or 
            request.url.path not in self.cacheable_routes):
            return await call_next(request)
            
        # Check cache
        cache_key = f"{request.method}:{request.url.path}:{request.url.query}"
        current_time = time.time()
        
        if cache_key in self.cache:
            cached_data, cached_time = self.cache[cache_key]
            
            if current_time - cached_time < self.cache_ttl:
                # Cache hit
                return Response(
                    content=cached_data,
                    headers={
                        "Content-Type": "application/json",
                        "X-Cache": "HIT"
                    }
                )
                
        # Cache miss - process request
        response = await call_next(request)
        
        # Cache successful responses
        if response.status_code == 200:
            # Read response body
            body = b""
            async for chunk in response.body_iterator:
                body += chunk
                
            # Cache the response
            self.cache[cache_key] = (body.decode(), current_time)
            
            # Create new response
            return Response(
                content=body,
                status_code=response.status_code,
                headers={
                    **dict(response.headers),
                    "X-Cache": "MISS"
                }
            )
            
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses"""
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "connect-src 'self'"
        )
        
        return response