"""
Performance optimization middleware for Music Gen AI API.

This middleware implements various performance optimizations including:
- Response compression
- ETag generation for caching
- Query optimization hints
- Memory usage monitoring
"""

import gzip
import hashlib
import io
import json
import time
from typing import Callable, Optional, Set

from fastapi import Request, Response
from fastapi.responses import StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from ...core.resource_manager import ResourceMonitor


class PerformanceMiddleware(BaseHTTPMiddleware):
    """Middleware for performance optimization."""

    def __init__(
        self,
        app: ASGIApp,
        compress_threshold: int = 1024,  # Compress responses larger than 1KB
        enable_etag: bool = True,
        enable_compression: bool = True,
        monitor_memory: bool = True,
    ):
        super().__init__(app)
        self.compress_threshold = compress_threshold
        self.enable_etag = enable_etag
        self.enable_compression = enable_compression
        self.monitor_memory = monitor_memory

        # Paths to exclude from compression
        self.compression_exclude_paths: Set[str] = {
            "/api/v1/stream/ws",  # WebSocket
            "/download",  # Audio downloads (already compressed)
        }

        # Initialize resource monitor if needed
        if self.monitor_memory:
            self.resource_monitor = ResourceMonitor(sampling_interval=60)
            self.resource_monitor.start_monitoring()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with performance optimizations."""

        # Start timing
        start_time = time.time()

        # Add performance hints to request state
        request.state.perf_hints = {
            "use_cache": True,
            "optimize_queries": True,
            "stream_large_responses": True,
        }

        # Get response
        response = await call_next(request)

        # Calculate processing time
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = f"{process_time:.3f}"

        # Skip optimization for excluded paths
        if any(request.url.path.startswith(path) for path in self.compression_exclude_paths):
            return response

        # Apply optimizations for successful responses
        if response.status_code == 200:
            # Handle streaming responses differently
            if isinstance(response, StreamingResponse):
                return await self._optimize_streaming_response(request, response)
            else:
                return await self._optimize_standard_response(request, response)

        return response

    async def _optimize_standard_response(self, request: Request, response: Response) -> Response:
        """Optimize standard responses."""

        # Read response body
        body = b""
        async for chunk in response.body_iterator:
            body += chunk

        # Generate ETag if enabled
        if self.enable_etag and request.method in ("GET", "HEAD"):
            etag = self._generate_etag(body)
            response.headers["ETag"] = etag

            # Check if client has matching ETag
            if_none_match = request.headers.get("if-none-match")
            if if_none_match == etag:
                return Response(status_code=304, headers=dict(response.headers))

        # Compress response if enabled and beneficial
        if (
            self.enable_compression
            and len(body) > self.compress_threshold
            and self._should_compress(request, response)
        ):
            compressed_body = gzip.compress(body)

            # Only use compression if it reduces size
            if len(compressed_body) < len(body):
                response.headers["Content-Encoding"] = "gzip"
                response.headers["Content-Length"] = str(len(compressed_body))
                response.headers["Vary"] = "Accept-Encoding"
                return Response(
                    content=compressed_body,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type=response.media_type,
                )

        # Return original response with body
        return Response(
            content=body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
        )

    async def _optimize_streaming_response(
        self, request: Request, response: StreamingResponse
    ) -> Response:
        """Optimize streaming responses."""

        # For streaming responses, we can't generate ETags or compress easily
        # But we can add cache headers
        if request.method == "GET":
            # Add cache headers for static content
            if "/static/" in request.url.path:
                response.headers["Cache-Control"] = "public, max-age=31536000"  # 1 year
            elif "/api/v1/models" in request.url.path:
                response.headers["Cache-Control"] = "public, max-age=3600"  # 1 hour

        return response

    def _should_compress(self, request: Request, response: Response) -> bool:
        """Check if response should be compressed."""

        # Check if client accepts gzip
        accept_encoding = request.headers.get("accept-encoding", "")
        if "gzip" not in accept_encoding:
            return False

        # Check content type
        content_type = response.headers.get("content-type", "")
        compressible_types = [
            "application/json",
            "text/",
            "application/javascript",
            "application/xml",
        ]

        return any(ct in content_type for ct in compressible_types)

    def _generate_etag(self, content: bytes) -> str:
        """Generate ETag for content."""
        return f'"{hashlib.md5(content).hexdigest()}"'


class LazyLoadingMiddleware(BaseHTTPMiddleware):
    """Middleware for implementing lazy loading patterns."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add lazy loading hints to request."""

        # Parse query parameters for pagination
        limit = request.query_params.get("limit", "10")
        offset = request.query_params.get("offset", "0")

        # Add lazy loading configuration to request state
        request.state.lazy_loading = {
            "enabled": True,
            "default_limit": int(limit),
            "default_offset": int(offset),
            "max_limit": 100,  # Prevent excessive data fetching
            "prefetch_related": [],  # Relationships to prefetch
            "select_related": [],  # Relationships to join
        }

        # For specific endpoints, configure prefetching
        if "/api/v1/tracks" in request.url.path:
            request.state.lazy_loading["prefetch_related"] = ["user", "stats"]
        elif "/api/v1/users" in request.url.path:
            request.state.lazy_loading["select_related"] = ["profile"]

        response = await call_next(request)
        return response


class DatabaseOptimizationMiddleware(BaseHTTPMiddleware):
    """Middleware for database query optimization."""

    def __init__(self, app: ASGIApp, enable_query_logging: bool = False):
        super().__init__(app)
        self.enable_query_logging = enable_query_logging
        self.slow_query_threshold = 0.1  # 100ms

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add database optimization hints."""

        # Add query optimization hints
        request.state.db_optimization = {
            "use_read_replica": request.method in ("GET", "HEAD"),
            "enable_query_cache": True,
            "batch_size": 100,
            "connection_pool_size": 10,
            "statement_timeout": 30000,  # 30 seconds
            "log_slow_queries": self.enable_query_logging,
        }

        # For write operations, use primary database
        if request.method in ("POST", "PUT", "PATCH", "DELETE"):
            request.state.db_optimization["use_read_replica"] = False
            request.state.db_optimization["enable_query_cache"] = False

        response = await call_next(request)

        # Log slow queries if enabled
        if hasattr(request.state, "db_queries") and self.enable_query_logging:
            for query in request.state.db_queries:
                if query["duration"] > self.slow_query_threshold:
                    print(f"Slow query ({query['duration']:.3f}s): {query['sql']}")

        return response


class MemoryOptimizationMiddleware(BaseHTTPMiddleware):
    """Middleware for memory usage optimization."""

    def __init__(
        self,
        app: ASGIApp,
        max_request_size: int = 100 * 1024 * 1024,  # 100MB
        enable_gc_collection: bool = True,
    ):
        super().__init__(app)
        self.max_request_size = max_request_size
        self.enable_gc_collection = enable_gc_collection

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Monitor and optimize memory usage."""

        # Check content length for large uploads
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_request_size:
            return Response(
                content="Request too large",
                status_code=413,
                headers={"X-Max-Size": str(self.max_request_size)},
            )

        # Add memory optimization hints
        request.state.memory_optimization = {
            "stream_large_files": True,
            "chunk_size": 8192,  # 8KB chunks
            "enable_gc": self.enable_gc_collection,
            "clear_caches_on_complete": True,
        }

        # Process request
        response = await call_next(request)

        # Force garbage collection for memory-intensive endpoints
        if self.enable_gc_collection and request.url.path.startswith("/api/v1/generate"):
            import gc

            gc.collect()

        return response


def create_performance_middleware_stack(app: ASGIApp) -> ASGIApp:
    """Create a stack of performance optimization middleware."""

    # Apply middleware in reverse order (last one runs first)
    app = MemoryOptimizationMiddleware(app)
    app = DatabaseOptimizationMiddleware(app)
    app = LazyLoadingMiddleware(app)
    app = PerformanceMiddleware(app)

    return app
