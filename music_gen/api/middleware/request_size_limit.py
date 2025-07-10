"""
Request size limiting middleware for Music Gen AI.

Protects against DoS attacks by limiting request sizes and implementing
comprehensive upload controls based on 2024 security best practices.
"""

import asyncio
import logging
import time
from typing import Callable, Dict, Optional

from fastapi import HTTPException, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


class RequestSizeLimitConfig:
    """Configuration for request size limits."""

    def __init__(self):
        # Default size limits (in bytes)
        self.default_max_size = 10 * 1024 * 1024  # 10MB default
        self.max_json_size = 1 * 1024 * 1024  # 1MB for JSON
        self.max_audio_size = 100 * 1024 * 1024  # 100MB for audio files
        self.max_image_size = 10 * 1024 * 1024  # 10MB for images

        # Endpoint-specific limits
        self.endpoint_limits = {
            "/api/v1/generate/text": 1024,  # 1KB for text generation
            "/api/v1/generate/audio": 100 * 1024 * 1024,  # 100MB for audio
            "/api/auth/register": 1024,  # 1KB for registration
            "/api/auth/login": 1024,  # 1KB for login
            "/api/tracks": 5 * 1024 * 1024,  # 5MB for track metadata
        }

        # Content-type specific limits
        self.content_type_limits = {
            "application/json": 1 * 1024 * 1024,  # 1MB
            "application/x-www-form-urlencoded": 1024,  # 1KB
            "multipart/form-data": 100 * 1024 * 1024,  # 100MB
            "audio/wav": 100 * 1024 * 1024,  # 100MB
            "audio/mp3": 100 * 1024 * 1024,  # 100MB
            "audio/mpeg": 100 * 1024 * 1024,  # 100MB
            "image/jpeg": 10 * 1024 * 1024,  # 10MB
            "image/png": 10 * 1024 * 1024,  # 10MB
        }

        # Rate limiting for large uploads
        self.upload_rate_limit = 5  # Max 5 large uploads per minute per IP
        self.large_upload_threshold = 10 * 1024 * 1024  # 10MB

        # Security settings
        self.block_empty_content_type = True
        self.require_content_length = True
        self.max_content_length_header = 500 * 1024 * 1024  # 500MB absolute max


class UploadTracker:
    """Tracks upload activity for rate limiting."""

    def __init__(self):
        self.uploads_by_ip: Dict[str, list] = {}
        self.cleanup_interval = 60  # Clean up every minute

    def record_upload(self, ip_address: str, size: int):
        """Record an upload attempt."""
        now = time.time()

        if ip_address not in self.uploads_by_ip:
            self.uploads_by_ip[ip_address] = []

        self.uploads_by_ip[ip_address].append((now, size))

        # Clean old entries (older than 1 minute)
        cutoff = now - 60
        self.uploads_by_ip[ip_address] = [
            (timestamp, size)
            for timestamp, size in self.uploads_by_ip[ip_address]
            if timestamp > cutoff
        ]

    def check_rate_limit(self, ip_address: str, upload_threshold: int, max_uploads: int) -> bool:
        """Check if IP is within upload rate limits."""
        if ip_address not in self.uploads_by_ip:
            return True

        # Count large uploads in the last minute
        large_uploads = [
            size for timestamp, size in self.uploads_by_ip[ip_address] if size >= upload_threshold
        ]

        return len(large_uploads) < max_uploads

    def cleanup_old_entries(self):
        """Clean up old tracking entries."""
        now = time.time()
        cutoff = now - 300  # 5 minutes

        for ip in list(self.uploads_by_ip.keys()):
            self.uploads_by_ip[ip] = [
                (timestamp, size)
                for timestamp, size in self.uploads_by_ip[ip]
                if timestamp > cutoff
            ]

            if not self.uploads_by_ip[ip]:
                del self.uploads_by_ip[ip]


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Middleware that enforces request size limits and upload controls."""

    def __init__(self, app: ASGIApp, config: Optional[RequestSizeLimitConfig] = None):
        super().__init__(app)
        self.config = config or RequestSizeLimitConfig()
        self.upload_tracker = UploadTracker()

        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address."""
        # Check for forwarded headers
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip

        return request.client.host if request.client else "unknown"

    def _get_size_limit(self, request: Request) -> int:
        """Determine the appropriate size limit for the request."""

        # Check endpoint-specific limits first
        path = request.url.path
        for endpoint, limit in self.config.endpoint_limits.items():
            if path.startswith(endpoint):
                return limit

        # Check content-type specific limits
        content_type = request.headers.get("content-type", "").split(";")[0].lower()
        if content_type in self.config.content_type_limits:
            return self.config.content_type_limits[content_type]

        # Default limit
        return self.config.default_max_size

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Enforce request size limits."""

        # Skip size checking for certain methods
        if request.method in ("GET", "HEAD", "OPTIONS"):
            return await call_next(request)

        # Get Content-Length header
        content_length_header = request.headers.get("content-length")

        # Require Content-Length for non-GET requests
        if self.config.require_content_length and not content_length_header:
            logger.warning(
                f"Request without Content-Length header from {self._get_client_ip(request)}"
            )
            raise HTTPException(status_code=411, detail="Content-Length header required")

        # Parse content length
        try:
            content_length = int(content_length_header) if content_length_header else 0
        except ValueError:
            logger.warning(f"Invalid Content-Length header: {content_length_header}")
            raise HTTPException(status_code=400, detail="Invalid Content-Length header")

        # Check absolute maximum
        if content_length > self.config.max_content_length_header:
            logger.warning(
                f"Request exceeds absolute maximum: {content_length} bytes "
                f"from {self._get_client_ip(request)}"
            )
            raise HTTPException(
                status_code=413,
                detail=f"Request too large. Maximum allowed: {self.config.max_content_length_header} bytes",
            )

        # Determine appropriate size limit
        size_limit = self._get_size_limit(request)

        # Check against size limit
        if content_length > size_limit:
            logger.warning(
                f"Request exceeds size limit: {content_length} bytes "
                f"(limit: {size_limit}) from {self._get_client_ip(request)} "
                f"for {request.url.path}"
            )
            raise HTTPException(
                status_code=413, detail=f"Request too large. Maximum allowed: {size_limit} bytes"
            )

        # Check content-type requirements
        content_type = request.headers.get("content-type", "").strip()
        if self.config.block_empty_content_type and not content_type and content_length > 0:
            logger.warning(
                f"Request with content but no Content-Type from {self._get_client_ip(request)}"
            )
            raise HTTPException(
                status_code=400, detail="Content-Type header required for requests with body"
            )

        # Rate limiting for large uploads
        client_ip = self._get_client_ip(request)
        if content_length >= self.config.large_upload_threshold:
            if not self.upload_tracker.check_rate_limit(
                client_ip, self.config.large_upload_threshold, self.config.upload_rate_limit
            ):
                logger.warning(f"Upload rate limit exceeded for IP: {client_ip}")
                raise HTTPException(
                    status_code=429,
                    detail="Upload rate limit exceeded. Try again later.",
                    headers={"Retry-After": "60"},
                )

            # Record the upload
            self.upload_tracker.record_upload(client_ip, content_length)

        # Add size information to request state for monitoring
        request.state.content_length = content_length
        request.state.size_limit = size_limit
        request.state.client_ip = client_ip

        # Process request with streaming body validation
        return await self._process_with_body_validation(request, call_next, size_limit)

    async def _process_with_body_validation(
        self, request: Request, call_next: Callable, size_limit: int
    ) -> Response:
        """Process request with streaming body size validation."""

        # For requests that might have streaming bodies (like file uploads),
        # we need to validate the actual body size as it's read

        original_receive = request.receive
        bytes_received = 0

        async def size_limited_receive():
            nonlocal bytes_received

            message = await original_receive()

            if message["type"] == "http.request":
                body = message.get("body", b"")
                bytes_received += len(body)

                if bytes_received > size_limit:
                    logger.warning(
                        f"Request body exceeds size limit during streaming: "
                        f"{bytes_received} bytes (limit: {size_limit})"
                    )
                    raise HTTPException(
                        status_code=413, detail=f"Request body too large. Limit: {size_limit} bytes"
                    )

            return message

        # Replace the receive callable
        request._receive = size_limited_receive

        try:
            response = await call_next(request)

            # Log successful large uploads
            if bytes_received >= self.config.large_upload_threshold:
                logger.info(
                    f"Large upload completed: {bytes_received} bytes "
                    f"from {request.state.client_ip} to {request.url.path}"
                )

            return response

        except HTTPException:
            # Re-raise HTTP exceptions (including our size limit errors)
            raise
        except Exception as e:
            # Log other errors that might be related to large uploads
            if bytes_received >= self.config.large_upload_threshold:
                logger.error(f"Error processing large upload ({bytes_received} bytes): {e}")
            raise

    async def _cleanup_loop(self):
        """Background cleanup task."""
        while True:
            try:
                await asyncio.sleep(self.upload_tracker.cleanup_interval)
                self.upload_tracker.cleanup_old_entries()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")


class StreamingUploadValidator:
    """Validates uploads as they stream to prevent memory exhaustion."""

    def __init__(self, max_size: int, chunk_size: int = 8192):
        self.max_size = max_size
        self.chunk_size = chunk_size
        self.bytes_read = 0

    async def validate_chunk(self, chunk: bytes) -> None:
        """Validate a chunk of uploaded data."""
        self.bytes_read += len(chunk)

        if self.bytes_read > self.max_size:
            raise HTTPException(
                status_code=413, detail=f"Upload too large. Maximum: {self.max_size} bytes"
            )

    def get_progress(self) -> Dict[str, int]:
        """Get upload progress information."""
        return {
            "bytes_uploaded": self.bytes_read,
            "max_size": self.max_size,
            "progress_percent": min(100, (self.bytes_read / self.max_size) * 100),
        }


# Utility functions for size limit management
def format_bytes(bytes_value: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_value < 1024:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024
    return f"{bytes_value:.1f} TB"


def parse_size_string(size_str: str) -> int:
    """Parse size string like '10MB' to bytes."""
    size_str = size_str.upper().strip()

    multipliers = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}

    for unit, multiplier in multipliers.items():
        if size_str.endswith(unit):
            try:
                number = float(size_str[: -len(unit)])
                return int(number * multiplier)
            except ValueError:
                raise ValueError(f"Invalid size format: {size_str}")

    # Try parsing as plain number (bytes)
    try:
        return int(size_str)
    except ValueError:
        raise ValueError(f"Invalid size format: {size_str}")


# Monitoring and statistics
class RequestSizeMonitor:
    """Monitors request sizes for analytics and alerting."""

    def __init__(self):
        self.size_stats = {
            "total_requests": 0,
            "total_bytes": 0,
            "large_requests": 0,
            "rejected_requests": 0,
            "avg_size": 0.0,
        }
        self.recent_sizes = []

    def record_request(self, size: int, rejected: bool = False):
        """Record a request size."""
        self.size_stats["total_requests"] += 1

        if not rejected:
            self.size_stats["total_bytes"] += size
            self.recent_sizes.append(size)

            # Keep only recent sizes for average calculation
            if len(self.recent_sizes) > 1000:
                self.recent_sizes = self.recent_sizes[-500:]

            if size >= 10 * 1024 * 1024:  # 10MB threshold
                self.size_stats["large_requests"] += 1

            # Update average
            if self.recent_sizes:
                self.size_stats["avg_size"] = sum(self.recent_sizes) / len(self.recent_sizes)
        else:
            self.size_stats["rejected_requests"] += 1

    def get_stats(self) -> Dict[str, any]:
        """Get current statistics."""
        return {
            **self.size_stats,
            "rejection_rate": (
                self.size_stats["rejected_requests"]
                / max(1, self.size_stats["total_requests"])
                * 100
            ),
            "avg_size_formatted": format_bytes(self.size_stats["avg_size"]),
        }
