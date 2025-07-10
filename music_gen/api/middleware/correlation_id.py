"""
Correlation ID middleware for request tracing.

Adds unique correlation IDs to all requests for distributed tracing
and log correlation across microservices.
"""

import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from ...core.logging_config import get_logger

logger = get_logger(__name__)


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Middleware that adds correlation IDs to all requests."""

    def __init__(
        self, app: ASGIApp, header_name: str = "X-Correlation-ID", generate_if_missing: bool = True
    ):
        super().__init__(app)
        self.header_name = header_name
        self.generate_if_missing = generate_if_missing

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add correlation ID to request and response."""

        # Get or generate correlation ID
        correlation_id = request.headers.get(self.header_name)

        if not correlation_id and self.generate_if_missing:
            correlation_id = str(uuid.uuid4())

        # Add to request state for access in route handlers
        request.state.correlation_id = correlation_id

        # Set up logging context
        logger_with_context = logger.bind(
            correlation_id=correlation_id,
            method=request.method,
            path=request.url.path,
            client_ip=self._get_client_ip(request),
        )

        # Log request start
        logger_with_context.info(
            "request_started",
            user_agent=request.headers.get("user-agent", "unknown"),
            query_params=dict(request.query_params),
        )

        # Process request
        start_time = time.time()

        try:
            response = await call_next(request)

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # ms

            # Add correlation ID to response headers
            if correlation_id:
                response.headers[self.header_name] = correlation_id

            # Log successful request
            logger_with_context.info(
                "request_completed",
                status_code=response.status_code,
                processing_time_ms=processing_time,
            )

            return response

        except Exception as e:
            # Calculate processing time for failed requests
            processing_time = (time.time() - start_time) * 1000  # ms

            # Log error
            logger_with_context.error(
                "request_failed",
                error=str(e),
                error_type=type(e).__name__,
                processing_time_ms=processing_time,
                exc_info=True,
            )

            raise

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check for forwarded headers (common with reverse proxies)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip

        # Fallback to client IP
        return request.client.host if request.client else "unknown"


# Utility functions for correlation ID access
def get_correlation_id_from_request(request: Request) -> str:
    """Get correlation ID from request state."""
    return getattr(request.state, "correlation_id", None)


def set_correlation_id_header(response: Response, correlation_id: str):
    """Set correlation ID in response header."""
    response.headers["X-Correlation-ID"] = correlation_id
