"""
Monitoring middleware for Music Gen AI API.
"""

import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware for collecting request metrics.
    """

    def __init__(self, app):
        super().__init__(app)
        self.request_count = 0
        self.error_count = 0
        self.total_response_time = 0.0

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and collect metrics.
        """
        # Start timer
        start_time = time.time()

        # Increment request count
        self.request_count += 1

        try:
            # Process request
            response = await call_next(request)

            # Check for errors
            if response.status_code >= 400:
                self.error_count += 1

            # Calculate response time
            response_time = time.time() - start_time
            self.total_response_time += response_time

            # Add metrics headers
            response.headers["X-Response-Time"] = f"{response_time:.3f}"
            response.headers["X-Request-Count"] = str(self.request_count)

            return response

        except Exception as e:
            # Increment error count
            self.error_count += 1

            # Calculate response time
            response_time = time.time() - start_time
            self.total_response_time += response_time

            # Re-raise exception
            raise e

    def get_metrics(self) -> dict:
        """
        Get current metrics.
        """
        avg_response_time = (
            self.total_response_time / self.request_count if self.request_count > 0 else 0.0
        )

        return {
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / self.request_count if self.request_count > 0 else 0.0,
            "average_response_time": avg_response_time,
            "total_response_time": self.total_response_time,
        }
