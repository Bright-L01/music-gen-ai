"""
Comprehensive monitoring middleware for Music Gen AI.

Integrates Prometheus metrics, SLI/SLO tracking, and automated
alerting into the FastAPI application pipeline.
"""

import time
from datetime import datetime
from typing import Callable, Dict, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from ...core.logging_config import get_logger

logger = get_logger(__name__)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Legacy metrics middleware - kept for compatibility."""

    def __init__(self, app):
        super().__init__(app)
        self.request_count = 0
        self.error_count = 0
        self.total_response_time = 0.0

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and collect basic metrics."""
        start_time = time.time()
        self.request_count += 1

        try:
            response = await call_next(request)

            if response.status_code >= 400:
                self.error_count += 1

            response_time = time.time() - start_time
            self.total_response_time += response_time

            response.headers["X-Response-Time"] = f"{response_time:.3f}"
            response.headers["X-Request-Count"] = str(self.request_count)

            return response

        except Exception as e:
            self.error_count += 1
            response_time = time.time() - start_time
            self.total_response_time += response_time
            raise e

    def get_metrics(self) -> dict:
        """Get current metrics."""
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


class PrometheusMetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting Prometheus metrics from HTTP requests."""

    def __init__(self, app: ASGIApp, metrics_path: str = "/metrics"):
        super().__init__(app)
        self.metrics_path = metrics_path

        # Import metrics collector with fallback
        try:
            from ...monitoring import get_metrics_collector

            self.metrics_collector = get_metrics_collector()
            logger.info("Prometheus metrics collector initialized")
        except ImportError:
            logger.warning("Monitoring module not available, using basic metrics")
            self.metrics_collector = None

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Collect metrics for each HTTP request."""

        # Serve metrics endpoint if metrics collector is available
        if request.url.path == self.metrics_path and self.metrics_collector:
            return Response(content=self.metrics_collector.get_metrics(), media_type="text/plain")

        if not self.metrics_collector:
            # If no metrics collector, just pass through
            return await call_next(request)

        # Extract request metadata
        method = request.method
        path = self._normalize_path(request.url.path)
        user_tier = self._get_user_tier(request)
        correlation_id = getattr(request.state, "correlation_id", "unknown")

        # Track concurrent requests
        self.metrics_collector.api.concurrent_requests.labels(endpoint=path).inc()

        # Record request size
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                request_size = int(content_length)
                self.metrics_collector.api.http_request_size_bytes.labels(
                    method=method, endpoint=path
                ).observe(request_size)
            except ValueError:
                pass

        start_time = time.time()
        status_code = 500  # Default for exceptions

        try:
            # Process request
            response = await call_next(request)
            status_code = response.status_code

            # Record successful request
            self.metrics_collector.api.http_requests_total.labels(
                method=method, endpoint=path, status_code=status_code, user_tier=user_tier
            ).inc()

            # Track response size
            response_size = self._get_response_size(response)
            if response_size:
                self.metrics_collector.api.http_response_size_bytes.labels(
                    method=method, endpoint=path, status_code=status_code
                ).observe(response_size)

            return response

        except Exception as e:
            # Record failed request
            status_code = 500
            self.metrics_collector.api.http_requests_total.labels(
                method=method, endpoint=path, status_code=status_code, user_tier=user_tier
            ).inc()

            logger.error(
                f"Request failed: {method} {path}", error=str(e), correlation_id=correlation_id
            )
            raise

        finally:
            # Record request duration
            duration = time.time() - start_time
            self.metrics_collector.api.http_request_duration_seconds.labels(
                method=method, endpoint=path, status_code=status_code
            ).observe(duration)

            # Update concurrent requests counter
            self.metrics_collector.api.concurrent_requests.labels(endpoint=path).dec()

    def _normalize_path(self, path: str) -> str:
        """Normalize URL path to reduce cardinality."""
        # Remove query parameters
        path = path.split("?")[0]

        # Replace UUIDs and IDs with placeholders
        import re

        # UUID pattern
        path = re.sub(
            r"/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", "/{uuid}", path
        )

        # Numeric IDs
        path = re.sub(r"/\d+", "/{id}", path)

        # Task IDs (alphanumeric)
        path = re.sub(r"/[a-zA-Z0-9]{8,}", "/{task_id}", path)

        return path

    def _get_user_tier(self, request: Request) -> str:
        """Extract user tier from request context."""
        # Try to get from user context
        if hasattr(request.state, "user"):
            user = request.state.user
            if hasattr(user, "tier"):
                return user.tier
            elif isinstance(user, dict):
                return user.get("tier", "free")

        # Default to free tier
        return "free"

    def _get_response_size(self, response: Response) -> Optional[int]:
        """Get response size from headers."""
        content_length = response.headers.get("content-length")
        if content_length:
            try:
                return int(content_length)
            except ValueError:
                pass
        return None


# Health check endpoint for monitoring
async def monitoring_health_check() -> Dict[str, any]:
    """Comprehensive health check for monitoring systems."""
    try:
        # Try to import monitoring components
        try:
            from ...monitoring import get_metrics_collector, get_sli_collector, get_alert_manager

            metrics_collector = get_metrics_collector()
            sli_collector = get_sli_collector()
            alert_manager = get_alert_manager()

            # Check metrics collection
            metrics_status = "healthy"
            try:
                metrics_output = metrics_collector.get_metrics()
                if not metrics_output:
                    metrics_status = "unhealthy"
            except Exception:
                metrics_status = "unhealthy"

            # Check SLO status
            slo_statuses = sli_collector.get_all_slo_statuses()
            slo_violations = sum(
                1 for status in slo_statuses.values() if not status.is_meeting_target
            )

            # Check active alerts
            active_alerts = len(alert_manager.active_alerts)

            # Overall health determination
            overall_health = "healthy"
            if metrics_status != "healthy" or slo_violations > 2 or active_alerts > 5:
                overall_health = "degraded"
            if metrics_status == "unhealthy" or slo_violations > 5 or active_alerts > 10:
                overall_health = "unhealthy"

            return {
                "status": overall_health,
                "timestamp": datetime.now().isoformat(),
                "components": {
                    "metrics_collection": {
                        "status": metrics_status,
                        "metrics_available": metrics_status == "healthy",
                    },
                    "slo_monitoring": {
                        "status": "healthy" if slo_violations <= 2 else "degraded",
                        "total_slos": len(slo_statuses),
                        "violations": slo_violations,
                    },
                    "alerting": {
                        "status": "healthy" if active_alerts <= 5 else "degraded",
                        "active_alerts": active_alerts,
                    },
                },
                "summary": {
                    "slo_violations": slo_violations,
                    "active_alerts": active_alerts,
                    "uptime_seconds": time.time() - metrics_collector._start_time,
                },
            }

        except ImportError:
            # Monitoring not available
            return {
                "status": "unavailable",
                "timestamp": datetime.now().isoformat(),
                "message": "Monitoring system not available",
                "components": {
                    "metrics_collection": {"status": "unavailable"},
                    "slo_monitoring": {"status": "unavailable"},
                    "alerting": {"status": "unavailable"},
                },
            }

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "timestamp": datetime.now().isoformat(), "error": str(e)}
