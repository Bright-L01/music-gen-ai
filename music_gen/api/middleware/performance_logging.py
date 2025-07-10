"""
Performance logging middleware for Music Gen AI.

Logs detailed performance metrics including response times,
database queries, memory usage, and resource utilization.
"""

import asyncio
import gc
import psutil
import time
from typing import Callable, Dict, Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from ...core.logging_config import get_performance_logger, get_logger

logger = get_logger(__name__)
performance_logger = get_performance_logger()


class PerformanceLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware that logs detailed performance metrics for all requests."""

    def __init__(
        self,
        app: ASGIApp,
        log_slow_requests_only: bool = False,
        slow_request_threshold_ms: float = 1000,
        include_memory_metrics: bool = True,
        include_database_metrics: bool = True,
    ):
        super().__init__(app)
        self.log_slow_requests_only = log_slow_requests_only
        self.slow_request_threshold_ms = slow_request_threshold_ms
        self.include_memory_metrics = include_memory_metrics
        self.include_database_metrics = include_database_metrics

        # Initialize process for memory monitoring
        self.process = psutil.Process()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Log performance metrics for the request."""

        # Get correlation ID from request
        correlation_id = getattr(request.state, "correlation_id", "unknown")

        # Start performance monitoring
        start_time = time.time()
        start_cpu_time = time.process_time()

        # Memory usage before request
        memory_before = None
        if self.include_memory_metrics:
            memory_info = self.process.memory_info()
            memory_before = {
                "rss": memory_info.rss,  # Resident Set Size
                "vms": memory_info.vms,  # Virtual Memory Size
                "percent": self.process.memory_percent(),
            }

        # Initialize performance tracking
        request.state.performance_metrics = {
            "start_time": start_time,
            "database_queries": [],
            "cache_operations": [],
            "external_calls": [],
        }

        try:
            # Process request
            response = await call_next(request)

            # Calculate performance metrics
            end_time = time.time()
            end_cpu_time = time.process_time()

            duration_ms = (end_time - start_time) * 1000
            cpu_time_ms = (end_cpu_time - start_cpu_time) * 1000

            # Memory usage after request
            memory_after = None
            memory_delta = None
            if self.include_memory_metrics:
                memory_info = self.process.memory_info()
                memory_after = {
                    "rss": memory_info.rss,
                    "vms": memory_info.vms,
                    "percent": self.process.memory_percent(),
                }
                memory_delta = {
                    "rss_mb": (memory_after["rss"] - memory_before["rss"]) / 1024 / 1024,
                    "vms_mb": (memory_after["vms"] - memory_before["vms"]) / 1024 / 1024,
                    "percent": memory_after["percent"] - memory_before["percent"],
                }

            # Get performance metrics from request state
            perf_metrics = getattr(request.state, "performance_metrics", {})

            # Build metrics payload
            metrics = {
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 2),
                "cpu_time_ms": round(cpu_time_ms, 2),
                "correlation_id": correlation_id,
                "user_agent": request.headers.get("user-agent", "unknown"),
                "content_length": response.headers.get("content-length", 0),
                "query_params_count": len(request.query_params),
            }

            # Add memory metrics if enabled
            if self.include_memory_metrics and memory_delta:
                metrics.update(
                    {
                        "memory_delta_mb": round(memory_delta["rss_mb"], 2),
                        "memory_percent_delta": round(memory_delta["percent"], 2),
                        "memory_after_mb": round(memory_after["rss"] / 1024 / 1024, 2),
                    }
                )

            # Add database metrics if available
            if self.include_database_metrics:
                db_queries = perf_metrics.get("database_queries", [])
                if db_queries:
                    metrics.update(
                        {
                            "database_query_count": len(db_queries),
                            "database_total_time_ms": sum(
                                q.get("duration_ms", 0) for q in db_queries
                            ),
                            "database_avg_time_ms": round(
                                sum(q.get("duration_ms", 0) for q in db_queries) / len(db_queries),
                                2,
                            )
                            if db_queries
                            else 0,
                        }
                    )

            # Add cache metrics if available
            cache_ops = perf_metrics.get("cache_operations", [])
            if cache_ops:
                hits = sum(1 for op in cache_ops if op.get("hit", False))
                metrics.update(
                    {
                        "cache_operations": len(cache_ops),
                        "cache_hits": hits,
                        "cache_hit_rate": round(hits / len(cache_ops) * 100, 1) if cache_ops else 0,
                    }
                )

            # Add external call metrics if available
            external_calls = perf_metrics.get("external_calls", [])
            if external_calls:
                metrics.update(
                    {
                        "external_calls": len(external_calls),
                        "external_calls_time_ms": sum(
                            call.get("duration_ms", 0) for call in external_calls
                        ),
                    }
                )

            # Log performance metrics
            should_log = True
            if self.log_slow_requests_only:
                should_log = duration_ms >= self.slow_request_threshold_ms

            if should_log:
                # Determine log level based on performance
                if duration_ms >= 5000:  # 5 seconds
                    log_level = "error"
                elif duration_ms >= 2000:  # 2 seconds
                    log_level = "warning"
                else:
                    log_level = "info"

                getattr(performance_logger.logger, log_level)(
                    "request_performance_detailed", **metrics
                )

            # Add performance headers to response
            response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
            response.headers["X-CPU-Time"] = f"{cpu_time_ms:.2f}ms"

            if self.include_memory_metrics and memory_delta:
                response.headers["X-Memory-Delta"] = f"{memory_delta['rss_mb']:.2f}MB"

            return response

        except Exception as e:
            # Log error with performance context
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000

            performance_logger.logger.error(
                "request_error_performance",
                method=request.method,
                path=request.url.path,
                duration_ms=round(duration_ms, 2),
                error=str(e),
                error_type=type(e).__name__,
                correlation_id=correlation_id,
            )

            raise


class DatabasePerformanceTracker:
    """Tracks database query performance within requests."""

    @staticmethod
    def log_query(
        request: Request,
        query_type: str,
        duration_ms: float,
        rows_affected: int = 0,
        **extra_context,
    ):
        """Log a database query performance metric."""

        if hasattr(request.state, "performance_metrics"):
            request.state.performance_metrics["database_queries"].append(
                {
                    "query_type": query_type,
                    "duration_ms": duration_ms,
                    "rows_affected": rows_affected,
                    "timestamp": time.time(),
                    **extra_context,
                }
            )

        # Also log to performance logger
        correlation_id = getattr(request.state, "correlation_id", "unknown")
        performance_logger.log_database_performance(
            query_type=query_type,
            duration_ms=duration_ms,
            rows_affected=rows_affected,
            correlation_id=correlation_id,
            **extra_context,
        )


class CachePerformanceTracker:
    """Tracks cache operation performance within requests."""

    @staticmethod
    def log_operation(
        request: Request,
        operation: str,  # get, set, delete, etc.
        hit: bool,
        duration_ms: float,
        key: str = None,
        **extra_context,
    ):
        """Log a cache operation performance metric."""

        if hasattr(request.state, "performance_metrics"):
            request.state.performance_metrics["cache_operations"].append(
                {
                    "operation": operation,
                    "hit": hit,
                    "duration_ms": duration_ms,
                    "key": key,
                    "timestamp": time.time(),
                    **extra_context,
                }
            )


class ExternalCallTracker:
    """Tracks external API call performance within requests."""

    @staticmethod
    def log_call(
        request: Request,
        service: str,
        operation: str,
        duration_ms: float,
        status_code: int = None,
        **extra_context,
    ):
        """Log an external API call performance metric."""

        if hasattr(request.state, "performance_metrics"):
            request.state.performance_metrics["external_calls"].append(
                {
                    "service": service,
                    "operation": operation,
                    "duration_ms": duration_ms,
                    "status_code": status_code,
                    "timestamp": time.time(),
                    **extra_context,
                }
            )


# Decorators for performance tracking
def track_database_query(query_type: str):
    """Decorator to track database query performance."""

    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            result = await func(*args, **kwargs)
            duration_ms = (time.time() - start_time) * 1000

            # Try to find request in args/kwargs to log performance
            request = None
            for arg in args:
                if hasattr(arg, "state") and hasattr(arg.state, "correlation_id"):
                    request = arg
                    break

            if request:
                DatabasePerformanceTracker.log_query(request, query_type, duration_ms)

            return result

        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            duration_ms = (time.time() - start_time) * 1000

            # Try to find request in args/kwargs
            request = None
            for arg in args:
                if hasattr(arg, "state") and hasattr(arg.state, "correlation_id"):
                    request = arg
                    break

            if request:
                DatabasePerformanceTracker.log_query(request, query_type, duration_ms)

            return result

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


def track_cache_operation(operation: str):
    """Decorator to track cache operation performance."""

    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            result = await func(*args, **kwargs)
            duration_ms = (time.time() - start_time) * 1000

            # Determine if it was a hit based on result
            hit = result is not None

            # Try to find request in context
            # This would need to be implemented based on your cache usage pattern

            return result

        return async_wrapper if asyncio.iscoroutinefunction(func) else func

    return decorator


# Resource monitoring utilities
class ResourceMonitor:
    """Monitors system resources and logs alerts."""

    def __init__(self):
        self.process = psutil.Process()
        self.logger = get_logger("resource_monitor")

    async def check_resources(self):
        """Check system resources and log alerts if necessary."""

        # Memory usage
        memory_percent = self.process.memory_percent()
        if memory_percent > 80:
            self.logger.warning(
                "high_memory_usage",
                memory_percent=memory_percent,
                memory_mb=self.process.memory_info().rss / 1024 / 1024,
            )

        # CPU usage
        cpu_percent = self.process.cpu_percent()
        if cpu_percent > 80:
            self.logger.warning("high_cpu_usage", cpu_percent=cpu_percent)

        # Disk usage
        disk_usage = psutil.disk_usage("/")
        disk_percent = (disk_usage.used / disk_usage.total) * 100
        if disk_percent > 85:
            self.logger.warning(
                "high_disk_usage",
                disk_percent=disk_percent,
                disk_free_gb=disk_usage.free / 1024 / 1024 / 1024,
            )

        # Open file descriptors (Linux/Mac)
        try:
            open_files = len(self.process.open_files())
            if open_files > 500:
                self.logger.warning("high_open_files", open_files=open_files)
        except Exception:
            pass  # Not available on all platforms


# Global resource monitor instance
resource_monitor = ResourceMonitor()
