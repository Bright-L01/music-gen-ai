"""
Comprehensive Prometheus metrics collection for Music Gen AI.

Implements custom metrics for business logic, API performance,
system resources, and task queue monitoring.
"""

import os
import time
import psutil
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Union

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Info,
    Summary,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST,
    multiprocess,
    start_http_server,
)

from ..core.logging_config import get_logger

logger = get_logger(__name__)


class MetricType(Enum):
    """Supported Prometheus metric types."""

    COUNTER = "counter"
    HISTOGRAM = "histogram"
    GAUGE = "gauge"
    SUMMARY = "summary"
    INFO = "info"


@dataclass
class MetricDefinition:
    """Definition for a custom metric."""

    name: str
    help_text: str
    metric_type: MetricType
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # For histograms
    quantiles: Optional[Dict[float, float]] = None  # For summaries


class BusinessMetrics:
    """Business logic metrics for Music Gen AI."""

    def __init__(self, registry: CollectorRegistry):
        self.registry = registry

        # Music generation metrics
        self.generation_requests_total = Counter(
            "musicgen_generation_requests_total",
            "Total number of music generation requests",
            ["model", "user_tier", "status"],
            registry=registry,
        )

        self.generation_duration_seconds = Histogram(
            "musicgen_generation_duration_seconds",
            "Time taken to generate music",
            ["model", "duration_category"],
            buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, float("inf")],
            registry=registry,
        )

        self.generated_audio_duration_seconds = Histogram(
            "musicgen_generated_audio_duration_seconds",
            "Duration of generated audio",
            ["model", "user_tier"],
            buckets=[5.0, 10.0, 15.0, 30.0, 60.0, 120.0, 300.0, float("inf")],
            registry=registry,
        )

        self.generation_errors_total = Counter(
            "musicgen_generation_errors_total",
            "Total number of generation errors",
            ["model", "error_type", "error_category"],
            registry=registry,
        )

        self.model_load_duration_seconds = Histogram(
            "musicgen_model_load_duration_seconds",
            "Time taken to load models",
            ["model", "cache_hit"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, float("inf")],
            registry=registry,
        )

        self.model_cache_hits_total = Counter(
            "musicgen_model_cache_hits_total",
            "Number of model cache hits",
            ["model"],
            registry=registry,
        )

        self.model_cache_misses_total = Counter(
            "musicgen_model_cache_misses_total",
            "Number of model cache misses",
            ["model"],
            registry=registry,
        )

        # User engagement metrics
        self.user_sessions_total = Counter(
            "musicgen_user_sessions_total",
            "Total number of user sessions",
            ["user_tier", "session_type"],
            registry=registry,
        )

        self.user_session_duration_seconds = Histogram(
            "musicgen_user_session_duration_seconds",
            "Duration of user sessions",
            ["user_tier"],
            buckets=[60.0, 300.0, 900.0, 1800.0, 3600.0, 7200.0, float("inf")],
            registry=registry,
        )

        # Quality metrics
        self.audio_quality_score = Histogram(
            "musicgen_audio_quality_score",
            "Audio quality scores from evaluation",
            ["model", "metric_type"],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            registry=registry,
        )

        self.user_satisfaction_score = Histogram(
            "musicgen_user_satisfaction_score",
            "User satisfaction ratings",
            ["model", "user_tier"],
            buckets=[1.0, 2.0, 3.0, 4.0, 5.0],
            registry=registry,
        )


class ApiMetrics:
    """API performance and behavior metrics."""

    def __init__(self, registry: CollectorRegistry):
        self.registry = registry

        # Request metrics
        self.http_requests_total = Counter(
            "musicgen_http_requests_total",
            "Total number of HTTP requests",
            ["method", "endpoint", "status_code", "user_tier"],
            registry=registry,
        )

        self.http_request_duration_seconds = Histogram(
            "musicgen_http_request_duration_seconds",
            "HTTP request duration",
            ["method", "endpoint", "status_code"],
            buckets=[
                0.001,
                0.005,
                0.01,
                0.025,
                0.05,
                0.1,
                0.25,
                0.5,
                1.0,
                2.5,
                5.0,
                10.0,
                float("inf"),
            ],
            registry=registry,
        )

        self.http_request_size_bytes = Histogram(
            "musicgen_http_request_size_bytes",
            "HTTP request size in bytes",
            ["method", "endpoint"],
            buckets=[64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, float("inf")],
            registry=registry,
        )

        self.http_response_size_bytes = Histogram(
            "musicgen_http_response_size_bytes",
            "HTTP response size in bytes",
            ["method", "endpoint", "status_code"],
            buckets=[64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 16777216, float("inf")],
            registry=registry,
        )

        # Authentication metrics
        self.auth_attempts_total = Counter(
            "musicgen_auth_attempts_total",
            "Total authentication attempts",
            ["method", "result", "user_tier"],
            registry=registry,
        )

        self.auth_token_validation_duration_seconds = Histogram(
            "musicgen_auth_token_validation_duration_seconds",
            "Time taken to validate authentication tokens",
            ["token_type", "result"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, float("inf")],
            registry=registry,
        )

        # Rate limiting metrics
        self.rate_limit_hits_total = Counter(
            "musicgen_rate_limit_hits_total",
            "Total rate limit hits",
            ["endpoint", "user_tier", "limit_type"],
            registry=registry,
        )

        self.concurrent_requests = Gauge(
            "musicgen_concurrent_requests",
            "Current number of concurrent requests",
            ["endpoint"],
            registry=registry,
        )


class SystemMetrics:
    """System resource and infrastructure metrics."""

    def __init__(self, registry: CollectorRegistry):
        self.registry = registry

        # CPU metrics
        self.cpu_usage_percent = Gauge(
            "musicgen_cpu_usage_percent",
            "CPU usage percentage",
            ["cpu_type"],  # 'total', 'user', 'system', 'iowait'
            registry=registry,
        )

        self.cpu_load_average = Gauge(
            "musicgen_cpu_load_average",
            "System load average",
            ["interval"],  # '1m', '5m', '15m'
            registry=registry,
        )

        # Memory metrics
        self.memory_usage_bytes = Gauge(
            "musicgen_memory_usage_bytes",
            "Memory usage in bytes",
            ["memory_type"],  # 'total', 'used', 'free', 'cached', 'buffers'
            registry=registry,
        )

        self.memory_usage_percent = Gauge(
            "musicgen_memory_usage_percent", "Memory usage percentage", registry=registry
        )

        # GPU metrics (if available)
        self.gpu_usage_percent = Gauge(
            "musicgen_gpu_usage_percent",
            "GPU usage percentage",
            ["gpu_id", "metric_type"],  # 'utilization', 'memory'
            registry=registry,
        )

        self.gpu_memory_usage_bytes = Gauge(
            "musicgen_gpu_memory_usage_bytes",
            "GPU memory usage in bytes",
            ["gpu_id", "memory_type"],  # 'used', 'free', 'total'
            registry=registry,
        )

        self.gpu_temperature_celsius = Gauge(
            "musicgen_gpu_temperature_celsius",
            "GPU temperature in Celsius",
            ["gpu_id"],
            registry=registry,
        )

        # Disk metrics
        self.disk_usage_bytes = Gauge(
            "musicgen_disk_usage_bytes",
            "Disk usage in bytes",
            ["mount_point", "usage_type"],  # 'used', 'free', 'total'
            registry=registry,
        )

        self.disk_usage_percent = Gauge(
            "musicgen_disk_usage_percent",
            "Disk usage percentage",
            ["mount_point"],
            registry=registry,
        )

        self.disk_io_operations_total = Counter(
            "musicgen_disk_io_operations_total",
            "Total disk I/O operations",
            ["device", "operation"],  # 'read', 'write'
            registry=registry,
        )

        self.disk_io_bytes_total = Counter(
            "musicgen_disk_io_bytes_total",
            "Total disk I/O bytes",
            ["device", "operation"],  # 'read', 'write'
            registry=registry,
        )

        # Network metrics
        self.network_bytes_total = Counter(
            "musicgen_network_bytes_total",
            "Total network bytes",
            ["interface", "direction"],  # 'sent', 'received'
            registry=registry,
        )

        self.network_packets_total = Counter(
            "musicgen_network_packets_total",
            "Total network packets",
            ["interface", "direction", "packet_type"],  # 'sent', 'received', 'errors', 'dropped'
            registry=registry,
        )

        # Database metrics
        self.database_connections_active = Gauge(
            "musicgen_database_connections_active",
            "Active database connections",
            ["database", "pool"],
            registry=registry,
        )

        self.database_connections_idle = Gauge(
            "musicgen_database_connections_idle",
            "Idle database connections",
            ["database", "pool"],
            registry=registry,
        )

        self.database_query_duration_seconds = Histogram(
            "musicgen_database_query_duration_seconds",
            "Database query duration",
            ["database", "query_type", "table"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, float("inf")],
            registry=registry,
        )

        # Cache metrics
        self.cache_operations_total = Counter(
            "musicgen_cache_operations_total",
            "Total cache operations",
            ["cache_type", "operation", "result"],  # 'hit', 'miss', 'set', 'delete'
            registry=registry,
        )

        self.cache_size_bytes = Gauge(
            "musicgen_cache_size_bytes", "Cache size in bytes", ["cache_type"], registry=registry
        )


class TaskQueueMetrics:
    """Task queue and background job metrics."""

    def __init__(self, registry: CollectorRegistry):
        self.registry = registry

        # Queue depth and processing
        self.queue_depth = Gauge(
            "musicgen_queue_depth",
            "Number of tasks in queue",
            ["queue_name", "priority"],
            registry=registry,
        )

        self.queue_processing_duration_seconds = Histogram(
            "musicgen_queue_processing_duration_seconds",
            "Time taken to process queue tasks",
            ["queue_name", "task_type", "result"],
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, float("inf")],
            registry=registry,
        )

        self.queue_tasks_total = Counter(
            "musicgen_queue_tasks_total",
            "Total number of queued tasks",
            ["queue_name", "task_type", "status"],  # 'pending', 'processing', 'completed', 'failed'
            registry=registry,
        )

        self.queue_retry_attempts_total = Counter(
            "musicgen_queue_retry_attempts_total",
            "Total number of task retry attempts",
            ["queue_name", "task_type", "retry_reason"],
            registry=registry,
        )

        # Worker metrics
        self.queue_workers_active = Gauge(
            "musicgen_queue_workers_active",
            "Number of active queue workers",
            ["queue_name", "worker_type"],
            registry=registry,
        )

        self.queue_worker_utilization_percent = Gauge(
            "musicgen_queue_worker_utilization_percent",
            "Queue worker utilization percentage",
            ["queue_name", "worker_id"],
            registry=registry,
        )

        # Dead letter queue
        self.dead_letter_queue_size = Gauge(
            "musicgen_dead_letter_queue_size",
            "Number of messages in dead letter queue",
            ["queue_name", "failure_reason"],
            registry=registry,
        )


class MetricsCollector:
    """Main metrics collector and manager."""

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        self.custom_metrics: Dict[str, Any] = {}
        self.collection_interval = 30  # seconds
        self._collection_thread: Optional[threading.Thread] = None
        self._stop_collection = threading.Event()

        # Initialize metric groups
        self.business = BusinessMetrics(self.registry)
        self.api = ApiMetrics(self.registry)
        self.system = SystemMetrics(self.registry)
        self.task_queue = TaskQueueMetrics(self.registry)

        # Service info
        self.service_info = Info(
            "musicgen_service_info", "Service information", registry=self.registry
        )
        self.service_info.info(
            {
                "version": os.getenv("SERVICE_VERSION", "1.0.0"),
                "environment": os.getenv("ENVIRONMENT", "development"),
                "build_time": os.getenv("BUILD_TIME", "unknown"),
                "git_commit": os.getenv("GIT_COMMIT", "unknown"),
            }
        )

        # Application uptime
        self._start_time = time.time()
        self.uptime_seconds = Gauge(
            "musicgen_uptime_seconds", "Service uptime in seconds", registry=self.registry
        )

        logger.info("Metrics collector initialized")

    def register_custom_metric(self, definition: MetricDefinition) -> Any:
        """Register a custom metric."""
        if definition.name in self.custom_metrics:
            logger.warning(f"Metric {definition.name} already exists")
            return self.custom_metrics[definition.name]

        metric_class = {
            MetricType.COUNTER: Counter,
            MetricType.HISTOGRAM: Histogram,
            MetricType.GAUGE: Gauge,
            MetricType.SUMMARY: Summary,
            MetricType.INFO: Info,
        }[definition.metric_type]

        kwargs = {
            "name": definition.name,
            "documentation": definition.help_text,
            "registry": self.registry,
        }

        if definition.labels:
            kwargs["labelnames"] = definition.labels

        if definition.metric_type == MetricType.HISTOGRAM and definition.buckets:
            kwargs["buckets"] = definition.buckets

        if definition.metric_type == MetricType.SUMMARY and definition.quantiles:
            kwargs["quantiles"] = definition.quantiles

        metric = metric_class(**kwargs)
        self.custom_metrics[definition.name] = metric

        logger.info(f"Registered custom metric: {definition.name}")
        return metric

    def collect_system_metrics(self):
        """Collect system resource metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None, percpu=False)
            self.system.cpu_usage_percent.labels(cpu_type="total").set(cpu_percent)

            cpu_times = psutil.cpu_times_percent(interval=None)
            if hasattr(cpu_times, "user"):
                self.system.cpu_usage_percent.labels(cpu_type="user").set(cpu_times.user)
            if hasattr(cpu_times, "system"):
                self.system.cpu_usage_percent.labels(cpu_type="system").set(cpu_times.system)
            if hasattr(cpu_times, "iowait"):
                self.system.cpu_usage_percent.labels(cpu_type="iowait").set(cpu_times.iowait)

            # Load average
            load_avg = psutil.getloadavg() if hasattr(psutil, "getloadavg") else (0, 0, 0)
            self.system.cpu_load_average.labels(interval="1m").set(load_avg[0])
            self.system.cpu_load_average.labels(interval="5m").set(load_avg[1])
            self.system.cpu_load_average.labels(interval="15m").set(load_avg[2])

            # Memory metrics
            memory = psutil.virtual_memory()
            self.system.memory_usage_bytes.labels(memory_type="total").set(memory.total)
            self.system.memory_usage_bytes.labels(memory_type="used").set(memory.used)
            self.system.memory_usage_bytes.labels(memory_type="free").set(memory.free)
            self.system.memory_usage_bytes.labels(memory_type="cached").set(
                getattr(memory, "cached", 0)
            )
            self.system.memory_usage_bytes.labels(memory_type="buffers").set(
                getattr(memory, "buffers", 0)
            )
            self.system.memory_usage_percent.set(memory.percent)

            # Disk metrics
            for disk in psutil.disk_partitions():
                try:
                    disk_usage = psutil.disk_usage(disk.mountpoint)
                    mount = disk.mountpoint

                    self.system.disk_usage_bytes.labels(mount_point=mount, usage_type="total").set(
                        disk_usage.total
                    )
                    self.system.disk_usage_bytes.labels(mount_point=mount, usage_type="used").set(
                        disk_usage.used
                    )
                    self.system.disk_usage_bytes.labels(mount_point=mount, usage_type="free").set(
                        disk_usage.free
                    )
                    self.system.disk_usage_percent.labels(mount_point=mount).set(disk_usage.percent)
                except (PermissionError, OSError):
                    continue

            # Network metrics
            network_io = psutil.net_io_counters(pernic=True)
            for interface, stats in network_io.items():
                self.system.network_bytes_total.labels(
                    interface=interface, direction="sent"
                )._value._value = stats.bytes_sent
                self.system.network_bytes_total.labels(
                    interface=interface, direction="received"
                )._value._value = stats.bytes_recv
                self.system.network_packets_total.labels(
                    interface=interface, direction="sent", packet_type="packets"
                )._value._value = stats.packets_sent
                self.system.network_packets_total.labels(
                    interface=interface, direction="received", packet_type="packets"
                )._value._value = stats.packets_recv
                self.system.network_packets_total.labels(
                    interface=interface, direction="sent", packet_type="errors"
                )._value._value = stats.errin
                self.system.network_packets_total.labels(
                    interface=interface, direction="received", packet_type="errors"
                )._value._value = stats.errout

            # GPU metrics (if available)
            self._collect_gpu_metrics()

            # Update uptime
            self.uptime_seconds.set(time.time() - self._start_time)

        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")

    def _collect_gpu_metrics(self):
        """Collect GPU metrics if available."""
        try:
            import nvidia_ml_py3 as nvml

            nvml.nvmlInit()

            device_count = nvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = nvml.nvmlDeviceGetHandleByIndex(i)

                # GPU utilization
                util = nvml.nvmlDeviceGetUtilizationRates(handle)
                self.system.gpu_usage_percent.labels(gpu_id=str(i), metric_type="utilization").set(
                    util.gpu
                )
                self.system.gpu_usage_percent.labels(gpu_id=str(i), metric_type="memory").set(
                    util.memory
                )

                # GPU memory
                memory = nvml.nvmlDeviceGetMemoryInfo(handle)
                self.system.gpu_memory_usage_bytes.labels(gpu_id=str(i), memory_type="total").set(
                    memory.total
                )
                self.system.gpu_memory_usage_bytes.labels(gpu_id=str(i), memory_type="used").set(
                    memory.used
                )
                self.system.gpu_memory_usage_bytes.labels(gpu_id=str(i), memory_type="free").set(
                    memory.free
                )

                # GPU temperature
                temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                self.system.gpu_temperature_celsius.labels(gpu_id=str(i)).set(temp)

        except (ImportError, Exception):
            # GPU monitoring not available
            pass

    def start_background_collection(self):
        """Start background thread for periodic metric collection."""
        if self._collection_thread and self._collection_thread.is_alive():
            logger.warning("Background collection already running")
            return

        self._stop_collection.clear()
        self._collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self._collection_thread.start()
        logger.info("Started background metric collection")

    def stop_background_collection(self):
        """Stop background metric collection."""
        if self._collection_thread:
            self._stop_collection.set()
            self._collection_thread.join(timeout=5.0)
            logger.info("Stopped background metric collection")

    def _collection_loop(self):
        """Background collection loop."""
        while not self._stop_collection.wait(self.collection_interval):
            try:
                self.collect_system_metrics()
            except Exception as e:
                logger.error(f"Error in collection loop: {e}")

    def get_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        return generate_latest(self.registry).decode("utf-8")

    def start_http_server(self, port: int = 8000):
        """Start HTTP server to serve metrics."""
        start_http_server(port, registry=self.registry)
        logger.info(f"Metrics HTTP server started on port {port}")


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def register_custom_metric(definition: MetricDefinition) -> Any:
    """Register a custom metric with the global collector."""
    return get_metrics_collector().register_custom_metric(definition)


# Decorators for automatic metric tracking


def track_operation(operation_name: str, labels: Optional[Dict[str, str]] = None):
    """Decorator to track operation metrics."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            metrics = get_metrics_collector()
            start_time = time.time()
            labels_dict = labels or {}
            labels_dict["operation"] = operation_name

            try:
                result = func(*args, **kwargs)
                labels_dict["status"] = "success"
                return result
            except Exception as e:
                labels_dict["status"] = "error"
                labels_dict["error_type"] = type(e).__name__
                raise
            finally:
                duration = time.time() - start_time
                # This would need a custom metric registered for operation tracking
                logger.debug(f"Operation {operation_name} took {duration:.3f}s")

        return wrapper

    return decorator


def track_api_request(endpoint: str):
    """Decorator to track API request metrics."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            metrics = get_metrics_collector()
            start_time = time.time()

            # Extract request info (this would need to be adapted based on your framework)
            method = "GET"  # Default, should extract from request
            status_code = 200  # Default, should extract from response
            user_tier = "free"  # Default, should extract from user context

            try:
                result = await func(*args, **kwargs)
                metrics.api.http_requests_total.labels(
                    method=method, endpoint=endpoint, status_code=status_code, user_tier=user_tier
                ).inc()

                return result
            except Exception as e:
                status_code = 500
                metrics.api.http_requests_total.labels(
                    method=method, endpoint=endpoint, status_code=status_code, user_tier=user_tier
                ).inc()
                raise
            finally:
                duration = time.time() - start_time
                metrics.api.http_request_duration_seconds.labels(
                    method=method, endpoint=endpoint, status_code=status_code
                ).observe(duration)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Similar implementation for sync functions
            return func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def track_business_event(event_type: str, labels: Optional[Dict[str, str]] = None):
    """Track business logic events."""
    metrics = get_metrics_collector()
    labels_dict = labels or {}

    if event_type == "generation_request":
        model = labels_dict.get("model", "unknown")
        user_tier = labels_dict.get("user_tier", "free")
        status = labels_dict.get("status", "unknown")
        metrics.business.generation_requests_total.labels(
            model=model, user_tier=user_tier, status=status
        ).inc()

    elif event_type == "model_load":
        model = labels_dict.get("model", "unknown")
        cache_hit = labels_dict.get("cache_hit", "false")
        duration = float(labels_dict.get("duration", 0))
        metrics.business.model_load_duration_seconds.labels(
            model=model, cache_hit=cache_hit
        ).observe(duration)

    logger.debug(f"Tracked business event: {event_type} with labels: {labels_dict}")


def track_system_resource(
    resource_type: str, value: float, labels: Optional[Dict[str, str]] = None
):
    """Track system resource usage."""
    metrics = get_metrics_collector()
    labels_dict = labels or {}

    if resource_type == "cpu_usage":
        cpu_type = labels_dict.get("cpu_type", "total")
        metrics.system.cpu_usage_percent.labels(cpu_type=cpu_type).set(value)

    elif resource_type == "memory_usage":
        memory_type = labels_dict.get("memory_type", "used")
        metrics.system.memory_usage_bytes.labels(memory_type=memory_type).set(value)

    elif resource_type == "gpu_usage":
        gpu_id = labels_dict.get("gpu_id", "0")
        metric_type = labels_dict.get("metric_type", "utilization")
        metrics.system.gpu_usage_percent.labels(gpu_id=gpu_id, metric_type=metric_type).set(value)


def track_task_queue_event(event_type: str, labels: Optional[Dict[str, str]] = None):
    """Track task queue events."""
    metrics = get_metrics_collector()
    labels_dict = labels or {}

    queue_name = labels_dict.get("queue_name", "default")
    task_type = labels_dict.get("task_type", "unknown")

    if event_type == "task_queued":
        status = labels_dict.get("status", "pending")
        metrics.task_queue.queue_tasks_total.labels(
            queue_name=queue_name, task_type=task_type, status=status
        ).inc()

    elif event_type == "task_processed":
        result = labels_dict.get("result", "unknown")
        duration = float(labels_dict.get("duration", 0))
        metrics.task_queue.queue_processing_duration_seconds.labels(
            queue_name=queue_name, task_type=task_type, result=result
        ).observe(duration)

    elif event_type == "queue_depth_update":
        priority = labels_dict.get("priority", "normal")
        depth = float(labels_dict.get("depth", 0))
        metrics.task_queue.queue_depth.labels(queue_name=queue_name, priority=priority).set(depth)


# Initialize metrics collection on module import
import asyncio

if os.getenv("ENABLE_METRICS", "true").lower() == "true":
    get_metrics_collector().start_background_collection()
