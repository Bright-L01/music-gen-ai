"""
Metrics collection module for baseline performance measurements.

Collects and analyzes performance metrics to establish baseline
performance characteristics and identify optimal operating parameters.
"""

import time
import threading
import json
import statistics
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
import psutil
import requests
from dataclasses import dataclass, asdict


@dataclass
class BaselineMetric:
    """Individual baseline metric measurement."""

    name: str
    value: float
    unit: str
    timestamp: float
    percentile: Optional[str] = None
    category: str = "general"


@dataclass
class PerformanceThreshold:
    """Performance threshold definition."""

    metric_name: str
    warning_threshold: float
    critical_threshold: float
    unit: str
    direction: str = "above"  # "above" or "below"


class MetricsCollector:
    """
    Comprehensive metrics collector for establishing performance baselines.

    Collects and analyzes metrics to create baseline performance
    characteristics for the Music Gen AI API.
    """

    def __init__(self):
        self.is_collecting = False
        self.start_time = None
        self.end_time = None

        # Metric storage
        self.metrics = []
        self.aggregated_metrics = defaultdict(list)

        # Real-time tracking
        self.request_latencies = defaultdict(list)
        self.error_rates = defaultdict(float)
        self.throughput_measurements = []
        self.resource_utilization = {
            "cpu": deque(maxlen=1000),
            "memory": deque(maxlen=1000),
            "disk_io": deque(maxlen=1000),
            "network_io": deque(maxlen=1000),
        }

        # Application-specific metrics
        self.generation_metrics = {
            "latency": [],
            "success_rate": [],
            "queue_depth": [],
            "model_load_time": [],
        }

        self.database_metrics = {
            "connection_pool_usage": [],
            "query_latency": [],
            "transaction_rate": [],
            "deadlock_count": 0,
        }

        self.redis_metrics = {
            "connection_pool_usage": [],
            "operation_latency": [],
            "cache_hit_rate": [],
            "memory_usage": [],
        }

        self.websocket_metrics = {
            "connection_latency": [],
            "message_throughput": [],
            "concurrent_connections": [],
            "reconnection_rate": [],
        }

        # Performance thresholds
        self.thresholds = self._define_performance_thresholds()

        # Thread safety
        self.lock = threading.Lock()
        self.collection_thread = None
        self.stop_collection_flag = threading.Event()

    def _define_performance_thresholds(self) -> List[PerformanceThreshold]:
        """Define performance thresholds for baseline metrics."""
        return [
            # Response time thresholds
            PerformanceThreshold("api_response_time_p95", 2000, 5000, "ms", "above"),
            PerformanceThreshold("generation_latency_avg", 3000, 8000, "ms", "above"),
            PerformanceThreshold("database_query_p90", 500, 1000, "ms", "above"),
            PerformanceThreshold("redis_operation_p90", 10, 50, "ms", "above"),
            # Throughput thresholds
            PerformanceThreshold("requests_per_second", 50, 10, "rps", "below"),
            PerformanceThreshold("generations_per_minute", 30, 10, "gpm", "below"),
            # Error rate thresholds
            PerformanceThreshold("error_rate", 5, 15, "percent", "above"),
            PerformanceThreshold("task_failure_rate", 2, 10, "percent", "above"),
            # Resource utilization thresholds
            PerformanceThreshold("cpu_usage_avg", 70, 90, "percent", "above"),
            PerformanceThreshold("memory_usage_avg", 80, 95, "percent", "above"),
            PerformanceThreshold("disk_io_rate", 100, 500, "MB/s", "above"),
            # Connection pool thresholds
            PerformanceThreshold("db_pool_usage", 70, 90, "percent", "above"),
            PerformanceThreshold("redis_pool_usage", 70, 90, "percent", "above"),
            # WebSocket thresholds
            PerformanceThreshold("websocket_connection_time", 1000, 3000, "ms", "above"),
            PerformanceThreshold("websocket_message_rate", 100, 50, "msg/s", "below"),
        ]

    def start_collection(self):
        """Start metrics collection."""
        with self.lock:
            if self.is_collecting:
                return

            self.is_collecting = True
            self.start_time = time.time()
            self.stop_collection_flag.clear()

            # Start collection thread
            self.collection_thread = threading.Thread(target=self._collect_metrics)
            self.collection_thread.daemon = True
            self.collection_thread.start()

    def stop_collection(self):
        """Stop metrics collection."""
        with self.lock:
            if not self.is_collecting:
                return

            self.is_collecting = False
            self.end_time = time.time()
            self.stop_collection_flag.set()

            if self.collection_thread:
                self.collection_thread.join(timeout=5)

    def _collect_metrics(self):
        """Collect metrics in background thread."""
        while not self.stop_collection_flag.is_set():
            try:
                timestamp = time.time()

                # System resource metrics
                self._collect_system_metrics(timestamp)

                # Application metrics (if API is accessible)
                self._collect_application_metrics(timestamp)

                # Database metrics (simulated)
                self._collect_database_metrics(timestamp)

                # Redis metrics (simulated)
                self._collect_redis_metrics(timestamp)

            except Exception as e:
                print(f"Metrics collection error: {e}")

            # Wait before next collection
            self.stop_collection_flag.wait(5)  # Collect every 5 seconds

    def _collect_system_metrics(self, timestamp: float):
        """Collect system resource metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            load_avg = (
                psutil.getloadavg()[0] if hasattr(psutil, "getloadavg") else cpu_percent / 100
            )

            self._add_metric(
                "cpu_usage_percent", cpu_percent, "percent", timestamp, category="system"
            )
            self._add_metric("cpu_load_average", load_avg, "ratio", timestamp, category="system")

            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()

            self._add_metric(
                "memory_usage_percent", memory.percent, "percent", timestamp, category="system"
            )
            self._add_metric(
                "memory_available_gb",
                memory.available / (1024**3),
                "GB",
                timestamp,
                category="system",
            )
            self._add_metric(
                "swap_usage_percent", swap.percent, "percent", timestamp, category="system"
            )

            # Disk I/O metrics
            disk_io = psutil.disk_io_counters()
            if disk_io:
                disk_usage = psutil.disk_usage("/")

                self._add_metric(
                    "disk_read_mb_per_sec",
                    disk_io.read_bytes / (1024**2),
                    "MB/s",
                    timestamp,
                    category="system",
                )
                self._add_metric(
                    "disk_write_mb_per_sec",
                    disk_io.write_bytes / (1024**2),
                    "MB/s",
                    timestamp,
                    category="system",
                )
                self._add_metric(
                    "disk_usage_percent",
                    (disk_usage.used / disk_usage.total) * 100,
                    "percent",
                    timestamp,
                    category="system",
                )

            # Network I/O metrics
            network_io = psutil.net_io_counters()
            if network_io:
                self._add_metric(
                    "network_sent_mb_per_sec",
                    network_io.bytes_sent / (1024**2),
                    "MB/s",
                    timestamp,
                    category="system",
                )
                self._add_metric(
                    "network_recv_mb_per_sec",
                    network_io.bytes_recv / (1024**2),
                    "MB/s",
                    timestamp,
                    category="system",
                )

            # Process-specific metrics
            current_process = psutil.Process()
            self._add_metric(
                "process_cpu_percent",
                current_process.cpu_percent(),
                "percent",
                timestamp,
                category="process",
            )
            self._add_metric(
                "process_memory_mb",
                current_process.memory_info().rss / (1024**2),
                "MB",
                timestamp,
                category="process",
            )

        except Exception as e:
            print(f"System metrics collection error: {e}")

    def _collect_application_metrics(self, timestamp: float):
        """Collect application-specific metrics via API calls."""
        try:
            # Health check response time
            start_time = time.time()
            response = requests.get("http://localhost:8000/health", timeout=5)
            health_latency = (time.time() - start_time) * 1000

            if response.status_code == 200:
                self._add_metric(
                    "health_check_latency_ms", health_latency, "ms", timestamp, category="api"
                )

                # Parse health data if available
                health_data = response.json()
                if "uptime" in health_data:
                    self._add_metric(
                        "api_uptime_seconds",
                        health_data["uptime"],
                        "seconds",
                        timestamp,
                        category="api",
                    )

            # Detailed health metrics
            start_time = time.time()
            response = requests.get("http://localhost:8000/health/detailed", timeout=5)
            detailed_latency = (time.time() - start_time) * 1000

            if response.status_code == 200:
                self._add_metric(
                    "detailed_health_latency_ms", detailed_latency, "ms", timestamp, category="api"
                )

                health_data = response.json()
                if "memory_usage" in health_data:
                    self._add_metric(
                        "api_memory_usage_percent",
                        health_data["memory_usage"],
                        "percent",
                        timestamp,
                        category="api",
                    )

                if "model_manager" in health_data:
                    model_status = health_data["model_manager"]
                    if isinstance(model_status, dict) and "status" in model_status:
                        model_loaded = 1 if model_status["status"] == "loaded" else 0
                        self._add_metric(
                            "model_loaded_status",
                            model_loaded,
                            "boolean",
                            timestamp,
                            category="api",
                        )

        except requests.RequestException:
            # API not available during testing
            pass
        except Exception as e:
            print(f"Application metrics collection error: {e}")

    def _collect_database_metrics(self, timestamp: float):
        """Collect database performance metrics (simulated for baseline)."""
        try:
            # Simulate database connection pool metrics
            # In real implementation, these would come from database monitoring

            # Simulated connection pool usage (varies between 20-80%)
            import random

            pool_usage = random.uniform(20, 80)
            self._add_metric(
                "db_connection_pool_usage_percent",
                pool_usage,
                "percent",
                timestamp,
                category="database",
            )

            # Simulated query latency (varies between 10-200ms)
            query_latency = random.uniform(10, 200)
            self._add_metric(
                "db_query_latency_ms", query_latency, "ms", timestamp, category="database"
            )

            # Simulated transaction rate (varies between 50-500 tps)
            transaction_rate = random.uniform(50, 500)
            self._add_metric(
                "db_transaction_rate_tps", transaction_rate, "tps", timestamp, category="database"
            )

            # Simulated active connections
            active_connections = random.randint(5, 50)
            self._add_metric(
                "db_active_connections", active_connections, "count", timestamp, category="database"
            )

        except Exception as e:
            print(f"Database metrics collection error: {e}")

    def _collect_redis_metrics(self, timestamp: float):
        """Collect Redis performance metrics (simulated for baseline)."""
        try:
            # Simulate Redis metrics
            import random

            # Connection pool usage
            redis_pool_usage = random.uniform(10, 60)
            self._add_metric(
                "redis_connection_pool_usage_percent",
                redis_pool_usage,
                "percent",
                timestamp,
                category="redis",
            )

            # Operation latency
            redis_latency = random.uniform(1, 20)
            self._add_metric(
                "redis_operation_latency_ms", redis_latency, "ms", timestamp, category="redis"
            )

            # Cache hit rate
            cache_hit_rate = random.uniform(70, 95)
            self._add_metric(
                "redis_cache_hit_rate_percent",
                cache_hit_rate,
                "percent",
                timestamp,
                category="redis",
            )

            # Memory usage
            redis_memory_mb = random.uniform(100, 1000)
            self._add_metric(
                "redis_memory_usage_mb", redis_memory_mb, "MB", timestamp, category="redis"
            )

            # Active connections
            redis_connections = random.randint(10, 100)
            self._add_metric(
                "redis_active_connections", redis_connections, "count", timestamp, category="redis"
            )

        except Exception as e:
            print(f"Redis metrics collection error: {e}")

    def _add_metric(
        self,
        name: str,
        value: float,
        unit: str,
        timestamp: float,
        percentile: Optional[str] = None,
        category: str = "general",
    ):
        """Add a metric measurement."""
        metric = BaselineMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=timestamp,
            percentile=percentile,
            category=category,
        )

        with self.lock:
            self.metrics.append(metric)
            self.aggregated_metrics[name].append(value)

    def record_request_latency(self, endpoint: str, latency_ms: float):
        """Record request latency for specific endpoint."""
        timestamp = time.time()
        self._add_metric(
            f"endpoint_latency_{endpoint}", latency_ms, "ms", timestamp, category="endpoint"
        )

    def record_generation_metrics(self, latency_ms: float, success: bool, queue_depth: int):
        """Record music generation metrics."""
        timestamp = time.time()
        self._add_metric(
            "generation_latency_ms", latency_ms, "ms", timestamp, category="generation"
        )
        self._add_metric(
            "generation_success", 1 if success else 0, "boolean", timestamp, category="generation"
        )
        self._add_metric(
            "generation_queue_depth", queue_depth, "count", timestamp, category="generation"
        )

    def record_websocket_metrics(
        self, connection_time_ms: float, message_rate: float, concurrent_connections: int
    ):
        """Record WebSocket performance metrics."""
        timestamp = time.time()
        self._add_metric(
            "websocket_connection_time_ms",
            connection_time_ms,
            "ms",
            timestamp,
            category="websocket",
        )
        self._add_metric(
            "websocket_message_rate", message_rate, "msg/s", timestamp, category="websocket"
        )
        self._add_metric(
            "websocket_concurrent_connections",
            concurrent_connections,
            "count",
            timestamp,
            category="websocket",
        )

    def _calculate_baseline_statistics(self) -> Dict[str, Any]:
        """Calculate baseline statistics for all metrics."""
        baseline_stats = {}

        with self.lock:
            for metric_name, values in self.aggregated_metrics.items():
                if not values:
                    continue

                stats = {
                    "count": len(values),
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
                    "min": min(values),
                    "max": max(values),
                }

                # Calculate percentiles
                sorted_values = sorted(values)
                if len(sorted_values) >= 4:
                    stats["percentiles"] = {
                        "p25": sorted_values[int(0.25 * len(sorted_values))],
                        "p50": sorted_values[int(0.5 * len(sorted_values))],
                        "p75": sorted_values[int(0.75 * len(sorted_values))],
                        "p90": sorted_values[int(0.9 * len(sorted_values))],
                        "p95": sorted_values[int(0.95 * len(sorted_values))],
                        "p99": sorted_values[int(0.99 * len(sorted_values))],
                    }

                baseline_stats[metric_name] = stats

        return baseline_stats

    def _evaluate_performance_thresholds(
        self, baseline_stats: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Evaluate performance against defined thresholds."""
        threshold_evaluations = []

        for threshold in self.thresholds:
            if threshold.metric_name in baseline_stats:
                stats = baseline_stats[threshold.metric_name]
                mean_value = stats["mean"]

                evaluation = {
                    "metric_name": threshold.metric_name,
                    "current_value": mean_value,
                    "unit": threshold.unit,
                    "warning_threshold": threshold.warning_threshold,
                    "critical_threshold": threshold.critical_threshold,
                    "direction": threshold.direction,
                }

                if threshold.direction == "above":
                    if mean_value >= threshold.critical_threshold:
                        evaluation["status"] = "critical"
                    elif mean_value >= threshold.warning_threshold:
                        evaluation["status"] = "warning"
                    else:
                        evaluation["status"] = "ok"
                else:  # below
                    if mean_value <= threshold.critical_threshold:
                        evaluation["status"] = "critical"
                    elif mean_value <= threshold.warning_threshold:
                        evaluation["status"] = "warning"
                    else:
                        evaluation["status"] = "ok"

                threshold_evaluations.append(evaluation)

        return threshold_evaluations

    def _generate_recommendations(
        self, baseline_stats: Dict[str, Any], threshold_evaluations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate performance optimization recommendations."""
        recommendations = []

        # Check for critical issues
        critical_issues = [eval for eval in threshold_evaluations if eval["status"] == "critical"]

        for issue in critical_issues:
            metric_name = issue["metric_name"]

            if "cpu_usage" in metric_name:
                recommendations.append(
                    {
                        "type": "optimization",
                        "priority": "high",
                        "metric": metric_name,
                        "issue": "High CPU usage detected",
                        "recommendation": "Consider scaling horizontally or optimizing CPU-intensive operations",
                        "estimated_impact": "high",
                    }
                )

            elif "memory_usage" in metric_name:
                recommendations.append(
                    {
                        "type": "optimization",
                        "priority": "high",
                        "metric": metric_name,
                        "issue": "High memory usage detected",
                        "recommendation": "Investigate memory leaks and consider increasing memory allocation",
                        "estimated_impact": "high",
                    }
                )

            elif "latency" in metric_name:
                recommendations.append(
                    {
                        "type": "performance",
                        "priority": "medium",
                        "metric": metric_name,
                        "issue": "High latency detected",
                        "recommendation": "Optimize database queries, add caching, or scale database",
                        "estimated_impact": "medium",
                    }
                )

            elif "error_rate" in metric_name:
                recommendations.append(
                    {
                        "type": "reliability",
                        "priority": "high",
                        "metric": metric_name,
                        "issue": "High error rate detected",
                        "recommendation": "Investigate error causes and improve error handling",
                        "estimated_impact": "high",
                    }
                )

        # Check for optimization opportunities
        if "db_connection_pool_usage_percent" in baseline_stats:
            pool_usage = baseline_stats["db_connection_pool_usage_percent"]["mean"]
            if pool_usage > 80:
                recommendations.append(
                    {
                        "type": "scaling",
                        "priority": "medium",
                        "metric": "database_connections",
                        "issue": "Database connection pool highly utilized",
                        "recommendation": "Increase database connection pool size or optimize connection usage",
                        "estimated_impact": "medium",
                    }
                )

        return recommendations

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics analysis."""
        if not self.start_time:
            return {"error": "Collection not started"}

        test_duration = (self.end_time or time.time()) - self.start_time
        baseline_stats = self._calculate_baseline_statistics()
        threshold_evaluations = self._evaluate_performance_thresholds(baseline_stats)
        recommendations = self._generate_recommendations(baseline_stats, threshold_evaluations)

        # Categorize metrics
        categorized_metrics = defaultdict(dict)
        for metric_name, stats in baseline_stats.items():
            # Determine category from metric name
            category = "general"
            if any(keyword in metric_name for keyword in ["cpu", "memory", "disk", "network"]):
                category = "system"
            elif any(keyword in metric_name for keyword in ["db", "database"]):
                category = "database"
            elif any(keyword in metric_name for keyword in ["redis", "cache"]):
                category = "redis"
            elif any(keyword in metric_name for keyword in ["websocket", "ws"]):
                category = "websocket"
            elif any(keyword in metric_name for keyword in ["api", "endpoint", "health"]):
                category = "api"
            elif any(keyword in metric_name for keyword in ["generation", "model"]):
                category = "generation"

            categorized_metrics[category][metric_name] = stats

        return {
            "collection_summary": {
                "duration_seconds": test_duration,
                "metrics_collected": len(self.metrics),
                "unique_metrics": len(baseline_stats),
                "collection_started": datetime.fromtimestamp(
                    self.start_time, timezone.utc
                ).isoformat(),
                "collection_ended": datetime.fromtimestamp(self.end_time, timezone.utc).isoformat()
                if self.end_time
                else None,
            },
            "baseline_statistics": baseline_stats,
            "categorized_metrics": dict(categorized_metrics),
            "performance_thresholds": threshold_evaluations,
            "recommendations": recommendations,
            "raw_metrics": [asdict(metric) for metric in self.metrics[-100:]],  # Last 100 metrics
        }
