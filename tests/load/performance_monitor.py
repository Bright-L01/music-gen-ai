"""
Performance monitoring module for load testing.

Collects detailed performance metrics during load tests including:
- Response times and latencies
- Resource utilization
- Error rates and patterns
- Bottleneck identification
- Custom application metrics
"""

import time
import threading
import statistics
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
import psutil
import json


class PerformanceMonitor:
    """
    Comprehensive performance monitoring for load tests.

    Tracks application performance, system resources,
    and identifies bottlenecks during load testing.
    """

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.is_monitoring = False

        # Request tracking
        self.request_times = defaultdict(list)
        self.request_counts = defaultdict(int)
        self.error_counts = defaultdict(int)
        self.response_sizes = defaultdict(list)

        # Generation-specific metrics
        self.generation_latencies = []
        self.batch_latencies = []
        self.task_completion_times = []
        self.task_failures = 0
        self.status_check_latencies = []

        # Streaming metrics
        self.streaming_session_latencies = []
        self.websocket_connection_times = []
        self.websocket_chunks_received = 0
        self.websocket_sessions_completed = 0
        self.websocket_errors = []
        self.active_sessions_history = []

        # Download metrics
        self.download_sizes = []
        self.download_speeds = []

        # Database metrics
        self.database_query_times = []
        self.database_insert_times = []
        self.database_batch_times = []
        self.database_batch_sizes = []

        # Redis metrics
        self.redis_operation_times = []
        self.redis_session_times = []
        self.rate_limit_hits = 0

        # System resource tracking
        self.cpu_usage_history = deque(maxlen=1000)
        self.memory_usage_history = deque(maxlen=1000)
        self.disk_io_history = deque(maxlen=1000)
        self.network_io_history = deque(maxlen=1000)

        # Concurrent tracking
        self.concurrent_requests = 0
        self.max_concurrent_requests = 0
        self.concurrent_history = deque(maxlen=1000)

        # Thread safety
        self.lock = threading.Lock()
        self.resource_monitor_thread = None
        self.stop_monitoring_flag = threading.Event()

    def start_monitoring(self):
        """Start performance monitoring."""
        with self.lock:
            if self.is_monitoring:
                return

            self.is_monitoring = True
            self.start_time = time.time()
            self.stop_monitoring_flag.clear()

            # Start resource monitoring thread
            self.resource_monitor_thread = threading.Thread(target=self._monitor_resources)
            self.resource_monitor_thread.daemon = True
            self.resource_monitor_thread.start()

    def stop_monitoring(self):
        """Stop performance monitoring."""
        with self.lock:
            if not self.is_monitoring:
                return

            self.is_monitoring = False
            self.end_time = time.time()
            self.stop_monitoring_flag.set()

            # Wait for resource monitoring thread to finish
            if self.resource_monitor_thread:
                self.resource_monitor_thread.join(timeout=5)

    def _monitor_resources(self):
        """Monitor system resources in background thread."""
        while not self.stop_monitoring_flag.is_set():
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)

                # Memory usage
                memory = psutil.virtual_memory()
                memory_percent = memory.percent

                # Disk I/O
                disk_io = psutil.disk_io_counters()
                disk_read_mb = disk_io.read_bytes / (1024 * 1024) if disk_io else 0
                disk_write_mb = disk_io.write_bytes / (1024 * 1024) if disk_io else 0

                # Network I/O
                network_io = psutil.net_io_counters()
                network_sent_mb = network_io.bytes_sent / (1024 * 1024) if network_io else 0
                network_recv_mb = network_io.bytes_recv / (1024 * 1024) if network_io else 0

                timestamp = time.time()

                with self.lock:
                    self.cpu_usage_history.append((timestamp, cpu_percent))
                    self.memory_usage_history.append((timestamp, memory_percent))
                    self.disk_io_history.append((timestamp, disk_read_mb, disk_write_mb))
                    self.network_io_history.append((timestamp, network_sent_mb, network_recv_mb))
                    self.concurrent_history.append((timestamp, self.concurrent_requests))

            except Exception as e:
                print(f"Resource monitoring error: {e}")

            # Wait before next measurement
            self.stop_monitoring_flag.wait(1)

    def record_success(
        self, request_type: str, name: str, response_time: float, response_length: int
    ):
        """Record successful request."""
        with self.lock:
            self.request_times[name].append(response_time)
            self.request_counts[name] += 1
            self.response_sizes[name].append(response_length)
            self.concurrent_requests -= 1

    def record_failure(
        self, request_type: str, name: str, response_time: float, exception: Exception
    ):
        """Record failed request."""
        with self.lock:
            self.error_counts[name] += 1
            self.concurrent_requests -= 1

    def record_request_start(self):
        """Record request start for concurrency tracking."""
        with self.lock:
            self.concurrent_requests += 1
            self.max_concurrent_requests = max(
                self.max_concurrent_requests, self.concurrent_requests
            )

    # Generation-specific metrics
    def record_generation_latency(self, latency: float):
        """Record music generation request latency."""
        with self.lock:
            self.generation_latencies.append(latency)

    def record_batch_latency(self, latency: float, batch_size: int):
        """Record batch generation latency."""
        with self.lock:
            self.batch_latencies.append(latency)

    def record_task_completion_time(self, completion_time: float):
        """Record task completion time."""
        with self.lock:
            self.task_completion_times.append(completion_time)

    def record_task_failure(self):
        """Record task failure."""
        with self.lock:
            self.task_failures += 1

    def record_status_check_latency(self, latency: float):
        """Record status check latency."""
        with self.lock:
            self.status_check_latencies.append(latency)

    def record_rate_limit(self):
        """Record rate limit hit."""
        with self.lock:
            self.rate_limit_hits += 1

    # Streaming metrics
    def record_streaming_session_latency(self, latency: float):
        """Record streaming session creation latency."""
        with self.lock:
            self.streaming_session_latencies.append(latency)

    def record_websocket_connection_time(self, connection_time: float):
        """Record WebSocket connection time."""
        with self.lock:
            self.websocket_connection_times.append(connection_time)

    def record_websocket_chunk_received(self):
        """Record WebSocket chunk received."""
        with self.lock:
            self.websocket_chunks_received += 1

    def record_websocket_session_complete(self):
        """Record completed WebSocket session."""
        with self.lock:
            self.websocket_sessions_completed += 1

    def record_websocket_error(self, error: str):
        """Record WebSocket error."""
        with self.lock:
            self.websocket_errors.append({"timestamp": time.time(), "error": error})

    def record_active_sessions(self, count: int):
        """Record active session count."""
        with self.lock:
            self.active_sessions_history.append({"timestamp": time.time(), "count": count})

    def record_session_stopped(self):
        """Record session stop."""
        pass  # Could add specific metrics here

    # Download metrics
    def record_download_metrics(self, size_bytes: int, speed_bytes_per_sec: float):
        """Record download metrics."""
        with self.lock:
            self.download_sizes.append(size_bytes)
            self.download_speeds.append(speed_bytes_per_sec)

    # Database metrics
    def record_database_query_time(self, query_time: float):
        """Record database query time."""
        with self.lock:
            self.database_query_times.append(query_time)

    def record_database_insert_time(self, insert_time: float):
        """Record database insert time."""
        with self.lock:
            self.database_insert_times.append(insert_time)

    def record_database_batch_time(self, batch_time: float, batch_size: int):
        """Record database batch operation time."""
        with self.lock:
            self.database_batch_times.append(batch_time)
            self.database_batch_sizes.append(batch_size)

    # Redis metrics
    def record_redis_operation_time(self, operation_time: float):
        """Record Redis operation time."""
        with self.lock:
            self.redis_operation_times.append(operation_time)

    def record_redis_session_time(self, session_time: float):
        """Record Redis session operation time."""
        with self.lock:
            self.redis_session_times.append(session_time)

    def record_rate_limit_hit(self):
        """Record rate limit hit."""
        with self.lock:
            self.rate_limit_hits += 1

    def _calculate_percentiles(self, data: List[float]) -> Dict[str, float]:
        """Calculate percentile statistics."""
        if not data:
            return {"p50": 0, "p90": 0, "p95": 0, "p99": 0}

        sorted_data = sorted(data)
        return {
            "p50": statistics.median(sorted_data),
            "p90": sorted_data[int(0.9 * len(sorted_data))],
            "p95": sorted_data[int(0.95 * len(sorted_data))],
            "p99": sorted_data[int(0.99 * len(sorted_data))],
        }

    def _identify_bottlenecks(self) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks."""
        bottlenecks = []

        with self.lock:
            # High response times
            for endpoint, times in self.request_times.items():
                if times:
                    p95_time = self._calculate_percentiles(times)["p95"]
                    if p95_time > 5000:  # > 5 seconds
                        bottlenecks.append(
                            {
                                "type": "high_response_time",
                                "endpoint": endpoint,
                                "p95_time_ms": p95_time,
                                "severity": "high" if p95_time > 10000 else "medium",
                            }
                        )

            # High error rates
            for endpoint in self.error_counts:
                total_requests = self.request_counts.get(endpoint, 0) + self.error_counts[endpoint]
                if total_requests > 0:
                    error_rate = (self.error_counts[endpoint] / total_requests) * 100
                    if error_rate > 5:  # > 5% error rate
                        bottlenecks.append(
                            {
                                "type": "high_error_rate",
                                "endpoint": endpoint,
                                "error_rate_percent": error_rate,
                                "severity": "high" if error_rate > 20 else "medium",
                            }
                        )

            # Database performance issues
            if self.database_query_times:
                avg_db_time = statistics.mean(self.database_query_times)
                if avg_db_time > 1.0:  # > 1 second average
                    bottlenecks.append(
                        {
                            "type": "slow_database_queries",
                            "avg_time_seconds": avg_db_time,
                            "severity": "high" if avg_db_time > 3.0 else "medium",
                        }
                    )

            # Redis performance issues
            if self.redis_operation_times:
                avg_redis_time = statistics.mean(self.redis_operation_times)
                if avg_redis_time > 0.1:  # > 100ms average
                    bottlenecks.append(
                        {
                            "type": "slow_redis_operations",
                            "avg_time_seconds": avg_redis_time,
                            "severity": "medium",
                        }
                    )

            # High rate limiting
            total_requests = sum(self.request_counts.values())
            if total_requests > 0 and self.rate_limit_hits > 0:
                rate_limit_percent = (self.rate_limit_hits / total_requests) * 100
                if rate_limit_percent > 10:  # > 10% rate limited
                    bottlenecks.append(
                        {
                            "type": "excessive_rate_limiting",
                            "rate_limit_percent": rate_limit_percent,
                            "severity": "medium",
                        }
                    )

            # Resource utilization issues
            if self.cpu_usage_history:
                avg_cpu = statistics.mean([cpu for _, cpu in self.cpu_usage_history])
                if avg_cpu > 80:  # > 80% CPU
                    bottlenecks.append(
                        {
                            "type": "high_cpu_usage",
                            "avg_cpu_percent": avg_cpu,
                            "severity": "high" if avg_cpu > 95 else "medium",
                        }
                    )

            if self.memory_usage_history:
                avg_memory = statistics.mean([mem for _, mem in self.memory_usage_history])
                if avg_memory > 85:  # > 85% memory
                    bottlenecks.append(
                        {
                            "type": "high_memory_usage",
                            "avg_memory_percent": avg_memory,
                            "severity": "high" if avg_memory > 95 else "medium",
                        }
                    )

        return bottlenecks

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        if not self.start_time:
            return {"error": "Monitoring not started"}

        with self.lock:
            test_duration = (self.end_time or time.time()) - self.start_time

            # Calculate overall statistics
            total_requests = sum(self.request_counts.values())
            total_errors = sum(self.error_counts.values())
            error_rate = (total_errors / max(total_requests, 1)) * 100

            # Response time statistics
            all_response_times = []
            for times in self.request_times.values():
                all_response_times.extend(times)

            response_time_stats = self._calculate_percentiles(all_response_times)
            avg_response_time = statistics.mean(all_response_times) if all_response_times else 0

            # Throughput
            throughput = total_requests / test_duration if test_duration > 0 else 0

            # Generation-specific statistics
            generation_stats = {}
            if self.generation_latencies:
                generation_stats = {
                    "avg_latency_ms": statistics.mean(self.generation_latencies) * 1000,
                    "percentiles": self._calculate_percentiles(
                        [l * 1000 for l in self.generation_latencies]
                    ),
                }

            # Task completion statistics
            task_completion_stats = {}
            if self.task_completion_times:
                task_completion_stats = {
                    "avg_completion_time_seconds": statistics.mean(self.task_completion_times),
                    "percentiles": self._calculate_percentiles(self.task_completion_times),
                    "total_failures": self.task_failures,
                }

            # Database statistics
            database_stats = {}
            if self.database_query_times:
                database_stats = {
                    "avg_query_time_ms": statistics.mean(self.database_query_times) * 1000,
                    "query_percentiles": self._calculate_percentiles(
                        [t * 1000 for t in self.database_query_times]
                    ),
                    "total_queries": len(self.database_query_times),
                }

            if self.database_insert_times:
                database_stats["avg_insert_time_ms"] = (
                    statistics.mean(self.database_insert_times) * 1000
                )
                database_stats["insert_percentiles"] = self._calculate_percentiles(
                    [t * 1000 for t in self.database_insert_times]
                )

            # Redis statistics
            redis_stats = {}
            if self.redis_operation_times:
                redis_stats = {
                    "avg_operation_time_ms": statistics.mean(self.redis_operation_times) * 1000,
                    "operation_percentiles": self._calculate_percentiles(
                        [t * 1000 for t in self.redis_operation_times]
                    ),
                    "total_operations": len(self.redis_operation_times),
                    "rate_limit_hits": self.rate_limit_hits,
                }

            # WebSocket statistics
            websocket_stats = {}
            if self.websocket_connection_times:
                websocket_stats = {
                    "avg_connection_time_ms": statistics.mean(self.websocket_connection_times)
                    * 1000,
                    "connection_percentiles": self._calculate_percentiles(
                        [t * 1000 for t in self.websocket_connection_times]
                    ),
                    "total_chunks_received": self.websocket_chunks_received,
                    "sessions_completed": self.websocket_sessions_completed,
                    "total_errors": len(self.websocket_errors),
                }

            # System resource statistics
            resource_stats = {}
            if self.cpu_usage_history:
                cpu_values = [cpu for _, cpu in self.cpu_usage_history]
                resource_stats["cpu"] = {
                    "avg_percent": statistics.mean(cpu_values),
                    "max_percent": max(cpu_values),
                    "min_percent": min(cpu_values),
                }

            if self.memory_usage_history:
                memory_values = [mem for _, mem in self.memory_usage_history]
                resource_stats["memory"] = {
                    "avg_percent": statistics.mean(memory_values),
                    "max_percent": max(memory_values),
                    "min_percent": min(memory_values),
                }

            # Identify bottlenecks
            bottlenecks = self._identify_bottlenecks()

            report = {
                "test_summary": {
                    "duration_seconds": test_duration,
                    "total_requests": total_requests,
                    "total_errors": total_errors,
                    "error_rate_percent": error_rate,
                    "avg_response_time_ms": avg_response_time,
                    "throughput_rps": throughput,
                    "max_concurrent_requests": self.max_concurrent_requests,
                },
                "response_time_statistics": response_time_stats,
                "generation_performance": generation_stats,
                "task_completion_performance": task_completion_stats,
                "database_performance": database_stats,
                "redis_performance": redis_stats,
                "websocket_performance": websocket_stats,
                "system_resources": resource_stats,
                "performance_bottlenecks": bottlenecks,
                "endpoint_breakdown": {
                    endpoint: {
                        "total_requests": self.request_counts.get(endpoint, 0),
                        "total_errors": self.error_counts.get(endpoint, 0),
                        "avg_response_time_ms": statistics.mean(times) if times else 0,
                        "percentiles": self._calculate_percentiles(times),
                    }
                    for endpoint, times in self.request_times.items()
                },
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "test_configuration": {
                    "monitoring_started": datetime.fromtimestamp(
                        self.start_time, timezone.utc
                    ).isoformat(),
                    "monitoring_ended": datetime.fromtimestamp(
                        self.end_time, timezone.utc
                    ).isoformat()
                    if self.end_time
                    else None,
                },
            }

            return report
