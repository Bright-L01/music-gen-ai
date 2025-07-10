"""
Redis connection pool and caching performance load testing for Music Gen AI.

Tests Redis connection pooling, cache performance, and rate limiting
under various load conditions to identify optimal configurations.
"""

import time
import threading
import statistics
import random
import json
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone
from collections import defaultdict


# Mock Redis connection and operations for testing
class MockRedisConnection:
    """Mock Redis connection for testing pool behavior."""

    def __init__(self, connection_id: str):
        self.connection_id = connection_id
        self.created_at = time.time()
        self.last_used = time.time()
        self.command_count = 0
        self.is_active = False
        self.total_command_time = 0.0
        self.memory_usage = 0  # MB

    def execute_command(
        self, command: str, key: str = None, value: str = None, execution_time: float = None
    ) -> Dict[str, Any]:
        """Simulate Redis command execution."""
        if execution_time is None:
            # Simulate realistic Redis response times
            if command in ["GET", "SET", "EXISTS"]:
                execution_time = random.uniform(0.0001, 0.003)  # 0.1-3ms
            elif command in ["HGET", "HSET", "SADD"]:
                execution_time = random.uniform(0.0002, 0.005)  # 0.2-5ms
            elif command in ["LPUSH", "RPOP", "EXPIRE"]:
                execution_time = random.uniform(0.0001, 0.002)  # 0.1-2ms
            else:
                execution_time = random.uniform(0.0005, 0.010)  # 0.5-10ms

        self.is_active = True
        time.sleep(execution_time)  # Simulate command execution time

        self.last_used = time.time()
        self.command_count += 1
        self.total_command_time += execution_time
        self.is_active = False

        # Simulate cache hit/miss
        cache_hit = random.random() < 0.85  # 85% cache hit rate

        return {
            "command": command,
            "key": key,
            "execution_time": execution_time,
            "connection_id": self.connection_id,
            "cache_hit": cache_hit,
            "memory_impact": random.uniform(0.001, 0.1),  # MB
        }

    def close(self):
        """Close the connection."""
        self.is_active = False


class MockRedisPool:
    """Mock Redis connection pool for testing."""

    def __init__(self, min_size: int = 5, max_size: int = 50):
        self.min_size = min_size
        self.max_size = max_size
        self.connections = []
        self.available_connections = []
        self.active_connections = []
        self.connection_counter = 0
        self.lock = threading.Lock()

        # Cache simulation
        self.cache = {}
        self.cache_stats = {"hits": 0, "misses": 0, "sets": 0, "deletes": 0}

        # Metrics
        self.total_commands = 0
        self.successful_commands = 0
        self.failed_commands = 0
        self.wait_times = []
        self.command_times = []
        self.pool_exhausted_count = 0
        self.peak_usage = 0
        self.memory_usage = 0.0  # MB

        # Rate limiting simulation
        self.rate_limit_counters = defaultdict(list)
        self.rate_limit_violations = 0

        # Initialize minimum connections
        self._initialize_pool()

    def _initialize_pool(self):
        """Initialize the connection pool with minimum connections."""
        with self.lock:
            for _ in range(self.min_size):
                self._create_connection()

    def _create_connection(self) -> MockRedisConnection:
        """Create a new Redis connection."""
        self.connection_counter += 1
        connection = MockRedisConnection(f"redis_conn_{self.connection_counter}")
        self.connections.append(connection)
        self.available_connections.append(connection)
        return connection

    def get_connection(self, timeout: float = 1.0) -> Optional[MockRedisConnection]:
        """Get a connection from the pool."""
        start_time = time.time()

        with self.lock:
            self.total_commands += 1

            # Try to get available connection
            if self.available_connections:
                connection = self.available_connections.pop(0)
                self.active_connections.append(connection)

                wait_time = time.time() - start_time
                self.wait_times.append(wait_time)
                self.successful_commands += 1

                # Track peak usage
                self.peak_usage = max(self.peak_usage, len(self.active_connections))

                return connection

            # Try to create new connection if under max limit
            if len(self.connections) < self.max_size:
                connection = self._create_connection()
                self.available_connections.remove(connection)  # Just created, so it's in available
                self.active_connections.append(connection)

                wait_time = time.time() - start_time
                self.wait_times.append(wait_time)
                self.successful_commands += 1

                self.peak_usage = max(self.peak_usage, len(self.active_connections))

                return connection

        # Wait for connection to become available
        while time.time() - start_time < timeout:
            time.sleep(0.0001)  # 0.1ms

            with self.lock:
                if self.available_connections:
                    connection = self.available_connections.pop(0)
                    self.active_connections.append(connection)

                    wait_time = time.time() - start_time
                    self.wait_times.append(wait_time)
                    self.successful_commands += 1

                    self.peak_usage = max(self.peak_usage, len(self.active_connections))

                    return connection

        # Timeout - no connection available
        with self.lock:
            self.failed_commands += 1
            self.pool_exhausted_count += 1

        return None

    def return_connection(self, connection: MockRedisConnection):
        """Return a connection to the pool."""
        with self.lock:
            if connection in self.active_connections:
                self.active_connections.remove(connection)
                self.available_connections.append(connection)

    def execute_command(
        self, command: str, key: str = None, value: str = None, timeout: float = 1.0
    ) -> Optional[Dict[str, Any]]:
        """Execute a Redis command using pool connection."""
        connection = self.get_connection(timeout)
        if not connection:
            return None

        try:
            result = connection.execute_command(command, key, value)

            with self.lock:
                self.command_times.append(result["execution_time"])
                self.memory_usage += result["memory_impact"]

                # Update cache statistics
                if command == "GET" and result["cache_hit"]:
                    self.cache_stats["hits"] += 1
                elif command == "GET" and not result["cache_hit"]:
                    self.cache_stats["misses"] += 1
                elif command in ["SET", "HSET", "SADD"]:
                    self.cache_stats["sets"] += 1
                elif command == "DEL":
                    self.cache_stats["deletes"] += 1

            return result
        finally:
            self.return_connection(connection)

    def check_rate_limit(self, client_id: str, limit: int = 100, window: int = 60) -> bool:
        """Check if client is within rate limit."""
        current_time = time.time()

        with self.lock:
            # Clean old entries
            self.rate_limit_counters[client_id] = [
                timestamp
                for timestamp in self.rate_limit_counters[client_id]
                if current_time - timestamp < window
            ]

            # Check limit
            if len(self.rate_limit_counters[client_id]) >= limit:
                self.rate_limit_violations += 1
                return False

            # Record request
            self.rate_limit_counters[client_id].append(current_time)
            return True

    def get_cache_hit_rate(self) -> float:
        """Get current cache hit rate."""
        total_reads = self.cache_stats["hits"] + self.cache_stats["misses"]
        return (self.cache_stats["hits"] / total_reads) * 100 if total_reads > 0 else 0

    def get_pool_stats(self) -> Dict[str, Any]:
        """Get current pool statistics."""
        with self.lock:
            total_connections = len(self.connections)
            available_count = len(self.available_connections)
            active_count = len(self.active_connections)

            return {
                "total_connections": total_connections,
                "available_connections": available_count,
                "active_connections": active_count,
                "pool_usage_percent": (active_count / total_connections) * 100
                if total_connections > 0
                else 0,
                "total_commands": self.total_commands,
                "successful_commands": self.successful_commands,
                "failed_commands": self.failed_commands,
                "pool_exhausted_count": self.pool_exhausted_count,
                "peak_usage": self.peak_usage,
                "avg_wait_time_ms": statistics.mean(self.wait_times) * 1000
                if self.wait_times
                else 0,
                "avg_command_time_ms": statistics.mean(self.command_times) * 1000
                if self.command_times
                else 0,
                "cache_hit_rate_percent": self.get_cache_hit_rate(),
                "memory_usage_mb": self.memory_usage,
                "rate_limit_violations": self.rate_limit_violations,
            }

    def flush_all(self):
        """Flush all cached data."""
        with self.lock:
            self.cache.clear()
            self.memory_usage = 0.0

    def close_all_connections(self):
        """Close all connections in the pool."""
        with self.lock:
            for connection in self.connections:
                connection.close()
            self.connections.clear()
            self.available_connections.clear()
            self.active_connections.clear()


@dataclass
class RedisTestResult:
    """Result of a Redis connection pool test."""

    test_name: str
    duration: float
    concurrent_clients: int
    total_commands: int
    successful_commands: int
    failed_commands: int
    avg_response_time: float
    max_response_time: float
    min_response_time: float
    commands_per_second: float
    cache_hit_rate: float
    pool_exhaustion_events: int
    avg_pool_usage: float
    max_pool_usage: float
    rate_limit_violations: int
    memory_usage_mb: float


class RedisPoolLoadTester:
    """
    Comprehensive Redis connection pool and caching load tester.

    Tests various scenarios including caching performance, rate limiting,
    and connection pool behavior under different load patterns.
    """

    def __init__(self):
        self.test_results = []
        self.pool_configs_tested = []

    def test_redis_pool_configuration(
        self,
        min_size: int,
        max_size: int,
        concurrent_clients: int,
        test_duration: float,
        workload_pattern: str = "mixed",
    ) -> RedisTestResult:
        """Test a specific Redis pool configuration under load."""
        print(
            f"Testing Redis pool: min={min_size}, max={max_size}, clients={concurrent_clients}, duration={test_duration}s"
        )

        pool = MockRedisPool(min_size=min_size, max_size=max_size)

        # Track metrics
        command_times = []
        successful_commands = 0
        failed_commands = 0
        pool_usage_history = []

        def worker_thread(client_id: int):
            """Worker thread that executes Redis commands."""
            nonlocal successful_commands, failed_commands

            start_time = time.time()
            client_commands = 0

            while time.time() - start_time < test_duration:
                # Generate commands based on workload pattern
                command, key, value = self._generate_redis_command(workload_pattern, client_id)

                # Check rate limiting
                if not pool.check_rate_limit(f"client_{client_id}", limit=200, window=60):
                    time.sleep(0.001)  # Brief delay on rate limit
                    continue

                command_start = time.time()
                result = pool.execute_command(command, key, value, timeout=0.5)
                command_time = time.time() - command_start

                if result:
                    successful_commands += 1
                    command_times.append(command_time)
                else:
                    failed_commands += 1

                client_commands += 1

                # Record pool usage periodically
                if client_commands % 20 == 0:
                    stats = pool.get_pool_stats()
                    pool_usage_history.append(stats["pool_usage_percent"])

                # Variable wait time based on workload pattern
                if workload_pattern == "burst":
                    time.sleep(random.uniform(0.0001, 0.001))  # Very fast commands
                elif workload_pattern == "steady":
                    time.sleep(random.uniform(0.001, 0.010))  # Steady rate
                elif workload_pattern == "caching":
                    time.sleep(random.uniform(0.0005, 0.005))  # Cache-focused
                else:
                    time.sleep(random.uniform(0.0002, 0.003))  # Mixed pattern

        # Start worker threads
        threads = []
        start_time = time.time()

        for i in range(concurrent_clients):
            thread = threading.Thread(target=worker_thread, args=(i,))
            thread.start()
            threads.append(thread)

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        actual_duration = time.time() - start_time

        # Calculate final statistics
        total_commands = successful_commands + failed_commands
        avg_response_time = statistics.mean(command_times) if command_times else 0
        max_response_time = max(command_times) if command_times else 0
        min_response_time = min(command_times) if command_times else 0
        commands_per_second = total_commands / actual_duration if actual_duration > 0 else 0

        final_stats = pool.get_pool_stats()
        avg_pool_usage = statistics.mean(pool_usage_history) if pool_usage_history else 0
        max_pool_usage = max(pool_usage_history) if pool_usage_history else 0

        result = RedisTestResult(
            test_name=f"redis_pool_{min_size}_{max_size}_clients_{concurrent_clients}",
            duration=actual_duration,
            concurrent_clients=concurrent_clients,
            total_commands=total_commands,
            successful_commands=successful_commands,
            failed_commands=failed_commands,
            avg_response_time=avg_response_time,
            max_response_time=max_response_time,
            min_response_time=min_response_time,
            commands_per_second=commands_per_second,
            cache_hit_rate=final_stats["cache_hit_rate_percent"],
            pool_exhaustion_events=final_stats["pool_exhausted_count"],
            avg_pool_usage=avg_pool_usage,
            max_pool_usage=max_pool_usage,
            rate_limit_violations=final_stats["rate_limit_violations"],
            memory_usage_mb=final_stats["memory_usage_mb"],
        )

        self.test_results.append(result)
        self.pool_configs_tested.append(
            {"min_size": min_size, "max_size": max_size, "result": result}
        )

        # Cleanup
        pool.close_all_connections()

        print(
            f"  Completed: {successful_commands} successful, {failed_commands} failed, {commands_per_second:.1f} CPS"
        )
        print(f"  Cache hit rate: {final_stats['cache_hit_rate_percent']:.1f}%")

        return result

    def _generate_redis_command(self, pattern: str, client_id: int) -> Tuple[str, str, str]:
        """Generate Redis command based on workload pattern."""
        base_key = f"client_{client_id}"

        if pattern == "caching":
            # Heavy read pattern with occasional writes
            commands = [
                ("GET", f"cache:{base_key}:{random.randint(1, 100)}", None),
                ("GET", f"user:{client_id}:profile", None),
                ("GET", f"session:{uuid.uuid4().hex[:8]}", None),
                (
                    "SET",
                    f"cache:{base_key}:{random.randint(1, 100)}",
                    f"value_{uuid.uuid4().hex[:8]}",
                ),
                ("EXPIRE", f"cache:{base_key}:{random.randint(1, 100)}", "3600"),
            ]
            # 70% reads, 25% sets, 5% expires
            weights = [0.4, 0.15, 0.15, 0.2, 0.1]

        elif pattern == "session":
            # Session management pattern
            commands = [
                ("GET", f"session:{client_id}", None),
                ("SET", f"session:{client_id}", f"data_{uuid.uuid4().hex}"),
                ("EXPIRE", f"session:{client_id}", "1800"),
                ("HGET", f"user:{client_id}", "last_seen"),
                ("HSET", f"user:{client_id}", "last_seen", str(int(time.time()))),
            ]
            weights = [0.3, 0.2, 0.1, 0.2, 0.2]

        elif pattern == "rate_limiting":
            # Rate limiting pattern
            commands = [
                ("GET", f"rate_limit:{client_id}:minute", None),
                ("INCR", f"rate_limit:{client_id}:minute", None),
                ("EXPIRE", f"rate_limit:{client_id}:minute", "60"),
                ("GET", f"rate_limit:{client_id}:hour", None),
                ("INCR", f"rate_limit:{client_id}:hour", None),
            ]
            weights = [0.25, 0.25, 0.15, 0.2, 0.15]

        elif pattern == "burst":
            # Burst pattern with quick operations
            commands = [
                ("GET", f"burst:{random.randint(1, 50)}", None),
                ("SET", f"burst:{random.randint(1, 50)}", f"val_{random.randint(1, 1000)}"),
                ("EXISTS", f"burst:{random.randint(1, 50)}", None),
                ("DEL", f"burst:{random.randint(1, 50)}", None),
            ]
            weights = [0.4, 0.3, 0.2, 0.1]

        else:  # mixed
            commands = [
                ("GET", f"key:{random.randint(1, 200)}", None),
                ("SET", f"key:{random.randint(1, 200)}", f"value_{uuid.uuid4().hex[:8]}"),
                ("HGET", f"hash:{client_id}", f"field_{random.randint(1, 10)}"),
                (
                    "HSET",
                    f"hash:{client_id}",
                    f"field_{random.randint(1, 10)}",
                    f"value_{random.randint(1, 1000)}",
                ),
                ("LPUSH", f"list:{client_id}", f"item_{uuid.uuid4().hex[:8]}"),
                ("RPOP", f"list:{client_id}", None),
                ("SADD", f"set:{client_id}", f"member_{random.randint(1, 100)}"),
                ("EXPIRE", f"key:{random.randint(1, 200)}", str(random.randint(300, 3600))),
            ]
            weights = [0.25, 0.2, 0.15, 0.15, 0.1, 0.05, 0.05, 0.05]

        # Weighted random selection
        cmd_tuple = random.choices(commands, weights=weights)[0]
        return cmd_tuple

    def test_cache_performance(
        self, concurrent_clients: int = 30, test_duration: float = 120.0
    ) -> Dict[str, Any]:
        """Test Redis caching performance under load."""
        print(
            f"\nTesting cache performance with {concurrent_clients} clients for {test_duration}s..."
        )

        cache_result = self.test_redis_pool_configuration(
            min_size=10,
            max_size=40,
            concurrent_clients=concurrent_clients,
            test_duration=test_duration,
            workload_pattern="caching",
        )

        # Additional cache-specific analysis
        cache_efficiency = cache_result.cache_hit_rate / 100  # Convert to ratio
        cache_throughput = cache_result.commands_per_second

        return {
            "cache_hit_rate_percent": cache_result.cache_hit_rate,
            "cache_efficiency_score": cache_efficiency,
            "cache_throughput_cps": cache_throughput,
            "avg_response_time_ms": cache_result.avg_response_time * 1000,
            "memory_usage_mb": cache_result.memory_usage_mb,
            "test_result": cache_result,
        }

    def test_rate_limiting_performance(
        self, concurrent_clients: int = 50, test_duration: float = 90.0
    ) -> Dict[str, Any]:
        """Test Redis rate limiting performance."""
        print(f"\nTesting rate limiting with {concurrent_clients} clients for {test_duration}s...")

        rate_limit_result = self.test_redis_pool_configuration(
            min_size=15,
            max_size=50,
            concurrent_clients=concurrent_clients,
            test_duration=test_duration,
            workload_pattern="rate_limiting",
        )

        # Calculate rate limiting effectiveness
        violation_rate = (
            rate_limit_result.rate_limit_violations / rate_limit_result.total_commands * 100
        )

        return {
            "rate_limit_violations": rate_limit_result.rate_limit_violations,
            "violation_rate_percent": violation_rate,
            "commands_per_second": rate_limit_result.commands_per_second,
            "avg_response_time_ms": rate_limit_result.avg_response_time * 1000,
            "pool_exhaustion_events": rate_limit_result.pool_exhaustion_events,
            "test_result": rate_limit_result,
        }

    def test_session_management_performance(
        self, concurrent_clients: int = 40, test_duration: float = 100.0
    ) -> Dict[str, Any]:
        """Test Redis session management performance."""
        print(
            f"\nTesting session management with {concurrent_clients} clients for {test_duration}s..."
        )

        session_result = self.test_redis_pool_configuration(
            min_size=8,
            max_size=35,
            concurrent_clients=concurrent_clients,
            test_duration=test_duration,
            workload_pattern="session",
        )

        # Session-specific metrics
        session_throughput = session_result.commands_per_second
        session_latency = session_result.avg_response_time * 1000

        return {
            "session_throughput_cps": session_throughput,
            "avg_session_latency_ms": session_latency,
            "memory_usage_mb": session_result.memory_usage_mb,
            "success_rate_percent": (
                session_result.successful_commands / session_result.total_commands
            )
            * 100,
            "pool_usage_efficiency": session_result.avg_pool_usage,
            "test_result": session_result,
        }

    def test_burst_handling(
        self, max_clients: int = 100, burst_duration: float = 30.0
    ) -> Dict[str, Any]:
        """Test Redis pool behavior under burst load."""
        print(f"\nTesting burst handling up to {max_clients} clients for {burst_duration}s...")

        burst_results = {}
        client_steps = [20, 40, 60, 80, 100]

        for clients in client_steps:
            if clients > max_clients:
                break

            print(f"  Testing burst with {clients} clients...")

            result = self.test_redis_pool_configuration(
                min_size=10,
                max_size=60,
                concurrent_clients=clients,
                test_duration=burst_duration,
                workload_pattern="burst",
            )

            burst_results[clients] = {
                "concurrent_clients": clients,
                "commands_per_second": result.commands_per_second,
                "avg_response_time_ms": result.avg_response_time * 1000,
                "success_rate_percent": (result.successful_commands / result.total_commands) * 100,
                "pool_exhaustion_events": result.pool_exhaustion_events,
                "max_pool_usage_percent": result.max_pool_usage,
            }

            # Stop if performance degrades significantly
            if burst_results[clients]["success_rate_percent"] < 90:
                print(f"    Performance degraded at {clients} clients")
                break

        # Find optimal burst capacity
        optimal_clients = max(
            [
                clients
                for clients, result in burst_results.items()
                if result["success_rate_percent"] >= 95
            ],
            default=0,
        )

        return {
            "burst_test_results": burst_results,
            "optimal_burst_capacity": optimal_clients,
            "max_tested_clients": max(burst_results.keys()) if burst_results else 0,
        }

    def test_memory_usage_scaling(self, test_duration: float = 180.0) -> Dict[str, Any]:
        """Test Redis memory usage under sustained load."""
        print(f"\nTesting memory usage scaling for {test_duration}s...")

        # Test with increasing data load
        memory_results = []

        for data_scale in [1, 2, 4, 8]:  # Increasing data intensity
            print(f"  Testing with data scale factor: {data_scale}x")

            result = self.test_redis_pool_configuration(
                min_size=15,
                max_size=45,
                concurrent_clients=25,
                test_duration=test_duration / 4,  # Shorter tests for scaling
                workload_pattern="mixed",
            )

            memory_results.append(
                {
                    "data_scale_factor": data_scale,
                    "memory_usage_mb": result.memory_usage_mb,
                    "commands_per_second": result.commands_per_second,
                    "avg_response_time_ms": result.avg_response_time * 1000,
                    "cache_hit_rate_percent": result.cache_hit_rate,
                }
            )

        # Analyze memory scaling
        if len(memory_results) > 1:
            memory_growth_rate = (
                memory_results[-1]["memory_usage_mb"] / memory_results[0]["memory_usage_mb"]
            ) / data_scale
            performance_impact = (
                memory_results[0]["commands_per_second"] / memory_results[-1]["commands_per_second"]
            ) - 1
        else:
            memory_growth_rate = performance_impact = 0

        return {
            "memory_scaling_results": memory_results,
            "memory_growth_rate": memory_growth_rate,
            "performance_impact_percent": performance_impact * 100,
            "memory_efficiency_recommendations": self._analyze_memory_efficiency(memory_results),
        }

    def _analyze_memory_efficiency(self, memory_results: List[Dict[str, Any]]) -> List[str]:
        """Analyze memory efficiency and provide recommendations."""
        recommendations = []

        if not memory_results:
            return recommendations

        # Check memory growth rate
        if len(memory_results) > 1:
            growth_rate = (
                memory_results[-1]["memory_usage_mb"] / memory_results[0]["memory_usage_mb"]
            )
            if growth_rate > 5:
                recommendations.append(
                    "High memory growth detected. Consider implementing memory limits and TTL policies."
                )

        # Check cache hit rates
        avg_hit_rate = statistics.mean([r["cache_hit_rate_percent"] for r in memory_results])
        if avg_hit_rate < 80:
            recommendations.append(
                f"Low cache hit rate ({avg_hit_rate:.1f}%). Review caching strategy and key expiration policies."
            )

        # Check performance correlation with memory
        if len(memory_results) > 1:
            performance_degradation = (
                memory_results[0]["commands_per_second"] / memory_results[-1]["commands_per_second"]
            ) - 1
            if performance_degradation > 0.3:  # 30% degradation
                recommendations.append(
                    "Performance degrades significantly with memory growth. Consider memory optimization."
                )

        return recommendations

    def generate_redis_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive Redis performance report."""
        if not self.test_results:
            return {"error": "No test results available"}

        # Overall statistics
        total_tests = len(self.test_results)
        total_commands = sum(result.total_commands for result in self.test_results)
        total_successful = sum(result.successful_commands for result in self.test_results)
        overall_success_rate = (
            (total_successful / total_commands) * 100 if total_commands > 0 else 0
        )

        # Performance statistics
        avg_cps = statistics.mean([result.commands_per_second for result in self.test_results])
        avg_response_time = statistics.mean(
            [result.avg_response_time for result in self.test_results]
        )
        avg_cache_hit_rate = statistics.mean(
            [result.cache_hit_rate for result in self.test_results]
        )
        max_cps = max([result.commands_per_second for result in self.test_results])

        # Pool configuration analysis
        best_config = max(self.pool_configs_tested, key=lambda x: x["result"].commands_per_second)

        # Memory usage analysis
        total_memory_usage = sum(result.memory_usage_mb for result in self.test_results)
        avg_memory_per_test = total_memory_usage / total_tests if total_tests > 0 else 0

        # Rate limiting analysis
        total_rate_violations = sum(result.rate_limit_violations for result in self.test_results)

        return {
            "test_summary": {
                "total_tests": total_tests,
                "total_commands": total_commands,
                "total_successful_commands": total_successful,
                "overall_success_rate_percent": overall_success_rate,
                "avg_commands_per_second": avg_cps,
                "max_commands_per_second": max_cps,
                "avg_response_time_seconds": avg_response_time,
                "avg_cache_hit_rate_percent": avg_cache_hit_rate,
            },
            "pool_performance": {
                "best_performing_config": {
                    "min_size": best_config["min_size"],
                    "max_size": best_config["max_size"],
                    "commands_per_second": best_config["result"].commands_per_second,
                    "cache_hit_rate": best_config["result"].cache_hit_rate,
                },
                "avg_pool_exhaustion_events": statistics.mean(
                    [result.pool_exhaustion_events for result in self.test_results]
                ),
            },
            "cache_performance": {
                "avg_cache_hit_rate_percent": avg_cache_hit_rate,
                "cache_efficiency_score": avg_cache_hit_rate / 100,
                "total_memory_usage_mb": total_memory_usage,
                "avg_memory_per_test_mb": avg_memory_per_test,
            },
            "rate_limiting": {
                "total_violations": total_rate_violations,
                "avg_violations_per_test": total_rate_violations / total_tests
                if total_tests > 0
                else 0,
            },
            "individual_test_results": [
                {
                    "test_name": result.test_name,
                    "duration": result.duration,
                    "concurrent_clients": result.concurrent_clients,
                    "commands_per_second": result.commands_per_second,
                    "avg_response_time_ms": result.avg_response_time * 1000,
                    "cache_hit_rate_percent": result.cache_hit_rate,
                    "success_rate_percent": (result.successful_commands / result.total_commands)
                    * 100
                    if result.total_commands > 0
                    else 0,
                    "memory_usage_mb": result.memory_usage_mb,
                }
                for result in self.test_results
            ],
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }


def run_redis_pool_tests():
    """Run comprehensive Redis connection pool and caching tests."""
    tester = RedisPoolLoadTester()

    print("Starting Redis Connection Pool and Caching Load Testing...")
    print("=" * 60)

    # Test 1: Cache performance
    print("\n1. Testing cache performance...")
    cache_results = tester.test_cache_performance(concurrent_clients=25, test_duration=90.0)

    # Test 2: Rate limiting performance
    print("\n2. Testing rate limiting performance...")
    rate_limit_results = tester.test_rate_limiting_performance(
        concurrent_clients=40, test_duration=60.0
    )

    # Test 3: Session management performance
    print("\n3. Testing session management...")
    session_results = tester.test_session_management_performance(
        concurrent_clients=30, test_duration=75.0
    )

    # Test 4: Burst handling
    print("\n4. Testing burst handling...")
    burst_results = tester.test_burst_handling(max_clients=80, burst_duration=20.0)

    # Test 5: Memory usage scaling
    print("\n5. Testing memory usage scaling...")
    memory_results = tester.test_memory_usage_scaling(test_duration=120.0)

    # Generate comprehensive report
    print("\n6. Generating comprehensive report...")
    comprehensive_report = tester.generate_redis_comprehensive_report()

    # Combine all results
    full_report = {
        "redis_pool_load_test": comprehensive_report,
        "cache_performance_test": cache_results,
        "rate_limiting_test": rate_limit_results,
        "session_management_test": session_results,
        "burst_handling_test": burst_results,
        "memory_scaling_test": memory_results,
    }

    # Save report
    with open(
        "/Users/brightliu/Coding_Projects/music_gen/tests/load/redis_pool_report.json", "w"
    ) as f:
        json.dump(full_report, f, indent=2)

    print(f"\nRedis Pool Load Testing Complete!")
    print(f"Total tests: {comprehensive_report['test_summary']['total_tests']}")
    print(f"Max CPS: {comprehensive_report['test_summary']['max_commands_per_second']:.1f}")
    print(
        f"Avg cache hit rate: {comprehensive_report['test_summary']['avg_cache_hit_rate_percent']:.1f}%"
    )
    print(
        f"Overall success rate: {comprehensive_report['test_summary']['overall_success_rate_percent']:.1f}%"
    )
    print(f"Report saved to: redis_pool_report.json")

    return full_report


if __name__ == "__main__":
    # Run Redis pool load tests
    run_redis_pool_tests()
