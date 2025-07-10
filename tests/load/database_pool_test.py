"""
Database connection pool load testing for Music Gen AI.

Tests database connection pooling behavior under various load conditions
to identify optimal pool configurations and performance bottlenecks.
"""

import time
import asyncio
import threading
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import random
import uuid


# Mock database operations for testing
class MockDatabaseConnection:
    """Mock database connection for testing pool behavior."""

    def __init__(self, connection_id: str):
        self.connection_id = connection_id
        self.created_at = time.time()
        self.last_used = time.time()
        self.query_count = 0
        self.is_active = False
        self.total_query_time = 0.0

    def execute_query(self, query: str, execution_time: float = None) -> Dict[str, Any]:
        """Simulate query execution."""
        if execution_time is None:
            execution_time = random.uniform(0.010, 0.200)  # 10-200ms

        self.is_active = True
        time.sleep(execution_time)  # Simulate query execution time

        self.last_used = time.time()
        self.query_count += 1
        self.total_query_time += execution_time
        self.is_active = False

        return {
            "query": query,
            "execution_time": execution_time,
            "connection_id": self.connection_id,
            "rows_affected": random.randint(1, 100),
        }

    def close(self):
        """Close the connection."""
        self.is_active = False


class MockConnectionPool:
    """Mock database connection pool for testing."""

    def __init__(self, min_size: int = 5, max_size: int = 20):
        self.min_size = min_size
        self.max_size = max_size
        self.connections = []
        self.available_connections = []
        self.active_connections = []
        self.connection_counter = 0
        self.lock = threading.Lock()

        # Metrics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.wait_times = []
        self.query_times = []
        self.pool_exhausted_count = 0
        self.peak_usage = 0

        # Initialize minimum connections
        self._initialize_pool()

    def _initialize_pool(self):
        """Initialize the connection pool with minimum connections."""
        with self.lock:
            for _ in range(self.min_size):
                self._create_connection()

    def _create_connection(self) -> MockDatabaseConnection:
        """Create a new database connection."""
        self.connection_counter += 1
        connection = MockDatabaseConnection(f"conn_{self.connection_counter}")
        self.connections.append(connection)
        self.available_connections.append(connection)
        return connection

    def get_connection(self, timeout: float = 5.0) -> Optional[MockDatabaseConnection]:
        """Get a connection from the pool."""
        start_time = time.time()

        with self.lock:
            self.total_requests += 1

            # Try to get available connection
            if self.available_connections:
                connection = self.available_connections.pop(0)
                self.active_connections.append(connection)

                wait_time = time.time() - start_time
                self.wait_times.append(wait_time)
                self.successful_requests += 1

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
                self.successful_requests += 1

                self.peak_usage = max(self.peak_usage, len(self.active_connections))

                return connection

        # Wait for connection to become available
        while time.time() - start_time < timeout:
            time.sleep(0.001)  # 1ms

            with self.lock:
                if self.available_connections:
                    connection = self.available_connections.pop(0)
                    self.active_connections.append(connection)

                    wait_time = time.time() - start_time
                    self.wait_times.append(wait_time)
                    self.successful_requests += 1

                    self.peak_usage = max(self.peak_usage, len(self.active_connections))

                    return connection

        # Timeout - no connection available
        with self.lock:
            self.failed_requests += 1
            self.pool_exhausted_count += 1

        return None

    def return_connection(self, connection: MockDatabaseConnection):
        """Return a connection to the pool."""
        with self.lock:
            if connection in self.active_connections:
                self.active_connections.remove(connection)
                self.available_connections.append(connection)

    def execute_query(self, query: str, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        """Execute a query using pool connection."""
        connection = self.get_connection(timeout)
        if not connection:
            return None

        try:
            result = connection.execute_query(query)
            with self.lock:
                self.query_times.append(result["execution_time"])
            return result
        finally:
            self.return_connection(connection)

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
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "pool_exhausted_count": self.pool_exhausted_count,
                "peak_usage": self.peak_usage,
                "avg_wait_time_ms": statistics.mean(self.wait_times) * 1000
                if self.wait_times
                else 0,
                "avg_query_time_ms": statistics.mean(self.query_times) * 1000
                if self.query_times
                else 0,
            }

    def close_all_connections(self):
        """Close all connections in the pool."""
        with self.lock:
            for connection in self.connections:
                connection.close()
            self.connections.clear()
            self.available_connections.clear()
            self.active_connections.clear()


@dataclass
class DatabaseTestResult:
    """Result of a database connection pool test."""

    test_name: str
    duration: float
    concurrent_users: int
    total_queries: int
    successful_queries: int
    failed_queries: int
    avg_response_time: float
    max_response_time: float
    min_response_time: float
    queries_per_second: float
    pool_exhaustion_events: int
    avg_pool_usage: float
    max_pool_usage: float


class DatabasePoolLoadTester:
    """
    Comprehensive database connection pool load tester.

    Tests various scenarios to identify optimal pool configurations
    and performance characteristics under different load patterns.
    """

    def __init__(self):
        self.test_results = []
        self.pool_configs_tested = []

    def test_pool_configuration(
        self,
        min_size: int,
        max_size: int,
        concurrent_users: int,
        test_duration: float,
        query_pattern: str = "mixed",
    ) -> DatabaseTestResult:
        """Test a specific pool configuration under load."""
        print(
            f"Testing pool config: min={min_size}, max={max_size}, users={concurrent_users}, duration={test_duration}s"
        )

        pool = MockConnectionPool(min_size=min_size, max_size=max_size)

        # Track metrics
        query_times = []
        successful_queries = 0
        failed_queries = 0
        pool_usage_history = []

        def worker_thread(worker_id: int):
            """Worker thread that executes database queries."""
            nonlocal successful_queries, failed_queries

            start_time = time.time()
            worker_queries = 0

            while time.time() - start_time < test_duration:
                # Generate query based on pattern
                query, params = self._generate_query(query_pattern, worker_id)

                query_start = time.time()
                result = pool.execute_query(query, params, timeout=2.0)
                query_time = time.time() - query_start

                if result:
                    successful_queries += 1
                    query_times.append(query_time)
                else:
                    failed_queries += 1

                worker_queries += 1

                # Record pool usage periodically
                if worker_queries % 10 == 0:
                    stats = pool.get_pool_stats()
                    pool_usage_history.append(stats["pool_usage_percent"])

                # Variable wait time based on query pattern
                if query_pattern == "burst":
                    time.sleep(random.uniform(0.001, 0.010))  # Fast queries
                elif query_pattern == "steady":
                    time.sleep(random.uniform(0.050, 0.150))  # Steady rate
                elif query_pattern == "mixed":
                    time.sleep(random.uniform(0.010, 0.100))  # Mixed pattern
                else:
                    time.sleep(random.uniform(0.020, 0.080))  # Default

        # Start worker threads
        threads = []
        start_time = time.time()

        for i in range(concurrent_users):
            thread = threading.Thread(target=worker_thread, args=(i,))
            thread.start()
            threads.append(thread)

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        actual_duration = time.time() - start_time

        # Calculate final statistics
        total_queries = successful_queries + failed_queries
        avg_response_time = statistics.mean(query_times) if query_times else 0
        max_response_time = max(query_times) if query_times else 0
        min_response_time = min(query_times) if query_times else 0
        queries_per_second = total_queries / actual_duration if actual_duration > 0 else 0

        final_stats = pool.get_pool_stats()
        avg_pool_usage = statistics.mean(pool_usage_history) if pool_usage_history else 0
        max_pool_usage = max(pool_usage_history) if pool_usage_history else 0

        result = DatabaseTestResult(
            test_name=f"pool_{min_size}_{max_size}_users_{concurrent_users}",
            duration=actual_duration,
            concurrent_users=concurrent_users,
            total_queries=total_queries,
            successful_queries=successful_queries,
            failed_queries=failed_queries,
            avg_response_time=avg_response_time,
            max_response_time=max_response_time,
            min_response_time=min_response_time,
            queries_per_second=queries_per_second,
            pool_exhaustion_events=final_stats["pool_exhausted_count"],
            avg_pool_usage=avg_pool_usage,
            max_pool_usage=max_pool_usage,
        )

        self.test_results.append(result)
        self.pool_configs_tested.append(
            {"min_size": min_size, "max_size": max_size, "result": result}
        )

        # Cleanup
        pool.close_all_connections()

        print(
            f"  Completed: {successful_queries} successful, {failed_queries} failed, {queries_per_second:.1f} QPS"
        )

        return result

    def _generate_query(self, pattern: str, worker_id: int) -> tuple:
        """Generate database query based on pattern with parameterized queries."""
        # Use parameterized queries to prevent SQL injection
        queries = [
            ("SELECT * FROM tasks WHERE user_id = $1 AND status = $2", (worker_id, "pending")),
            (
                "INSERT INTO tasks (user_id, prompt, status) VALUES ($1, $2, $3)",
                (worker_id, "test prompt", "pending"),
            ),
            ("UPDATE tasks SET status = $1 WHERE id = $2", ("processing", random.randint(1, 1000))),
            ("SELECT COUNT(*) FROM tasks WHERE created_at > NOW() - INTERVAL '1 HOUR'", ()),
            (
                "DELETE FROM tasks WHERE status = $1 AND created_at < NOW() - INTERVAL '24 HOUR'",
                ("completed",),
            ),
        ]

        if pattern == "read_heavy":
            # 80% reads, 20% writes
            return random.choice(queries[:2] + [queries[0]] * 4)
        elif pattern == "write_heavy":
            # 60% writes, 40% reads
            return random.choice(queries[1:3] + [queries[1]] * 2)
        elif pattern == "mixed":
            return random.choice(queries)
        else:
            return random.choice(queries)

    def test_optimal_pool_size(
        self, concurrent_users: int = 50, test_duration: float = 60.0
    ) -> Dict[str, Any]:
        """Test different pool sizes to find optimal configuration."""
        print(f"\nTesting optimal pool size with {concurrent_users} concurrent users...")

        pool_configurations = [
            (5, 10),  # Small pool
            (10, 20),  # Medium pool
            (15, 30),  # Large pool
            (20, 40),  # Extra large pool
            (25, 50),  # Very large pool
        ]

        optimization_results = {}

        for min_size, max_size in pool_configurations:
            result = self.test_pool_configuration(
                min_size=min_size,
                max_size=max_size,
                concurrent_users=concurrent_users,
                test_duration=test_duration,
                query_pattern="mixed",
            )

            optimization_results[f"{min_size}_{max_size}"] = {
                "min_size": min_size,
                "max_size": max_size,
                "queries_per_second": result.queries_per_second,
                "avg_response_time_ms": result.avg_response_time * 1000,
                "pool_exhaustion_events": result.pool_exhaustion_events,
                "success_rate": (result.successful_queries / result.total_queries) * 100
                if result.total_queries > 0
                else 0,
                "efficiency_score": self._calculate_efficiency_score(result),
            }

        # Find optimal configuration
        best_config = max(optimization_results.items(), key=lambda x: x[1]["efficiency_score"])

        return {
            "test_results": optimization_results,
            "optimal_configuration": {"config": best_config[0], "details": best_config[1]},
            "recommendations": self._generate_pool_recommendations(optimization_results),
        }

    def _calculate_efficiency_score(self, result: DatabaseTestResult) -> float:
        """Calculate efficiency score for pool configuration."""
        # Score based on QPS, response time, and success rate
        success_rate = (
            (result.successful_queries / result.total_queries) * 100
            if result.total_queries > 0
            else 0
        )

        # Normalize metrics (higher is better)
        qps_score = min(result.queries_per_second / 100, 1.0)  # Normalize to 100 QPS max
        response_time_score = max(
            0, 1.0 - (result.avg_response_time / 2.0)
        )  # Penalty for >2s response
        success_score = success_rate / 100

        # Penalty for pool exhaustion
        exhaustion_penalty = min(result.pool_exhaustion_events / 10, 0.5)  # Max 50% penalty

        efficiency_score = (
            qps_score * 0.4 + response_time_score * 0.3 + success_score * 0.3
        ) - exhaustion_penalty

        return max(0, efficiency_score)

    def _generate_pool_recommendations(self, optimization_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []

        # Find configurations with high exhaustion events
        high_exhaustion_configs = [
            config
            for config, results in optimization_results.items()
            if results["pool_exhaustion_events"] > 10
        ]

        if high_exhaustion_configs:
            recommendations.append(
                f"Pool configurations {high_exhaustion_configs} experienced high exhaustion events. "
                "Consider increasing max pool size."
            )

        # Find configurations with high response times
        slow_configs = [
            config
            for config, results in optimization_results.items()
            if results["avg_response_time_ms"] > 1000
        ]

        if slow_configs:
            recommendations.append(
                f"Pool configurations {slow_configs} had high response times. "
                "Consider optimizing queries or increasing pool size."
            )

        # Find best performing configuration
        best_config = max(optimization_results.items(), key=lambda x: x[1]["efficiency_score"])
        recommendations.append(
            f"Optimal pool configuration: {best_config[0]} "
            f"(efficiency score: {best_config[1]['efficiency_score']:.2f})"
        )

        return recommendations

    def test_connection_lifecycle(
        self, pool_size: int = 20, test_duration: float = 300.0
    ) -> Dict[str, Any]:
        """Test connection lifecycle and idle connection handling."""
        print(f"\nTesting connection lifecycle with pool size {pool_size} for {test_duration}s...")

        pool = MockConnectionPool(min_size=5, max_size=pool_size)

        # Track connection metrics over time
        lifecycle_metrics = []

        def monitor_pool():
            """Monitor pool state periodically."""
            start_time = time.time()

            while time.time() - start_time < test_duration:
                stats = pool.get_pool_stats()
                stats["timestamp"] = time.time() - start_time
                lifecycle_metrics.append(stats)

                time.sleep(5)  # Check every 5 seconds

        def variable_load_worker():
            """Worker with variable load pattern."""
            start_time = time.time()

            while time.time() - start_time < test_duration:
                # Variable load pattern: bursts and quiet periods
                current_time = time.time() - start_time

                if current_time % 60 < 20:  # Burst period (20s every minute)
                    for _ in range(random.randint(5, 15)):
                        pool.execute_query("SELECT * FROM tasks LIMIT 10")
                        time.sleep(random.uniform(0.01, 0.05))
                else:  # Quiet period
                    time.sleep(random.uniform(1.0, 3.0))

        # Start monitoring and worker threads
        monitor_thread = threading.Thread(target=monitor_pool)
        monitor_thread.start()

        worker_threads = []
        for _ in range(10):  # 10 variable load workers
            thread = threading.Thread(target=variable_load_worker)
            thread.start()
            worker_threads.append(thread)

        # Wait for completion
        for thread in worker_threads:
            thread.join()

        monitor_thread.join()

        # Analyze lifecycle metrics
        if lifecycle_metrics:
            avg_pool_usage = statistics.mean([m["pool_usage_percent"] for m in lifecycle_metrics])
            max_pool_usage = max([m["pool_usage_percent"] for m in lifecycle_metrics])
            avg_available = statistics.mean([m["available_connections"] for m in lifecycle_metrics])

            # Connection utilization over time
            utilization_trend = [m["pool_usage_percent"] for m in lifecycle_metrics]
        else:
            avg_pool_usage = max_pool_usage = avg_available = 0
            utilization_trend = []

        pool.close_all_connections()

        return {
            "test_duration": test_duration,
            "pool_size": pool_size,
            "avg_pool_usage_percent": avg_pool_usage,
            "max_pool_usage_percent": max_pool_usage,
            "avg_available_connections": avg_available,
            "utilization_trend": utilization_trend,
            "lifecycle_metrics": lifecycle_metrics[-20:],  # Last 20 measurements
        }

    def test_concurrent_scaling(self, max_users: int = 100, step_size: int = 10) -> Dict[str, Any]:
        """Test how pool performs with increasing concurrent load."""
        print(f"\nTesting concurrent scaling up to {max_users} users...")

        scaling_results = {}
        pool_config = (10, 30)  # Fixed pool configuration for scaling test

        current_users = step_size

        while current_users <= max_users:
            print(f"Testing with {current_users} concurrent users...")

            result = self.test_pool_configuration(
                min_size=pool_config[0],
                max_size=pool_config[1],
                concurrent_users=current_users,
                test_duration=30.0,  # Shorter tests for scaling
                query_pattern="mixed",
            )

            scaling_results[current_users] = {
                "concurrent_users": current_users,
                "queries_per_second": result.queries_per_second,
                "avg_response_time_ms": result.avg_response_time * 1000,
                "success_rate_percent": (result.successful_queries / result.total_queries) * 100
                if result.total_queries > 0
                else 0,
                "pool_exhaustion_events": result.pool_exhaustion_events,
                "max_pool_usage": result.max_pool_usage,
            }

            # Stop if success rate drops significantly
            if scaling_results[current_users]["success_rate_percent"] < 80:
                print(f"Success rate dropped below 80% at {current_users} users")
                break

            current_users += step_size

        # Identify scaling characteristics
        max_effective_users = max(
            [
                users
                for users, result in scaling_results.items()
                if result["success_rate_percent"] >= 95
            ],
            default=0,
        )

        return {
            "scaling_results": scaling_results,
            "max_effective_concurrent_users": max_effective_users,
            "pool_configuration": pool_config,
            "scaling_characteristics": self._analyze_scaling_characteristics(scaling_results),
        }

    def _analyze_scaling_characteristics(self, scaling_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze scaling characteristics from test results."""
        if not scaling_results:
            return {}

        user_counts = list(scaling_results.keys())
        qps_values = [scaling_results[users]["queries_per_second"] for users in user_counts]
        response_times = [scaling_results[users]["avg_response_time_ms"] for users in user_counts]

        # Find sweet spot (best QPS with acceptable response time)
        sweet_spot_users = 0
        best_efficiency = 0

        for users in user_counts:
            result = scaling_results[users]
            if result["avg_response_time_ms"] < 500:  # Acceptable response time
                efficiency = result["queries_per_second"] / users  # QPS per user
                if efficiency > best_efficiency:
                    best_efficiency = efficiency
                    sweet_spot_users = users

        return {
            "sweet_spot_concurrent_users": sweet_spot_users,
            "max_qps_achieved": max(qps_values) if qps_values else 0,
            "qps_scaling_factor": max(qps_values) / min(qps_values)
            if qps_values and min(qps_values) > 0
            else 0,
            "response_time_degradation": max(response_times) / min(response_times)
            if response_times and min(response_times) > 0
            else 0,
        }

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive database pool performance report."""
        if not self.test_results:
            return {"error": "No test results available"}

        # Overall statistics
        total_tests = len(self.test_results)
        total_queries = sum(result.total_queries for result in self.test_results)
        total_successful = sum(result.successful_queries for result in self.test_results)
        overall_success_rate = (total_successful / total_queries) * 100 if total_queries > 0 else 0

        # Performance statistics
        avg_qps = statistics.mean([result.queries_per_second for result in self.test_results])
        avg_response_time = statistics.mean(
            [result.avg_response_time for result in self.test_results]
        )
        max_qps = max([result.queries_per_second for result in self.test_results])

        # Pool configuration analysis
        best_config = max(self.pool_configs_tested, key=lambda x: x["result"].queries_per_second)
        worst_config = min(self.pool_configs_tested, key=lambda x: x["result"].queries_per_second)

        return {
            "test_summary": {
                "total_tests": total_tests,
                "total_queries": total_queries,
                "total_successful_queries": total_successful,
                "overall_success_rate_percent": overall_success_rate,
                "avg_queries_per_second": avg_qps,
                "max_queries_per_second": max_qps,
                "avg_response_time_seconds": avg_response_time,
            },
            "pool_configuration_analysis": {
                "best_performing_config": {
                    "min_size": best_config["min_size"],
                    "max_size": best_config["max_size"],
                    "queries_per_second": best_config["result"].queries_per_second,
                    "success_rate": (
                        best_config["result"].successful_queries
                        / best_config["result"].total_queries
                    )
                    * 100,
                },
                "worst_performing_config": {
                    "min_size": worst_config["min_size"],
                    "max_size": worst_config["max_size"],
                    "queries_per_second": worst_config["result"].queries_per_second,
                    "success_rate": (
                        worst_config["result"].successful_queries
                        / worst_config["result"].total_queries
                    )
                    * 100,
                },
            },
            "individual_test_results": [
                {
                    "test_name": result.test_name,
                    "duration": result.duration,
                    "concurrent_users": result.concurrent_users,
                    "queries_per_second": result.queries_per_second,
                    "avg_response_time_ms": result.avg_response_time * 1000,
                    "success_rate_percent": (result.successful_queries / result.total_queries) * 100
                    if result.total_queries > 0
                    else 0,
                    "pool_exhaustion_events": result.pool_exhaustion_events,
                }
                for result in self.test_results
            ],
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }


def run_database_pool_tests():
    """Run comprehensive database connection pool tests."""
    tester = DatabasePoolLoadTester()

    print("Starting Database Connection Pool Load Testing...")
    print("=" * 50)

    # Test 1: Optimal pool size
    print("\n1. Testing optimal pool size...")
    optimization_results = tester.test_optimal_pool_size(concurrent_users=30, test_duration=45.0)

    # Test 2: Connection lifecycle
    print("\n2. Testing connection lifecycle...")
    lifecycle_results = tester.test_connection_lifecycle(pool_size=25, test_duration=120.0)

    # Test 3: Concurrent scaling
    print("\n3. Testing concurrent scaling...")
    scaling_results = tester.test_concurrent_scaling(max_users=60, step_size=10)

    # Generate comprehensive report
    print("\n4. Generating comprehensive report...")
    comprehensive_report = tester.generate_comprehensive_report()

    # Combine all results
    full_report = {
        "database_pool_load_test": comprehensive_report,
        "optimization_results": optimization_results,
        "lifecycle_results": lifecycle_results,
        "scaling_results": scaling_results,
    }

    # Save report
    with open(
        "/Users/brightliu/Coding_Projects/music_gen/tests/load/database_pool_report.json", "w"
    ) as f:
        json.dump(full_report, f, indent=2)

    print(f"\nDatabase Pool Load Testing Complete!")
    print(f"Total tests: {comprehensive_report['test_summary']['total_tests']}")
    print(f"Max QPS: {comprehensive_report['test_summary']['max_queries_per_second']:.1f}")
    print(
        f"Overall success rate: {comprehensive_report['test_summary']['overall_success_rate_percent']:.1f}%"
    )
    print(f"Report saved to: database_pool_report.json")

    return full_report


if __name__ == "__main__":
    # Run database pool load tests
    run_database_pool_tests()
