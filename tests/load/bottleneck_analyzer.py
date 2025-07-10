"""
Performance bottleneck identification and analysis for Music Gen AI.

Analyzes load test results to identify performance bottlenecks,
resource constraints, and optimization opportunities across the system.
"""

import json
import statistics
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class BottleneckIssue:
    """Represents a performance bottleneck issue."""

    category: str
    severity: str  # "critical", "high", "medium", "low"
    component: str
    description: str
    impact: str
    evidence: Dict[str, Any]
    recommendations: List[str]
    estimated_improvement: str  # e.g., "20-30% latency reduction"
    priority_score: float  # 0-100


@dataclass
class PerformanceInsight:
    """Represents a performance insight or optimization opportunity."""

    type: str  # "optimization", "scaling", "configuration", "architecture"
    component: str
    insight: str
    potential_benefit: str
    implementation_effort: str  # "low", "medium", "high"
    confidence: float  # 0-100


class BottleneckAnalyzer:
    """
    Comprehensive performance bottleneck analyzer.

    Analyzes load test results from multiple components to identify
    system-wide performance bottlenecks and optimization opportunities.
    """

    def __init__(self):
        self.bottlenecks = []
        self.insights = []
        self.analysis_timestamp = datetime.now(timezone.utc)

        # Thresholds for bottleneck detection
        self.thresholds = {
            "response_time": {"warning": 2000, "critical": 5000},  # ms  # ms
            "error_rate": {"warning": 5, "critical": 15},  # percent  # percent
            "cpu_usage": {"warning": 70, "critical": 90},  # percent  # percent
            "memory_usage": {"warning": 80, "critical": 95},  # percent  # percent
            "database_query_time": {"warning": 500, "critical": 2000},  # ms  # ms
            "redis_operation_time": {"warning": 10, "critical": 50},  # ms  # ms
            "connection_pool_usage": {"warning": 70, "critical": 90},  # percent  # percent
            "websocket_connection_time": {"warning": 1000, "critical": 3000},  # ms  # ms
        }

    def analyze_load_test_results(
        self, results_dir: str = "/Users/brightliu/Coding_Projects/music_gen/tests/load"
    ) -> Dict[str, Any]:
        """Analyze all load test results and identify bottlenecks."""
        print("Analyzing load test results for performance bottlenecks...")

        # Load all test results
        test_results = self._load_test_results(results_dir)

        # Analyze each component
        if "locust_performance" in test_results:
            self._analyze_api_performance(test_results["locust_performance"])

        if "websocket_load" in test_results:
            self._analyze_websocket_performance(test_results["websocket_load"])

        if "database_pool" in test_results:
            self._analyze_database_performance(test_results["database_pool"])

        if "redis_pool" in test_results:
            self._analyze_redis_performance(test_results["redis_pool"])

        # Cross-component analysis
        self._analyze_cross_component_issues(test_results)

        # Generate optimization insights
        self._generate_optimization_insights(test_results)

        # Create comprehensive analysis report
        return self._create_analysis_report(test_results)

    def _load_test_results(self, results_dir: str) -> Dict[str, Any]:
        """Load all available test results."""
        results = {}
        results_path = Path(results_dir)

        # Load Locust performance report
        locust_file = results_path / "performance_report.json"
        if locust_file.exists():
            with open(locust_file, "r") as f:
                results["locust_performance"] = json.load(f)

        # Load WebSocket load test report
        websocket_file = results_path / "websocket_load_report.json"
        if websocket_file.exists():
            with open(websocket_file, "r") as f:
                results["websocket_load"] = json.load(f)

        # Load database pool report
        database_file = results_path / "database_pool_report.json"
        if database_file.exists():
            with open(database_file, "r") as f:
                results["database_pool"] = json.load(f)

        # Load Redis pool report
        redis_file = results_path / "redis_pool_report.json"
        if redis_file.exists():
            with open(redis_file, "r") as f:
                results["redis_pool"] = json.load(f)

        # Load baseline metrics
        baseline_file = results_path / "metrics_baseline.json"
        if baseline_file.exists():
            with open(baseline_file, "r") as f:
                results["baseline_metrics"] = json.load(f)

        return results

    def _analyze_api_performance(self, performance_data: Dict[str, Any]):
        """Analyze API performance for bottlenecks."""
        if "test_summary" not in performance_data:
            return

        summary = performance_data["test_summary"]

        # Check response time bottlenecks
        avg_response_time = summary.get("avg_response_time_ms", 0)
        if avg_response_time > self.thresholds["response_time"]["critical"]:
            self.bottlenecks.append(
                BottleneckIssue(
                    category="api_performance",
                    severity="critical",
                    component="api_endpoints",
                    description=f"Extremely high average response time: {avg_response_time:.1f}ms",
                    impact="Severe user experience degradation, potential timeout issues",
                    evidence={
                        "avg_response_time_ms": avg_response_time,
                        "threshold": self.thresholds["response_time"]["critical"],
                    },
                    recommendations=[
                        "Implement response caching for frequently accessed endpoints",
                        "Optimize database queries and add connection pooling",
                        "Add CDN for static content delivery",
                        "Implement request queuing and rate limiting",
                        "Consider horizontal scaling of API servers",
                    ],
                    estimated_improvement="40-60% latency reduction",
                    priority_score=95,
                )
            )
        elif avg_response_time > self.thresholds["response_time"]["warning"]:
            self.bottlenecks.append(
                BottleneckIssue(
                    category="api_performance",
                    severity="high",
                    component="api_endpoints",
                    description=f"High average response time: {avg_response_time:.1f}ms",
                    impact="Noticeable performance impact on user experience",
                    evidence={"avg_response_time_ms": avg_response_time},
                    recommendations=[
                        "Profile slow endpoints and optimize database queries",
                        "Implement caching for expensive operations",
                        "Add response compression",
                        "Optimize JSON serialization",
                    ],
                    estimated_improvement="25-40% latency reduction",
                    priority_score=75,
                )
            )

        # Check error rate bottlenecks
        error_rate = summary.get("error_rate_percent", 0)
        if error_rate > self.thresholds["error_rate"]["critical"]:
            self.bottlenecks.append(
                BottleneckIssue(
                    category="reliability",
                    severity="critical",
                    component="api_endpoints",
                    description=f"Critical error rate: {error_rate:.1f}%",
                    impact="Significant service degradation, user trust issues",
                    evidence={"error_rate_percent": error_rate},
                    recommendations=[
                        "Implement comprehensive error handling and retry logic",
                        "Add circuit breakers for external dependencies",
                        "Improve input validation and sanitization",
                        "Scale backend services to handle load",
                        "Implement graceful degradation mechanisms",
                    ],
                    estimated_improvement="80-95% error reduction",
                    priority_score=100,
                )
            )

        # Check throughput bottlenecks
        throughput = summary.get("throughput_rps", 0)
        max_concurrent = summary.get("max_concurrent_requests", 0)
        if throughput < 50 and max_concurrent > 20:  # Low throughput with high concurrency
            self.bottlenecks.append(
                BottleneckIssue(
                    category="throughput",
                    severity="high",
                    component="api_scaling",
                    description=f"Low throughput: {throughput:.1f} RPS with {max_concurrent} concurrent requests",
                    impact="Poor scalability, inability to handle peak load",
                    evidence={"throughput_rps": throughput, "max_concurrent": max_concurrent},
                    recommendations=[
                        "Implement async request processing",
                        "Add load balancing across multiple API instances",
                        "Optimize critical code paths",
                        "Implement request queuing",
                        "Scale database and caching layers",
                    ],
                    estimated_improvement="200-300% throughput increase",
                    priority_score=85,
                )
            )

        # Analyze endpoint-specific issues
        if "endpoint_breakdown" in performance_data:
            self._analyze_endpoint_performance(performance_data["endpoint_breakdown"])

    def _analyze_endpoint_performance(self, endpoint_data: Dict[str, Any]):
        """Analyze individual endpoint performance."""
        for endpoint, stats in endpoint_data.items():
            avg_time = stats.get("avg_response_time_ms", 0)
            error_rate = stats.get("total_errors", 0) / max(stats.get("total_requests", 1), 1) * 100

            if avg_time > 3000:  # 3 second threshold for individual endpoints
                self.bottlenecks.append(
                    BottleneckIssue(
                        category="endpoint_performance",
                        severity="high",
                        component=f"endpoint_{endpoint}",
                        description=f"Slow endpoint {endpoint}: {avg_time:.1f}ms average",
                        impact="Poor user experience for specific functionality",
                        evidence={"endpoint": endpoint, "avg_response_time_ms": avg_time},
                        recommendations=[
                            f"Optimize {endpoint} endpoint implementation",
                            "Add endpoint-specific caching",
                            "Profile and optimize database queries for this endpoint",
                            "Consider async processing for heavy operations",
                        ],
                        estimated_improvement="50-70% latency reduction for this endpoint",
                        priority_score=70,
                    )
                )

            if error_rate > 10:  # 10% error rate for individual endpoints
                self.bottlenecks.append(
                    BottleneckIssue(
                        category="endpoint_reliability",
                        severity="high",
                        component=f"endpoint_{endpoint}",
                        description=f"High error rate for {endpoint}: {error_rate:.1f}%",
                        impact="Functionality degradation for specific features",
                        evidence={"endpoint": endpoint, "error_rate_percent": error_rate},
                        recommendations=[
                            f"Fix error handling in {endpoint}",
                            "Add comprehensive input validation",
                            "Implement retry mechanisms",
                            "Add better error logging and monitoring",
                        ],
                        estimated_improvement="90% error reduction for this endpoint",
                        priority_score=80,
                    )
                )

    def _analyze_websocket_performance(self, websocket_data: Dict[str, Any]):
        """Analyze WebSocket performance for bottlenecks."""
        if "websocket_load_test_report" not in websocket_data:
            return

        ws_report = websocket_data["websocket_load_test_report"]

        if "test_summary" in ws_report:
            summary = ws_report["test_summary"]

            # Check connection success rate
            success_rate = summary.get("success_rate_percent", 0)
            if success_rate < 90:
                severity = "critical" if success_rate < 70 else "high"
                self.bottlenecks.append(
                    BottleneckIssue(
                        category="websocket_reliability",
                        severity=severity,
                        component="websocket_connections",
                        description=f"Low WebSocket success rate: {success_rate:.1f}%",
                        impact="Streaming functionality degradation",
                        evidence={"success_rate_percent": success_rate},
                        recommendations=[
                            "Investigate WebSocket connection failures",
                            "Implement connection retry mechanisms",
                            "Optimize WebSocket server configuration",
                            "Add connection pooling for WebSocket servers",
                            "Implement graceful connection handling",
                        ],
                        estimated_improvement="20-40% success rate improvement",
                        priority_score=85,
                    )
                )

        if "connection_performance" in ws_report:
            conn_perf = ws_report["connection_performance"]
            avg_conn_time = conn_perf.get("avg_connection_time_ms", 0)

            if avg_conn_time > self.thresholds["websocket_connection_time"]["critical"]:
                self.bottlenecks.append(
                    BottleneckIssue(
                        category="websocket_performance",
                        severity="critical",
                        component="websocket_connections",
                        description=f"Extremely slow WebSocket connections: {avg_conn_time:.1f}ms",
                        impact="Poor real-time streaming experience",
                        evidence={"avg_connection_time_ms": avg_conn_time},
                        recommendations=[
                            "Optimize WebSocket handshake process",
                            "Implement connection keep-alive mechanisms",
                            "Add WebSocket connection pooling",
                            "Optimize network configuration",
                            "Consider WebSocket clustering",
                        ],
                        estimated_improvement="60-80% connection time reduction",
                        priority_score=90,
                    )
                )

    def _analyze_database_performance(self, database_data: Dict[str, Any]):
        """Analyze database performance for bottlenecks."""
        if "database_pool_load_test" not in database_data:
            return

        db_report = database_data["database_pool_load_test"]

        if "test_summary" in db_report:
            summary = db_report["test_summary"]

            # Check database response time
            avg_response_time = summary.get("avg_response_time_seconds", 0) * 1000  # Convert to ms
            if avg_response_time > self.thresholds["database_query_time"]["critical"]:
                self.bottlenecks.append(
                    BottleneckIssue(
                        category="database_performance",
                        severity="critical",
                        component="database_queries",
                        description=f"Extremely slow database queries: {avg_response_time:.1f}ms",
                        impact="Severe application performance degradation",
                        evidence={"avg_query_time_ms": avg_response_time},
                        recommendations=[
                            "Add database indexes for frequently queried columns",
                            "Optimize slow query patterns",
                            "Implement query result caching",
                            "Consider database sharding or read replicas",
                            "Optimize database connection pooling",
                        ],
                        estimated_improvement="70-90% query time reduction",
                        priority_score=95,
                    )
                )

            # Check query success rate
            total_queries = summary.get("total_queries", 0)
            successful_queries = summary.get("total_successful_queries", 0)
            if total_queries > 0:
                success_rate = (successful_queries / total_queries) * 100
                if success_rate < 95:
                    self.bottlenecks.append(
                        BottleneckIssue(
                            category="database_reliability",
                            severity="high",
                            component="database_connections",
                            description=f"Database query failure rate: {100 - success_rate:.1f}%",
                            impact="Data consistency and reliability issues",
                            evidence={"success_rate_percent": success_rate},
                            recommendations=[
                                "Investigate database connection timeouts",
                                "Increase database connection pool size",
                                "Add database connection retry logic",
                                "Monitor database resource utilization",
                                "Implement database failover mechanisms",
                            ],
                            estimated_improvement="95%+ query success rate",
                            priority_score=85,
                        )
                    )

        # Analyze connection pool performance
        if "pool_configuration_analysis" in db_report:
            pool_analysis = db_report["pool_configuration_analysis"]
            best_config = pool_analysis.get("best_performing_config", {})
            worst_config = pool_analysis.get("worst_performing_config", {})

            if best_config and worst_config:
                performance_gap = best_config.get("queries_per_second", 0) / max(
                    worst_config.get("queries_per_second", 1), 1
                )
                if performance_gap > 2:  # 2x performance difference
                    self.insights.append(
                        PerformanceInsight(
                            type="configuration",
                            component="database_pool",
                            insight=f"Database pool configuration significantly impacts performance. "
                            f"Best config ({best_config.get('min_size')}-{best_config.get('max_size')}) "
                            f"performs {performance_gap:.1f}x better than worst config.",
                            potential_benefit=f"{(performance_gap - 1) * 100:.0f}% performance improvement",
                            implementation_effort="low",
                            confidence=95,
                        )
                    )

    def _analyze_redis_performance(self, redis_data: Dict[str, Any]):
        """Analyze Redis performance for bottlenecks."""
        if "redis_pool_load_test" not in redis_data:
            return

        redis_report = redis_data["redis_pool_load_test"]

        if "test_summary" in redis_report:
            summary = redis_report["test_summary"]

            # Check Redis response time
            avg_response_time = summary.get("avg_response_time_seconds", 0) * 1000  # Convert to ms
            if avg_response_time > self.thresholds["redis_operation_time"]["critical"]:
                self.bottlenecks.append(
                    BottleneckIssue(
                        category="redis_performance",
                        severity="critical",
                        component="redis_operations",
                        description=f"Extremely slow Redis operations: {avg_response_time:.1f}ms",
                        impact="Caching ineffectiveness, session management issues",
                        evidence={"avg_operation_time_ms": avg_response_time},
                        recommendations=[
                            "Optimize Redis data structures and operations",
                            "Implement Redis pipelining for bulk operations",
                            "Add Redis clustering for horizontal scaling",
                            "Optimize Redis memory usage and eviction policies",
                            "Consider Redis connection pooling optimization",
                        ],
                        estimated_improvement="80-95% operation time reduction",
                        priority_score=85,
                    )
                )

            # Check cache performance
            if "cache_performance" in redis_report:
                cache_perf = redis_report["cache_performance"]
                cache_hit_rate = cache_perf.get("avg_cache_hit_rate_percent", 0)

                if cache_hit_rate < 70:
                    self.bottlenecks.append(
                        BottleneckIssue(
                            category="caching_efficiency",
                            severity="high",
                            component="redis_cache",
                            description=f"Low cache hit rate: {cache_hit_rate:.1f}%",
                            impact="Increased database load, slower response times",
                            evidence={"cache_hit_rate_percent": cache_hit_rate},
                            recommendations=[
                                "Review caching strategy and key patterns",
                                "Implement smarter cache eviction policies",
                                "Increase cache TTL for stable data",
                                "Add cache warming mechanisms",
                                "Optimize cache key design",
                            ],
                            estimated_improvement="20-40% response time improvement",
                            priority_score=75,
                        )
                    )

    def _analyze_cross_component_issues(self, test_results: Dict[str, Any]):
        """Analyze cross-component performance issues."""
        # Check for cascading failures
        components_with_high_errors = []

        # API errors
        if "locust_performance" in test_results:
            api_summary = test_results["locust_performance"].get("test_summary", {})
            if api_summary.get("error_rate_percent", 0) > 10:
                components_with_high_errors.append("api")

        # Database errors
        if "database_pool" in test_results:
            db_summary = (
                test_results["database_pool"]
                .get("database_pool_load_test", {})
                .get("test_summary", {})
            )
            total_queries = db_summary.get("total_queries", 0)
            successful_queries = db_summary.get("total_successful_queries", 0)
            if total_queries > 0 and (successful_queries / total_queries) < 0.9:
                components_with_high_errors.append("database")

        # WebSocket errors
        if "websocket_load" in test_results:
            ws_summary = (
                test_results["websocket_load"]
                .get("websocket_load_test_report", {})
                .get("test_summary", {})
            )
            if ws_summary.get("success_rate_percent", 100) < 90:
                components_with_high_errors.append("websocket")

        if len(components_with_high_errors) > 1:
            self.bottlenecks.append(
                BottleneckIssue(
                    category="system_wide",
                    severity="critical",
                    component="multiple_components",
                    description=f"Cascading failures detected across components: {', '.join(components_with_high_errors)}",
                    impact="System-wide reliability issues, potential total service degradation",
                    evidence={"affected_components": components_with_high_errors},
                    recommendations=[
                        "Implement circuit breakers between components",
                        "Add comprehensive health checks",
                        "Implement graceful degradation mechanisms",
                        "Add component isolation and bulkheads",
                        "Implement distributed tracing for root cause analysis",
                    ],
                    estimated_improvement="System-wide stability improvement",
                    priority_score=100,
                )
            )

        # Check for resource contention
        self._analyze_resource_contention(test_results)

    def _analyze_resource_contention(self, test_results: Dict[str, Any]):
        """Analyze resource contention issues across components."""
        # Check for memory pressure across components
        memory_usage_components = []

        if "baseline_metrics" in test_results:
            baseline = test_results["baseline_metrics"]
            if "categorized_metrics" in baseline:
                system_metrics = baseline["categorized_metrics"].get("system", {})
                for metric_name, stats in system_metrics.items():
                    if "memory" in metric_name and "percent" in metric_name:
                        avg_usage = stats.get("mean", 0)
                        if avg_usage > 85:
                            memory_usage_components.append(f"{metric_name}: {avg_usage:.1f}%")

        if memory_usage_components:
            self.bottlenecks.append(
                BottleneckIssue(
                    category="resource_contention",
                    severity="high",
                    component="system_memory",
                    description=f"High memory usage detected: {', '.join(memory_usage_components)}",
                    impact="Potential OOM errors, performance degradation",
                    evidence={"memory_usage": memory_usage_components},
                    recommendations=[
                        "Implement memory monitoring and alerting",
                        "Optimize application memory usage",
                        "Add memory limits and garbage collection tuning",
                        "Consider horizontal scaling",
                        "Implement memory-efficient data structures",
                    ],
                    estimated_improvement="20-40% memory usage reduction",
                    priority_score=80,
                )
            )

        # Check for connection pool exhaustion
        self._analyze_connection_pool_contention(test_results)

    def _analyze_connection_pool_contention(self, test_results: Dict[str, Any]):
        """Analyze connection pool contention issues."""
        pool_issues = []

        # Database pool analysis
        if "database_pool" in test_results:
            db_results = test_results["database_pool"]
            for test_key, test_data in db_results.items():
                if isinstance(test_data, dict) and "individual_test_results" in test_data:
                    for test_result in test_data["individual_test_results"]:
                        pool_exhaustion = test_result.get("pool_exhaustion_events", 0)
                        if pool_exhaustion > 5:
                            pool_issues.append(
                                f"Database pool exhaustion: {pool_exhaustion} events"
                            )

        # Redis pool analysis
        if "redis_pool" in test_results:
            redis_results = test_results["redis_pool"]
            for test_key, test_data in redis_results.items():
                if isinstance(test_data, dict) and "individual_test_results" in test_data:
                    for test_result in test_data["individual_test_results"]:
                        if test_result.get("avg_response_time_ms", 0) > 20:  # High for Redis
                            pool_issues.append(
                                f"Redis high latency: {test_result['avg_response_time_ms']:.1f}ms"
                            )

        if pool_issues:
            self.bottlenecks.append(
                BottleneckIssue(
                    category="connection_pooling",
                    severity="high",
                    component="connection_pools",
                    description=f"Connection pool issues detected: {'; '.join(pool_issues)}",
                    impact="Connection timeouts, degraded performance",
                    evidence={"pool_issues": pool_issues},
                    recommendations=[
                        "Increase connection pool sizes",
                        "Implement connection pool monitoring",
                        "Add connection timeout and retry logic",
                        "Optimize connection lifecycle management",
                        "Consider connection multiplexing",
                    ],
                    estimated_improvement="50-80% connection reliability improvement",
                    priority_score=75,
                )
            )

    def _generate_optimization_insights(self, test_results: Dict[str, Any]):
        """Generate optimization insights based on test results."""
        # Scaling insights
        self._generate_scaling_insights(test_results)

        # Configuration insights
        self._generate_configuration_insights(test_results)

        # Architecture insights
        self._generate_architecture_insights(test_results)

    def _generate_scaling_insights(self, test_results: Dict[str, Any]):
        """Generate scaling-related insights."""
        # Database scaling insights
        if "database_pool" in test_results:
            db_data = test_results["database_pool"]
            if "scaling_results" in db_data:
                scaling_results = db_data["scaling_results"]
                if "max_effective_concurrent_users" in scaling_results:
                    max_users = scaling_results["max_effective_concurrent_users"]
                    if max_users < 50:  # Low concurrent capacity
                        self.insights.append(
                            PerformanceInsight(
                                type="scaling",
                                component="database",
                                insight=f"Database can effectively handle only {max_users} concurrent users. "
                                "This indicates a scaling bottleneck.",
                                potential_benefit="2-3x concurrent user capacity increase",
                                implementation_effort="medium",
                                confidence=85,
                            )
                        )

        # API scaling insights
        if "locust_performance" in test_results:
            api_data = test_results["locust_performance"]
            if "test_summary" in api_data:
                max_concurrent = api_data["test_summary"].get("max_concurrent_requests", 0)
                throughput = api_data["test_summary"].get("throughput_rps", 0)

                if max_concurrent > 0 and throughput > 0:
                    efficiency = throughput / max_concurrent
                    if efficiency < 1:  # Less than 1 RPS per concurrent request
                        self.insights.append(
                            PerformanceInsight(
                                type="scaling",
                                component="api",
                                insight=f"API efficiency is low: {efficiency:.2f} RPS per concurrent request. "
                                "This suggests scaling bottlenecks in API layer.",
                                potential_benefit="100-200% throughput improvement",
                                implementation_effort="medium",
                                confidence=80,
                            )
                        )

    def _generate_configuration_insights(self, test_results: Dict[str, Any]):
        """Generate configuration optimization insights."""
        # Redis configuration insights
        if "redis_pool" in test_results:
            redis_data = test_results["redis_pool"]
            if "cache_performance_test" in redis_data:
                cache_test = redis_data["cache_performance_test"]
                hit_rate = cache_test.get("cache_hit_rate_percent", 0)

                if 60 <= hit_rate < 85:  # Moderate cache hit rate
                    self.insights.append(
                        PerformanceInsight(
                            type="configuration",
                            component="redis_cache",
                            insight=f"Cache hit rate is moderate ({hit_rate:.1f}%). "
                            "Optimizing cache configuration could improve performance.",
                            potential_benefit="15-25% response time improvement",
                            implementation_effort="low",
                            confidence=75,
                        )
                    )

        # Connection pool insights
        self._generate_connection_pool_insights(test_results)

    def _generate_connection_pool_insights(self, test_results: Dict[str, Any]):
        """Generate connection pool configuration insights."""
        # Database pool insights
        if "database_pool" in test_results:
            db_data = test_results["database_pool"]
            if "optimization_results" in db_data:
                opt_results = db_data["optimization_results"]
                if "optimal_configuration" in opt_results:
                    optimal = opt_results["optimal_configuration"]
                    self.insights.append(
                        PerformanceInsight(
                            type="configuration",
                            component="database_pool",
                            insight=f"Optimal database pool configuration identified: {optimal['config']}. "
                            f"This configuration shows {optimal['details']['efficiency_score']:.2f} efficiency score.",
                            potential_benefit="20-40% database performance improvement",
                            implementation_effort="low",
                            confidence=90,
                        )
                    )

    def _generate_architecture_insights(self, test_results: Dict[str, Any]):
        """Generate architecture-level insights."""
        # Check for async processing opportunities
        if "locust_performance" in test_results:
            api_data = test_results["locust_performance"]
            if "generation_performance" in api_data:
                gen_perf = api_data["generation_performance"]
                if "avg_latency_ms" in gen_perf:
                    avg_latency = gen_perf["avg_latency_ms"]
                    if avg_latency > 5000:  # 5+ second generation time
                        self.insights.append(
                            PerformanceInsight(
                                type="architecture",
                                component="music_generation",
                                insight=f"Music generation latency is high ({avg_latency:.0f}ms). "
                                "Consider implementing async processing with task queues.",
                                potential_benefit="10x improvement in perceived response time",
                                implementation_effort="high",
                                confidence=85,
                            )
                        )

        # Check for caching opportunities
        if "redis_pool" in test_results and "database_pool" in test_results:
            redis_hit_rate = 0
            if "redis_pool_load_test" in test_results["redis_pool"]:
                redis_summary = test_results["redis_pool"]["redis_pool_load_test"]["test_summary"]
                redis_hit_rate = redis_summary.get("avg_cache_hit_rate_percent", 0)

            db_query_time = 0
            if "database_pool_load_test" in test_results["database_pool"]:
                db_summary = test_results["database_pool"]["database_pool_load_test"][
                    "test_summary"
                ]
                db_query_time = db_summary.get("avg_response_time_seconds", 0) * 1000

            if redis_hit_rate < 80 and db_query_time > 100:
                self.insights.append(
                    PerformanceInsight(
                        type="architecture",
                        component="caching_layer",
                        insight=f"Low cache hit rate ({redis_hit_rate:.1f}%) combined with slow database "
                        f"queries ({db_query_time:.1f}ms) suggests need for improved caching strategy.",
                        potential_benefit="30-50% overall response time improvement",
                        implementation_effort="medium",
                        confidence=80,
                    )
                )

    def _create_analysis_report(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive bottleneck analysis report."""
        # Sort bottlenecks by priority score
        sorted_bottlenecks = sorted(self.bottlenecks, key=lambda x: x.priority_score, reverse=True)

        # Sort insights by confidence
        sorted_insights = sorted(self.insights, key=lambda x: x.confidence, reverse=True)

        # Create summary statistics
        severity_counts = {}
        category_counts = {}
        for bottleneck in self.bottlenecks:
            severity_counts[bottleneck.severity] = severity_counts.get(bottleneck.severity, 0) + 1
            category_counts[bottleneck.category] = category_counts.get(bottleneck.category, 0) + 1

        # Generate executive summary
        critical_issues = len([b for b in self.bottlenecks if b.severity == "critical"])
        high_issues = len([b for b in self.bottlenecks if b.severity == "high"])
        total_issues = len(self.bottlenecks)

        executive_summary = self._generate_executive_summary(
            critical_issues, high_issues, total_issues
        )

        # Create prioritized action plan
        action_plan = self._create_action_plan(sorted_bottlenecks[:10])  # Top 10 issues

        return {
            "analysis_metadata": {
                "analysis_timestamp": self.analysis_timestamp.isoformat(),
                "total_bottlenecks_identified": total_issues,
                "total_optimization_insights": len(self.insights),
                "data_sources": list(test_results.keys()),
            },
            "executive_summary": executive_summary,
            "severity_breakdown": severity_counts,
            "category_breakdown": category_counts,
            "critical_bottlenecks": [
                asdict(b) for b in sorted_bottlenecks if b.severity == "critical"
            ],
            "high_priority_bottlenecks": [
                asdict(b) for b in sorted_bottlenecks if b.severity == "high"
            ],
            "all_bottlenecks": [asdict(b) for b in sorted_bottlenecks],
            "optimization_insights": [asdict(i) for i in sorted_insights],
            "prioritized_action_plan": action_plan,
            "performance_recommendations": self._generate_performance_recommendations(),
            "estimated_improvements": self._calculate_total_improvements(sorted_bottlenecks),
        }

    def _generate_executive_summary(self, critical: int, high: int, total: int) -> str:
        """Generate executive summary of bottleneck analysis."""
        if critical > 0:
            summary = f"CRITICAL: {critical} critical performance bottlenecks require immediate attention. "
        else:
            summary = "No critical bottlenecks identified. "

        if high > 0:
            summary += f"{high} high-priority issues should be addressed soon. "

        if total == 0:
            summary += "System performance appears to be within acceptable thresholds."
        elif critical == 0 and high == 0:
            summary += f"{total} minor performance optimizations identified."
        else:
            summary += f"Total of {total} performance issues identified across the system."

        return summary

    def _create_action_plan(self, top_bottlenecks: List[BottleneckIssue]) -> List[Dict[str, Any]]:
        """Create prioritized action plan for addressing bottlenecks."""
        action_plan = []

        for i, bottleneck in enumerate(top_bottlenecks, 1):
            action_item = {
                "priority": i,
                "title": f"Address {bottleneck.component} {bottleneck.category}",
                "description": bottleneck.description,
                "severity": bottleneck.severity,
                "estimated_effort": self._estimate_implementation_effort(bottleneck),
                "estimated_timeline": self._estimate_timeline(bottleneck),
                "success_criteria": self._define_success_criteria(bottleneck),
                "recommended_actions": bottleneck.recommendations[:3],  # Top 3 recommendations
                "expected_improvement": bottleneck.estimated_improvement,
            }
            action_plan.append(action_item)

        return action_plan

    def _estimate_implementation_effort(self, bottleneck: BottleneckIssue) -> str:
        """Estimate implementation effort for addressing bottleneck."""
        if bottleneck.category in ["configuration", "connection_pooling"]:
            return "Low"
        elif bottleneck.category in ["api_performance", "caching_efficiency"]:
            return "Medium"
        elif bottleneck.category in ["system_wide", "architecture"]:
            return "High"
        else:
            return "Medium"

    def _estimate_timeline(self, bottleneck: BottleneckIssue) -> str:
        """Estimate timeline for addressing bottleneck."""
        effort = self._estimate_implementation_effort(bottleneck)

        if bottleneck.severity == "critical":
            if effort == "Low":
                return "1-2 days"
            elif effort == "Medium":
                return "3-5 days"
            else:
                return "1-2 weeks"
        elif bottleneck.severity == "high":
            if effort == "Low":
                return "2-3 days"
            elif effort == "Medium":
                return "1-2 weeks"
            else:
                return "2-4 weeks"
        else:
            if effort == "Low":
                return "1 week"
            elif effort == "Medium":
                return "2-3 weeks"
            else:
                return "1-2 months"

    def _define_success_criteria(self, bottleneck: BottleneckIssue) -> List[str]:
        """Define success criteria for addressing bottleneck."""
        criteria = []

        if "response_time" in bottleneck.description.lower():
            criteria.append("Response time reduced by 50% or below 1 second")

        if "error_rate" in bottleneck.description.lower():
            criteria.append("Error rate reduced below 1%")

        if "success_rate" in bottleneck.description.lower():
            criteria.append("Success rate improved to 99%+")

        if "throughput" in bottleneck.description.lower():
            criteria.append("Throughput increased by 100%+")

        if not criteria:
            criteria.append("Performance metrics within acceptable thresholds")
            criteria.append("No related error alerts in monitoring")

        return criteria

    def _generate_performance_recommendations(self) -> List[str]:
        """Generate overall performance recommendations."""
        recommendations = [
            "Implement comprehensive performance monitoring with real-time alerts",
            "Establish performance baselines and SLA thresholds",
            "Create automated load testing as part of CI/CD pipeline",
            "Implement distributed tracing for end-to-end performance visibility",
            "Add capacity planning based on growth projections",
        ]

        # Add specific recommendations based on identified issues
        if any(b.category == "api_performance" for b in self.bottlenecks):
            recommendations.append("Implement API response caching strategy")

        if any(b.category == "database_performance" for b in self.bottlenecks):
            recommendations.append("Optimize database schema and query patterns")

        if any(b.category == "websocket_performance" for b in self.bottlenecks):
            recommendations.append("Implement WebSocket connection management and failover")

        return recommendations

    def _calculate_total_improvements(self, bottlenecks: List[BottleneckIssue]) -> Dict[str, str]:
        """Calculate estimated total improvements from addressing bottlenecks."""
        improvements = {
            "response_time_improvement": "30-60% faster API responses",
            "error_rate_improvement": "90% reduction in errors",
            "throughput_improvement": "200-300% increase in requests per second",
            "user_experience_improvement": "Significantly improved application responsiveness",
            "operational_improvement": "Reduced operational overhead and support burden",
        }

        return improvements


def run_bottleneck_analysis():
    """Run comprehensive bottleneck analysis on load test results."""
    analyzer = BottleneckAnalyzer()

    print("Running Comprehensive Performance Bottleneck Analysis...")
    print("=" * 60)

    # Analyze all test results
    analysis_report = analyzer.analyze_load_test_results()

    # Save analysis report
    with open(
        "/Users/brightliu/Coding_Projects/music_gen/tests/load/bottleneck_analysis_report.json", "w"
    ) as f:
        json.dump(analysis_report, f, indent=2)

    # Print summary
    print(f"\nBottleneck Analysis Complete!")
    print(f"Analysis timestamp: {analysis_report['analysis_metadata']['analysis_timestamp']}")
    print(
        f"Total bottlenecks identified: {analysis_report['analysis_metadata']['total_bottlenecks_identified']}"
    )
    print(f"Critical issues: {len(analysis_report['critical_bottlenecks'])}")
    print(f"High priority issues: {len(analysis_report['high_priority_bottlenecks'])}")
    print(
        f"Optimization insights: {analysis_report['analysis_metadata']['total_optimization_insights']}"
    )
    print(f"\nExecutive Summary:")
    print(analysis_report["executive_summary"])
    print(f"\nDetailed analysis saved to: bottleneck_analysis_report.json")

    return analysis_report


if __name__ == "__main__":
    # Run bottleneck analysis
    run_bottleneck_analysis()
