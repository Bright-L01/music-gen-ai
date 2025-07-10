"""
Performance baseline metrics report generator for Music Gen AI.

Creates comprehensive baseline performance metrics and benchmarks
based on load testing results to establish performance standards
and monitoring thresholds for production systems.
"""

import json
import statistics
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class BaselineMetric:
    """Represents a baseline performance metric."""

    name: str
    value: float
    unit: str
    percentile: Optional[str] = None
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    category: str = "general"
    description: str = ""
    measurement_method: str = ""


@dataclass
class PerformanceBenchmark:
    """Represents a performance benchmark for comparison."""

    component: str
    metric_name: str
    baseline_value: float
    target_value: float
    current_status: str  # "meets_target", "below_target", "exceeds_target"
    improvement_needed: Optional[float] = None
    confidence_level: float = 95.0


class BaselineMetricsReportGenerator:
    """
    Generates comprehensive baseline metrics report from load testing results.

    Creates performance baselines, benchmarks, and monitoring thresholds
    for production system performance management.
    """

    def __init__(self):
        self.baseline_metrics = []
        self.benchmarks = []
        self.report_timestamp = datetime.now(timezone.utc)

        # Industry standard benchmarks for comparison
        self.industry_standards = {
            "api_response_time_p95": {
                "value": 2000,
                "unit": "ms",
                "description": "95th percentile API response time",
            },
            "api_response_time_p99": {
                "value": 5000,
                "unit": "ms",
                "description": "99th percentile API response time",
            },
            "api_throughput": {
                "value": 100,
                "unit": "rps",
                "description": "Requests per second capacity",
            },
            "error_rate": {
                "value": 1,
                "unit": "percent",
                "description": "Maximum acceptable error rate",
            },
            "database_query_p95": {
                "value": 500,
                "unit": "ms",
                "description": "95th percentile database query time",
            },
            "cache_hit_rate": {
                "value": 80,
                "unit": "percent",
                "description": "Minimum cache hit rate",
            },
            "websocket_connection_time": {
                "value": 1000,
                "unit": "ms",
                "description": "WebSocket connection establishment",
            },
            "concurrent_users": {
                "value": 1000,
                "unit": "users",
                "description": "Concurrent user capacity",
            },
            "cpu_utilization": {
                "value": 70,
                "unit": "percent",
                "description": "Maximum sustained CPU usage",
            },
            "memory_utilization": {
                "value": 80,
                "unit": "percent",
                "description": "Maximum memory usage",
            },
        }

    def generate_baseline_report(
        self, results_dir: str = "/Users/brightliu/Coding_Projects/music_gen/tests/load"
    ) -> Dict[str, Any]:
        """Generate comprehensive baseline metrics report."""
        print("Generating Performance Baseline Metrics Report...")

        # Load all test results
        test_results = self._load_test_results(results_dir)

        # Extract baseline metrics from each component
        self._extract_api_baselines(test_results.get("locust_performance", {}))
        self._extract_websocket_baselines(test_results.get("websocket_load", {}))
        self._extract_database_baselines(test_results.get("database_pool", {}))
        self._extract_redis_baselines(test_results.get("redis_pool", {}))
        self._extract_system_baselines(test_results.get("baseline_metrics", {}))

        # Create performance benchmarks
        self._create_performance_benchmarks()

        # Generate monitoring thresholds
        monitoring_thresholds = self._generate_monitoring_thresholds()

        # Create capacity planning metrics
        capacity_metrics = self._generate_capacity_metrics(test_results)

        # Generate SLA recommendations
        sla_recommendations = self._generate_sla_recommendations()

        # Create comprehensive report
        report = self._create_comprehensive_report(
            test_results, monitoring_thresholds, capacity_metrics, sla_recommendations
        )

        return report

    def _load_test_results(self, results_dir: str) -> Dict[str, Any]:
        """Load all available test results."""
        results = {}
        results_path = Path(results_dir)

        # Load all available test result files
        test_files = {
            "locust_performance": "performance_report.json",
            "websocket_load": "websocket_load_report.json",
            "database_pool": "database_pool_report.json",
            "redis_pool": "redis_pool_report.json",
            "baseline_metrics": "metrics_baseline.json",
            "bottleneck_analysis": "bottleneck_analysis_report.json",
        }

        for key, filename in test_files.items():
            file_path = results_path / filename
            if file_path.exists():
                try:
                    with open(file_path, "r") as f:
                        results[key] = json.load(f)
                except Exception as e:
                    print(f"Warning: Could not load {filename}: {e}")

        return results

    def _extract_api_baselines(self, api_data: Dict[str, Any]):
        """Extract API performance baselines."""
        if not api_data or "test_summary" not in api_data:
            return

        summary = api_data["test_summary"]

        # Response time baselines
        avg_response_time = summary.get("avg_response_time_ms", 0)
        self.baseline_metrics.append(
            BaselineMetric(
                name="api_response_time_avg",
                value=avg_response_time,
                unit="ms",
                threshold_warning=2000,
                threshold_critical=5000,
                category="api_performance",
                description="Average API response time across all endpoints",
                measurement_method="Load testing with Locust",
            )
        )

        # Throughput baselines
        throughput = summary.get("throughput_rps", 0)
        self.baseline_metrics.append(
            BaselineMetric(
                name="api_throughput",
                value=throughput,
                unit="rps",
                threshold_warning=50,
                threshold_critical=20,
                category="api_performance",
                description="API requests per second capacity",
                measurement_method="Sustained load testing",
            )
        )

        # Error rate baselines
        error_rate = summary.get("error_rate_percent", 0)
        self.baseline_metrics.append(
            BaselineMetric(
                name="api_error_rate",
                value=error_rate,
                unit="percent",
                threshold_warning=5,
                threshold_critical=15,
                category="api_reliability",
                description="API error rate under normal load",
                measurement_method="Load testing error tracking",
            )
        )

        # Concurrent capacity
        max_concurrent = summary.get("max_concurrent_requests", 0)
        self.baseline_metrics.append(
            BaselineMetric(
                name="api_concurrent_capacity",
                value=max_concurrent,
                unit="requests",
                threshold_warning=100,
                threshold_critical=50,
                category="api_capacity",
                description="Maximum concurrent request handling capacity",
                measurement_method="Concurrent load testing",
            )
        )

        # Extract percentile data if available
        if "response_time_statistics" in api_data:
            percentiles = api_data["response_time_statistics"]

            for percentile in ["p50", "p90", "p95", "p99"]:
                if percentile in percentiles:
                    self.baseline_metrics.append(
                        BaselineMetric(
                            name=f"api_response_time_{percentile}",
                            value=percentiles[percentile],
                            unit="ms",
                            percentile=percentile,
                            threshold_warning=2000 if percentile in ["p50", "p90"] else 5000,
                            threshold_critical=5000 if percentile in ["p50", "p90"] else 10000,
                            category="api_performance",
                            description=f"{percentile.upper()} percentile API response time",
                            measurement_method="Statistical analysis of load test results",
                        )
                    )

        # Generation-specific baselines
        if "generation_performance" in api_data:
            gen_perf = api_data["generation_performance"]
            if "avg_latency_ms" in gen_perf:
                self.baseline_metrics.append(
                    BaselineMetric(
                        name="music_generation_latency",
                        value=gen_perf["avg_latency_ms"],
                        unit="ms",
                        threshold_warning=10000,
                        threshold_critical=30000,
                        category="generation_performance",
                        description="Average music generation processing time",
                        measurement_method="Generation request timing",
                    )
                )

    def _extract_websocket_baselines(self, websocket_data: Dict[str, Any]):
        """Extract WebSocket performance baselines."""
        if not websocket_data or "websocket_load_test_report" not in websocket_data:
            return

        ws_report = websocket_data["websocket_load_test_report"]

        if "test_summary" in ws_report:
            summary = ws_report["test_summary"]

            # Connection success rate
            success_rate = summary.get("success_rate_percent", 0)
            self.baseline_metrics.append(
                BaselineMetric(
                    name="websocket_success_rate",
                    value=success_rate,
                    unit="percent",
                    threshold_warning=95,
                    threshold_critical=90,
                    category="websocket_reliability",
                    description="WebSocket connection success rate",
                    measurement_method="WebSocket load testing",
                )
            )

            # Maximum concurrent connections
            max_concurrent = summary.get("max_concurrent_connections", 0)
            self.baseline_metrics.append(
                BaselineMetric(
                    name="websocket_concurrent_capacity",
                    value=max_concurrent,
                    unit="connections",
                    threshold_warning=100,
                    threshold_critical=50,
                    category="websocket_capacity",
                    description="Maximum concurrent WebSocket connections",
                    measurement_method="WebSocket concurrent load testing",
                )
            )

        if "connection_performance" in ws_report:
            conn_perf = ws_report["connection_performance"]

            # Connection time
            avg_conn_time = conn_perf.get("avg_connection_time_ms", 0)
            self.baseline_metrics.append(
                BaselineMetric(
                    name="websocket_connection_time",
                    value=avg_conn_time,
                    unit="ms",
                    threshold_warning=1000,
                    threshold_critical=3000,
                    category="websocket_performance",
                    description="Average WebSocket connection establishment time",
                    measurement_method="WebSocket connection timing",
                )
            )

        if "message_performance" in ws_report:
            msg_perf = ws_report["message_performance"]

            # Message throughput
            total_messages = msg_perf.get("total_messages", 0)
            test_duration = ws_report.get("test_summary", {}).get("duration_seconds", 1)
            if test_duration > 0:
                message_rate = total_messages / test_duration
                self.baseline_metrics.append(
                    BaselineMetric(
                        name="websocket_message_rate",
                        value=message_rate,
                        unit="msg/s",
                        threshold_warning=100,
                        threshold_critical=50,
                        category="websocket_performance",
                        description="WebSocket message processing rate",
                        measurement_method="WebSocket message counting",
                    )
                )

    def _extract_database_baselines(self, database_data: Dict[str, Any]):
        """Extract database performance baselines."""
        if not database_data or "database_pool_load_test" not in database_data:
            return

        db_report = database_data["database_pool_load_test"]

        if "test_summary" in db_report:
            summary = db_report["test_summary"]

            # Query response time
            avg_response_time = summary.get("avg_response_time_seconds", 0) * 1000  # Convert to ms
            self.baseline_metrics.append(
                BaselineMetric(
                    name="database_query_time_avg",
                    value=avg_response_time,
                    unit="ms",
                    threshold_warning=500,
                    threshold_critical=2000,
                    category="database_performance",
                    description="Average database query response time",
                    measurement_method="Database load testing",
                )
            )

            # Query throughput
            max_qps = summary.get("max_queries_per_second", 0)
            self.baseline_metrics.append(
                BaselineMetric(
                    name="database_throughput",
                    value=max_qps,
                    unit="qps",
                    threshold_warning=100,
                    threshold_critical=50,
                    category="database_performance",
                    description="Database queries per second capacity",
                    measurement_method="Database throughput testing",
                )
            )

            # Success rate
            total_queries = summary.get("total_queries", 0)
            successful_queries = summary.get("total_successful_queries", 0)
            if total_queries > 0:
                success_rate = (successful_queries / total_queries) * 100
                self.baseline_metrics.append(
                    BaselineMetric(
                        name="database_success_rate",
                        value=success_rate,
                        unit="percent",
                        threshold_warning=99,
                        threshold_critical=95,
                        category="database_reliability",
                        description="Database query success rate",
                        measurement_method="Database error tracking",
                    )
                )

        # Connection pool metrics
        if "pool_configuration_analysis" in database_data:
            pool_analysis = database_data["pool_configuration_analysis"]
            best_config = pool_analysis.get("best_performing_config", {})

            if best_config:
                optimal_pool_size = best_config.get("max_size", 20)
                self.baseline_metrics.append(
                    BaselineMetric(
                        name="database_optimal_pool_size",
                        value=optimal_pool_size,
                        unit="connections",
                        category="database_configuration",
                        description="Optimal database connection pool size",
                        measurement_method="Pool configuration testing",
                    )
                )

    def _extract_redis_baselines(self, redis_data: Dict[str, Any]):
        """Extract Redis performance baselines."""
        if not redis_data or "redis_pool_load_test" not in redis_data:
            return

        redis_report = redis_data["redis_pool_load_test"]

        if "test_summary" in redis_report:
            summary = redis_report["test_summary"]

            # Operation latency
            avg_response_time = summary.get("avg_response_time_seconds", 0) * 1000  # Convert to ms
            self.baseline_metrics.append(
                BaselineMetric(
                    name="redis_operation_time_avg",
                    value=avg_response_time,
                    unit="ms",
                    threshold_warning=10,
                    threshold_critical=50,
                    category="redis_performance",
                    description="Average Redis operation response time",
                    measurement_method="Redis load testing",
                )
            )

            # Commands per second
            max_cps = summary.get("max_commands_per_second", 0)
            self.baseline_metrics.append(
                BaselineMetric(
                    name="redis_throughput",
                    value=max_cps,
                    unit="cps",
                    threshold_warning=1000,
                    threshold_critical=500,
                    category="redis_performance",
                    description="Redis commands per second capacity",
                    measurement_method="Redis throughput testing",
                )
            )

            # Cache hit rate
            cache_hit_rate = summary.get("avg_cache_hit_rate_percent", 0)
            self.baseline_metrics.append(
                BaselineMetric(
                    name="redis_cache_hit_rate",
                    value=cache_hit_rate,
                    unit="percent",
                    threshold_warning=80,
                    threshold_critical=60,
                    category="redis_performance",
                    description="Redis cache hit rate",
                    measurement_method="Cache performance testing",
                )
            )

        # Memory usage
        if "cache_performance" in redis_report:
            cache_perf = redis_report["cache_performance"]
            memory_usage = cache_perf.get("total_memory_usage_mb", 0)
            if memory_usage > 0:
                self.baseline_metrics.append(
                    BaselineMetric(
                        name="redis_memory_usage",
                        value=memory_usage,
                        unit="MB",
                        threshold_warning=1000,
                        threshold_critical=2000,
                        category="redis_resources",
                        description="Redis memory usage under load",
                        measurement_method="Memory monitoring during load testing",
                    )
                )

    def _extract_system_baselines(self, system_data: Dict[str, Any]):
        """Extract system resource baselines."""
        if not system_data or "categorized_metrics" not in system_data:
            return

        categorized = system_data["categorized_metrics"]

        # System metrics
        if "system" in categorized:
            system_metrics = categorized["system"]

            for metric_name, stats in system_metrics.items():
                if "cpu_usage_percent" in metric_name:
                    self.baseline_metrics.append(
                        BaselineMetric(
                            name="system_cpu_usage_avg",
                            value=stats.get("mean", 0),
                            unit="percent",
                            threshold_warning=70,
                            threshold_critical=90,
                            category="system_resources",
                            description="Average CPU utilization under load",
                            measurement_method="System monitoring during load testing",
                        )
                    )

                elif "memory_usage_percent" in metric_name:
                    self.baseline_metrics.append(
                        BaselineMetric(
                            name="system_memory_usage_avg",
                            value=stats.get("mean", 0),
                            unit="percent",
                            threshold_warning=80,
                            threshold_critical=95,
                            category="system_resources",
                            description="Average memory utilization under load",
                            measurement_method="System monitoring during load testing",
                        )
                    )

    def _create_performance_benchmarks(self):
        """Create performance benchmarks comparing baselines to industry standards."""
        for metric in self.baseline_metrics:
            if metric.name in self.industry_standards:
                standard = self.industry_standards[metric.name]

                # Determine if metric is "higher is better" or "lower is better"
                higher_is_better = metric.name in [
                    "api_throughput",
                    "database_throughput",
                    "redis_throughput",
                    "redis_cache_hit_rate",
                    "websocket_success_rate",
                    "database_success_rate",
                ]

                if higher_is_better:
                    if metric.value >= standard["value"]:
                        status = "meets_target"
                        improvement_needed = None
                    else:
                        status = "below_target"
                        improvement_needed = standard["value"] - metric.value
                else:
                    if metric.value <= standard["value"]:
                        status = "meets_target"
                        improvement_needed = None
                    else:
                        status = "below_target"
                        improvement_needed = metric.value - standard["value"]

                self.benchmarks.append(
                    PerformanceBenchmark(
                        component=metric.category,
                        metric_name=metric.name,
                        baseline_value=metric.value,
                        target_value=standard["value"],
                        current_status=status,
                        improvement_needed=improvement_needed,
                        confidence_level=95.0,
                    )
                )

    def _generate_monitoring_thresholds(self) -> Dict[str, Any]:
        """Generate monitoring thresholds based on baseline metrics."""
        thresholds = {
            "api_monitoring": {},
            "database_monitoring": {},
            "redis_monitoring": {},
            "websocket_monitoring": {},
            "system_monitoring": {},
        }

        for metric in self.baseline_metrics:
            category_mapping = {
                "api_performance": "api_monitoring",
                "api_reliability": "api_monitoring",
                "api_capacity": "api_monitoring",
                "database_performance": "database_monitoring",
                "database_reliability": "database_monitoring",
                "redis_performance": "redis_monitoring",
                "websocket_performance": "websocket_monitoring",
                "websocket_reliability": "websocket_monitoring",
                "system_resources": "system_monitoring",
            }

            monitoring_category = category_mapping.get(metric.category, "general_monitoring")

            if monitoring_category not in thresholds:
                thresholds[monitoring_category] = {}

            thresholds[monitoring_category][metric.name] = {
                "baseline": metric.value,
                "warning_threshold": metric.threshold_warning,
                "critical_threshold": metric.threshold_critical,
                "unit": metric.unit,
                "description": metric.description,
            }

        return thresholds

    def _generate_capacity_metrics(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate capacity planning metrics."""
        capacity_metrics = {
            "current_capacity": {},
            "scaling_projections": {},
            "bottleneck_analysis": {},
        }

        # Current capacity metrics
        for metric in self.baseline_metrics:
            if "capacity" in metric.name or "throughput" in metric.name:
                capacity_metrics["current_capacity"][metric.name] = {
                    "current_value": metric.value,
                    "unit": metric.unit,
                    "confidence_level": "high",
                }

        # Scaling projections
        api_throughput = next(
            (m.value for m in self.baseline_metrics if m.name == "api_throughput"), 0
        )
        if api_throughput > 0:
            capacity_metrics["scaling_projections"]["users_supported"] = {
                "current_concurrent_users": api_throughput * 2,  # Rough estimate
                "peak_capacity": api_throughput * 3,
                "recommended_scaling_threshold": api_throughput * 1.5,
                "scaling_recommendation": "horizontal" if api_throughput < 100 else "vertical",
            }

        # Extract bottleneck information if available
        if "bottleneck_analysis" in test_results:
            bottleneck_data = test_results["bottleneck_analysis"]
            if "critical_bottlenecks" in bottleneck_data:
                capacity_metrics["bottleneck_analysis"]["critical_constraints"] = [
                    {
                        "component": b.get("component", "unknown"),
                        "description": b.get("description", ""),
                        "impact": b.get("impact", ""),
                    }
                    for b in bottleneck_data["critical_bottlenecks"][:3]  # Top 3
                ]

        return capacity_metrics

    def _generate_sla_recommendations(self) -> Dict[str, Any]:
        """Generate SLA recommendations based on baseline performance."""
        sla_recommendations = {
            "availability_targets": {},
            "performance_targets": {},
            "error_rate_targets": {},
        }

        # Availability targets
        websocket_success = next(
            (m.value for m in self.baseline_metrics if m.name == "websocket_success_rate"), 95
        )
        db_success = next(
            (m.value for m in self.baseline_metrics if m.name == "database_success_rate"), 99
        )

        overall_availability = min(websocket_success, db_success)
        sla_recommendations["availability_targets"] = {
            "recommended_sla": f"{overall_availability:.1f}%",
            "conservative_sla": f"{overall_availability * 0.95:.1f}%",
            "aggressive_sla": f"{min(overall_availability * 1.02, 99.9):.1f}%",
            "basis": "Based on measured component reliability",
        }

        # Performance targets
        api_p95 = next(
            (m.value for m in self.baseline_metrics if m.name == "api_response_time_p95"), 2000
        )
        sla_recommendations["performance_targets"] = {
            "api_response_time_p95": f"{api_p95 * 1.2:.0f}ms",  # 20% buffer
            "api_response_time_p99": f"{api_p95 * 2:.0f}ms",  # Conservative p99
            "music_generation_time": "30s",  # Fixed target for generation
            "basis": "Based on load testing results with safety margins",
        }

        # Error rate targets
        api_error_rate = next(
            (m.value for m in self.baseline_metrics if m.name == "api_error_rate"), 1
        )
        sla_recommendations["error_rate_targets"] = {
            "max_error_rate": f"{max(api_error_rate * 2, 1):.1f}%",
            "target_error_rate": f"{max(api_error_rate, 0.1):.1f}%",
            "basis": "Based on measured error rates with operational margins",
        }

        return sla_recommendations

    def _create_comprehensive_report(
        self,
        test_results: Dict[str, Any],
        monitoring_thresholds: Dict[str, Any],
        capacity_metrics: Dict[str, Any],
        sla_recommendations: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create comprehensive baseline metrics report."""

        # Calculate summary statistics
        total_metrics = len(self.baseline_metrics)
        metrics_by_category = {}
        for metric in self.baseline_metrics:
            if metric.category not in metrics_by_category:
                metrics_by_category[metric.category] = []
            metrics_by_category[metric.category].append(metric)

        # Performance score calculation
        performance_score = self._calculate_overall_performance_score()

        # Create recommendations
        recommendations = self._generate_baseline_recommendations()

        return {
            "report_metadata": {
                "generated_at": self.report_timestamp.isoformat(),
                "total_metrics": total_metrics,
                "categories_covered": list(metrics_by_category.keys()),
                "data_sources": list(test_results.keys()),
                "methodology": "Load testing and system monitoring baseline establishment",
            },
            "executive_summary": {
                "overall_performance_score": performance_score,
                "total_benchmarks": len(self.benchmarks),
                "benchmarks_meeting_targets": len(
                    [b for b in self.benchmarks if b.current_status == "meets_target"]
                ),
                "key_strengths": self._identify_key_strengths(),
                "areas_for_improvement": self._identify_improvement_areas(),
                "readiness_assessment": self._assess_production_readiness(),
            },
            "baseline_metrics": {
                "by_category": {
                    category: [asdict(metric) for metric in metrics]
                    for category, metrics in metrics_by_category.items()
                },
                "all_metrics": [asdict(metric) for metric in self.baseline_metrics],
            },
            "performance_benchmarks": [asdict(benchmark) for benchmark in self.benchmarks],
            "monitoring_thresholds": monitoring_thresholds,
            "capacity_planning": capacity_metrics,
            "sla_recommendations": sla_recommendations,
            "production_recommendations": recommendations,
            "next_steps": self._generate_next_steps(),
        }

    def _calculate_overall_performance_score(self) -> float:
        """Calculate overall performance score (0-100)."""
        if not self.benchmarks:
            return 50.0  # Neutral score if no benchmarks

        scores = []
        for benchmark in self.benchmarks:
            if benchmark.current_status == "meets_target":
                scores.append(100)
            elif benchmark.current_status == "exceeds_target":
                scores.append(110)  # Bonus for exceeding
            else:  # below_target
                # Calculate partial score based on how close to target
                if benchmark.improvement_needed and benchmark.target_value > 0:
                    gap_ratio = benchmark.improvement_needed / benchmark.target_value
                    score = max(0, 100 - (gap_ratio * 100))
                    scores.append(score)
                else:
                    scores.append(50)  # Default partial score

        return min(100, statistics.mean(scores))

    def _identify_key_strengths(self) -> List[str]:
        """Identify key performance strengths."""
        strengths = []

        # Find metrics that exceed industry standards
        exceeding_benchmarks = [b for b in self.benchmarks if b.current_status == "exceeds_target"]
        if exceeding_benchmarks:
            strengths.append(
                f"Exceeds industry standards in {len(exceeding_benchmarks)} key performance areas"
            )

        # Find strong performance areas
        strong_metrics = [
            m
            for m in self.baseline_metrics
            if m.threshold_warning and m.value < m.threshold_warning
        ]
        if len(strong_metrics) > len(self.baseline_metrics) * 0.7:
            strengths.append("Strong performance across most measured components")

        # Specific strengths
        cache_hit_rate = next(
            (m.value for m in self.baseline_metrics if m.name == "redis_cache_hit_rate"), 0
        )
        if cache_hit_rate > 85:
            strengths.append(f"Excellent caching performance ({cache_hit_rate:.1f}% hit rate)")

        return strengths or ["Baseline performance established for monitoring"]

    def _identify_improvement_areas(self) -> List[str]:
        """Identify areas needing improvement."""
        improvements = []

        # Find metrics exceeding critical thresholds
        critical_issues = [
            m
            for m in self.baseline_metrics
            if m.threshold_critical and m.value > m.threshold_critical
        ]
        if critical_issues:
            improvements.append(f"Critical performance issues in {len(critical_issues)} areas")

        # Find benchmarks below target
        below_target = [b for b in self.benchmarks if b.current_status == "below_target"]
        if below_target:
            improvements.append(
                f"Performance below industry standards in {len(below_target)} areas"
            )

        # Specific improvements
        api_throughput = next(
            (m.value for m in self.baseline_metrics if m.name == "api_throughput"), 0
        )
        if api_throughput < 50:
            improvements.append("API throughput requires optimization for production scale")

        return improvements or ["Performance metrics within acceptable ranges"]

    def _assess_production_readiness(self) -> str:
        """Assess production readiness based on metrics."""
        critical_issues = len(
            [
                m
                for m in self.baseline_metrics
                if m.threshold_critical and m.value > m.threshold_critical
            ]
        )

        below_target_benchmarks = len(
            [b for b in self.benchmarks if b.current_status == "below_target"]
        )

        if critical_issues > 0:
            return "Not Ready - Critical performance issues must be addressed"
        elif below_target_benchmarks > len(self.benchmarks) * 0.5:
            return "Conditional - Significant optimizations recommended before production"
        elif below_target_benchmarks > 0:
            return "Ready with Monitoring - Performance acceptable with close monitoring"
        else:
            return "Production Ready - Performance meets all targets"

    def _generate_baseline_recommendations(self) -> List[str]:
        """Generate recommendations based on baseline analysis."""
        recommendations = [
            "Implement continuous performance monitoring with established thresholds",
            "Set up automated alerting for critical performance metrics",
            "Establish regular performance testing as part of CI/CD pipeline",
            "Create performance dashboards for stakeholder visibility",
        ]

        # Add specific recommendations based on findings
        critical_issues = [
            m
            for m in self.baseline_metrics
            if m.threshold_critical and m.value > m.threshold_critical
        ]
        if critical_issues:
            recommendations.append(
                "Address critical performance issues before production deployment"
            )

        # Capacity recommendations
        api_throughput = next(
            (m.value for m in self.baseline_metrics if m.name == "api_throughput"), 0
        )
        if api_throughput < 100:
            recommendations.append("Plan for horizontal scaling to meet production traffic demands")

        return recommendations

    def _generate_next_steps(self) -> List[str]:
        """Generate next steps for performance optimization."""
        return [
            "Deploy monitoring infrastructure with established thresholds",
            "Create performance regression testing suite",
            "Implement gradual production rollout with performance monitoring",
            "Schedule regular performance reviews and optimization cycles",
            "Establish performance budgets for new feature development",
            "Create incident response procedures for performance degradation",
        ]


def generate_baseline_metrics_report():
    """Generate comprehensive baseline metrics report."""
    generator = BaselineMetricsReportGenerator()

    print("Generating Comprehensive Performance Baseline Metrics Report...")
    print("=" * 70)

    # Generate report
    report = generator.generate_baseline_report()

    # Save report
    with open(
        "/Users/brightliu/Coding_Projects/music_gen/tests/load/baseline_metrics_report.json", "w"
    ) as f:
        json.dump(report, f, indent=2)

    # Save summary for quick reference
    summary = {
        "generated_at": report["report_metadata"]["generated_at"],
        "performance_score": report["executive_summary"]["overall_performance_score"],
        "production_readiness": report["executive_summary"]["readiness_assessment"],
        "total_metrics": report["report_metadata"]["total_metrics"],
        "benchmarks_meeting_targets": report["executive_summary"]["benchmarks_meeting_targets"],
        "total_benchmarks": report["executive_summary"]["total_benchmarks"],
        "key_recommendations": report["production_recommendations"][:5],
    }

    with open(
        "/Users/brightliu/Coding_Projects/music_gen/tests/load/baseline_summary.json", "w"
    ) as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print(f"\nBaseline Metrics Report Generated!")
    print(f"Report timestamp: {report['report_metadata']['generated_at']}")
    print(f"Total metrics collected: {report['report_metadata']['total_metrics']}")
    print(f"Performance categories: {len(report['report_metadata']['categories_covered'])}")
    print(
        f"Overall performance score: {report['executive_summary']['overall_performance_score']:.1f}/100"
    )
    print(
        f"Benchmarks meeting targets: {report['executive_summary']['benchmarks_meeting_targets']}/{report['executive_summary']['total_benchmarks']}"
    )
    print(f"Production readiness: {report['executive_summary']['readiness_assessment']}")
    print(f"\nKey Strengths:")
    for strength in report["executive_summary"]["key_strengths"]:
        print(f"  • {strength}")
    print(f"\nAreas for Improvement:")
    for improvement in report["executive_summary"]["areas_for_improvement"]:
        print(f"  • {improvement}")
    print(f"\nDetailed report saved to: baseline_metrics_report.json")
    print(f"Quick summary saved to: baseline_summary.json")

    return report


if __name__ == "__main__":
    # Generate baseline metrics report
    generate_baseline_metrics_report()
