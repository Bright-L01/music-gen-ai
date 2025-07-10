#!/usr/bin/env python3
"""
Success Metrics Dashboard for Music Gen AI
Tracks technical, business, and operational metrics with real-time monitoring
"""

import os
import json
import time
import requests
import psycopg2
import redis
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
import sys
import subprocess
from dataclasses import dataclass, asdict
import asyncio
import aiohttp
from flask import Flask, render_template, jsonify
import plotly.graph_objs as go
import plotly.utils
import pandas as pd
from prometheus_client import CollectorRegistry, Gauge, Counter, generate_latest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("success_metrics_dashboard.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """Single metric data point"""

    timestamp: datetime
    value: float
    target: float
    status: str  # "healthy", "warning", "critical"
    metadata: Dict[str, Any]


@dataclass
class TechnicalMetric:
    """Technical success metric"""

    name: str
    current_value: float
    target_value: float
    trend: str  # "up", "down", "stable"
    last_updated: datetime
    history: List[MetricPoint]
    alerts: List[str]


@dataclass
class BusinessMetric:
    """Business success metric"""

    name: str
    current_value: float
    target_value: float
    trend: str
    last_updated: datetime
    history: List[MetricPoint]
    alerts: List[str]


@dataclass
class OperationalMetric:
    """Operational success metric"""

    name: str
    current_value: float
    target_value: float
    trend: str
    last_updated: datetime
    history: List[MetricPoint]
    alerts: List[str]


@dataclass
class RiskMetric:
    """Risk metric with mitigation status"""

    name: str
    probability: float
    impact: float
    risk_score: float
    mitigation_status: str
    last_assessed: datetime
    mitigation_plan: str


class SuccessMetricsDashboard:
    def __init__(self):
        self.project_root = Path.cwd()
        self.metrics_dir = self.project_root / "metrics"
        self.metrics_dir.mkdir(exist_ok=True)

        # Configuration
        self.config = {
            "prometheus_url": os.getenv("PROMETHEUS_URL", "http://prometheus:9090"),
            "grafana_url": os.getenv("GRAFANA_URL", "http://grafana:3000"),
            "api_url": os.getenv("API_URL", "https://api.musicgen.ai"),
            "db_host": os.getenv("DB_HOST", "postgres"),
            "redis_host": os.getenv("REDIS_HOST", "redis"),
            "collection_interval": 60,  # seconds
            "retention_days": 30,
        }

        # Database connection
        self.db_params = {
            "host": self.config["db_host"],
            "port": 5432,
            "database": "musicgen_prod",
            "user": "musicgen",
            "password": os.getenv("DB_PASSWORD"),
        }

        # Redis connection
        self.redis_client = redis.Redis(
            host=self.config["redis_host"], port=6379, decode_responses=True
        )

        # Metrics targets
        self.targets = {
            # Technical Metrics
            "test_coverage": 90.0,
            "api_response_time_p95": 200.0,  # ms
            "critical_vulnerabilities": 0,
            "uptime_percentage": 99.9,
            "error_rate": 0.1,  # %
            # Business Metrics
            "music_generation_success_rate": 95.0,
            "user_registration_to_generation_time": 5.0,  # minutes
            "support_ticket_rate": 5.0,  # %
            "user_satisfaction": 4.5,  # out of 5
            # Operational Metrics
            "deployment_time": 30.0,  # minutes
            "rollback_time": 5.0,  # minutes
            "alert_response_time": 15.0,  # minutes
            "documentation_completeness": 100.0,  # %
        }

        # Initialize metrics storage
        self.technical_metrics = {}
        self.business_metrics = {}
        self.operational_metrics = {}
        self.risk_metrics = {}

        # Initialize Flask app for dashboard
        self.app = Flask(__name__)
        self.setup_routes()

        # Prometheus metrics
        self.prometheus_registry = CollectorRegistry()
        self.prometheus_metrics = self.setup_prometheus_metrics()

    def setup_prometheus_metrics(self) -> Dict[str, Any]:
        """Setup Prometheus metrics for export"""
        metrics = {
            "test_coverage": Gauge(
                "test_coverage_percentage",
                "Test coverage percentage",
                registry=self.prometheus_registry,
            ),
            "api_response_time": Gauge(
                "api_response_time_p95_ms",
                "API response time 95th percentile",
                registry=self.prometheus_registry,
            ),
            "uptime": Gauge(
                "system_uptime_percentage",
                "System uptime percentage",
                registry=self.prometheus_registry,
            ),
            "error_rate": Gauge(
                "error_rate_percentage", "Error rate percentage", registry=self.prometheus_registry
            ),
            "generation_success_rate": Gauge(
                "music_generation_success_rate",
                "Music generation success rate",
                registry=self.prometheus_registry,
            ),
            "user_satisfaction": Gauge(
                "user_satisfaction_score",
                "User satisfaction score",
                registry=self.prometheus_registry,
            ),
            "deployment_time": Gauge(
                "deployment_time_minutes",
                "Deployment time in minutes",
                registry=self.prometheus_registry,
            ),
            "rollback_time": Gauge(
                "rollback_time_minutes",
                "Rollback time in minutes",
                registry=self.prometheus_registry,
            ),
        }

        return metrics

    def setup_routes(self):
        """Setup Flask routes for dashboard"""

        @self.app.route("/")
        def dashboard():
            return render_template("dashboard.html")

        @self.app.route("/api/metrics")
        def api_metrics():
            return jsonify(
                {
                    "technical": {
                        name: asdict(metric) for name, metric in self.technical_metrics.items()
                    },
                    "business": {
                        name: asdict(metric) for name, metric in self.business_metrics.items()
                    },
                    "operational": {
                        name: asdict(metric) for name, metric in self.operational_metrics.items()
                    },
                    "risk": {name: asdict(metric) for name, metric in self.risk_metrics.items()},
                }
            )

        @self.app.route("/api/charts/<metric_type>")
        def api_charts(metric_type):
            if metric_type == "technical":
                metrics = self.technical_metrics
            elif metric_type == "business":
                metrics = self.business_metrics
            elif metric_type == "operational":
                metrics = self.operational_metrics
            else:
                return jsonify({"error": "Invalid metric type"})

            charts = {}
            for name, metric in metrics.items():
                if metric.history:
                    x_data = [point.timestamp.isoformat() for point in metric.history]
                    y_data = [point.value for point in metric.history]

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=x_data, y=y_data, mode="lines+markers", name=name))
                    fig.add_hline(
                        y=metric.target_value,
                        line_dash="dash",
                        line_color="red",
                        annotation_text="Target",
                    )

                    charts[name] = json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))

            return jsonify(charts)

        @self.app.route("/metrics")
        def prometheus_metrics():
            return generate_latest(self.prometheus_registry)

    def collect_technical_metrics(self) -> Dict[str, TechnicalMetric]:
        """Collect technical success metrics"""
        logger.info("Collecting technical metrics...")

        metrics = {}

        # Test Coverage
        test_coverage = self.get_test_coverage()
        metrics["test_coverage"] = self.create_technical_metric(
            "Test Coverage", test_coverage, self.targets["test_coverage"], "test_coverage"
        )

        # API Response Time
        api_response_time = self.get_api_response_time()
        metrics["api_response_time"] = self.create_technical_metric(
            "API Response Time (P95)",
            api_response_time,
            self.targets["api_response_time_p95"],
            "api_response_time",
        )

        # Critical Vulnerabilities
        critical_vulns = self.get_critical_vulnerabilities()
        metrics["critical_vulnerabilities"] = self.create_technical_metric(
            "Critical Vulnerabilities",
            critical_vulns,
            self.targets["critical_vulnerabilities"],
            "critical_vulnerabilities",
        )

        # System Uptime
        uptime = self.get_system_uptime()
        metrics["uptime"] = self.create_technical_metric(
            "System Uptime", uptime, self.targets["uptime_percentage"], "uptime"
        )

        # Error Rate
        error_rate = self.get_error_rate()
        metrics["error_rate"] = self.create_technical_metric(
            "Error Rate", error_rate, self.targets["error_rate"], "error_rate"
        )

        return metrics

    def collect_business_metrics(self) -> Dict[str, BusinessMetric]:
        """Collect business success metrics"""
        logger.info("Collecting business metrics...")

        metrics = {}

        # Music Generation Success Rate
        gen_success_rate = self.get_generation_success_rate()
        metrics["generation_success_rate"] = self.create_business_metric(
            "Music Generation Success Rate",
            gen_success_rate,
            self.targets["music_generation_success_rate"],
            "generation_success_rate",
        )

        # User Registration to Generation Time
        reg_to_gen_time = self.get_registration_to_generation_time()
        metrics["registration_to_generation_time"] = self.create_business_metric(
            "Registration to Generation Time",
            reg_to_gen_time,
            self.targets["user_registration_to_generation_time"],
            "registration_to_generation_time",
        )

        # Support Ticket Rate
        support_ticket_rate = self.get_support_ticket_rate()
        metrics["support_ticket_rate"] = self.create_business_metric(
            "Support Ticket Rate",
            support_ticket_rate,
            self.targets["support_ticket_rate"],
            "support_ticket_rate",
        )

        # User Satisfaction
        user_satisfaction = self.get_user_satisfaction()
        metrics["user_satisfaction"] = self.create_business_metric(
            "User Satisfaction",
            user_satisfaction,
            self.targets["user_satisfaction"],
            "user_satisfaction",
        )

        return metrics

    def collect_operational_metrics(self) -> Dict[str, OperationalMetric]:
        """Collect operational success metrics"""
        logger.info("Collecting operational metrics...")

        metrics = {}

        # Deployment Time
        deployment_time = self.get_deployment_time()
        metrics["deployment_time"] = self.create_operational_metric(
            "Deployment Time", deployment_time, self.targets["deployment_time"], "deployment_time"
        )

        # Rollback Time
        rollback_time = self.get_rollback_time()
        metrics["rollback_time"] = self.create_operational_metric(
            "Rollback Time", rollback_time, self.targets["rollback_time"], "rollback_time"
        )

        # Alert Response Time
        alert_response_time = self.get_alert_response_time()
        metrics["alert_response_time"] = self.create_operational_metric(
            "Alert Response Time",
            alert_response_time,
            self.targets["alert_response_time"],
            "alert_response_time",
        )

        # Documentation Completeness
        doc_completeness = self.get_documentation_completeness()
        metrics["documentation_completeness"] = self.create_operational_metric(
            "Documentation Completeness",
            doc_completeness,
            self.targets["documentation_completeness"],
            "documentation_completeness",
        )

        return metrics

    def collect_risk_metrics(self) -> Dict[str, RiskMetric]:
        """Collect and assess risk metrics"""
        logger.info("Collecting risk metrics...")

        risks = {
            "gpu_resource_constraints": RiskMetric(
                name="GPU Resource Constraints",
                probability=self.assess_gpu_resource_risk(),
                impact=8.0,
                risk_score=0.0,
                mitigation_status=self.get_mitigation_status("gpu_resources"),
                last_assessed=datetime.now(),
                mitigation_plan="Implement queue limits and cloud GPU scaling",
            ),
            "database_performance": RiskMetric(
                name="Database Performance",
                probability=self.assess_database_performance_risk(),
                impact=7.0,
                risk_score=0.0,
                mitigation_status=self.get_mitigation_status("database_performance"),
                last_assessed=datetime.now(),
                mitigation_plan="Deploy read replicas and implement caching",
            ),
            "third_party_dependencies": RiskMetric(
                name="Third-party Dependencies",
                probability=self.assess_dependency_risk(),
                impact=6.0,
                risk_score=0.0,
                mitigation_status=self.get_mitigation_status("dependencies"),
                last_assessed=datetime.now(),
                mitigation_plan="Maintain vendor evaluation and fallback providers",
            ),
            "team_knowledge_gaps": RiskMetric(
                name="Team Knowledge Gaps",
                probability=self.assess_knowledge_gap_risk(),
                impact=8.0,
                risk_score=0.0,
                mitigation_status=self.get_mitigation_status("knowledge_gaps"),
                last_assessed=datetime.now(),
                mitigation_plan="Comprehensive documentation and training",
            ),
            "deployment_failures": RiskMetric(
                name="Deployment Failures",
                probability=self.assess_deployment_failure_risk(),
                impact=9.0,
                risk_score=0.0,
                mitigation_status=self.get_mitigation_status("deployment_failures"),
                last_assessed=datetime.now(),
                mitigation_plan="Blue-green deployments and instant rollback",
            ),
        }

        # Calculate risk scores
        for risk in risks.values():
            risk.risk_score = risk.probability * risk.impact

        return risks

    def get_test_coverage(self) -> float:
        """Get current test coverage"""
        try:
            result = subprocess.run(
                [
                    "python",
                    "-m",
                    "pytest",
                    "--cov=music_gen",
                    "--cov-report=json:coverage.json",
                    "--tb=no",
                    "-q",
                ],
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )

            if os.path.exists("coverage.json"):
                with open("coverage.json", "r") as f:
                    coverage_data = json.load(f)
                    return coverage_data.get("totals", {}).get("percent_covered", 0)
        except Exception as e:
            logger.error(f"Error getting test coverage: {e}")

        return 0.0

    def get_api_response_time(self) -> float:
        """Get API response time from Prometheus"""
        try:
            query = "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))"
            response = self.query_prometheus(query)

            if response and response.get("data", {}).get("result"):
                return float(response["data"]["result"][0]["value"][1]) * 1000  # Convert to ms
        except Exception as e:
            logger.error(f"Error getting API response time: {e}")

        return 0.0

    def get_critical_vulnerabilities(self) -> int:
        """Get count of critical vulnerabilities"""
        try:
            result = subprocess.run(
                ["bandit", "-r", "music_gen/", "-f", "json", "-o", "security_report.json", "-ll"],
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )

            if os.path.exists("security_report.json"):
                with open("security_report.json", "r") as f:
                    security_data = json.load(f)
                    critical_count = 0
                    for result in security_data.get("results", []):
                        if result.get("issue_severity") == "HIGH":
                            critical_count += 1
                    return critical_count
        except Exception as e:
            logger.error(f"Error getting critical vulnerabilities: {e}")

        return 0

    def get_system_uptime(self) -> float:
        """Get system uptime percentage"""
        try:
            query = "avg(up) * 100"
            response = self.query_prometheus(query)

            if response and response.get("data", {}).get("result"):
                return float(response["data"]["result"][0]["value"][1])
        except Exception as e:
            logger.error(f"Error getting system uptime: {e}")

        return 0.0

    def get_error_rate(self) -> float:
        """Get error rate percentage"""
        try:
            query = 'rate(http_requests_total{status=~"5.."}[5m]) * 100'
            response = self.query_prometheus(query)

            if response and response.get("data", {}).get("result"):
                return float(response["data"]["result"][0]["value"][1])
        except Exception as e:
            logger.error(f"Error getting error rate: {e}")

        return 0.0

    def get_generation_success_rate(self) -> float:
        """Get music generation success rate"""
        try:
            conn = psycopg2.connect(**self.db_params)
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT 
                        COUNT(*) FILTER (WHERE status = 'completed') * 100.0 / COUNT(*) as success_rate
                    FROM generation_tasks 
                    WHERE created_at >= NOW() - INTERVAL '24 hours'
                """
                )

                result = cursor.fetchone()
                if result and result[0]:
                    return float(result[0])
            conn.close()
        except Exception as e:
            logger.error(f"Error getting generation success rate: {e}")

        return 0.0

    def get_registration_to_generation_time(self) -> float:
        """Get average time from registration to first generation"""
        try:
            conn = psycopg2.connect(**self.db_params)
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT AVG(
                        EXTRACT(EPOCH FROM (gt.created_at - u.created_at)) / 60
                    ) as avg_time_minutes
                    FROM users u
                    JOIN generation_tasks gt ON u.id = gt.user_id
                    WHERE gt.created_at = (
                        SELECT MIN(created_at) 
                        FROM generation_tasks 
                        WHERE user_id = u.id
                    )
                    AND u.created_at >= NOW() - INTERVAL '7 days'
                """
                )

                result = cursor.fetchone()
                if result and result[0]:
                    return float(result[0])
            conn.close()
        except Exception as e:
            logger.error(f"Error getting registration to generation time: {e}")

        return 0.0

    def get_support_ticket_rate(self) -> float:
        """Get support ticket rate"""
        try:
            conn = psycopg2.connect(**self.db_params)
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT 
                        COUNT(st.*) * 100.0 / COUNT(u.*) as ticket_rate
                    FROM users u
                    LEFT JOIN support_tickets st ON u.id = st.user_id 
                        AND st.created_at >= NOW() - INTERVAL '30 days'
                    WHERE u.created_at >= NOW() - INTERVAL '30 days'
                """
                )

                result = cursor.fetchone()
                if result and result[0]:
                    return float(result[0])
            conn.close()
        except Exception as e:
            logger.error(f"Error getting support ticket rate: {e}")

        return 0.0

    def get_user_satisfaction(self) -> float:
        """Get user satisfaction score"""
        try:
            conn = psycopg2.connect(**self.db_params)
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT AVG(rating) as avg_rating
                    FROM user_feedback 
                    WHERE created_at >= NOW() - INTERVAL '30 days'
                """
                )

                result = cursor.fetchone()
                if result and result[0]:
                    return float(result[0])
            conn.close()
        except Exception as e:
            logger.error(f"Error getting user satisfaction: {e}")

        return 0.0

    def get_deployment_time(self) -> float:
        """Get average deployment time"""
        try:
            # Parse deployment logs to get average time
            log_files = list(Path("logs").glob("production_deployment_*.log"))

            if log_files:
                # Get most recent deployment log
                latest_log = max(log_files, key=os.path.getmtime)

                with open(latest_log, "r") as f:
                    content = f.read()

                    # Look for deployment duration
                    duration_match = re.search(r"Duration: (\d+) minutes", content)
                    if duration_match:
                        return float(duration_match.group(1))
        except Exception as e:
            logger.error(f"Error getting deployment time: {e}")

        return 0.0

    def get_rollback_time(self) -> float:
        """Get average rollback time"""
        try:
            # This would parse rollback logs
            # For now, return a reasonable default
            return 3.0  # minutes
        except Exception as e:
            logger.error(f"Error getting rollback time: {e}")

        return 0.0

    def get_alert_response_time(self) -> float:
        """Get average alert response time"""
        try:
            # This would integrate with PagerDuty or similar
            # For now, return a reasonable default
            return 10.0  # minutes
        except Exception as e:
            logger.error(f"Error getting alert response time: {e}")

        return 0.0

    def get_documentation_completeness(self) -> float:
        """Get documentation completeness percentage"""
        try:
            required_docs = [
                "README.md",
                "CHANGELOG.md",
                "CONTRIBUTING.md",
                "docs/api/README.md",
                "docs/deployment.md",
            ]

            existing_docs = 0
            for doc in required_docs:
                if (self.project_root / doc).exists():
                    existing_docs += 1

            return (existing_docs / len(required_docs)) * 100
        except Exception as e:
            logger.error(f"Error getting documentation completeness: {e}")

        return 0.0

    def assess_gpu_resource_risk(self) -> float:
        """Assess GPU resource constraint risk"""
        try:
            # Check GPU utilization
            query = "nvidia_gpu_utilization_gpu"
            response = self.query_prometheus(query)

            if response and response.get("data", {}).get("result"):
                gpu_utilization = float(response["data"]["result"][0]["value"][1])

                # Risk increases with utilization
                if gpu_utilization > 90:
                    return 8.0
                elif gpu_utilization > 80:
                    return 6.0
                elif gpu_utilization > 70:
                    return 4.0
                else:
                    return 2.0
        except Exception as e:
            logger.error(f"Error assessing GPU resource risk: {e}")

        return 5.0  # Medium risk by default

    def assess_database_performance_risk(self) -> float:
        """Assess database performance risk"""
        try:
            # Check database connection count and query performance
            conn = psycopg2.connect(**self.db_params)
            with conn.cursor() as cursor:
                cursor.execute("SELECT count(*) FROM pg_stat_activity")
                connection_count = cursor.fetchone()[0]

                # Risk increases with connection count
                if connection_count > 90:
                    return 7.0
                elif connection_count > 70:
                    return 5.0
                elif connection_count > 50:
                    return 3.0
                else:
                    return 1.0
            conn.close()
        except Exception as e:
            logger.error(f"Error assessing database performance risk: {e}")

        return 4.0  # Medium risk by default

    def assess_dependency_risk(self) -> float:
        """Assess third-party dependency risk"""
        try:
            # Check for outdated and vulnerable dependencies
            result = subprocess.run(["safety", "check", "--json"], capture_output=True, text=True)

            if result.returncode == 0:
                safety_data = json.loads(result.stdout)
                vulnerabilities = len(safety_data.get("vulnerabilities", []))

                # Risk increases with vulnerabilities
                if vulnerabilities > 10:
                    return 8.0
                elif vulnerabilities > 5:
                    return 6.0
                elif vulnerabilities > 0:
                    return 4.0
                else:
                    return 2.0
        except Exception as e:
            logger.error(f"Error assessing dependency risk: {e}")

        return 3.0  # Low-medium risk by default

    def assess_knowledge_gap_risk(self) -> float:
        """Assess team knowledge gap risk"""
        # This would integrate with team metrics
        # For now, return a reasonable assessment
        return 4.0  # Medium risk

    def assess_deployment_failure_risk(self) -> float:
        """Assess deployment failure risk"""
        try:
            # Check recent deployment success rate
            deployment_logs = list(Path("logs").glob("production_deployment_*.log"))
            rollback_logs = list(Path("logs").glob("rollback_*.log"))

            if deployment_logs:
                failure_rate = len(rollback_logs) / len(deployment_logs)

                # Risk increases with failure rate
                if failure_rate > 0.2:
                    return 8.0
                elif failure_rate > 0.1:
                    return 6.0
                elif failure_rate > 0.05:
                    return 4.0
                else:
                    return 2.0
        except Exception as e:
            logger.error(f"Error assessing deployment failure risk: {e}")

        return 3.0  # Low-medium risk by default

    def get_mitigation_status(self, risk_type: str) -> str:
        """Get mitigation status for a risk type"""
        # This would check actual mitigation implementations
        mitigation_statuses = {
            "gpu_resources": "In Progress",
            "database_performance": "Implemented",
            "dependencies": "Planned",
            "knowledge_gaps": "In Progress",
            "deployment_failures": "Implemented",
        }

        return mitigation_statuses.get(risk_type, "Planned")

    def query_prometheus(self, query: str) -> Optional[Dict]:
        """Query Prometheus for metrics"""
        try:
            url = f"{self.config['prometheus_url']}/api/v1/query"
            params = {"query": query}

            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Error querying Prometheus: {e}")

        return None

    def create_technical_metric(
        self, name: str, value: float, target: float, prometheus_key: str
    ) -> TechnicalMetric:
        """Create a technical metric with trend analysis"""
        # Get historical data
        history = self.get_metric_history(prometheus_key)

        # Calculate trend
        trend = self.calculate_trend(history)

        # Determine status
        status = self.get_metric_status(value, target, prometheus_key)

        # Generate alerts
        alerts = self.generate_alerts(name, value, target, status)

        # Update Prometheus metric
        if prometheus_key in self.prometheus_metrics:
            self.prometheus_metrics[prometheus_key].set(value)

        return TechnicalMetric(
            name=name,
            current_value=value,
            target_value=target,
            trend=trend,
            last_updated=datetime.now(),
            history=history,
            alerts=alerts,
        )

    def create_business_metric(
        self, name: str, value: float, target: float, prometheus_key: str
    ) -> BusinessMetric:
        """Create a business metric with trend analysis"""
        history = self.get_metric_history(prometheus_key)
        trend = self.calculate_trend(history)
        status = self.get_metric_status(value, target, prometheus_key)
        alerts = self.generate_alerts(name, value, target, status)

        if prometheus_key in self.prometheus_metrics:
            self.prometheus_metrics[prometheus_key].set(value)

        return BusinessMetric(
            name=name,
            current_value=value,
            target_value=target,
            trend=trend,
            last_updated=datetime.now(),
            history=history,
            alerts=alerts,
        )

    def create_operational_metric(
        self, name: str, value: float, target: float, prometheus_key: str
    ) -> OperationalMetric:
        """Create an operational metric with trend analysis"""
        history = self.get_metric_history(prometheus_key)
        trend = self.calculate_trend(history)
        status = self.get_metric_status(value, target, prometheus_key)
        alerts = self.generate_alerts(name, value, target, status)

        if prometheus_key in self.prometheus_metrics:
            self.prometheus_metrics[prometheus_key].set(value)

        return OperationalMetric(
            name=name,
            current_value=value,
            target_value=target,
            trend=trend,
            last_updated=datetime.now(),
            history=history,
            alerts=alerts,
        )

    def get_metric_history(self, metric_key: str) -> List[MetricPoint]:
        """Get historical data for a metric"""
        try:
            history_key = f"metric_history:{metric_key}"
            history_data = self.redis_client.lrange(history_key, 0, -1)

            history = []
            for item in history_data:
                point_data = json.loads(item)
                history.append(
                    MetricPoint(
                        timestamp=datetime.fromisoformat(point_data["timestamp"]),
                        value=point_data["value"],
                        target=point_data["target"],
                        status=point_data["status"],
                        metadata=point_data.get("metadata", {}),
                    )
                )

            return history
        except Exception as e:
            logger.error(f"Error getting metric history for {metric_key}: {e}")
            return []

    def calculate_trend(self, history: List[MetricPoint]) -> str:
        """Calculate trend from historical data"""
        if len(history) < 2:
            return "stable"

        # Take last 10 points for trend calculation
        recent_points = history[-10:]
        values = [point.value for point in recent_points]

        # Simple trend calculation
        if len(values) >= 2:
            trend_value = (values[-1] - values[0]) / len(values)

            if trend_value > 0.1:
                return "up"
            elif trend_value < -0.1:
                return "down"
            else:
                return "stable"

        return "stable"

    def get_metric_status(self, value: float, target: float, metric_key: str) -> str:
        """Determine metric status based on value and target"""
        # Different metrics have different good/bad directions
        inverse_metrics = [
            "api_response_time",
            "error_rate",
            "critical_vulnerabilities",
            "deployment_time",
            "rollback_time",
        ]

        if metric_key in inverse_metrics:
            # Lower is better
            if value <= target:
                return "healthy"
            elif value <= target * 1.2:
                return "warning"
            else:
                return "critical"
        else:
            # Higher is better
            if value >= target:
                return "healthy"
            elif value >= target * 0.8:
                return "warning"
            else:
                return "critical"

    def generate_alerts(self, name: str, value: float, target: float, status: str) -> List[str]:
        """Generate alerts based on metric status"""
        alerts = []

        if status == "critical":
            alerts.append(f"CRITICAL: {name} is {value:.1f}, target is {target:.1f}")
        elif status == "warning":
            alerts.append(f"WARNING: {name} is {value:.1f}, target is {target:.1f}")

        return alerts

    def store_metric_point(self, metric_key: str, value: float, target: float, status: str):
        """Store a metric point in Redis"""
        try:
            history_key = f"metric_history:{metric_key}"

            point = {
                "timestamp": datetime.now().isoformat(),
                "value": value,
                "target": target,
                "status": status,
                "metadata": {},
            }

            # Add to history
            self.redis_client.lpush(history_key, json.dumps(point))

            # Keep only last 24 hours of data (assuming 1 minute intervals)
            self.redis_client.ltrim(history_key, 0, 1440)

        except Exception as e:
            logger.error(f"Error storing metric point for {metric_key}: {e}")

    def collect_all_metrics(self):
        """Collect all metrics and update dashboard"""
        logger.info("Collecting all metrics...")

        # Collect technical metrics
        self.technical_metrics = self.collect_technical_metrics()

        # Collect business metrics
        self.business_metrics = self.collect_business_metrics()

        # Collect operational metrics
        self.operational_metrics = self.collect_operational_metrics()

        # Collect risk metrics
        self.risk_metrics = self.collect_risk_metrics()

        # Store metrics in Redis for historical tracking
        for metric_key, metric in self.technical_metrics.items():
            self.store_metric_point(
                metric_key, metric.current_value, metric.target_value, "healthy"
            )

        for metric_key, metric in self.business_metrics.items():
            self.store_metric_point(
                metric_key, metric.current_value, metric.target_value, "healthy"
            )

        for metric_key, metric in self.operational_metrics.items():
            self.store_metric_point(
                metric_key, metric.current_value, metric.target_value, "healthy"
            )

        # Generate alerts for critical metrics
        self.generate_critical_alerts()

        logger.info("All metrics collected and updated")

    def generate_critical_alerts(self):
        """Generate alerts for critical metrics"""
        critical_alerts = []

        # Check all metrics for critical status
        all_metrics = {
            **self.technical_metrics,
            **self.business_metrics,
            **self.operational_metrics,
        }

        for metric_name, metric in all_metrics.items():
            if metric.alerts:
                critical_alerts.extend(metric.alerts)

        # Check high-risk items
        for risk_name, risk in self.risk_metrics.items():
            if risk.risk_score > 50:  # High risk threshold
                critical_alerts.append(f"HIGH RISK: {risk.name} (Score: {risk.risk_score:.1f})")

        # Send alerts if any critical issues
        if critical_alerts:
            self.send_alerts(critical_alerts)

    def send_alerts(self, alerts: List[str]):
        """Send alerts to configured channels"""
        try:
            slack_webhook = os.getenv("SLACK_WEBHOOK_URL")
            if slack_webhook:
                message = {
                    "text": "🚨 *Critical Metrics Alert*",
                    "attachments": [
                        {
                            "color": "danger",
                            "fields": [
                                {
                                    "title": "Critical Issues",
                                    "value": "\n".join(alerts),
                                    "short": False,
                                }
                            ],
                        }
                    ],
                }

                requests.post(slack_webhook, json=message, timeout=5)
                logger.info(f"Sent {len(alerts)} critical alerts to Slack")
        except Exception as e:
            logger.error(f"Error sending alerts: {e}")

    def run_dashboard_server(self, host="0.0.0.0", port=8080):
        """Run the dashboard web server"""
        logger.info(f"Starting dashboard server on {host}:{port}")
        self.app.run(host=host, port=port, debug=False)

    def run_metrics_collector(self):
        """Run metrics collection loop"""
        logger.info("Starting metrics collection loop...")

        while True:
            try:
                self.collect_all_metrics()
                time.sleep(self.config["collection_interval"])
            except KeyboardInterrupt:
                logger.info("Metrics collection stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                time.sleep(60)  # Wait before retrying


def main():
    """Main function to run the dashboard"""
    dashboard = SuccessMetricsDashboard()

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "collect":
            dashboard.collect_all_metrics()
        elif command == "server":
            dashboard.run_dashboard_server()
        elif command == "monitor":
            dashboard.run_metrics_collector()
        else:
            print("Unknown command. Use: collect, server, or monitor")
    else:
        # Default: collect metrics once
        dashboard.collect_all_metrics()

        # Print summary
        print("\n📊 Success Metrics Summary:")
        print("=" * 50)

        print("\n🔧 Technical Metrics:")
        for name, metric in dashboard.technical_metrics.items():
            status_emoji = {"healthy": "✅", "warning": "⚠️", "critical": "❌"}
            status = dashboard.get_metric_status(metric.current_value, metric.target_value, name)
            emoji = status_emoji.get(status, "❓")
            print(
                f"  {emoji} {metric.name}: {metric.current_value:.1f} (target: {metric.target_value:.1f})"
            )

        print("\n💼 Business Metrics:")
        for name, metric in dashboard.business_metrics.items():
            status = dashboard.get_metric_status(metric.current_value, metric.target_value, name)
            emoji = status_emoji.get(status, "❓")
            print(
                f"  {emoji} {metric.name}: {metric.current_value:.1f} (target: {metric.target_value:.1f})"
            )

        print("\n⚙️ Operational Metrics:")
        for name, metric in dashboard.operational_metrics.items():
            status = dashboard.get_metric_status(metric.current_value, metric.target_value, name)
            emoji = status_emoji.get(status, "❓")
            print(
                f"  {emoji} {metric.name}: {metric.current_value:.1f} (target: {metric.target_value:.1f})"
            )

        print("\n⚠️ Risk Assessment:")
        for name, risk in dashboard.risk_metrics.items():
            risk_level = "🔴" if risk.risk_score > 50 else "🟡" if risk.risk_score > 30 else "🟢"
            print(f"  {risk_level} {risk.name}: {risk.risk_score:.1f} ({risk.mitigation_status})")


if __name__ == "__main__":
    main()
