#!/usr/bin/env python3
"""
Production Validation Script for Music Gen AI
Monitors and validates production deployment for 24 hours post-deployment
"""

import os
import json
import time
import requests
import psycopg2
import redis
import threading
import signal
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
import subprocess
from dataclasses import dataclass, asdict, field
from collections import deque
import statistics
import asyncio
import aiohttp
import websockets

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s] %(message)s",
    handlers=[
        logging.FileHandler(f"production_validation_{int(time.time())}.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


@dataclass
class MetricSnapshot:
    """Single metric measurement"""

    timestamp: datetime
    error_rate: float
    response_time_p50: float
    response_time_p95: float
    response_time_p99: float
    requests_per_second: float
    active_users: int
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    database_connections: int
    redis_memory_usage: float
    queue_depth: int


@dataclass
class ValidationAlert:
    """Alert generated during validation"""

    timestamp: datetime
    severity: str  # info, warning, error, critical
    category: str  # performance, error, resource, security
    title: str
    description: str
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    action_required: bool = False


@dataclass
class IntegrationStatus:
    """Status of external integration"""

    name: str
    healthy: bool
    last_check: datetime
    response_time: Optional[float] = None
    error_message: Optional[str] = None


class ProductionValidator:
    def __init__(self):
        self.api_url = os.getenv("API_URL", "https://api.musicgen.ai")
        self.prometheus_url = os.getenv("PROMETHEUS_URL", "http://prometheus:9090")
        self.grafana_url = os.getenv("GRAFANA_URL", "http://grafana:3000")
        self.elastic_url = os.getenv("ELASTIC_URL", "http://elasticsearch:9200")

        # Alert thresholds
        self.thresholds = {
            "error_rate_warning": 0.01,  # 1%
            "error_rate_critical": 0.05,  # 5%
            "response_time_p95_warning": 2.0,  # 2 seconds
            "response_time_p95_critical": 5.0,  # 5 seconds
            "cpu_usage_warning": 70,  # 70%
            "cpu_usage_critical": 85,  # 85%
            "memory_usage_warning": 80,  # 80%
            "memory_usage_critical": 90,  # 90%
            "disk_usage_warning": 75,  # 75%
            "disk_usage_critical": 85,  # 85%
        }

        # Monitoring data
        self.metrics_history: deque = deque(maxlen=8640)  # 24 hours of data (10s intervals)
        self.alerts: List[ValidationAlert] = []
        self.integration_status: Dict[str, IntegrationStatus] = {}

        # Control flags
        self.monitoring_active = False
        self.validation_start_time = None
        self.standby_team_alerted = False

        # Alert channels
        self.slack_webhook = os.getenv("SLACK_WEBHOOK_URL")
        self.pagerduty_token = os.getenv("PAGERDUTY_TOKEN")
        self.email_alerts = os.getenv("EMAIL_ALERTS", "").split(",")

        # Database connections
        self.db_params = {
            "host": os.getenv("DB_HOST", "postgres"),
            "port": 5432,
            "database": "musicgen_prod",
            "user": "musicgen",
            "password": os.getenv("DB_PASSWORD"),
        }

        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "redis"),
            port=6379,
            password=os.getenv("REDIS_PASSWORD"),
            decode_responses=True,
        )

        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, stopping validation...")
        self.monitoring_active = False
        self.generate_final_report()
        sys.exit(0)

    def collect_metrics(self) -> MetricSnapshot:
        """Collect current system metrics"""
        try:
            # Error rate
            error_rate_query = 'rate(http_requests_total{status=~"5.."}[5m])'
            error_rate = self._query_prometheus(error_rate_query)

            # Response times
            p50_query = "histogram_quantile(0.5, rate(http_request_duration_seconds_bucket[5m]))"
            p95_query = "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))"
            p99_query = "histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))"

            response_p50 = self._query_prometheus(p50_query)
            response_p95 = self._query_prometheus(p95_query)
            response_p99 = self._query_prometheus(p99_query)

            # Throughput
            rps_query = "rate(http_requests_total[5m])"
            requests_per_second = self._query_prometheus(rps_query)

            # Active users
            active_users_query = 'count(rate(http_requests_total{user_id!=""}[5m]) > 0)'
            active_users = int(self._query_prometheus(active_users_query))

            # Resource usage
            cpu_query = 'avg(100 - rate(node_cpu_seconds_total{mode="idle"}[5m]) * 100)'
            memory_query = (
                "avg((1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) * 100)"
            )
            disk_query = "max((node_filesystem_size_bytes - node_filesystem_free_bytes) / node_filesystem_size_bytes * 100)"

            cpu_usage = self._query_prometheus(cpu_query)
            memory_usage = self._query_prometheus(memory_query)
            disk_usage = self._query_prometheus(disk_query)

            # Database metrics
            db_connections = self._get_database_connections()

            # Redis metrics
            redis_info = self.redis_client.info("memory")
            redis_memory_mb = redis_info["used_memory"] / 1024 / 1024

            # Queue depth
            queue_depth = self.redis_client.llen("celery:queue:default")

            return MetricSnapshot(
                timestamp=datetime.now(),
                error_rate=error_rate,
                response_time_p50=response_p50,
                response_time_p95=response_p95,
                response_time_p99=response_p99,
                requests_per_second=requests_per_second,
                active_users=active_users,
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                database_connections=db_connections,
                redis_memory_usage=redis_memory_mb,
                queue_depth=queue_depth,
            )

        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
            return None

    def _query_prometheus(self, query: str) -> float:
        """Query Prometheus and return single value"""
        try:
            response = requests.get(
                f"{self.prometheus_url}/api/v1/query", params={"query": query}, timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                if data["data"]["result"]:
                    return float(data["data"]["result"][0]["value"][1])
            return 0.0

        except Exception as e:
            logger.error(f"Prometheus query failed: {e}")
            return 0.0

    def _get_database_connections(self) -> int:
        """Get active database connection count"""
        try:
            conn = psycopg2.connect(**self.db_params)
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT count(*) FROM pg_stat_activity 
                    WHERE datname = 'musicgen_prod' 
                    AND state != 'idle'
                """
                )
                count = cursor.fetchone()[0]
            conn.close()
            return count
        except Exception as e:
            logger.error(f"Failed to get database connections: {e}")
            return -1

    def check_error_logs(self) -> List[Dict[str, Any]]:
        """Check application logs for errors"""
        try:
            # Query Elasticsearch for recent errors
            query = {
                "query": {
                    "bool": {
                        "must": [
                            {"match": {"level": "ERROR"}},
                            {"range": {"@timestamp": {"gte": "now-5m"}}},
                        ]
                    }
                },
                "size": 100,
                "sort": [{"@timestamp": {"order": "desc"}}],
            }

            response = requests.post(
                f"{self.elastic_url}/musicgen-prod-*/_search", json=query, timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                errors = []
                for hit in data["hits"]["hits"]:
                    source = hit["_source"]
                    errors.append(
                        {
                            "timestamp": source.get("@timestamp"),
                            "level": source.get("level"),
                            "message": source.get("message"),
                            "service": source.get("service"),
                            "trace_id": source.get("trace_id"),
                        }
                    )
                return errors

        except Exception as e:
            logger.error(f"Failed to check error logs: {e}")

        return []

    def check_integrations(self):
        """Check health of all external integrations"""
        integrations = [
            {
                "name": "Payment Gateway",
                "url": "https://api.stripe.com/v1/charges",
                "headers": {"Authorization": f'Bearer {os.getenv("STRIPE_API_KEY")}'},
                "expected_status": [200, 401],
            },
            {
                "name": "Email Service",
                "url": "https://api.sendgrid.com/v3/mail/send",
                "headers": {"Authorization": f'Bearer {os.getenv("SENDGRID_API_KEY")}'},
                "expected_status": [200, 401],
            },
            {"name": "CDN", "url": "https://cdn.musicgen.ai/health", "expected_status": [200, 204]},
            {
                "name": "Object Storage",
                "url": f'https://s3.amazonaws.com/{os.getenv("S3_BUCKET")}',
                "expected_status": [200, 403, 404],
            },
        ]

        for integration in integrations:
            try:
                start_time = time.time()
                response = requests.get(
                    integration["url"], headers=integration.get("headers", {}), timeout=10
                )
                response_time = time.time() - start_time

                healthy = response.status_code in integration["expected_status"]

                self.integration_status[integration["name"]] = IntegrationStatus(
                    name=integration["name"],
                    healthy=healthy,
                    last_check=datetime.now(),
                    response_time=response_time,
                    error_message=None if healthy else f"Status: {response.status_code}",
                )

                if not healthy:
                    self.create_alert(
                        severity="warning",
                        category="integration",
                        title=f"{integration['name']} unhealthy",
                        description=f"Integration returned status {response.status_code}",
                    )

            except Exception as e:
                self.integration_status[integration["name"]] = IntegrationStatus(
                    name=integration["name"],
                    healthy=False,
                    last_check=datetime.now(),
                    error_message=str(e),
                )

                self.create_alert(
                    severity="error",
                    category="integration",
                    title=f"{integration['name']} connection failed",
                    description=str(e),
                    action_required=True,
                )

    def check_security_events(self) -> List[Dict[str, Any]]:
        """Check for security-related events"""
        security_events = []

        try:
            # Check for failed authentication attempts
            auth_failures_query = "rate(auth_failures_total[5m])"
            auth_failure_rate = self._query_prometheus(auth_failures_query)

            if auth_failure_rate > 10:  # More than 10 failures per 5 minutes
                security_events.append(
                    {
                        "type": "auth_failures",
                        "severity": "warning",
                        "rate": auth_failure_rate,
                        "message": f"High authentication failure rate: {auth_failure_rate:.2f}/5m",
                    }
                )

            # Check for suspicious patterns in logs
            suspicious_patterns = [
                "SQL injection",
                "XSS attempt",
                "Path traversal",
                "Command injection",
                "Unauthorized access",
            ]

            for pattern in suspicious_patterns:
                query = {
                    "query": {
                        "bool": {
                            "must": [
                                {"match_phrase": {"message": pattern}},
                                {"range": {"@timestamp": {"gte": "now-1h"}}},
                            ]
                        }
                    },
                    "size": 0,
                }

                response = requests.post(
                    f"{self.elastic_url}/musicgen-prod-*/_count", json=query, timeout=10
                )

                if response.status_code == 200:
                    count = response.json()["count"]
                    if count > 0:
                        security_events.append(
                            {
                                "type": "suspicious_pattern",
                                "severity": "critical",
                                "pattern": pattern,
                                "count": count,
                                "message": f'Detected {count} instances of "{pattern}" in logs',
                            }
                        )

            # Check for rate limit violations
            rate_limit_violations = self._query_prometheus(
                "sum(rate(rate_limit_exceeded_total[5m]))"
            )
            if rate_limit_violations > 100:
                security_events.append(
                    {
                        "type": "rate_limit_violations",
                        "severity": "warning",
                        "rate": rate_limit_violations,
                        "message": f"High rate limit violations: {rate_limit_violations:.0f}/5m",
                    }
                )

        except Exception as e:
            logger.error(f"Failed to check security events: {e}")

        return security_events

    def analyze_user_activity(self) -> Dict[str, Any]:
        """Analyze user activity patterns"""
        try:
            # Get user activity metrics
            new_users_query = "increase(user_registrations_total[1h])"
            active_users_query = 'count(rate(http_requests_total{user_id!=""}[5m]) > 0)'
            generation_requests_query = "sum(rate(generation_requests_total[5m]))"

            new_users = int(self._query_prometheus(new_users_query))
            active_users = int(self._query_prometheus(active_users_query))
            generation_rate = self._query_prometheus(generation_requests_query)

            # Get user feedback from database
            conn = psycopg2.connect(**self.db_params)
            with conn.cursor() as cursor:
                # Recent feedback
                cursor.execute(
                    """
                    SELECT 
                        sentiment,
                        COUNT(*) as count
                    FROM user_feedback
                    WHERE created_at > NOW() - INTERVAL '1 hour'
                    GROUP BY sentiment
                """
                )
                feedback_counts = dict(cursor.fetchall())

                # Error reports from users
                cursor.execute(
                    """
                    SELECT COUNT(*)
                    FROM support_tickets
                    WHERE 
                        created_at > NOW() - INTERVAL '1 hour'
                        AND category = 'bug_report'
                """
                )
                bug_reports = cursor.fetchone()[0]

            conn.close()

            activity_summary = {
                "new_users_hourly": new_users,
                "active_users": active_users,
                "generation_rate_per_second": generation_rate,
                "user_feedback": feedback_counts,
                "bug_reports_hourly": bug_reports,
            }

            # Check for anomalies
            if generation_rate < 0.1:  # Less than 0.1 generations per second
                self.create_alert(
                    severity="warning",
                    category="usage",
                    title="Low generation activity",
                    description=f"Generation rate: {generation_rate:.2f}/s",
                    metric_value=generation_rate,
                    threshold=0.1,
                )

            negative_feedback = feedback_counts.get("negative", 0)
            total_feedback = sum(feedback_counts.values())

            if total_feedback > 0 and negative_feedback / total_feedback > 0.2:
                self.create_alert(
                    severity="warning",
                    category="user_experience",
                    title="High negative feedback ratio",
                    description=f"{negative_feedback}/{total_feedback} negative feedback",
                    action_required=True,
                )

            return activity_summary

        except Exception as e:
            logger.error(f"Failed to analyze user activity: {e}")
            return {}

    def create_alert(
        self,
        severity: str,
        category: str,
        title: str,
        description: str,
        metric_value: Optional[float] = None,
        threshold: Optional[float] = None,
        action_required: bool = False,
    ):
        """Create and send alert"""
        alert = ValidationAlert(
            timestamp=datetime.now(),
            severity=severity,
            category=category,
            title=title,
            description=description,
            metric_value=metric_value,
            threshold=threshold,
            action_required=action_required,
        )

        self.alerts.append(alert)

        # Send notifications based on severity
        if severity in ["critical", "error"] or action_required:
            self._send_alert_notification(alert)

    def _send_alert_notification(self, alert: ValidationAlert):
        """Send alert through various channels"""
        # Slack notification
        if self.slack_webhook:
            emoji = {"critical": "🚨", "error": "❌", "warning": "⚠️", "info": "ℹ️"}.get(
                alert.severity, "📌"
            )

            slack_message = {
                "text": f"{emoji} *Production Alert*",
                "attachments": [
                    {
                        "color": {
                            "critical": "danger",
                            "error": "danger",
                            "warning": "warning",
                            "info": "good",
                        }.get(alert.severity, "gray"),
                        "fields": [
                            {"title": "Title", "value": alert.title, "short": True},
                            {"title": "Severity", "value": alert.severity.upper(), "short": True},
                            {"title": "Category", "value": alert.category, "short": True},
                            {
                                "title": "Time",
                                "value": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                                "short": True,
                            },
                            {"title": "Description", "value": alert.description, "short": False},
                        ],
                    }
                ],
            }

            if alert.metric_value is not None and alert.threshold is not None:
                slack_message["attachments"][0]["fields"].append(
                    {
                        "title": "Metric",
                        "value": f"{alert.metric_value:.2f} (threshold: {alert.threshold:.2f})",
                        "short": True,
                    }
                )

            if alert.action_required:
                slack_message["attachments"][0]["fields"].append(
                    {
                        "title": "⚡ Action Required",
                        "value": "Immediate attention needed",
                        "short": False,
                    }
                )

            try:
                requests.post(self.slack_webhook, json=slack_message, timeout=5)
            except Exception as e:
                logger.error(f"Failed to send Slack alert: {e}")

        # PagerDuty for critical alerts
        if alert.severity == "critical" and self.pagerduty_token:
            try:
                pagerduty_event = {
                    "routing_key": self.pagerduty_token,
                    "event_action": "trigger",
                    "payload": {
                        "summary": f"Production: {alert.title}",
                        "severity": "critical",
                        "source": "production-validator",
                        "custom_details": {
                            "category": alert.category,
                            "description": alert.description,
                            "metric_value": alert.metric_value,
                            "threshold": alert.threshold,
                        },
                    },
                }

                requests.post(
                    "https://events.pagerduty.com/v2/enqueue", json=pagerduty_event, timeout=5
                )
            except Exception as e:
                logger.error(f"Failed to send PagerDuty alert: {e}")

    def evaluate_metrics(self, metrics: MetricSnapshot):
        """Evaluate metrics against thresholds and create alerts"""
        if not metrics:
            return

        # Error rate checks
        if metrics.error_rate > self.thresholds["error_rate_critical"]:
            self.create_alert(
                severity="critical",
                category="error",
                title="Critical error rate",
                description=f"Error rate: {metrics.error_rate*100:.2f}%",
                metric_value=metrics.error_rate * 100,
                threshold=self.thresholds["error_rate_critical"] * 100,
                action_required=True,
            )
        elif metrics.error_rate > self.thresholds["error_rate_warning"]:
            self.create_alert(
                severity="warning",
                category="error",
                title="Elevated error rate",
                description=f"Error rate: {metrics.error_rate*100:.2f}%",
                metric_value=metrics.error_rate * 100,
                threshold=self.thresholds["error_rate_warning"] * 100,
            )

        # Response time checks
        if metrics.response_time_p95 > self.thresholds["response_time_p95_critical"]:
            self.create_alert(
                severity="critical",
                category="performance",
                title="Critical response time",
                description=f"P95 response time: {metrics.response_time_p95:.2f}s",
                metric_value=metrics.response_time_p95,
                threshold=self.thresholds["response_time_p95_critical"],
                action_required=True,
            )
        elif metrics.response_time_p95 > self.thresholds["response_time_p95_warning"]:
            self.create_alert(
                severity="warning",
                category="performance",
                title="Slow response time",
                description=f"P95 response time: {metrics.response_time_p95:.2f}s",
                metric_value=metrics.response_time_p95,
                threshold=self.thresholds["response_time_p95_warning"],
            )

        # Resource usage checks
        if metrics.cpu_usage > self.thresholds["cpu_usage_critical"]:
            self.create_alert(
                severity="critical",
                category="resource",
                title="Critical CPU usage",
                description=f"CPU usage: {metrics.cpu_usage:.1f}%",
                metric_value=metrics.cpu_usage,
                threshold=self.thresholds["cpu_usage_critical"],
                action_required=True,
            )
        elif metrics.cpu_usage > self.thresholds["cpu_usage_warning"]:
            self.create_alert(
                severity="warning",
                category="resource",
                title="High CPU usage",
                description=f"CPU usage: {metrics.cpu_usage:.1f}%",
                metric_value=metrics.cpu_usage,
                threshold=self.thresholds["cpu_usage_warning"],
            )

        if metrics.memory_usage > self.thresholds["memory_usage_critical"]:
            self.create_alert(
                severity="critical",
                category="resource",
                title="Critical memory usage",
                description=f"Memory usage: {metrics.memory_usage:.1f}%",
                metric_value=metrics.memory_usage,
                threshold=self.thresholds["memory_usage_critical"],
                action_required=True,
            )
        elif metrics.memory_usage > self.thresholds["memory_usage_warning"]:
            self.create_alert(
                severity="warning",
                category="resource",
                title="High memory usage",
                description=f"Memory usage: {metrics.memory_usage:.1f}%",
                metric_value=metrics.memory_usage,
                threshold=self.thresholds["memory_usage_warning"],
            )

        if metrics.disk_usage > self.thresholds["disk_usage_critical"]:
            self.create_alert(
                severity="critical",
                category="resource",
                title="Critical disk usage",
                description=f"Disk usage: {metrics.disk_usage:.1f}%",
                metric_value=metrics.disk_usage,
                threshold=self.thresholds["disk_usage_critical"],
                action_required=True,
            )
        elif metrics.disk_usage > self.thresholds["disk_usage_warning"]:
            self.create_alert(
                severity="warning",
                category="resource",
                title="High disk usage",
                description=f"Disk usage: {metrics.disk_usage:.1f}%",
                metric_value=metrics.disk_usage,
                threshold=self.thresholds["disk_usage_warning"],
            )

        # Queue depth check
        if metrics.queue_depth > 1000:
            self.create_alert(
                severity="warning",
                category="performance",
                title="High queue depth",
                description=f"Queue depth: {metrics.queue_depth} tasks",
                metric_value=metrics.queue_depth,
                threshold=1000,
            )

    def run_critical_path_tests(self):
        """Test critical user paths"""
        critical_paths = [
            {"name": "User Registration", "test": self._test_user_registration},
            {"name": "User Login", "test": self._test_user_login},
            {"name": "Music Generation", "test": self._test_music_generation},
            {"name": "File Download", "test": self._test_file_download},
            {"name": "Payment Processing", "test": self._test_payment_processing},
        ]

        for path in critical_paths:
            try:
                start_time = time.time()
                success, message = path["test"]()
                duration = time.time() - start_time

                if not success:
                    self.create_alert(
                        severity="critical",
                        category="functionality",
                        title=f'{path["name"]} failed',
                        description=message,
                        action_required=True,
                    )

                logger.info(
                    f"Critical path test '{path['name']}': {'PASS' if success else 'FAIL'} ({duration:.2f}s)"
                )

            except Exception as e:
                self.create_alert(
                    severity="critical",
                    category="functionality",
                    title=f'{path["name"]} test error',
                    description=str(e),
                    action_required=True,
                )

    def _test_user_registration(self) -> Tuple[bool, str]:
        """Test user registration flow"""
        test_email = f"test_{int(time.time())}@validator.musicgen.ai"

        response = requests.post(
            f"{self.api_url}/api/v1/auth/register",
            json={
                "email": test_email,
                "password": "TestPassword123!",
                "username": f"testuser_{int(time.time())}",
            },
            timeout=30,
        )

        if response.status_code in [200, 201]:
            return True, "Registration successful"
        else:
            return False, f"Registration failed with status {response.status_code}"

    def _test_user_login(self) -> Tuple[bool, str]:
        """Test user login flow"""
        response = requests.post(
            f"{self.api_url}/api/v1/auth/login",
            json={"email": "test@musicgen.ai", "password": "TestPassword123!"},
            timeout=30,
        )

        if response.status_code == 200:
            data = response.json()
            if "access_token" in data:
                return True, "Login successful"
            else:
                return False, "Login response missing access token"
        else:
            return False, f"Login failed with status {response.status_code}"

    def _test_music_generation(self) -> Tuple[bool, str]:
        """Test music generation flow"""
        # Get test token
        test_token = os.getenv("TEST_API_TOKEN")
        if not test_token:
            return False, "Test API token not configured"

        response = requests.post(
            f"{self.api_url}/api/v1/generate",
            headers={"Authorization": f"Bearer {test_token}"},
            json={"prompt": "Validation test: upbeat jazz", "duration": 5, "test_mode": True},
            timeout=60,
        )

        if response.status_code in [200, 202]:
            return True, "Generation request accepted"
        else:
            return False, f"Generation failed with status {response.status_code}"

    def _test_file_download(self) -> Tuple[bool, str]:
        """Test file download flow"""
        test_token = os.getenv("TEST_API_TOKEN")
        test_file_id = os.getenv("TEST_FILE_ID", "test-sample")

        response = requests.get(
            f"{self.api_url}/api/v1/files/{test_file_id}",
            headers={"Authorization": f"Bearer {test_token}"},
            timeout=30,
        )

        if response.status_code == 200:
            return True, "File download successful"
        elif response.status_code == 404:
            return True, "File not found (expected for test file)"
        else:
            return False, f"File download failed with status {response.status_code}"

    def _test_payment_processing(self) -> Tuple[bool, str]:
        """Test payment processing flow (in test mode)"""
        test_token = os.getenv("TEST_API_TOKEN")

        response = requests.post(
            f"{self.api_url}/api/v1/payments/test",
            headers={"Authorization": f"Bearer {test_token}"},
            json={"amount": 100, "test_mode": True},  # $1.00 in cents
            timeout=30,
        )

        if response.status_code in [200, 402]:  # 402 = Payment Required (expected in test)
            return True, "Payment endpoint responding correctly"
        else:
            return False, f"Payment test failed with status {response.status_code}"

    def monitoring_loop(self):
        """Main monitoring loop for 24 hours"""
        self.monitoring_active = True
        self.validation_start_time = datetime.now()

        logger.info("Starting 24-hour production validation monitoring...")

        # Alert team that validation has started
        self.create_alert(
            severity="info",
            category="monitoring",
            title="Production validation started",
            description="24-hour production monitoring is now active",
        )

        last_integration_check = datetime.now()
        last_security_check = datetime.now()
        last_critical_path_test = datetime.now()
        last_activity_analysis = datetime.now()

        while self.monitoring_active:
            try:
                # Collect metrics every 10 seconds
                metrics = self.collect_metrics()
                if metrics:
                    self.metrics_history.append(metrics)
                    self.evaluate_metrics(metrics)

                # Check integrations every 5 minutes
                if (datetime.now() - last_integration_check).seconds > 300:
                    self.check_integrations()
                    last_integration_check = datetime.now()

                # Check security events every 15 minutes
                if (datetime.now() - last_security_check).seconds > 900:
                    security_events = self.check_security_events()
                    for event in security_events:
                        self.create_alert(
                            severity=event["severity"],
                            category="security",
                            title=event["message"],
                            description=json.dumps(event),
                        )
                    last_security_check = datetime.now()

                # Run critical path tests every hour
                if (datetime.now() - last_critical_path_test).seconds > 3600:
                    self.run_critical_path_tests()
                    last_critical_path_test = datetime.now()

                # Analyze user activity every 30 minutes
                if (datetime.now() - last_activity_analysis).seconds > 1800:
                    self.analyze_user_activity()
                    last_activity_analysis = datetime.now()

                # Check if 24 hours have elapsed
                elapsed = (datetime.now() - self.validation_start_time).total_seconds()
                if elapsed > 86400:  # 24 hours
                    logger.info("24-hour validation period completed")
                    self.monitoring_active = False
                    break

                # Display status
                hours_elapsed = elapsed / 3600
                hours_remaining = 24 - hours_elapsed

                if int(elapsed) % 300 == 0:  # Every 5 minutes
                    logger.info(
                        f"Validation status: {hours_elapsed:.1f}h elapsed, {hours_remaining:.1f}h remaining"
                    )
                    logger.info(
                        f"Alerts: {len([a for a in self.alerts if a.severity in ['critical', 'error']])} critical/error, "
                        f"{len([a for a in self.alerts if a.severity == 'warning'])} warning"
                    )

                # Brief sleep
                time.sleep(10)

            except KeyboardInterrupt:
                logger.info("Validation interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(30)  # Wait before retrying

    def generate_final_report(self):
        """Generate comprehensive validation report"""
        logger.info("Generating final validation report...")

        # Calculate statistics
        total_metrics = len(self.metrics_history)

        if total_metrics > 0:
            # Error rate statistics
            error_rates = [m.error_rate for m in self.metrics_history if m]
            avg_error_rate = statistics.mean(error_rates) if error_rates else 0
            max_error_rate = max(error_rates) if error_rates else 0

            # Response time statistics
            response_times = [m.response_time_p95 for m in self.metrics_history if m]
            avg_response_time = statistics.mean(response_times) if response_times else 0
            max_response_time = max(response_times) if response_times else 0

            # Resource usage statistics
            cpu_usage = [m.cpu_usage for m in self.metrics_history if m]
            avg_cpu = statistics.mean(cpu_usage) if cpu_usage else 0
            max_cpu = max(cpu_usage) if cpu_usage else 0

            memory_usage = [m.memory_usage for m in self.metrics_history if m]
            avg_memory = statistics.mean(memory_usage) if memory_usage else 0
            max_memory = max(memory_usage) if memory_usage else 0
        else:
            avg_error_rate = max_error_rate = 0
            avg_response_time = max_response_time = 0
            avg_cpu = max_cpu = 0
            avg_memory = max_memory = 0

        # Alert summary
        critical_alerts = [a for a in self.alerts if a.severity == "critical"]
        error_alerts = [a for a in self.alerts if a.severity == "error"]
        warning_alerts = [a for a in self.alerts if a.severity == "warning"]

        # Integration health summary
        healthy_integrations = [
            name for name, status in self.integration_status.items() if status.healthy
        ]
        unhealthy_integrations = [
            name for name, status in self.integration_status.items() if not status.healthy
        ]

        report = {
            "validation_summary": {
                "start_time": self.validation_start_time.isoformat()
                if self.validation_start_time
                else None,
                "end_time": datetime.now().isoformat(),
                "duration_hours": (datetime.now() - self.validation_start_time).total_seconds()
                / 3600
                if self.validation_start_time
                else 0,
                "metrics_collected": total_metrics,
                "validation_result": "PASSED" if len(critical_alerts) == 0 else "FAILED",
            },
            "performance_metrics": {
                "error_rate": {
                    "average": avg_error_rate * 100,
                    "maximum": max_error_rate * 100,
                    "threshold": self.thresholds["error_rate_critical"] * 100,
                },
                "response_time_p95": {
                    "average": avg_response_time,
                    "maximum": max_response_time,
                    "threshold": self.thresholds["response_time_p95_critical"],
                },
            },
            "resource_utilization": {
                "cpu": {
                    "average": avg_cpu,
                    "maximum": max_cpu,
                    "threshold": self.thresholds["cpu_usage_critical"],
                },
                "memory": {
                    "average": avg_memory,
                    "maximum": max_memory,
                    "threshold": self.thresholds["memory_usage_critical"],
                },
            },
            "alerts": {
                "total": len(self.alerts),
                "critical": len(critical_alerts),
                "error": len(error_alerts),
                "warning": len(warning_alerts),
                "critical_alerts": [asdict(a) for a in critical_alerts[:10]],  # Top 10
            },
            "integrations": {
                "healthy": healthy_integrations,
                "unhealthy": unhealthy_integrations,
                "total_checked": len(self.integration_status),
            },
            "recommendations": self._generate_recommendations(),
        }

        # Save report
        report_file = f"production_validation_report_{int(time.time())}.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Validation report saved to {report_file}")

        # Print summary
        self._print_summary(report)

        # Send final notification
        self.create_alert(
            severity="info",
            category="monitoring",
            title="Production validation completed",
            description=f"Validation result: {report['validation_summary']['validation_result']}",
        )

        return report

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []

        # Analyze alerts
        critical_alerts = [a for a in self.alerts if a.severity == "critical"]
        if critical_alerts:
            recommendations.append(f"Address {len(critical_alerts)} critical issues immediately")

        # Performance recommendations
        if self.metrics_history:
            error_rates = [m.error_rate for m in self.metrics_history if m]
            if error_rates and max(error_rates) > self.thresholds["error_rate_warning"]:
                recommendations.append("Investigate and fix sources of errors")

            response_times = [m.response_time_p95 for m in self.metrics_history if m]
            if response_times and statistics.mean(response_times) > 1.5:
                recommendations.append("Optimize application performance to reduce response times")

        # Resource recommendations
        if self.metrics_history:
            cpu_usage = [m.cpu_usage for m in self.metrics_history if m]
            if cpu_usage and statistics.mean(cpu_usage) > 60:
                recommendations.append("Consider scaling out to handle CPU load")

            memory_usage = [m.memory_usage for m in self.metrics_history if m]
            if memory_usage and max(memory_usage) > 85:
                recommendations.append(
                    "Monitor memory usage closely and consider increasing resources"
                )

        # Integration recommendations
        unhealthy = [name for name, status in self.integration_status.items() if not status.healthy]
        if unhealthy:
            recommendations.append(f"Fix integration issues with: {', '.join(unhealthy)}")

        # General recommendations
        recommendations.extend(
            [
                "Continue monitoring key metrics for the next 48 hours",
                "Review and address any user-reported issues",
                "Schedule a post-deployment review meeting",
                "Update runbooks based on any issues encountered",
            ]
        )

        return recommendations

    def _print_summary(self, report: Dict[str, Any]):
        """Print validation summary"""
        print("\n" + "=" * 80)
        print("PRODUCTION VALIDATION SUMMARY")
        print("=" * 80)

        summary = report["validation_summary"]
        print(f"Duration: {summary['duration_hours']:.1f} hours")
        print(f"Result: {summary['validation_result']}")
        print(f"Metrics Collected: {summary['metrics_collected']:,}")
        print()

        print("PERFORMANCE METRICS:")
        perf = report["performance_metrics"]
        print(
            f"  Error Rate: avg={perf['error_rate']['average']:.2f}%, max={perf['error_rate']['maximum']:.2f}%"
        )
        print(
            f"  Response Time (p95): avg={perf['response_time_p95']['average']:.2f}s, max={perf['response_time_p95']['maximum']:.2f}s"
        )
        print()

        print("RESOURCE UTILIZATION:")
        resources = report["resource_utilization"]
        print(
            f"  CPU: avg={resources['cpu']['average']:.1f}%, max={resources['cpu']['maximum']:.1f}%"
        )
        print(
            f"  Memory: avg={resources['memory']['average']:.1f}%, max={resources['memory']['maximum']:.1f}%"
        )
        print()

        alerts = report["alerts"]
        print(f"ALERTS: {alerts['total']} total")
        print(f"  Critical: {alerts['critical']}")
        print(f"  Error: {alerts['error']}")
        print(f"  Warning: {alerts['warning']}")
        print()

        integrations = report["integrations"]
        print(f"INTEGRATIONS: {integrations['total_checked']} checked")
        print(f"  Healthy: {len(integrations['healthy'])}")
        print(f"  Unhealthy: {len(integrations['unhealthy'])}")
        if integrations["unhealthy"]:
            print(f"  Failed: {', '.join(integrations['unhealthy'])}")
        print()

        print("RECOMMENDATIONS:")
        for i, rec in enumerate(report["recommendations"], 1):
            print(f"  {i}. {rec}")

        print("=" * 80)


def main():
    """Main validation execution"""
    validator = ProductionValidator()

    try:
        # Run monitoring loop
        validator.monitoring_loop()

        # Generate final report
        report = validator.generate_final_report()

        # Exit code based on validation result
        exit_code = 0 if report["validation_summary"]["validation_result"] == "PASSED" else 1
        sys.exit(exit_code)

    except Exception as e:
        logger.error(f"Production validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
