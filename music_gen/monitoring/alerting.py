"""
Comprehensive alerting system for Music Gen AI monitoring.

Implements alert rules, conditions, escalation policies, and
notification channels for proactive incident response.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union

import aiohttp
import requests
from prometheus_client import Counter, Gauge

from .metrics import get_metrics_collector
from .sli_slo import get_sli_collector, SLOStatus
from ..core.logging_config import get_logger

logger = get_logger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertState(Enum):
    """Alert states."""

    FIRING = "firing"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"
    ACKNOWLEDGED = "acknowledged"


class ConditionOperator(Enum):
    """Condition operators for alert rules."""

    GREATER_THAN = "gt"
    GREATER_EQUAL = "ge"
    LESS_THAN = "lt"
    LESS_EQUAL = "le"
    EQUAL = "eq"
    NOT_EQUAL = "ne"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"


@dataclass
class AlertCondition:
    """Condition for triggering an alert."""

    metric_name: str
    operator: ConditionOperator
    threshold: Union[float, str]
    duration: timedelta = timedelta(minutes=5)  # How long condition must be true
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class NotificationChannel:
    """Configuration for alert notification channels."""

    name: str
    type: str  # 'email', 'slack', 'webhook', 'pagerduty', 'discord'
    config: Dict[str, Any]
    enabled: bool = True
    severity_filter: Set[AlertSeverity] = field(default_factory=lambda: set(AlertSeverity))


@dataclass
class AlertRule:
    """Definition of an alert rule."""

    name: str
    description: str
    conditions: List[AlertCondition]
    severity: AlertSeverity
    notification_channels: List[str]  # Channel names
    enabled: bool = True
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)

    # Advanced options
    cooldown_duration: timedelta = timedelta(minutes=15)  # Minimum time between alerts
    max_firing_time: timedelta = timedelta(hours=24)  # Auto-resolve after this time
    escalation_rules: List[Dict[str, Any]] = field(default_factory=list)

    # SLO integration
    slo_name: Optional[str] = None  # Associated SLO for error budget alerts


@dataclass
class Alert:
    """Active alert instance."""

    rule_name: str
    severity: AlertSeverity
    state: AlertState
    message: str
    details: Dict[str, Any]
    labels: Dict[str, str]
    annotations: Dict[str, str]

    # Timing
    first_fired: datetime
    last_fired: datetime
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None

    # Tracking
    firing_count: int = 1
    notification_count: int = 0
    last_notification: Optional[datetime] = None


class AlertManager:
    """Main alert management system."""

    def __init__(self):
        self.rules: Dict[str, AlertRule] = {}
        self.channels: Dict[str, NotificationChannel] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.metrics_collector = get_metrics_collector()
        self.sli_collector = get_sli_collector()

        # Prometheus metrics for alerting
        self.alerts_total = Counter(
            "musicgen_alerts_total",
            "Total number of alerts fired",
            ["rule_name", "severity", "state"],
            registry=self.metrics_collector.registry,
        )

        self.alerts_active = Gauge(
            "musicgen_alerts_active",
            "Number of active alerts",
            ["severity"],
            registry=self.metrics_collector.registry,
        )

        self.alert_notifications_total = Counter(
            "musicgen_alert_notifications_total",
            "Total alert notifications sent",
            ["channel_type", "severity", "status"],
            registry=self.metrics_collector.registry,
        )

        # Background task for checking alerts
        self._check_interval = 30  # seconds
        self._running = False
        self._check_task: Optional[asyncio.Task] = None

        # Register default alert rules and channels
        self._register_default_channels()
        self._register_default_alert_rules()

        logger.info("Alert manager initialized")

    def register_channel(self, channel: NotificationChannel):
        """Register a notification channel."""
        self.channels[channel.name] = channel
        logger.info(f"Registered notification channel: {channel.name} ({channel.type})")

    def register_alert_rule(self, rule: AlertRule):
        """Register an alert rule."""
        self.rules[rule.name] = rule
        logger.info(f"Registered alert rule: {rule.name} (severity: {rule.severity.value})")

    def _register_default_channels(self):
        """Register default notification channels."""

        # Slack channel
        slack_channel = NotificationChannel(
            name="slack_alerts",
            type="slack",
            config={
                "webhook_url": "${SLACK_WEBHOOK_URL}",
                "channel": "#alerts",
                "username": "MusicGen Monitor",
                "icon_emoji": ":warning:",
            },
            severity_filter={AlertSeverity.CRITICAL, AlertSeverity.HIGH, AlertSeverity.MEDIUM},
        )

        # Email channel
        email_channel = NotificationChannel(
            name="email_alerts",
            type="email",
            config={
                "smtp_host": "${SMTP_HOST}",
                "smtp_port": 587,
                "smtp_user": "${SMTP_USER}",
                "smtp_password": "${SMTP_PASSWORD}",
                "from_address": "alerts@musicgen.ai",
                "to_addresses": ["ops@musicgen.ai", "engineering@musicgen.ai"],
            },
            severity_filter={AlertSeverity.CRITICAL, AlertSeverity.HIGH},
        )

        # PagerDuty for critical alerts
        pagerduty_channel = NotificationChannel(
            name="pagerduty_critical",
            type="pagerduty",
            config={
                "integration_key": "${PAGERDUTY_INTEGRATION_KEY}",
                "severity_mapping": {
                    "critical": "critical",
                    "high": "error",
                    "medium": "warning",
                    "low": "info",
                },
            },
            severity_filter={AlertSeverity.CRITICAL},
        )

        # Webhook for custom integrations
        webhook_channel = NotificationChannel(
            name="webhook_monitoring",
            type="webhook",
            config={
                "url": "${MONITORING_WEBHOOK_URL}",
                "method": "POST",
                "headers": {
                    "Content-Type": "application/json",
                    "Authorization": "Bearer ${MONITORING_AUTH_TOKEN}",
                },
                "timeout": 30,
            },
        )

        for channel in [slack_channel, email_channel, pagerduty_channel, webhook_channel]:
            self.register_channel(channel)

    def _register_default_alert_rules(self):
        """Register default alert rules."""

        # High error rate alert
        high_error_rate = AlertRule(
            name="high_error_rate",
            description="High HTTP error rate detected",
            conditions=[
                AlertCondition(
                    metric_name="musicgen_http_requests_total",
                    operator=ConditionOperator.GREATER_THAN,
                    threshold=0.05,  # 5% error rate
                    duration=timedelta(minutes=5),
                )
            ],
            severity=AlertSeverity.HIGH,
            notification_channels=["slack_alerts", "email_alerts"],
            annotations={
                "summary": "High error rate detected in API",
                "runbook": "https://docs.musicgen.ai/runbooks/high-error-rate",
            },
        )

        # API latency alert
        high_latency = AlertRule(
            name="high_api_latency",
            description="API response time is too high",
            conditions=[
                AlertCondition(
                    metric_name="musicgen_http_request_duration_seconds",
                    operator=ConditionOperator.GREATER_THAN,
                    threshold=5.0,  # 5 seconds
                    duration=timedelta(minutes=3),
                )
            ],
            severity=AlertSeverity.MEDIUM,
            notification_channels=["slack_alerts"],
            annotations={
                "summary": "API latency is above acceptable threshold",
                "impact": "Users may experience slow response times",
            },
        )

        # Memory usage alert
        high_memory_usage = AlertRule(
            name="high_memory_usage",
            description="System memory usage is critically high",
            conditions=[
                AlertCondition(
                    metric_name="musicgen_memory_usage_percent",
                    operator=ConditionOperator.GREATER_THAN,
                    threshold=85.0,  # 85% memory usage
                    duration=timedelta(minutes=2),
                )
            ],
            severity=AlertSeverity.CRITICAL,
            notification_channels=["pagerduty_critical", "slack_alerts", "email_alerts"],
            annotations={
                "summary": "Critical memory usage detected",
                "runbook": "https://docs.musicgen.ai/runbooks/high-memory",
            },
        )

        # Generation failure rate alert
        generation_failures = AlertRule(
            name="generation_failure_rate",
            description="High music generation failure rate",
            conditions=[
                AlertCondition(
                    metric_name="musicgen_generation_errors_total",
                    operator=ConditionOperator.GREATER_THAN,
                    threshold=0.02,  # 2% failure rate
                    duration=timedelta(minutes=10),
                )
            ],
            severity=AlertSeverity.HIGH,
            notification_channels=["slack_alerts", "email_alerts"],
            labels={"service": "music_generation"},
            annotations={
                "summary": "High failure rate in music generation",
                "impact": "Users unable to generate music successfully",
            },
        )

        # Queue depth alert
        queue_backup = AlertRule(
            name="task_queue_backup",
            description="Task queue is backing up",
            conditions=[
                AlertCondition(
                    metric_name="musicgen_queue_depth",
                    operator=ConditionOperator.GREATER_THAN,
                    threshold=100,  # 100 tasks in queue
                    duration=timedelta(minutes=5),
                )
            ],
            severity=AlertSeverity.MEDIUM,
            notification_channels=["slack_alerts"],
            annotations={
                "summary": "Task queue depth is high",
                "impact": "Increased processing latency",
            },
        )

        # Disk space alert
        low_disk_space = AlertRule(
            name="low_disk_space",
            description="Disk space is running low",
            conditions=[
                AlertCondition(
                    metric_name="musicgen_disk_usage_percent",
                    operator=ConditionOperator.GREATER_THAN,
                    threshold=90.0,  # 90% disk usage
                    duration=timedelta(minutes=1),
                )
            ],
            severity=AlertSeverity.HIGH,
            notification_channels=["slack_alerts", "email_alerts"],
            annotations={
                "summary": "Disk space critically low",
                "runbook": "https://docs.musicgen.ai/runbooks/disk-space",
            },
        )

        # SLO violation alerts
        slo_availability_alert = AlertRule(
            name="slo_availability_violation",
            description="API availability SLO is being violated",
            conditions=[],  # Will be populated by SLO monitoring
            severity=AlertSeverity.CRITICAL,
            notification_channels=["pagerduty_critical", "slack_alerts"],
            slo_name="api_availability_99_9",
            annotations={
                "summary": "API availability SLO violation",
                "impact": "Service availability below target",
            },
        )

        # Register all default rules
        for rule in [
            high_error_rate,
            high_latency,
            high_memory_usage,
            generation_failures,
            queue_backup,
            low_disk_space,
            slo_availability_alert,
        ]:
            self.register_alert_rule(rule)

    async def check_alert_conditions(self):
        """Check all alert rule conditions."""
        for rule_name, rule in self.rules.items():
            if not rule.enabled:
                continue

            try:
                # Check SLO-based alerts
                if rule.slo_name:
                    await self._check_slo_alert(rule)
                else:
                    # Check metric-based alerts
                    await self._check_metric_alert(rule)

            except Exception as e:
                logger.error(f"Error checking alert rule {rule_name}: {e}")

    async def _check_slo_alert(self, rule: AlertRule):
        """Check SLO-based alert conditions."""
        if not rule.slo_name:
            return

        slo_status = self.sli_collector.check_slo_status(rule.slo_name)
        if not slo_status:
            return

        # Check if SLO is being violated
        is_violating = not slo_status.is_meeting_target

        # Check burn rate alerts
        is_burn_rate_alerting, burn_rate_reason = self.sli_collector.is_burn_rate_alerting(
            rule.slo_name
        )

        should_fire = is_violating or is_burn_rate_alerting

        if should_fire:
            message = f"SLO violation: {rule.slo_name}"
            if is_violating:
                message += f" (current: {slo_status.current_percent:.2f}%, target: {slo_status.target_percent}%)"
            if is_burn_rate_alerting:
                message += f" - {burn_rate_reason}"

            await self._fire_alert(
                rule,
                message,
                {
                    "slo_name": rule.slo_name,
                    "current_percent": slo_status.current_percent,
                    "target_percent": slo_status.target_percent,
                    "error_budget_remaining": slo_status.error_budget.remaining_budget,
                    "burn_rate_1h": slo_status.error_budget.burn_rate_1h,
                },
            )
        else:
            await self._resolve_alert(rule.name)

    async def _check_metric_alert(self, rule: AlertRule):
        """Check metric-based alert conditions."""
        # In a real implementation, this would query Prometheus
        # For now, we'll simulate metric values
        all_conditions_met = True

        for condition in rule.conditions:
            # Simulate metric evaluation
            metric_value = self._simulate_metric_value(condition.metric_name)
            condition_met = self._evaluate_condition(metric_value, condition)

            if not condition_met:
                all_conditions_met = False
                break

        if all_conditions_met:
            await self._fire_alert(
                rule,
                f"Alert condition met for {rule.name}",
                {"conditions": [c.metric_name for c in rule.conditions]},
            )
        else:
            await self._resolve_alert(rule.name)

    def _simulate_metric_value(self, metric_name: str) -> float:
        """Simulate metric values for testing."""
        # This would be replaced with actual Prometheus queries
        simulated_values = {
            "musicgen_http_requests_total": 0.02,  # 2% error rate
            "musicgen_http_request_duration_seconds": 2.5,  # 2.5s latency
            "musicgen_memory_usage_percent": 75.0,  # 75% memory
            "musicgen_generation_errors_total": 0.01,  # 1% generation errors
            "musicgen_queue_depth": 50,  # 50 tasks in queue
            "musicgen_disk_usage_percent": 80.0,  # 80% disk usage
        }
        return simulated_values.get(metric_name, 0.0)

    def _evaluate_condition(self, value: float, condition: AlertCondition) -> bool:
        """Evaluate if a condition is met."""
        threshold = float(condition.threshold)

        if condition.operator == ConditionOperator.GREATER_THAN:
            return value > threshold
        elif condition.operator == ConditionOperator.GREATER_EQUAL:
            return value >= threshold
        elif condition.operator == ConditionOperator.LESS_THAN:
            return value < threshold
        elif condition.operator == ConditionOperator.LESS_EQUAL:
            return value <= threshold
        elif condition.operator == ConditionOperator.EQUAL:
            return value == threshold
        elif condition.operator == ConditionOperator.NOT_EQUAL:
            return value != threshold
        else:
            return False

    async def _fire_alert(self, rule: AlertRule, message: str, details: Dict[str, Any]):
        """Fire an alert."""
        alert_id = f"{rule.name}_{int(time.time())}"
        now = datetime.now()

        # Check if alert already exists
        existing_alert = self.active_alerts.get(rule.name)

        if existing_alert:
            # Update existing alert
            existing_alert.last_fired = now
            existing_alert.firing_count += 1
            existing_alert.state = AlertState.FIRING
            alert = existing_alert
        else:
            # Create new alert
            alert = Alert(
                rule_name=rule.name,
                severity=rule.severity,
                state=AlertState.FIRING,
                message=message,
                details=details,
                labels=rule.labels,
                annotations=rule.annotations,
                first_fired=now,
                last_fired=now,
            )
            self.active_alerts[rule.name] = alert

        # Update metrics
        self.alerts_total.labels(
            rule_name=rule.name, severity=rule.severity.value, state=AlertState.FIRING.value
        ).inc()

        self._update_active_alerts_metrics()

        # Send notifications
        await self._send_notifications(alert, rule)

        logger.warning(f"Alert fired: {rule.name} - {message}")

    async def _resolve_alert(self, rule_name: str):
        """Resolve an active alert."""
        alert = self.active_alerts.get(rule_name)
        if not alert or alert.state == AlertState.RESOLVED:
            return

        alert.state = AlertState.RESOLVED
        alert.resolved_at = datetime.now()

        # Move to history and remove from active
        self.alert_history.append(alert)
        del self.active_alerts[rule_name]

        # Update metrics
        self.alerts_total.labels(
            rule_name=rule_name, severity=alert.severity.value, state=AlertState.RESOLVED.value
        ).inc()

        self._update_active_alerts_metrics()

        logger.info(f"Alert resolved: {rule_name}")

    async def _send_notifications(self, alert: Alert, rule: AlertRule):
        """Send alert notifications to configured channels."""
        # Check cooldown
        if (
            alert.last_notification
            and datetime.now() - alert.last_notification < rule.cooldown_duration
        ):
            return

        for channel_name in rule.notification_channels:
            channel = self.channels.get(channel_name)
            if not channel or not channel.enabled:
                continue

            # Check severity filter
            if channel.severity_filter and alert.severity not in channel.severity_filter:
                continue

            try:
                success = await self._send_notification(alert, channel)

                self.alert_notifications_total.labels(
                    channel_type=channel.type,
                    severity=alert.severity.value,
                    status="success" if success else "failed",
                ).inc()

                if success:
                    alert.notification_count += 1
                    alert.last_notification = datetime.now()

            except Exception as e:
                logger.error(f"Failed to send notification via {channel_name}: {e}")
                self.alert_notifications_total.labels(
                    channel_type=channel.type, severity=alert.severity.value, status="error"
                ).inc()

    async def _send_notification(self, alert: Alert, channel: NotificationChannel) -> bool:
        """Send notification to a specific channel."""
        if channel.type == "slack":
            return await self._send_slack_notification(alert, channel)
        elif channel.type == "email":
            return await self._send_email_notification(alert, channel)
        elif channel.type == "webhook":
            return await self._send_webhook_notification(alert, channel)
        elif channel.type == "pagerduty":
            return await self._send_pagerduty_notification(alert, channel)
        else:
            logger.warning(f"Unsupported notification channel type: {channel.type}")
            return False

    async def _send_slack_notification(self, alert: Alert, channel: NotificationChannel) -> bool:
        """Send Slack notification."""
        try:
            webhook_url = channel.config.get("webhook_url", "").replace("${SLACK_WEBHOOK_URL}", "")
            if not webhook_url:
                return False

            color_map = {
                AlertSeverity.CRITICAL: "#FF0000",
                AlertSeverity.HIGH: "#FF8000",
                AlertSeverity.MEDIUM: "#FFFF00",
                AlertSeverity.LOW: "#00FF00",
                AlertSeverity.INFO: "#0080FF",
            }

            payload = {
                "channel": channel.config.get("channel", "#alerts"),
                "username": channel.config.get("username", "MusicGen Monitor"),
                "icon_emoji": channel.config.get("icon_emoji", ":warning:"),
                "attachments": [
                    {
                        "color": color_map.get(alert.severity, "#808080"),
                        "title": f"🚨 {alert.severity.value.upper()} Alert",
                        "text": alert.message,
                        "fields": [
                            {"title": "Rule", "value": alert.rule_name, "short": True},
                            {"title": "Severity", "value": alert.severity.value, "short": True},
                            {"title": "State", "value": alert.state.value, "short": True},
                            {
                                "title": "First Fired",
                                "value": alert.first_fired.strftime("%Y-%m-%d %H:%M:%S"),
                                "short": True,
                            },
                        ],
                        "footer": "MusicGen AI Monitoring",
                        "ts": int(alert.last_fired.timestamp()),
                    }
                ],
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as response:
                    return response.status == 200

        except Exception as e:
            logger.error(f"Slack notification failed: {e}")
            return False

    async def _send_webhook_notification(self, alert: Alert, channel: NotificationChannel) -> bool:
        """Send webhook notification."""
        try:
            url = channel.config.get("url", "").replace("${MONITORING_WEBHOOK_URL}", "")
            if not url:
                return False

            payload = {
                "alert": {
                    "rule_name": alert.rule_name,
                    "severity": alert.severity.value,
                    "state": alert.state.value,
                    "message": alert.message,
                    "details": alert.details,
                    "labels": alert.labels,
                    "annotations": alert.annotations,
                    "first_fired": alert.first_fired.isoformat(),
                    "last_fired": alert.last_fired.isoformat(),
                },
                "timestamp": datetime.now().isoformat(),
                "service": "musicgen-ai",
            }

            headers = channel.config.get("headers", {})
            timeout = channel.config.get("timeout", 30)

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, json=payload, headers=headers, timeout=timeout
                ) as response:
                    return response.status < 400

        except Exception as e:
            logger.error(f"Webhook notification failed: {e}")
            return False

    async def _send_email_notification(self, alert: Alert, channel: NotificationChannel) -> bool:
        """Send email notification."""
        try:
            # Email implementation would go here
            # For now, just log that it would send an email
            logger.info(f"Would send email notification for alert: {alert.rule_name}")
            return True
        except Exception as e:
            logger.error(f"Email notification failed: {e}")
            return False

    async def _send_pagerduty_notification(
        self, alert: Alert, channel: NotificationChannel
    ) -> bool:
        """Send PagerDuty notification."""
        try:
            # PagerDuty implementation would go here
            logger.info(f"Would send PagerDuty notification for alert: {alert.rule_name}")
            return True
        except Exception as e:
            logger.error(f"PagerDuty notification failed: {e}")
            return False

    def _update_active_alerts_metrics(self):
        """Update active alerts metrics."""
        # Reset all severity counters
        for severity in AlertSeverity:
            self.alerts_active.labels(severity=severity.value).set(0)

        # Count active alerts by severity
        severity_counts = {}
        for alert in self.active_alerts.values():
            severity_counts[alert.severity] = severity_counts.get(alert.severity, 0) + 1

        # Update metrics
        for severity, count in severity_counts.items():
            self.alerts_active.labels(severity=severity.value).set(count)

    async def start_monitoring(self):
        """Start the alert monitoring loop."""
        if self._running:
            logger.warning("Alert monitoring already running")
            return

        self._running = True
        self._check_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Started alert monitoring")

    async def stop_monitoring(self):
        """Stop the alert monitoring loop."""
        if not self._running:
            return

        self._running = False
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass

        logger.info("Stopped alert monitoring")

    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                await self.check_alert_conditions()
                await asyncio.sleep(self._check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self._check_interval)

    def acknowledge_alert(self, rule_name: str, acknowledged_by: str) -> bool:
        """Acknowledge an active alert."""
        alert = self.active_alerts.get(rule_name)
        if not alert:
            return False

        alert.state = AlertState.ACKNOWLEDGED
        alert.acknowledged_at = datetime.now()
        alert.acknowledged_by = acknowledged_by

        logger.info(f"Alert acknowledged: {rule_name} by {acknowledged_by}")
        return True

    def suppress_alert(self, rule_name: str, duration: timedelta) -> bool:
        """Suppress an alert for a specified duration."""
        alert = self.active_alerts.get(rule_name)
        if not alert:
            return False

        alert.state = AlertState.SUPPRESSED
        # In a real implementation, you'd track suppression end time

        logger.info(f"Alert suppressed: {rule_name} for {duration}")
        return True

    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of all alerts."""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "active_alerts": len(self.active_alerts),
            "alert_history_count": len(self.alert_history),
            "alerts_by_severity": {},
            "alerts_by_state": {},
            "recent_alerts": [],
        }

        # Count by severity and state
        for alert in self.active_alerts.values():
            severity = alert.severity.value
            state = alert.state.value

            summary["alerts_by_severity"][severity] = (
                summary["alerts_by_severity"].get(severity, 0) + 1
            )
            summary["alerts_by_state"][state] = summary["alerts_by_state"].get(state, 0) + 1

        # Recent alerts (last 10)
        recent_alerts = sorted(
            list(self.active_alerts.values()) + self.alert_history[-10:],
            key=lambda x: x.last_fired,
            reverse=True,
        )[:10]

        summary["recent_alerts"] = [
            {
                "rule_name": alert.rule_name,
                "severity": alert.severity.value,
                "state": alert.state.value,
                "message": alert.message,
                "last_fired": alert.last_fired.isoformat(),
            }
            for alert in recent_alerts
        ]

        return summary


# Global alert manager instance
_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """Get the global alert manager instance."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager


def register_alert_rule(rule: AlertRule):
    """Register an alert rule with the global manager."""
    get_alert_manager().register_alert_rule(rule)


async def check_alert_conditions():
    """Check all alert conditions."""
    await get_alert_manager().check_alert_conditions()


async def send_alert(rule_name: str, message: str, details: Dict[str, Any] = None):
    """Manually trigger an alert."""
    manager = get_alert_manager()
    rule = manager.rules.get(rule_name)
    if rule:
        await manager._fire_alert(rule, message, details or {})
    else:
        logger.error(f"Alert rule not found: {rule_name}")


# Auto-start monitoring if enabled and running in an async context
import os

if os.getenv("ENABLE_ALERTING", "true").lower() == "true":
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(get_alert_manager().start_monitoring())
    except RuntimeError:
        # No event loop running, monitoring will be started manually
        pass
