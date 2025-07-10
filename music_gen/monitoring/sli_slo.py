"""
Service Level Indicators (SLI) and Service Level Objectives (SLO) system.

Implements comprehensive SLI/SLO monitoring with error budgets,
burn rate calculations, and automated SLO compliance tracking.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from prometheus_client import Counter, Gauge, Histogram

from .metrics import get_metrics_collector
from ..core.logging_config import get_logger

logger = get_logger(__name__)


class SLIType(Enum):
    """Types of Service Level Indicators."""

    AVAILABILITY = "availability"
    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"
    QUALITY = "quality"
    CUSTOM = "custom"


class TimePeriod(Enum):
    """Time periods for SLO evaluation."""

    HOUR = "1h"
    DAY = "24h"
    WEEK = "7d"
    MONTH = "30d"
    QUARTER = "90d"


@dataclass
class SLIDefinition:
    """Definition of a Service Level Indicator."""

    name: str
    description: str
    sli_type: SLIType
    query: str  # Prometheus query for calculating the SLI
    unit: str = ""
    good_threshold: Optional[float] = None  # For binary SLIs (good/bad)
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class SLODefinition:
    """Definition of a Service Level Objective."""

    name: str
    description: str
    sli: SLIDefinition
    target_percent: float  # Target percentage (e.g., 99.9)
    time_period: TimePeriod
    alert_burn_rate_1h: float = 14.4  # 1h burn rate for alerting (for 30d SLO)
    alert_burn_rate_6h: float = 6.0  # 6h burn rate for alerting (for 30d SLO)
    labels: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class ErrorBudget:
    """Error budget calculation and tracking."""

    slo_name: str
    total_budget: float  # Total error budget (e.g., 0.1% for 99.9% SLO)
    consumed_budget: float  # Amount of budget consumed
    remaining_budget: float  # Remaining budget
    burn_rate_1h: float  # Current 1-hour burn rate
    burn_rate_6h: float  # Current 6-hour burn rate
    burn_rate_24h: float  # Current 24-hour burn rate
    budget_exhaustion_time: Optional[datetime]  # When budget will be exhausted
    last_updated: datetime


@dataclass
class SLOStatus:
    """Current status of an SLO."""

    slo_name: str
    target_percent: float
    current_percent: float
    is_meeting_target: bool
    error_budget: ErrorBudget
    time_period: TimePeriod
    evaluation_time: datetime
    historical_compliance: List[Tuple[datetime, float]]  # History of compliance


class SLICollector:
    """Collects and calculates Service Level Indicators."""

    def __init__(self):
        self.slis: Dict[str, SLIDefinition] = {}
        self.slos: Dict[str, SLODefinition] = {}
        self.metrics_collector = get_metrics_collector()

        # Prometheus metrics for SLI/SLO tracking
        self.sli_value = Gauge(
            "musicgen_sli_value",
            "Current SLI value",
            ["sli_name", "sli_type"],
            registry=self.metrics_collector.registry,
        )

        self.slo_target = Gauge(
            "musicgen_slo_target_percent",
            "SLO target percentage",
            ["slo_name", "time_period"],
            registry=self.metrics_collector.registry,
        )

        self.slo_current = Gauge(
            "musicgen_slo_current_percent",
            "Current SLO compliance percentage",
            ["slo_name", "time_period"],
            registry=self.metrics_collector.registry,
        )

        self.slo_compliance = Gauge(
            "musicgen_slo_compliance",
            "SLO compliance status (1 = compliant, 0 = non-compliant)",
            ["slo_name", "time_period"],
            registry=self.metrics_collector.registry,
        )

        self.error_budget_remaining = Gauge(
            "musicgen_error_budget_remaining_percent",
            "Remaining error budget percentage",
            ["slo_name", "time_period"],
            registry=self.metrics_collector.registry,
        )

        self.error_budget_burn_rate = Gauge(
            "musicgen_error_budget_burn_rate",
            "Error budget burn rate",
            ["slo_name", "time_period", "interval"],  # interval: 1h, 6h, 24h
            registry=self.metrics_collector.registry,
        )

        # Initialize default SLIs and SLOs
        self._register_default_slos()

        logger.info("SLI/SLO collector initialized")

    def register_sli(self, sli: SLIDefinition):
        """Register a new SLI."""
        self.slis[sli.name] = sli
        logger.info(f"Registered SLI: {sli.name}")

    def register_slo(self, slo: SLODefinition):
        """Register a new SLO."""
        if slo.sli.name not in self.slis:
            self.register_sli(slo.sli)

        self.slos[slo.name] = slo

        # Set target metric
        self.slo_target.labels(slo_name=slo.name, time_period=slo.time_period.value).set(
            slo.target_percent
        )

        logger.info(f"Registered SLO: {slo.name} (target: {slo.target_percent}%)")

    def _register_default_slos(self):
        """Register default SLOs for Music Gen AI."""

        # API Availability SLO
        api_availability_sli = SLIDefinition(
            name="api_availability",
            description="API endpoint availability",
            sli_type=SLIType.AVAILABILITY,
            query="(sum(rate(musicgen_http_requests_total{status_code!~'5..'}[5m])) / sum(rate(musicgen_http_requests_total[5m]))) * 100",
            unit="percent",
            good_threshold=200,  # HTTP status codes < 500 are considered "good"
        )

        api_availability_slo = SLODefinition(
            name="api_availability_99_9",
            description="API should be available 99.9% of the time",
            sli=api_availability_sli,
            target_percent=99.9,
            time_period=TimePeriod.MONTH,
            alert_burn_rate_1h=14.4,
            alert_burn_rate_6h=6.0,
        )

        # API Latency SLO (P95)
        api_latency_sli = SLIDefinition(
            name="api_latency_p95",
            description="95th percentile API response time",
            sli_type=SLIType.LATENCY,
            query="histogram_quantile(0.95, sum(rate(musicgen_http_request_duration_seconds_bucket[5m])) by (le))",
            unit="seconds",
            good_threshold=2.0,  # Responses under 2s are considered "good"
        )

        api_latency_slo = SLODefinition(
            name="api_latency_p95_2s",
            description="95% of API requests should complete within 2 seconds",
            sli=api_latency_sli,
            target_percent=95.0,
            time_period=TimePeriod.DAY,
            alert_burn_rate_1h=36.0,
            alert_burn_rate_6h=6.0,
        )

        # Music Generation Success Rate SLO
        generation_success_sli = SLIDefinition(
            name="generation_success_rate",
            description="Music generation success rate",
            sli_type=SLIType.ERROR_RATE,
            query="(sum(rate(musicgen_generation_requests_total{status='success'}[5m])) / sum(rate(musicgen_generation_requests_total[5m]))) * 100",
            unit="percent",
            good_threshold=1,  # Successful generations are "good"
        )

        generation_success_slo = SLODefinition(
            name="generation_success_99_5",
            description="Music generation should succeed 99.5% of the time",
            sli=generation_success_sli,
            target_percent=99.5,
            time_period=TimePeriod.WEEK,
            alert_burn_rate_1h=10.08,
            alert_burn_rate_6h=2.52,
        )

        # Generation Latency SLO
        generation_latency_sli = SLIDefinition(
            name="generation_latency_p90",
            description="90th percentile music generation time",
            sli_type=SLIType.LATENCY,
            query="histogram_quantile(0.90, sum(rate(musicgen_generation_duration_seconds_bucket[5m])) by (le))",
            unit="seconds",
            good_threshold=30.0,  # Generations under 30s are considered "good"
        )

        generation_latency_slo = SLODefinition(
            name="generation_latency_p90_30s",
            description="90% of music generations should complete within 30 seconds",
            sli=generation_latency_sli,
            target_percent=90.0,
            time_period=TimePeriod.DAY,
            alert_burn_rate_1h=36.0,
            alert_burn_rate_6h=6.0,
        )

        # Model Load Time SLO
        model_load_sli = SLIDefinition(
            name="model_load_time_p99",
            description="99th percentile model load time",
            sli_type=SLIType.LATENCY,
            query="histogram_quantile(0.99, sum(rate(musicgen_model_load_duration_seconds_bucket[5m])) by (le))",
            unit="seconds",
            good_threshold=10.0,  # Model loads under 10s are considered "good"
        )

        model_load_slo = SLODefinition(
            name="model_load_time_p99_10s",
            description="99% of model loads should complete within 10 seconds",
            sli=model_load_sli,
            target_percent=99.0,
            time_period=TimePeriod.DAY,
            alert_burn_rate_1h=24.0,
            alert_burn_rate_6h=6.0,
        )

        # Register all default SLOs
        for slo in [
            api_availability_slo,
            api_latency_slo,
            generation_success_slo,
            generation_latency_slo,
            model_load_slo,
        ]:
            self.register_slo(slo)

    def calculate_sli_value(self, sli_name: str) -> Optional[float]:
        """Calculate current SLI value."""
        if sli_name not in self.slis:
            logger.error(f"SLI {sli_name} not found")
            return None

        sli = self.slis[sli_name]

        try:
            # In a real implementation, this would query Prometheus
            # For now, we'll simulate based on SLI type
            if sli.sli_type == SLIType.AVAILABILITY:
                # Simulate 99.95% availability
                value = 99.95
            elif sli.sli_type == SLIType.LATENCY:
                # Simulate latency values
                if "p95" in sli.name:
                    value = 1.2  # 1.2 seconds
                elif "p90" in sli.name:
                    value = 25.0  # 25 seconds
                else:
                    value = 8.0  # 8 seconds
            elif sli.sli_type == SLIType.ERROR_RATE:
                # Simulate 99.8% success rate
                value = 99.8
            else:
                value = 95.0  # Default

            # Update Prometheus metric
            self.sli_value.labels(sli_name=sli_name, sli_type=sli.sli_type.value).set(value)

            return value

        except Exception as e:
            logger.error(f"Error calculating SLI {sli_name}: {e}")
            return None

    def calculate_error_budget(self, slo_name: str) -> Optional[ErrorBudget]:
        """Calculate error budget for an SLO."""
        if slo_name not in self.slos:
            logger.error(f"SLO {slo_name} not found")
            return None

        slo = self.slos[slo_name]
        current_sli = self.calculate_sli_value(slo.sli.name)

        if current_sli is None:
            return None

        # Calculate error budget
        total_budget = 100.0 - slo.target_percent

        if slo.sli.sli_type == SLIType.LATENCY:
            # For latency SLOs, calculate based on threshold
            if slo.sli.good_threshold:
                is_good = current_sli <= slo.sli.good_threshold
                consumed_budget = 0.0 if is_good else total_budget * 0.1  # Simulate consumption
            else:
                consumed_budget = total_budget * 0.05  # Default consumption
        else:
            # For availability/error rate SLOs
            current_error_rate = 100.0 - current_sli
            consumed_budget = min(current_error_rate, total_budget)

        remaining_budget = max(0.0, total_budget - consumed_budget)

        # Calculate burn rates (simplified calculation)
        burn_rate_1h = consumed_budget / total_budget * 24.0  # Hourly rate
        burn_rate_6h = consumed_budget / total_budget * 4.0  # 6-hour rate
        burn_rate_24h = consumed_budget / total_budget  # Daily rate

        # Calculate budget exhaustion time
        if burn_rate_1h > 0:
            hours_remaining = remaining_budget / (burn_rate_1h / 24.0)
            budget_exhaustion_time = datetime.now() + timedelta(hours=hours_remaining)
        else:
            budget_exhaustion_time = None

        error_budget = ErrorBudget(
            slo_name=slo_name,
            total_budget=total_budget,
            consumed_budget=consumed_budget,
            remaining_budget=remaining_budget,
            burn_rate_1h=burn_rate_1h,
            burn_rate_6h=burn_rate_6h,
            burn_rate_24h=burn_rate_24h,
            budget_exhaustion_time=budget_exhaustion_time,
            last_updated=datetime.now(),
        )

        # Update Prometheus metrics
        self.error_budget_remaining.labels(
            slo_name=slo_name, time_period=slo.time_period.value
        ).set(remaining_budget)

        self.error_budget_burn_rate.labels(
            slo_name=slo_name, time_period=slo.time_period.value, interval="1h"
        ).set(burn_rate_1h)

        self.error_budget_burn_rate.labels(
            slo_name=slo_name, time_period=slo.time_period.value, interval="6h"
        ).set(burn_rate_6h)

        self.error_budget_burn_rate.labels(
            slo_name=slo_name, time_period=slo.time_period.value, interval="24h"
        ).set(burn_rate_24h)

        return error_budget

    def check_slo_status(self, slo_name: str) -> Optional[SLOStatus]:
        """Check current SLO status."""
        if slo_name not in self.slos:
            logger.error(f"SLO {slo_name} not found")
            return None

        slo = self.slos[slo_name]
        current_sli = self.calculate_sli_value(slo.sli.name)
        error_budget = self.calculate_error_budget(slo_name)

        if current_sli is None or error_budget is None:
            return None

        # Determine compliance based on SLI type
        if slo.sli.sli_type == SLIType.LATENCY:
            # For latency, check if within threshold
            if slo.sli.good_threshold:
                current_percent = 100.0 if current_sli <= slo.sli.good_threshold else 0.0
            else:
                current_percent = max(0.0, 100.0 - (current_sli / slo.sli.good_threshold * 100.0))
        else:
            # For availability/error rate
            current_percent = current_sli

        is_meeting_target = current_percent >= slo.target_percent

        status = SLOStatus(
            slo_name=slo_name,
            target_percent=slo.target_percent,
            current_percent=current_percent,
            is_meeting_target=is_meeting_target,
            error_budget=error_budget,
            time_period=slo.time_period,
            evaluation_time=datetime.now(),
            historical_compliance=[],  # Would be populated from historical data
        )

        # Update Prometheus metrics
        self.slo_current.labels(slo_name=slo_name, time_period=slo.time_period.value).set(
            current_percent
        )

        self.slo_compliance.labels(slo_name=slo_name, time_period=slo.time_period.value).set(
            1.0 if is_meeting_target else 0.0
        )

        return status

    def get_all_slo_statuses(self) -> Dict[str, SLOStatus]:
        """Get status for all registered SLOs."""
        statuses = {}
        for slo_name in self.slos:
            if self.slos[slo_name].enabled:
                status = self.check_slo_status(slo_name)
                if status:
                    statuses[slo_name] = status
        return statuses

    def calculate_error_budget_burn_rate(
        self, slo_name: str, time_window: str = "1h"
    ) -> Optional[float]:
        """Calculate error budget burn rate for a specific time window."""
        error_budget = self.calculate_error_budget(slo_name)
        if not error_budget:
            return None

        if time_window == "1h":
            return error_budget.burn_rate_1h
        elif time_window == "6h":
            return error_budget.burn_rate_6h
        elif time_window == "24h":
            return error_budget.burn_rate_24h
        else:
            logger.error(f"Unsupported time window: {time_window}")
            return None

    def is_burn_rate_alerting(self, slo_name: str) -> Tuple[bool, str]:
        """Check if burn rate exceeds alerting thresholds."""
        if slo_name not in self.slos:
            return False, f"SLO {slo_name} not found"

        slo = self.slos[slo_name]
        error_budget = self.calculate_error_budget(slo_name)

        if not error_budget:
            return False, "Could not calculate error budget"

        # Check 1-hour burn rate
        if error_budget.burn_rate_1h >= slo.alert_burn_rate_1h:
            return (
                True,
                f"1-hour burn rate ({error_budget.burn_rate_1h:.2f}) exceeds threshold ({slo.alert_burn_rate_1h})",
            )

        # Check 6-hour burn rate
        if error_budget.burn_rate_6h >= slo.alert_burn_rate_6h:
            return (
                True,
                f"6-hour burn rate ({error_budget.burn_rate_6h:.2f}) exceeds threshold ({slo.alert_burn_rate_6h})",
            )

        return False, "Burn rates within acceptable limits"

    def generate_slo_report(self) -> Dict[str, Any]:
        """Generate comprehensive SLO report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "slos": {},
            "summary": {
                "total_slos": len(self.slos),
                "compliant_slos": 0,
                "non_compliant_slos": 0,
                "alerting_slos": 0,
            },
        }

        for slo_name in self.slos:
            if not self.slos[slo_name].enabled:
                continue

            status = self.check_slo_status(slo_name)
            if not status:
                continue

            is_alerting, alert_reason = self.is_burn_rate_alerting(slo_name)

            report["slos"][slo_name] = {
                "target_percent": status.target_percent,
                "current_percent": status.current_percent,
                "is_meeting_target": status.is_meeting_target,
                "error_budget": {
                    "total": status.error_budget.total_budget,
                    "consumed": status.error_budget.consumed_budget,
                    "remaining": status.error_budget.remaining_budget,
                    "burn_rate_1h": status.error_budget.burn_rate_1h,
                    "burn_rate_6h": status.error_budget.burn_rate_6h,
                    "burn_rate_24h": status.error_budget.burn_rate_24h,
                    "exhaustion_time": status.error_budget.budget_exhaustion_time.isoformat()
                    if status.error_budget.budget_exhaustion_time
                    else None,
                },
                "is_alerting": is_alerting,
                "alert_reason": alert_reason if is_alerting else None,
                "time_period": status.time_period.value,
            }

            # Update summary
            if status.is_meeting_target:
                report["summary"]["compliant_slos"] += 1
            else:
                report["summary"]["non_compliant_slos"] += 1

            if is_alerting:
                report["summary"]["alerting_slos"] += 1

        return report


# Global SLI collector instance
_sli_collector: Optional[SLICollector] = None


def get_sli_collector() -> SLICollector:
    """Get the global SLI collector instance."""
    global _sli_collector
    if _sli_collector is None:
        _sli_collector = SLICollector()
    return _sli_collector


def register_slo(slo: SLODefinition):
    """Register an SLO with the global collector."""
    get_sli_collector().register_slo(slo)


def check_slo_status(slo_name: str) -> Optional[SLOStatus]:
    """Check status of a specific SLO."""
    return get_sli_collector().check_slo_status(slo_name)


def calculate_error_budget_burn_rate(slo_name: str, time_window: str = "1h") -> Optional[float]:
    """Calculate error budget burn rate for an SLO."""
    return get_sli_collector().calculate_error_budget_burn_rate(slo_name, time_window)
