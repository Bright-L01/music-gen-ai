"""
Comprehensive monitoring system for Music Gen AI.

Provides Prometheus metrics, custom business logic monitoring,
and production-ready observability following 2024 best practices.
"""

from .metrics import (
    MetricsCollector,
    BusinessMetrics,
    ApiMetrics,
    SystemMetrics,
    TaskQueueMetrics,
    get_metrics_collector,
    register_custom_metric,
    track_operation,
    track_api_request,
    track_business_event,
    track_system_resource,
    track_task_queue_event,
)

from .sli_slo import (
    SLICollector,
    SLODefinition,
    ErrorBudget,
    SLOStatus,
    get_sli_collector,
    register_slo,
    check_slo_status,
    calculate_error_budget_burn_rate,
)

from .alerting import (
    AlertManager,
    AlertRule,
    AlertSeverity,
    AlertCondition,
    get_alert_manager,
    register_alert_rule,
    check_alert_conditions,
    send_alert,
)

__all__ = [
    "MetricsCollector",
    "BusinessMetrics",
    "ApiMetrics",
    "SystemMetrics",
    "TaskQueueMetrics",
    "get_metrics_collector",
    "register_custom_metric",
    "track_operation",
    "track_api_request",
    "track_business_event",
    "track_system_resource",
    "track_task_queue_event",
    "SLICollector",
    "SLODefinition",
    "ErrorBudget",
    "SLOStatus",
    "get_sli_collector",
    "register_slo",
    "check_slo_status",
    "calculate_error_budget_burn_rate",
    "AlertManager",
    "AlertRule",
    "AlertSeverity",
    "AlertCondition",
    "get_alert_manager",
    "register_alert_rule",
    "check_alert_conditions",
    "send_alert",
]
