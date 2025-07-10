"""
Structured logging configuration for Music Gen AI.

Implements comprehensive logging with JSON format, correlation IDs,
performance metrics, and audit logging following 2024 best practices.
"""

import json
import logging
import logging.config
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import structlog
from pythonjsonlogger import jsonlogger

from .config import get_config


class CorrelationIdProcessor:
    """Processor to add correlation IDs to all log entries."""

    def __call__(self, logger, method_name, event_dict):
        """Add correlation ID from context or generate new one."""
        # Try to get correlation ID from various sources
        correlation_id = (
            event_dict.get("correlation_id")
            or getattr(logger, "_correlation_id", None)
            or str(uuid.uuid4())
        )

        event_dict["correlation_id"] = correlation_id
        return event_dict


class TimestampProcessor:
    """Processor to add consistent timestamps."""

    def __call__(self, logger, method_name, event_dict):
        """Add ISO format timestamp."""
        event_dict["timestamp"] = datetime.utcnow().isoformat() + "Z"
        return event_dict


class EnvironmentProcessor:
    """Processor to add environment information."""

    def __init__(self):
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.service_name = os.getenv("SERVICE_NAME", "musicgen-api")
        self.version = os.getenv("SERVICE_VERSION", "1.0.0")

    def __call__(self, logger, method_name, event_dict):
        """Add environment context."""
        event_dict.update(
            {"environment": self.environment, "service": self.service_name, "version": self.version}
        )
        return event_dict


class SensitiveDataFilter:
    """Filter to redact sensitive data from logs."""

    SENSITIVE_KEYS = {
        "password",
        "token",
        "secret",
        "key",
        "authorization",
        "credit_card",
        "ssn",
        "email",
        "phone",
        "api_key",
        "jwt",
        "session",
        "cookie",
        "csrf_token",
    }

    def __call__(self, logger, method_name, event_dict):
        """Redact sensitive data from log entries."""
        return self._redact_dict(event_dict)

    def _redact_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively redact sensitive data from dictionary."""
        if not isinstance(data, dict):
            return data

        redacted = {}
        for key, value in data.items():
            key_lower = key.lower()

            if any(sensitive in key_lower for sensitive in self.SENSITIVE_KEYS):
                # Redact but keep some info for debugging
                if isinstance(value, str) and len(value) > 8:
                    redacted[key] = f"{value[:4]}...{value[-4:]}"
                else:
                    redacted[key] = "[REDACTED]"
            elif isinstance(value, dict):
                redacted[key] = self._redact_dict(value)
            elif isinstance(value, list):
                redacted[key] = [
                    self._redact_dict(item) if isinstance(item, dict) else item for item in value
                ]
            else:
                redacted[key] = value

        return redacted


class PerformanceLogger:
    """Logger for performance metrics."""

    def __init__(self):
        self.logger = structlog.get_logger("performance")

    def log_request_performance(
        self,
        method: str,
        path: str,
        status_code: int,
        duration_ms: float,
        correlation_id: str,
        user_id: Optional[str] = None,
        **extra_metrics,
    ):
        """Log request performance metrics."""
        self.logger.info(
            "request_performance",
            method=method,
            path=path,
            status_code=status_code,
            duration_ms=duration_ms,
            correlation_id=correlation_id,
            user_id=user_id,
            **extra_metrics,
        )

    def log_database_performance(
        self,
        query_type: str,
        duration_ms: float,
        rows_affected: int,
        correlation_id: str,
        **extra_metrics,
    ):
        """Log database performance metrics."""
        self.logger.info(
            "database_performance",
            query_type=query_type,
            duration_ms=duration_ms,
            rows_affected=rows_affected,
            correlation_id=correlation_id,
            **extra_metrics,
        )

    def log_generation_performance(
        self,
        model_id: str,
        duration_seconds: float,
        audio_length_seconds: float,
        correlation_id: str,
        user_id: Optional[str] = None,
        **extra_metrics,
    ):
        """Log music generation performance metrics."""
        self.logger.info(
            "generation_performance",
            model_id=model_id,
            duration_seconds=duration_seconds,
            audio_length_seconds=audio_length_seconds,
            generation_ratio=duration_seconds / max(audio_length_seconds, 0.1),
            correlation_id=correlation_id,
            user_id=user_id,
            **extra_metrics,
        )


class AuditLogger:
    """Logger for security-sensitive operations."""

    def __init__(self):
        self.logger = structlog.get_logger("audit")

    def log_authentication(
        self,
        event_type: str,  # login, logout, token_refresh, etc.
        user_id: Optional[str],
        ip_address: str,
        user_agent: str,
        success: bool,
        correlation_id: str,
        failure_reason: Optional[str] = None,
    ):
        """Log authentication events."""
        self.logger.info(
            "authentication_event",
            event_type=event_type,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            failure_reason=failure_reason,
            correlation_id=correlation_id,
        )

    def log_authorization(
        self,
        user_id: str,
        resource: str,
        action: str,
        granted: bool,
        ip_address: str,
        correlation_id: str,
        denial_reason: Optional[str] = None,
    ):
        """Log authorization decisions."""
        self.logger.info(
            "authorization_event",
            user_id=user_id,
            resource=resource,
            action=action,
            granted=granted,
            denial_reason=denial_reason,
            ip_address=ip_address,
            correlation_id=correlation_id,
        )

    def log_data_access(
        self,
        user_id: str,
        data_type: str,
        action: str,  # read, create, update, delete
        resource_id: str,
        ip_address: str,
        correlation_id: str,
        **extra_context,
    ):
        """Log data access events."""
        self.logger.info(
            "data_access_event",
            user_id=user_id,
            data_type=data_type,
            action=action,
            resource_id=resource_id,
            ip_address=ip_address,
            correlation_id=correlation_id,
            **extra_context,
        )

    def log_security_event(
        self,
        event_type: str,
        severity: str,  # low, medium, high, critical
        description: str,
        ip_address: str,
        correlation_id: str,
        user_id: Optional[str] = None,
        **extra_context,
    ):
        """Log security-related events."""
        self.logger.warning(
            "security_event",
            event_type=event_type,
            severity=severity,
            description=description,
            ip_address=ip_address,
            user_id=user_id,
            correlation_id=correlation_id,
            **extra_context,
        )


class LoggingConfig:
    """Main logging configuration class."""

    def __init__(self):
        self.config = get_config()
        self.environment = getattr(self.config, "environment", "development")
        self.log_level = self._get_log_level()
        self.log_dir = Path(getattr(self.config, "log_dir", "logs"))
        self.log_dir.mkdir(exist_ok=True)

    def _get_log_level(self) -> str:
        """Get log level based on environment."""
        env_levels = {
            "development": "DEBUG",
            "testing": "INFO",
            "staging": "INFO",
            "production": "WARNING",
        }

        # Allow override via environment variable
        return os.getenv("LOG_LEVEL", env_levels.get(self.environment, "INFO"))

    def setup_logging(self):
        """Configure structured logging for the application."""

        # Configure structlog
        structlog.configure(
            processors=[
                # Built-in processors
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                # Custom processors
                CorrelationIdProcessor(),
                TimestampProcessor(),
                EnvironmentProcessor(),
                SensitiveDataFilter(),
                # Stack info for exceptions
                structlog.processors.StackInfoRenderer(),
                structlog.dev.set_exc_info,
                # JSON serialization
                structlog.processors.JSONRenderer(),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        # Configure standard library logging
        logging_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {
                    "()": jsonlogger.JsonFormatter,
                    "format": "%(asctime)s %(name)s %(levelname)s %(message)s",
                },
                "structured": {"()": "music_gen.core.logging_config.StructuredFormatter"},
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "structured" if self.environment == "development" else "json",
                    "stream": sys.stdout,
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": str(self.log_dir / "app.log"),
                    "maxBytes": 100 * 1024 * 1024,  # 100MB
                    "backupCount": 10,
                    "formatter": "json",
                },
                "audit_file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": str(self.log_dir / "audit.log"),
                    "maxBytes": 100 * 1024 * 1024,  # 100MB
                    "backupCount": 20,  # Keep more audit logs
                    "formatter": "json",
                },
                "performance_file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": str(self.log_dir / "performance.log"),
                    "maxBytes": 100 * 1024 * 1024,  # 100MB
                    "backupCount": 5,
                    "formatter": "json",
                },
                "error_file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": str(self.log_dir / "error.log"),
                    "maxBytes": 50 * 1024 * 1024,  # 50MB
                    "backupCount": 10,
                    "formatter": "json",
                    "level": "ERROR",
                },
            },
            "loggers": {
                # Root logger
                "": {"handlers": ["console", "file"], "level": self.log_level, "propagate": False},
                # Application loggers
                "music_gen": {
                    "handlers": ["console", "file"],
                    "level": self.log_level,
                    "propagate": False,
                },
                # Audit logger
                "audit": {
                    "handlers": ["audit_file", "console"],
                    "level": "INFO",
                    "propagate": False,
                },
                # Performance logger
                "performance": {
                    "handlers": ["performance_file"],
                    "level": "INFO",
                    "propagate": False,
                },
                # Error logger
                "error": {
                    "handlers": ["error_file", "console"],
                    "level": "ERROR",
                    "propagate": False,
                },
                # Third-party libraries
                "uvicorn": {"handlers": ["console", "file"], "level": "INFO", "propagate": False},
                "fastapi": {"handlers": ["console", "file"], "level": "INFO", "propagate": False},
                "asyncio": {"handlers": ["file"], "level": "WARNING", "propagate": False},
            },
        }

        # Apply configuration
        logging.config.dictConfig(logging_config)

        # Set up log aggregation if configured
        self._setup_log_aggregation()

        # Create global logger instances
        self._create_global_loggers()

    def _json_serializer(self, obj, **kwargs):
        """Custom JSON serializer for structlog."""
        try:
            # Remove any kwargs that json.dumps doesn't expect
            json_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k
                in [
                    "skipkeys",
                    "ensure_ascii",
                    "check_circular",
                    "allow_nan",
                    "cls",
                    "indent",
                    "separators",
                    "default",
                    "sort_keys",
                ]
            }
            json_kwargs.setdefault("default", str)
            json_kwargs.setdefault("ensure_ascii", False)
            return json.dumps(obj, **json_kwargs)
        except (TypeError, ValueError):
            return json.dumps({"serialization_error": str(obj)})

    def _setup_log_aggregation(self):
        """Set up log aggregation based on configuration."""

        # ELK Stack (Elasticsearch, Logstash, Kibana)
        elk_host = getattr(self.config, "elk_host", None)
        if elk_host:
            self._setup_elk_logging(elk_host)

        # Fluentd
        fluentd_host = getattr(self.config, "fluentd_host", None)
        if fluentd_host:
            self._setup_fluentd_logging(fluentd_host)

        # Syslog
        syslog_host = getattr(self.config, "syslog_host", None)
        if syslog_host:
            self._setup_syslog_logging(syslog_host)

    def _setup_elk_logging(self, elk_host: str):
        """Set up Elasticsearch logging."""
        try:
            from elasticsearch import Elasticsearch
            from pythonjsonlogger.jsonlogger import JsonFormatter

            class ElasticsearchHandler(logging.Handler):
                def __init__(self, es_host, index_prefix="musicgen-logs"):
                    super().__init__()
                    self.es = Elasticsearch([es_host])
                    self.index_prefix = index_prefix

                def emit(self, record):
                    try:
                        doc = json.loads(self.format(record))
                        index = f"{self.index_prefix}-{datetime.now().strftime('%Y.%m.%d')}"
                        self.es.index(index=index, document=doc)
                    except Exception:
                        self.handleError(record)

            # Add Elasticsearch handler
            es_handler = ElasticsearchHandler(elk_host)
            es_handler.setFormatter(JsonFormatter())
            logging.getLogger("").addHandler(es_handler)

        except ImportError:
            logging.warning("Elasticsearch client not available for log aggregation")

    def _setup_fluentd_logging(self, fluentd_host: str):
        """Set up Fluentd logging."""
        try:
            from fluent import handler

            # Parse host and port
            if ":" in fluentd_host:
                host, port = fluentd_host.split(":")
                port = int(port)
            else:
                host, port = fluentd_host, 24224

            fluentd_handler = handler.FluentHandler("musicgen", host=host, port=port)

            logging.getLogger("").addHandler(fluentd_handler)

        except ImportError:
            logging.warning("Fluent client not available for log aggregation")

    def _setup_syslog_logging(self, syslog_host: str):
        """Set up syslog logging."""
        try:
            from logging.handlers import SysLogHandler

            # Parse host and port
            if ":" in syslog_host:
                host, port = syslog_host.split(":")
                port = int(port)
                address = (host, port)
            else:
                address = syslog_host

            syslog_handler = SysLogHandler(address=address)
            syslog_handler.setFormatter(logging.Formatter("musicgen: %(message)s"))

            logging.getLogger("").addHandler(syslog_handler)

        except Exception as e:
            logging.warning(f"Failed to setup syslog logging: {e}")

    def _create_global_loggers(self):
        """Create global logger instances."""
        global performance_logger, audit_logger
        performance_logger = PerformanceLogger()
        audit_logger = AuditLogger()


class StructuredFormatter(logging.Formatter):
    """Custom formatter for development environment."""

    def format(self, record):
        """Format log record with colors for development."""

        # Color codes
        colors = {
            "DEBUG": "\033[36m",  # Cyan
            "INFO": "\033[32m",  # Green
            "WARNING": "\033[33m",  # Yellow
            "ERROR": "\033[31m",  # Red
            "CRITICAL": "\033[41m",  # Red background
        }

        reset = "\033[0m"

        # Extract structured data
        if hasattr(record, "getMessage"):
            try:
                msg_data = json.loads(record.getMessage())
                correlation_id = msg_data.get("correlation_id", "N/A")[:8]
                event = msg_data.get("event", record.msg)

                # Format: [TIMESTAMP] [LEVEL] [CORRELATION_ID] EVENT - DETAILS
                color = colors.get(record.levelname, "")
                formatted = (
                    f"{color}[{record.asctime}] "
                    f"[{record.levelname:8}] "
                    f"[{correlation_id}] "
                    f"{record.name}: {event}{reset}"
                )

                # Add extra context in development
                if msg_data:
                    context = {
                        k: v
                        for k, v in msg_data.items()
                        if k not in ["event", "correlation_id", "timestamp"]
                    }
                    if context:
                        formatted += f" - {json.dumps(context, default=str)}"

                return formatted

            except (json.JSONDecodeError, AttributeError):
                pass

        # Fallback to standard formatting
        return super().format(record)


# Global logger instances
performance_logger: Optional[PerformanceLogger] = None
audit_logger: Optional[AuditLogger] = None


def setup_logging():
    """Initialize logging configuration."""
    config = LoggingConfig()
    config.setup_logging()


def get_logger(name: str = None):
    """Get a structured logger instance."""
    return structlog.get_logger(name or __name__)


def get_performance_logger() -> PerformanceLogger:
    """Get the global performance logger."""
    global performance_logger
    if performance_logger is None:
        performance_logger = PerformanceLogger()
    return performance_logger


def get_audit_logger() -> AuditLogger:
    """Get the global audit logger."""
    global audit_logger
    if audit_logger is None:
        audit_logger = AuditLogger()
    return audit_logger


# Context managers for correlation ID management
class CorrelationContext:
    """Context manager for correlation ID management."""

    def __init__(self, correlation_id: Optional[str] = None):
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.old_correlation_id = None

    def __enter__(self):
        # Store old correlation ID if any
        import contextvars

        if hasattr(contextvars, "correlation_id"):
            self.old_correlation_id = contextvars.correlation_id.get(None)
            contextvars.correlation_id.set(self.correlation_id)
        return self.correlation_id

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore old correlation ID
        import contextvars

        if hasattr(contextvars, "correlation_id") and self.old_correlation_id:
            contextvars.correlation_id.set(self.old_correlation_id)


def with_correlation_id(correlation_id: Optional[str] = None):
    """Decorator to add correlation ID to function execution."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            with CorrelationContext(correlation_id):
                return func(*args, **kwargs)

        return wrapper

    return decorator
