"""
Logging management API endpoints.

Provides endpoints for log aggregation, health monitoring,
and log configuration management.
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from ...core.logging_config import get_logger, get_performance_logger, get_audit_logger

logger = get_logger(__name__)
router = APIRouter()


class LogLevel(BaseModel):
    """Log level configuration."""

    logger_name: str
    level: str


class LogStats(BaseModel):
    """Log statistics."""

    total_logs: int
    error_count: int
    warning_count: int
    info_count: int
    debug_count: int
    last_error_time: Optional[datetime]
    last_warning_time: Optional[datetime]


class LogEntry(BaseModel):
    """Log entry model."""

    timestamp: datetime
    level: str
    logger: str
    message: str
    correlation_id: Optional[str]
    extra: Dict


@router.get("/logs/health")
async def get_logging_health():
    """Get logging system health status."""
    try:
        log_dir = Path("logs")

        # Check if log directory exists
        if not log_dir.exists():
            return {
                "status": "error",
                "message": "Log directory does not exist",
                "log_directory": str(log_dir.absolute()),
            }

        # Check log files
        log_files = {
            "app.log": log_dir / "app.log",
            "audit.log": log_dir / "audit.log",
            "performance.log": log_dir / "performance.log",
            "error.log": log_dir / "error.log",
        }

        file_status = {}
        total_size = 0

        for name, path in log_files.items():
            if path.exists():
                size = path.stat().st_size
                total_size += size
                file_status[name] = {
                    "exists": True,
                    "size_bytes": size,
                    "size_mb": round(size / 1024 / 1024, 2),
                    "last_modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
                }
            else:
                file_status[name] = {"exists": False}

        # Calculate disk usage
        disk_usage = {
            "total_log_size_mb": round(total_size / 1024 / 1024, 2),
            "log_directory": str(log_dir.absolute()),
        }

        # Check if we can write to logs
        test_logger = get_logger("health_check")
        test_logger.info("Logging health check")

        return {
            "status": "healthy",
            "log_files": file_status,
            "disk_usage": disk_usage,
            "writable": True,
        }

    except Exception as e:
        logger.error("Logging health check failed", error=str(e))
        return {"status": "error", "message": str(e), "writable": False}


@router.get("/logs/stats")
async def get_log_statistics(
    hours: int = Query(24, ge=1, le=168, description="Hours of logs to analyze")
):
    """Get log statistics for the specified time period."""
    try:
        log_dir = Path("logs")
        since_time = datetime.now() - timedelta(hours=hours)

        stats = {"period_hours": hours, "since": since_time.isoformat(), "files": {}}

        # Analyze each log file
        for log_file in ["app.log", "audit.log", "performance.log", "error.log"]:
            file_path = log_dir / log_file

            if not file_path.exists():
                stats["files"][log_file] = {"exists": False}
                continue

            file_stats = {
                "exists": True,
                "total_lines": 0,
                "error_count": 0,
                "warning_count": 0,
                "info_count": 0,
                "debug_count": 0,
                "recent_errors": [],
            }

            try:
                with open(file_path, "r") as f:
                    for line in f:
                        file_stats["total_lines"] += 1

                        # Parse JSON log entries
                        try:
                            log_entry = json.loads(line.strip())

                            # Count by level
                            level = log_entry.get("levelname", "").lower()
                            if level == "error":
                                file_stats["error_count"] += 1
                                # Keep recent errors
                                if len(file_stats["recent_errors"]) < 5:
                                    file_stats["recent_errors"].append(
                                        {
                                            "timestamp": log_entry.get("timestamp"),
                                            "message": log_entry.get("message", "")[:100],
                                        }
                                    )
                            elif level == "warning":
                                file_stats["warning_count"] += 1
                            elif level == "info":
                                file_stats["info_count"] += 1
                            elif level == "debug":
                                file_stats["debug_count"] += 1

                        except json.JSONDecodeError:
                            # Skip non-JSON lines
                            continue

            except Exception as e:
                file_stats["error"] = f"Failed to read file: {str(e)}"

            stats["files"][log_file] = file_stats

        return stats

    except Exception as e:
        logger.error("Failed to get log statistics", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs/recent")
async def get_recent_logs(
    limit: int = Query(100, ge=1, le=1000, description="Number of recent log entries"),
    level: Optional[str] = Query(None, description="Filter by log level"),
    logger_name: Optional[str] = Query(None, description="Filter by logger name"),
    correlation_id: Optional[str] = Query(None, description="Filter by correlation ID"),
):
    """Get recent log entries with optional filtering."""
    try:
        log_dir = Path("logs")
        app_log = log_dir / "app.log"

        if not app_log.exists():
            return {"logs": [], "total": 0}

        logs = []

        # Read log file in reverse to get most recent entries first
        with open(app_log, "r") as f:
            lines = f.readlines()

        for line in reversed(lines[-limit * 2 :]):  # Read more lines to account for filtering
            try:
                log_entry = json.loads(line.strip())

                # Apply filters
                if level and log_entry.get("levelname", "").lower() != level.lower():
                    continue

                if logger_name and logger_name not in log_entry.get("name", ""):
                    continue

                if correlation_id and log_entry.get("correlation_id") != correlation_id:
                    continue

                logs.append(
                    {
                        "timestamp": log_entry.get("timestamp"),
                        "level": log_entry.get("levelname"),
                        "logger": log_entry.get("name"),
                        "message": log_entry.get("message"),
                        "correlation_id": log_entry.get("correlation_id"),
                        "extra": {
                            k: v
                            for k, v in log_entry.items()
                            if k
                            not in ["timestamp", "levelname", "name", "message", "correlation_id"]
                        },
                    }
                )

                if len(logs) >= limit:
                    break

            except json.JSONDecodeError:
                continue

        return {
            "logs": logs,
            "total": len(logs),
            "filters": {
                "level": level,
                "logger_name": logger_name,
                "correlation_id": correlation_id,
            },
        }

    except Exception as e:
        logger.error("Failed to get recent logs", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs/performance")
async def get_performance_metrics(
    hours: int = Query(1, ge=1, le=24, description="Hours of metrics to retrieve")
):
    """Get performance metrics from logs."""
    try:
        log_dir = Path("logs")
        perf_log = log_dir / "performance.log"

        if not perf_log.exists():
            return {"metrics": [], "summary": {}}

        since_time = datetime.now() - timedelta(hours=hours)
        metrics = []

        # Aggregate metrics
        total_requests = 0
        total_response_time = 0
        slow_requests = 0
        errors = 0

        with open(perf_log, "r") as f:
            for line in f:
                try:
                    log_entry = json.loads(line.strip())
                    timestamp_str = log_entry.get("timestamp")

                    if timestamp_str:
                        # Parse timestamp
                        timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))

                        if timestamp >= since_time:
                            duration_ms = log_entry.get("duration_ms", 0)
                            status_code = log_entry.get("status_code", 200)

                            metrics.append(
                                {
                                    "timestamp": timestamp_str,
                                    "method": log_entry.get("method"),
                                    "path": log_entry.get("path"),
                                    "duration_ms": duration_ms,
                                    "status_code": status_code,
                                    "correlation_id": log_entry.get("correlation_id"),
                                }
                            )

                            total_requests += 1
                            total_response_time += duration_ms

                            if duration_ms > 1000:  # >1 second
                                slow_requests += 1

                            if status_code >= 400:
                                errors += 1

                except (json.JSONDecodeError, ValueError):
                    continue

        # Calculate summary statistics
        summary = {
            "total_requests": total_requests,
            "avg_response_time_ms": round(total_response_time / max(total_requests, 1), 2),
            "slow_requests": slow_requests,
            "slow_request_percentage": round(slow_requests / max(total_requests, 1) * 100, 2),
            "error_requests": errors,
            "error_percentage": round(errors / max(total_requests, 1) * 100, 2),
            "period_hours": hours,
        }

        return {"metrics": metrics[-1000:], "summary": summary}  # Limit to recent 1000 entries

    except Exception as e:
        logger.error("Failed to get performance metrics", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs/audit")
async def get_audit_logs(
    limit: int = Query(100, ge=1, le=500, description="Number of audit entries"),
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
):
    """Get audit log entries for security monitoring."""
    try:
        log_dir = Path("logs")
        audit_log = log_dir / "audit.log"

        if not audit_log.exists():
            return {"audit_logs": [], "total": 0}

        logs = []

        with open(audit_log, "r") as f:
            lines = f.readlines()

        for line in reversed(lines[-limit * 2 :]):
            try:
                log_entry = json.loads(line.strip())

                # Apply filters
                if event_type and log_entry.get("event") != event_type:
                    continue

                if user_id and log_entry.get("user_id") != user_id:
                    continue

                logs.append(
                    {
                        "timestamp": log_entry.get("timestamp"),
                        "event": log_entry.get("event"),
                        "user_id": log_entry.get("user_id"),
                        "ip_address": log_entry.get("ip_address"),
                        "success": log_entry.get("success"),
                        "details": {
                            k: v
                            for k, v in log_entry.items()
                            if k not in ["timestamp", "event", "user_id", "ip_address", "success"]
                        },
                    }
                )

                if len(logs) >= limit:
                    break

            except json.JSONDecodeError:
                continue

        return {
            "audit_logs": logs,
            "total": len(logs),
            "filters": {"event_type": event_type, "user_id": user_id},
        }

    except Exception as e:
        logger.error("Failed to get audit logs", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/logs/test")
async def test_logging(request: Request):
    """Test logging functionality by generating sample log entries."""
    correlation_id = getattr(request.state, "correlation_id", "test")

    try:
        # Test different log levels
        test_logger = get_logger("test")
        perf_logger = get_performance_logger()
        audit_logger = get_audit_logger()

        # Generate sample logs
        test_logger.debug("Debug test message", correlation_id=correlation_id)
        test_logger.info("Info test message", correlation_id=correlation_id)
        test_logger.warning("Warning test message", correlation_id=correlation_id)
        test_logger.error("Error test message", correlation_id=correlation_id)

        # Generate performance log
        perf_logger.log_request_performance(
            method="POST",
            path="/logs/test",
            status_code=200,
            duration_ms=150.5,
            correlation_id=correlation_id,
            test=True,
        )

        # Generate audit log
        audit_logger.log_authentication(
            event_type="test_login",
            user_id="test_user",
            ip_address="127.0.0.1",
            user_agent="test-agent",
            success=True,
            correlation_id=correlation_id,
        )

        return {
            "success": True,
            "message": "Test logs generated successfully",
            "correlation_id": correlation_id,
            "logs_generated": {
                "debug": 1,
                "info": 1,
                "warning": 1,
                "error": 1,
                "performance": 1,
                "audit": 1,
            },
        }

    except Exception as e:
        logger.error("Logging test failed", error=str(e), correlation_id=correlation_id)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs/config")
async def get_logging_config():
    """Get current logging configuration."""
    try:
        import logging

        # Get all loggers
        loggers_info = {}

        for name in ["", "music_gen", "audit", "performance", "error"]:
            logger_obj = logging.getLogger(name)
            loggers_info[name or "root"] = {
                "level": logging.getLevelName(logger_obj.level),
                "handlers": [
                    {
                        "type": type(handler).__name__,
                        "level": logging.getLevelName(handler.level),
                        "formatter": type(handler.formatter).__name__
                        if handler.formatter
                        else None,
                    }
                    for handler in logger_obj.handlers
                ],
                "propagate": logger_obj.propagate,
            }

        # Environment configuration
        env_config = {
            "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
            "ENVIRONMENT": os.getenv("ENVIRONMENT", "development"),
            "SERVICE_NAME": os.getenv("SERVICE_NAME", "musicgen-api"),
            "SERVICE_VERSION": os.getenv("SERVICE_VERSION", "1.0.0"),
            "ELK_HOST": os.getenv("ELK_HOST"),
            "FLUENTD_HOST": os.getenv("FLUENTD_HOST"),
            "SYSLOG_HOST": os.getenv("SYSLOG_HOST"),
        }

        return {
            "loggers": loggers_info,
            "environment": env_config,
            "log_directory": str(Path("logs").absolute()),
        }

    except Exception as e:
        logger.error("Failed to get logging config", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
