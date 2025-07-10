"""
Comprehensive integration tests for the structured logging system.

Tests correlation IDs, performance logging, audit logging, log aggregation,
and OpenTelemetry integration following 2024 best practices.
"""

import asyncio
import json
import logging
import os
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import AsyncClient

from music_gen.api.app import create_app
from music_gen.api.middleware.correlation_id import CorrelationIdMiddleware
from music_gen.api.middleware.performance_logging import PerformanceLoggingMiddleware
from music_gen.api.middleware.audit_logging import AuditLoggingMiddleware
from music_gen.core.logging_config import (
    setup_logging,
    get_logger,
    get_performance_logger,
    get_audit_logger,
    LoggingConfig,
)


class TestStructuredLogging:
    """Test suite for structured logging implementation."""

    @pytest.fixture
    def temp_log_dir(self):
        """Create temporary directory for test logs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"
            log_dir.mkdir()

            # Set log directory environment variable
            with patch.dict(os.environ, {"LOG_DIR": str(log_dir)}):
                yield log_dir

    @pytest.fixture
    def test_app(self, temp_log_dir):
        """Create test FastAPI app with logging middleware."""
        app = create_app(title="Test Music Gen AI")
        return app

    @pytest.fixture
    def client(self, test_app):
        """Create test client."""
        return TestClient(test_app)

    def test_logging_initialization(self, temp_log_dir):
        """Test that logging system initializes correctly."""
        with patch.dict(
            os.environ, {"LOG_DIR": str(temp_log_dir), "LOG_LEVEL": "DEBUG", "ENVIRONMENT": "test"}
        ):
            # Initialize logging
            config = LoggingConfig()
            config.setup_logging()

            # Verify log files are created
            app_log = temp_log_dir / "app.log"
            audit_log = temp_log_dir / "audit.log"
            performance_log = temp_log_dir / "performance.log"
            error_log = temp_log_dir / "error.log"

            # Test logging to each file
            logger = get_logger("test")
            perf_logger = get_performance_logger()
            audit_logger = get_audit_logger()

            logger.info("Test message")
            perf_logger.log_request_performance(
                method="GET",
                path="/test",
                status_code=200,
                duration_ms=100.5,
                correlation_id="test-123",
            )
            audit_logger.log_authentication(
                event_type="test_login",
                user_id="test_user",
                ip_address="127.0.0.1",
                user_agent="test-agent",
                success=True,
                correlation_id="test-123",
            )

            # Force log flushing
            for handler in logging.root.handlers:
                if hasattr(handler, "flush"):
                    handler.flush()

            # Verify log files exist and contain data
            assert app_log.exists()
            assert audit_log.exists()
            assert performance_log.exists()

            # Read and verify log content
            app_content = app_log.read_text()
            assert "Test message" in app_content
            assert "test-123" in app_content  # Correlation ID

            perf_content = performance_log.read_text()
            assert "test-123" in perf_content
            assert "100.5" in perf_content

            audit_content = audit_log.read_text()
            assert "test_login" in audit_content
            assert "test_user" in audit_content

    def test_correlation_id_generation(self, client):
        """Test that correlation IDs are generated and propagated."""
        response = client.get("/health")
        assert response.status_code == 200

        # Check that correlation ID header is returned
        assert "x-correlation-id" in response.headers
        correlation_id = response.headers["x-correlation-id"]

        # Verify format (UUID-like)
        assert len(correlation_id) == 36
        assert correlation_id.count("-") == 4

    def test_correlation_id_propagation(self, client):
        """Test that provided correlation IDs are used and propagated."""
        correlation_id = "custom-correlation-123"

        response = client.get("/health", headers={"x-correlation-id": correlation_id})

        assert response.status_code == 200
        assert response.headers["x-correlation-id"] == correlation_id

    def test_performance_logging_middleware(self, client, temp_log_dir):
        """Test performance logging middleware captures metrics."""
        with patch.dict(os.environ, {"LOG_DIR": str(temp_log_dir)}):
            # Initialize logging
            setup_logging()

            # Make request
            start_time = time.time()
            response = client.get("/health")
            end_time = time.time()

            assert response.status_code == 200

            # Check performance log
            perf_log = temp_log_dir / "performance.log"
            assert perf_log.exists()

            # Wait briefly for log writing
            time.sleep(0.1)

            content = perf_log.read_text()
            log_entries = [json.loads(line) for line in content.strip().split("\n") if line]

            # Find our request log entry
            health_logs = [log for log in log_entries if log.get("path") == "/health"]
            assert len(health_logs) >= 1

            log_entry = health_logs[0]
            assert log_entry["method"] == "GET"
            assert log_entry["status_code"] == 200
            assert "duration_ms" in log_entry
            assert "correlation_id" in log_entry
            assert "timestamp" in log_entry

    def test_audit_logging_for_auth_endpoints(self, client, temp_log_dir):
        """Test audit logging for authentication endpoints."""
        with patch.dict(os.environ, {"LOG_DIR": str(temp_log_dir)}):
            setup_logging()

            # Test login attempt (will fail but should be logged)
            response = client.post("/api/auth/login", json={"username": "test", "password": "test"})

            # Wait for log writing
            time.sleep(0.1)

            # Check audit log
            audit_log = temp_log_dir / "audit.log"
            assert audit_log.exists()

            content = audit_log.read_text()
            log_entries = [json.loads(line) for line in content.strip().split("\n") if line]

            # Find authentication log entries
            auth_logs = [log for log in log_entries if log.get("event") == "login"]
            assert len(auth_logs) >= 1

            log_entry = auth_logs[0]
            assert "ip_address" in log_entry
            assert "correlation_id" in log_entry
            assert "timestamp" in log_entry

    def test_sensitive_data_filtering(self, client, temp_log_dir):
        """Test that sensitive data is filtered from logs."""
        with patch.dict(
            os.environ, {"LOG_DIR": str(temp_log_dir), "AUDIT_LOG_REQUEST_BODY": "true"}
        ):
            setup_logging()

            # Send request with sensitive data
            response = client.post(
                "/api/auth/login",
                json={
                    "username": "test@example.com",
                    "password": "supersecret123",
                    "api_key": "sk-1234567890abcdef",
                },
            )

            time.sleep(0.1)

            # Check that sensitive data is redacted
            audit_log = temp_log_dir / "audit.log"
            content = audit_log.read_text()

            # Password should be redacted
            assert "supersecret123" not in content
            assert "sk-1234567890abcdef" not in content
            assert "[REDACTED]" in content or "...abcdef" in content

    def test_json_log_format(self, temp_log_dir):
        """Test that logs are properly formatted as JSON."""
        with patch.dict(os.environ, {"LOG_DIR": str(temp_log_dir)}):
            setup_logging()

            logger = get_logger("test")
            logger.info("Test JSON formatting", extra={"custom_field": "value"})

            # Force flush
            for handler in logging.root.handlers:
                if hasattr(handler, "flush"):
                    handler.flush()

            app_log = temp_log_dir / "app.log"
            content = app_log.read_text().strip()

            if content:
                lines = content.split("\n")
                for line in lines:
                    if line.strip():
                        # Should be valid JSON
                        log_entry = json.loads(line)

                        # Should have required fields
                        assert "timestamp" in log_entry
                        assert "levelname" in log_entry
                        assert "name" in log_entry
                        assert "message" in log_entry

                        # Should have custom field if present
                        if "Test JSON formatting" in log_entry.get("message", ""):
                            assert log_entry.get("custom_field") == "value"

    def test_log_level_filtering(self, temp_log_dir):
        """Test log level filtering works correctly."""
        with patch.dict(os.environ, {"LOG_DIR": str(temp_log_dir), "LOG_LEVEL": "WARNING"}):
            setup_logging()

            logger = get_logger("test")
            logger.debug("Debug message - should not appear")
            logger.info("Info message - should not appear")
            logger.warning("Warning message - should appear")
            logger.error("Error message - should appear")

            # Force flush
            for handler in logging.root.handlers:
                if hasattr(handler, "flush"):
                    handler.flush()

            app_log = temp_log_dir / "app.log"
            content = app_log.read_text()

            # Only WARNING and ERROR should appear
            assert "Debug message" not in content
            assert "Info message" not in content
            assert "Warning message" in content
            assert "Error message" in content

    def test_environment_specific_configuration(self, temp_log_dir):
        """Test environment-specific logging configuration."""
        # Test production environment
        with patch.dict(
            os.environ,
            {"LOG_DIR": str(temp_log_dir), "ENVIRONMENT": "production", "LOG_LEVEL": "INFO"},
        ):
            config = LoggingConfig()
            config.setup_logging()

            logger = get_logger("test")
            logger.debug("Debug in production")
            logger.info("Info in production")

            # Force flush
            for handler in logging.root.handlers:
                if hasattr(handler, "flush"):
                    handler.flush()

            app_log = temp_log_dir / "app.log"
            content = app_log.read_text()

            # Debug should be filtered out in production
            assert "Debug in production" not in content
            assert "Info in production" in content

    def test_error_logging_to_separate_file(self, temp_log_dir):
        """Test that errors are logged to separate error file."""
        with patch.dict(os.environ, {"LOG_DIR": str(temp_log_dir)}):
            setup_logging()

            logger = get_logger("test")
            logger.error("Test error message", exc_info=True)

            # Force flush
            for handler in logging.root.handlers:
                if hasattr(handler, "flush"):
                    handler.flush()

            error_log = temp_log_dir / "error.log"
            assert error_log.exists()

            content = error_log.read_text()
            assert "Test error message" in content

    def test_concurrent_logging(self, temp_log_dir):
        """Test logging under concurrent load."""
        with patch.dict(os.environ, {"LOG_DIR": str(temp_log_dir)}):
            setup_logging()

            logger = get_logger("test")

            async def log_messages(worker_id: int, count: int):
                for i in range(count):
                    logger.info(
                        f"Worker {worker_id} message {i}", worker_id=worker_id, message_id=i
                    )
                    await asyncio.sleep(0.001)  # Small delay

            async def run_concurrent_test():
                tasks = []
                for worker_id in range(5):
                    task = asyncio.create_task(log_messages(worker_id, 10))
                    tasks.append(task)

                await asyncio.gather(*tasks)

            # Run concurrent logging
            asyncio.run(run_concurrent_test())

            # Force flush
            for handler in logging.root.handlers:
                if hasattr(handler, "flush"):
                    handler.flush()

            app_log = temp_log_dir / "app.log"
            content = app_log.read_text()
            lines = [line for line in content.split("\n") if line.strip()]

            # Should have 50 messages (5 workers * 10 messages)
            assert len(lines) >= 50

            # Each line should be valid JSON
            for line in lines:
                log_entry = json.loads(line)
                assert "worker_id" in log_entry
                assert "message_id" in log_entry

    @pytest.mark.asyncio
    async def test_async_logging_performance(self, temp_log_dir):
        """Test logging performance under async load."""
        with patch.dict(os.environ, {"LOG_DIR": str(temp_log_dir)}):
            setup_logging()

            logger = get_logger("performance_test")

            start_time = time.time()

            # Log 1000 messages as fast as possible
            for i in range(1000):
                logger.info(f"Performance test message {i}", test_id=i)

            # Force flush
            for handler in logging.root.handlers:
                if hasattr(handler, "flush"):
                    handler.flush()

            end_time = time.time()
            duration = end_time - start_time

            # Should complete in reasonable time (less than 2 seconds)
            assert duration < 2.0

            # Verify all messages were logged
            app_log = temp_log_dir / "app.log"
            content = app_log.read_text()
            lines = [line for line in content.split("\n") if line.strip()]

            performance_lines = [line for line in lines if "Performance test message" in line]
            assert len(performance_lines) == 1000


class TestLogAggregationIntegration:
    """Test log aggregation with external systems."""

    def test_elk_compatible_format(self, temp_log_dir):
        """Test that logs are compatible with ELK stack."""
        with patch.dict(os.environ, {"LOG_DIR": str(temp_log_dir), "ELK_HOST": "localhost:9200"}):
            setup_logging()

            logger = get_logger("elk_test")
            logger.info("ELK test message", service="musicgen-api", version="1.0.0")

            # Force flush
            for handler in logging.root.handlers:
                if hasattr(handler, "flush"):
                    handler.flush()

            app_log = temp_log_dir / "app.log"
            content = app_log.read_text().strip()

            if content:
                log_entry = json.loads(content.split("\n")[0])

                # Should have ELK-compatible fields
                assert "@timestamp" in log_entry or "timestamp" in log_entry
                assert "level" in log_entry or "levelname" in log_entry
                assert "message" in log_entry
                assert "service" in log_entry
                assert "version" in log_entry

    def test_opentelemetry_trace_correlation(self, temp_log_dir):
        """Test OpenTelemetry trace correlation in logs."""
        with patch.dict(os.environ, {"LOG_DIR": str(temp_log_dir)}):
            setup_logging()

            logger = get_logger("otel_test")

            # Simulate OpenTelemetry trace context
            trace_id = "12345678901234567890123456789012"
            span_id = "1234567890123456"

            logger.info(
                "OpenTelemetry test message", trace_id=trace_id, span_id=span_id, trace_flags="01"
            )

            # Force flush
            for handler in logging.root.handlers:
                if hasattr(handler, "flush"):
                    handler.flush()

            app_log = temp_log_dir / "app.log"
            content = app_log.read_text().strip()

            if content:
                log_entry = json.loads(content.split("\n")[0])

                # Should have OpenTelemetry trace fields
                assert log_entry.get("trace_id") == trace_id
                assert log_entry.get("span_id") == span_id
                assert "trace_flags" in log_entry


class TestLogManagementEndpoints:
    """Test log management API endpoints."""

    @pytest.fixture
    def client(self, temp_log_dir):
        """Create test client with logging."""
        with patch.dict(os.environ, {"LOG_DIR": str(temp_log_dir)}):
            app = create_app()
            return TestClient(app)

    def test_log_health_endpoint(self, client):
        """Test log health endpoint."""
        response = client.get("/api/v1/logs/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "log_files" in data
        assert "disk_usage" in data

    def test_log_stats_endpoint(self, client):
        """Test log statistics endpoint."""
        response = client.get("/api/v1/logs/stats?hours=1")
        assert response.status_code == 200

        data = response.json()
        assert "period_hours" in data
        assert "files" in data
        assert data["period_hours"] == 1

    def test_recent_logs_endpoint(self, client):
        """Test recent logs endpoint."""
        response = client.get("/api/v1/logs/recent?limit=10")
        assert response.status_code == 200

        data = response.json()
        assert "logs" in data
        assert "total" in data
        assert len(data["logs"]) <= 10

    def test_performance_metrics_endpoint(self, client):
        """Test performance metrics endpoint."""
        response = client.get("/api/v1/logs/performance?hours=1")
        assert response.status_code == 200

        data = response.json()
        assert "metrics" in data
        assert "summary" in data

    def test_audit_logs_endpoint(self, client):
        """Test audit logs endpoint."""
        response = client.get("/api/v1/logs/audit?limit=10")
        assert response.status_code == 200

        data = response.json()
        assert "audit_logs" in data
        assert "total" in data

    def test_logging_test_endpoint(self, client):
        """Test logging test endpoint."""
        response = client.post("/api/v1/logs/test")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert "correlation_id" in data
        assert "logs_generated" in data

    def test_logging_config_endpoint(self, client):
        """Test logging configuration endpoint."""
        response = client.get("/api/v1/logs/config")
        assert response.status_code == 200

        data = response.json()
        assert "loggers" in data
        assert "environment" in data
        assert "log_directory" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
