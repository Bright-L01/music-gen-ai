"""
Tests for music_gen.utils.logging module
"""

import json
import logging
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from music_gen.utils.logging import (
    ColoredFormatter,
    PerformanceLogger,
    StructuredFormatter,
    create_console_handler,
    create_file_handler,
    get_logger,
    log_execution_time,
    log_memory_usage,
    setup_logging,
)


class TestColoredFormatter:
    """Test ColoredFormatter class."""

    def test_colored_formatter_creation(self):
        """Test creating colored formatter."""
        formatter = ColoredFormatter()
        assert isinstance(formatter, logging.Formatter)

    def test_colored_formatter_format(self):
        """Test formatting with colors."""
        formatter = ColoredFormatter()

        # Create log records for different levels
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)
        assert "Test message" in formatted
        assert "\033[" in formatted  # ANSI color code

    def test_colored_formatter_levels(self):
        """Test different log levels have different colors."""
        formatter = ColoredFormatter()

        levels = [
            (logging.DEBUG, "DEBUG"),
            (logging.INFO, "INFO"),
            (logging.WARNING, "WARNING"),
            (logging.ERROR, "ERROR"),
            (logging.CRITICAL, "CRITICAL"),
        ]

        formatted_messages = []
        for level, level_name in levels:
            record = logging.LogRecord(
                name="test",
                level=level,
                pathname="test.py",
                lineno=10,
                msg=f"{level_name} message",
                args=(),
                exc_info=None,
            )
            formatted_messages.append(formatter.format(record))

        # Each level should have different formatting
        assert len(set(formatted_messages)) == len(levels)


class TestStructuredFormatter:
    """Test StructuredFormatter class."""

    def test_structured_formatter_json(self):
        """Test JSON structured formatting."""
        formatter = StructuredFormatter(format_type="json")

        record = logging.LogRecord(
            name="test.module",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.created = time.time()

        formatted = formatter.format(record)
        data = json.loads(formatted)

        assert data["message"] == "Test message"
        assert data["level"] == "INFO"
        assert data["logger"] == "test.module"
        assert "timestamp" in data
        assert data["file"] == "test.py"
        assert data["line"] == 10

    def test_structured_formatter_with_extra(self):
        """Test structured formatting with extra fields."""
        formatter = StructuredFormatter(format_type="json")

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        # Add extra fields
        record.user_id = "123"
        record.request_id = "abc-def"
        record.duration = 1.5

        formatted = formatter.format(record)
        data = json.loads(formatted)

        assert data["extra"]["user_id"] == "123"
        assert data["extra"]["request_id"] == "abc-def"
        assert data["extra"]["duration"] == 1.5

    def test_structured_formatter_key_value(self):
        """Test key-value structured formatting."""
        formatter = StructuredFormatter(format_type="key_value")

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)

        assert "level=INFO" in formatted
        assert "logger=test" in formatted
        assert 'message="Test message"' in formatted
        assert "file=test.py" in formatted
        assert "line=10" in formatted


class TestPerformanceLogger:
    """Test PerformanceLogger class."""

    def test_performance_logger_creation(self):
        """Test creating performance logger."""
        perf_logger = PerformanceLogger("test.performance")
        assert perf_logger.logger.name == "test.performance"

    def test_log_timing(self):
        """Test logging execution time."""
        perf_logger = PerformanceLogger("test")

        with patch.object(perf_logger.logger, "info") as mock_info:
            perf_logger.log_timing("test_operation", 1.234, {"param": "value"})

            mock_info.assert_called_once()
            call_args = mock_info.call_args[0][0]
            assert "test_operation" in call_args
            assert "1.234" in call_args or "1.23" in call_args

    def test_log_memory(self):
        """Test logging memory usage."""
        perf_logger = PerformanceLogger("test")

        with patch.object(perf_logger.logger, "info") as mock_info:
            perf_logger.log_memory("test_operation", 1024 * 1024 * 100)  # 100 MB

            mock_info.assert_called_once()
            call_args = mock_info.call_args[0][0]
            assert "test_operation" in call_args
            assert "100" in call_args or "MB" in call_args

    def test_log_metrics(self):
        """Test logging custom metrics."""
        perf_logger = PerformanceLogger("test")

        metrics = {"accuracy": 0.95, "loss": 0.05, "samples_per_second": 1000}

        with patch.object(perf_logger.logger, "info") as mock_info:
            perf_logger.log_metrics("training", metrics, {"epoch": 10})

            mock_info.assert_called_once()
            call_args = mock_info.call_args[0][0]
            assert "training" in call_args

            # Check extra data
            extra = mock_info.call_args[1].get("extra", {})
            assert extra.get("metrics") == metrics
            assert extra.get("metadata") == {"epoch": 10}


class TestLoggingSetup:
    """Test logging setup functions."""

    def test_setup_logging_default(self):
        """Test default logging setup."""
        with patch("logging.basicConfig") as mock_config:
            setup_logging()

            mock_config.assert_called_once()
            call_kwargs = mock_config.call_args[1]
            assert call_kwargs["level"] == logging.INFO
            assert "format" in call_kwargs

    def test_setup_logging_custom_level(self):
        """Test logging setup with custom level."""
        with patch("logging.basicConfig") as mock_config:
            setup_logging(level="DEBUG")

            call_kwargs = mock_config.call_args[1]
            assert call_kwargs["level"] == logging.DEBUG

    def test_setup_logging_with_file(self, tmp_path):
        """Test logging setup with file handler."""
        log_file = tmp_path / "test.log"

        with patch("logging.basicConfig") as mock_config:
            setup_logging(log_file=str(log_file))

            mock_config.assert_called_once()
            # File handler should be configured

    def test_get_logger(self):
        """Test getting logger instance."""
        logger = get_logger("test.module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test.module"

        # Should return same instance
        logger2 = get_logger("test.module")
        assert logger is logger2


class TestLogDecorators:
    """Test logging decorators."""

    def test_log_execution_time_success(self, caplog):
        """Test execution time logging decorator."""

        @log_execution_time()
        def slow_function(duration=0.1):
            time.sleep(duration)
            return "done"

        with caplog.at_level(logging.INFO):
            result = slow_function()

        assert result == "done"
        assert "slow_function" in caplog.text
        assert "execution time" in caplog.text.lower()
        assert "0.1" in caplog.text or "100" in caplog.text  # 0.1s or 100ms

    def test_log_execution_time_with_custom_logger(self, caplog):
        """Test execution time decorator with custom logger."""
        custom_logger = get_logger("custom.logger")

        @log_execution_time(logger=custom_logger)
        def test_function():
            return "result"

        with caplog.at_level(logging.INFO, logger="custom.logger"):
            result = test_function()

        assert result == "result"
        assert "test_function" in caplog.text

    def test_log_execution_time_with_prefix(self, caplog):
        """Test execution time decorator with prefix."""

        @log_execution_time(prefix="API Call")
        def api_function():
            return {"status": "ok"}

        with caplog.at_level(logging.INFO):
            result = api_function()

        assert result == {"status": "ok"}
        assert "API Call" in caplog.text
        assert "api_function" in caplog.text

    def test_log_memory_usage(self, caplog):
        """Test memory usage logging decorator."""

        @log_memory_usage()
        def memory_intensive_function():
            # Allocate some memory
            data = [0] * 1000000
            return len(data)

        with caplog.at_level(logging.INFO):
            result = memory_intensive_function()

        assert result == 1000000
        assert "memory_intensive_function" in caplog.text
        assert "memory" in caplog.text.lower()
        assert "MB" in caplog.text or "KB" in caplog.text

    def test_log_memory_usage_threshold(self, caplog):
        """Test memory usage decorator with threshold."""

        @log_memory_usage(threshold_mb=0.001)  # Very low threshold
        def small_function():
            return [1, 2, 3]

        with caplog.at_level(logging.INFO):
            result = small_function()

        assert result == [1, 2, 3]
        # Should log because we set a very low threshold
        assert "memory" in caplog.text.lower()


class TestHandlerCreation:
    """Test handler creation functions."""

    def test_create_file_handler(self, tmp_path):
        """Test creating file handler."""
        log_file = tmp_path / "test.log"

        handler = create_file_handler(
            str(log_file), level=logging.DEBUG, max_bytes=1024 * 1024, backup_count=3
        )

        assert isinstance(handler, logging.Handler)
        assert handler.level == logging.DEBUG

        # Test that it can write
        logger = logging.getLogger("test_file")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        logger.info("Test message")

        assert log_file.exists()
        assert "Test message" in log_file.read_text()

    def test_create_console_handler(self):
        """Test creating console handler."""
        handler = create_console_handler(level=logging.WARNING, use_color=True)

        assert isinstance(handler, logging.StreamHandler)
        assert handler.level == logging.WARNING
        assert isinstance(handler.formatter, ColoredFormatter)

    def test_create_console_handler_structured(self):
        """Test creating structured console handler."""
        handler = create_console_handler(level=logging.INFO, format_type="json")

        assert isinstance(handler, logging.StreamHandler)
        assert isinstance(handler.formatter, StructuredFormatter)


class TestLoggingIntegration:
    """Test integrated logging scenarios."""

    def test_performance_logging_context_manager(self, caplog):
        """Test performance logging as context manager."""
        perf_logger = PerformanceLogger("test.perf")

        with perf_logger.timer("database_query") as timer:
            time.sleep(0.05)
            timer.metadata = {"query": "SELECT * FROM users"}

        assert "database_query" in caplog.text
        assert "0.05" in caplog.text or "50" in caplog.text

    def test_structured_logging_with_context(self):
        """Test structured logging with context."""
        logger = get_logger("test.structured")
        handler = create_console_handler(format_type="json")
        logger.addHandler(handler)

        # Capture output
        import io

        stream = io.StringIO()
        handler.stream = stream

        # Log with extra context
        logger.info("User action", extra={"user_id": "123", "action": "login", "ip": "192.168.1.1"})

        output = stream.getvalue()
        data = json.loads(output.strip())

        assert data["message"] == "User action"
        assert data["extra"]["user_id"] == "123"
        assert data["extra"]["action"] == "login"
        assert data["extra"]["ip"] == "192.168.1.1"

    def test_exception_logging(self, caplog):
        """Test logging exceptions."""
        logger = get_logger("test.exceptions")

        try:
            raise ValueError("Test error")
        except ValueError:
            logger.exception("An error occurred")

        assert "An error occurred" in caplog.text
        assert "ValueError: Test error" in caplog.text
        assert "Traceback" in caplog.text

    def test_log_aggregation(self):
        """Test aggregating logs from multiple sources."""
        # Create parent logger
        parent_logger = get_logger("music_gen")

        # Create child loggers
        api_logger = get_logger("music_gen.api")
        model_logger = get_logger("music_gen.models")

        # Capture all logs
        import io

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        parent_logger.addHandler(handler)
        parent_logger.setLevel(logging.DEBUG)

        # Log from different modules
        api_logger.info("API request received")
        model_logger.debug("Loading model")
        api_logger.warning("Rate limit approaching")

        output = stream.getvalue()

        assert "API request received" in output
        assert "Loading model" in output
        assert "Rate limit approaching" in output
