"""
Tests for music_gen.utils.logging module
"""

import json
import logging
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from music_gen.utils.logging import (
    get_logger,
    setup_logging,
    LoggerMixin,
    log_function_call,
    log_gpu_memory,
)


class TestLoggerMixin:
    """Test LoggerMixin class."""

    def test_logger_mixin(self):
        """Test LoggerMixin provides logger property."""

        class TestClass(LoggerMixin):
            pass

        obj = TestClass()
        logger = obj.logger
        assert isinstance(logger, logging.Logger)
        assert logger.name.endswith("TestClass")

        # Should return same logger instance
        assert obj.logger is logger


class TestLogFunctionCall:
    """Test log_function_call decorator."""

    def test_log_function_call_success(self, caplog):
        """Test function call logging decorator."""

        @log_function_call
        def test_function(x, y):
            return x + y

        with caplog.at_level(logging.DEBUG):
            result = test_function(1, 2)

        assert result == 3
        assert "Calling test_function" in caplog.text
        assert "completed in" in caplog.text

    def test_log_function_call_error(self, caplog):
        """Test function call logging on error."""

        @log_function_call
        def failing_function():
            raise ValueError("Test error")

        with caplog.at_level(logging.DEBUG):
            with pytest.raises(ValueError):
                failing_function()

        assert "failed after" in caplog.text


class TestLogGPUMemory:
    """Test log_gpu_memory function."""

    def test_log_gpu_memory_no_cuda(self, caplog):
        """Test GPU memory logging when CUDA not available."""
        logger = get_logger("test")

        # This should not raise error even without CUDA
        log_gpu_memory(logger, "test operation")

        # Should be silent when no CUDA
        assert len(caplog.records) == 0 or "GPU Memory" not in caplog.text

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_log_gpu_memory_with_cuda(self, caplog):
        """Test GPU memory logging with CUDA available."""
        import torch

        logger = get_logger("test")

        with caplog.at_level(logging.DEBUG):
            log_gpu_memory(logger, "test operation")

        assert "GPU Memory test operation" in caplog.text
        assert "GB allocated" in caplog.text


class TestLoggingSetup:
    """Test logging setup functions."""

    def test_setup_logging_default(self):
        """Test default logging setup."""
        # Clear existing handlers
        logger = logging.getLogger("music_gen")
        logger.handlers = []

        setup_logging()

        # Check logger is configured
        assert len(logger.handlers) > 0
        assert logger.level == logging.INFO

    def test_setup_logging_custom_level(self):
        """Test logging setup with custom level."""
        # Clear existing handlers
        logger = logging.getLogger("music_gen")
        logger.handlers = []

        setup_logging(level="DEBUG")

        # Check logger level
        assert logger.level == logging.DEBUG

    def test_setup_logging_with_file(self, tmp_path):
        """Test logging setup with file handler."""
        log_file = tmp_path / "test.log"

        # Clear existing handlers
        logger = logging.getLogger("music_gen")
        logger.handlers = []

        setup_logging(log_file=str(log_file))

        # Check file handler exists
        file_handlers = [h for h in logger.handlers if hasattr(h, "baseFilename")]
        assert len(file_handlers) > 0

    def test_get_logger(self):
        """Test getting logger instance."""
        logger = get_logger("test.module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test.module"

        # Should return same instance
        logger2 = get_logger("test.module")
        assert logger is logger2


class TestLoggingIntegration:
    """Test integrated logging scenarios."""

    def test_custom_formatter_logging(self):
        """Test custom formatter functionality."""
        logger = get_logger("test.custom")

        # Create a string stream handler with custom formatter
        import io

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)

        # Use a simple custom formatter since we removed pythonjsonlogger
        class SimpleFormatter(logging.Formatter):
            def format(self, record):
                # Simple key=value format
                base = super().format(record)
                extras = []
                for key, value in record.__dict__.items():
                    if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 
                                   'pathname', 'filename', 'module', 'lineno', 
                                   'funcName', 'created', 'msecs', 'relativeCreated',
                                   'thread', 'threadName', 'processName', 'process',
                                   'stack_info', 'exc_info', 'exc_text', 'message']:
                        extras.append(f"{key}={value}")
                if extras:
                    return f"{base} | {' '.join(extras)}"
                return base

        formatter = SimpleFormatter()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        # Log a message with extra fields
        logger.info("Test message", extra={"user_id": "123", "action": "test"})

        # Check output contains our data
        output = stream.getvalue()
        assert "Test message" in output
        assert "user_id=123" in output
        assert "action=test" in output


class TestFileHandler:
    """Test file handler functionality."""

    def test_file_handler_creation(self, tmp_path):
        """Test creating file handler with setup_logging."""
        log_file = tmp_path / "test.log"

        # Clear existing handlers
        logger = logging.getLogger("music_gen")
        logger.handlers = []

        setup_logging(log_file=str(log_file))

        # Log a message
        test_logger = get_logger("music_gen.test")
        test_logger.info("Test message")

        # Check file was created and contains message
        assert log_file.exists()
        content = log_file.read_text()
        assert "Test message" in content

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
