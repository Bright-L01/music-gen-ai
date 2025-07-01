"""
Tests for music_gen.utils.exceptions module
"""

import pytest

from music_gen.utils.exceptions import (
    APIError,
    AudioProcessingError,
    AuthenticationError,
    CacheError,
    ConfigurationError,
    DatasetError,
    DependencyError,
    ExportError,
    GenerationError,
    InferenceError,
    InvalidRequestError,
    ModelLoadError,
    MusicGenException,
    NotFoundError,
)
from music_gen.utils.exceptions import PermissionError as MusicGenPermissionError
from music_gen.utils.exceptions import (
    RateLimitError,
    ResourceExhaustedError,
    ServiceUnavailableError,
    StreamingError,
)
from music_gen.utils.exceptions import TimeoutError as MusicGenTimeoutError
from music_gen.utils.exceptions import (
    TokenizationError,
    TrainingError,
    ValidationError,
    format_exception_message,
    handle_exception,
    log_exception,
    retry_on_exception,
)


class TestExceptions:
    """Test custom exception classes."""

    def test_base_exception(self):
        """Test base MusicGenException."""
        with pytest.raises(MusicGenException) as exc_info:
            raise MusicGenException("Test error")

        assert str(exc_info.value) == "Test error"
        assert exc_info.value.details is None
        assert exc_info.value.error_code is None

    def test_exception_with_details(self):
        """Test exception with details and error code."""
        details = {"key": "value", "code": 123}

        with pytest.raises(ModelLoadError) as exc_info:
            raise ModelLoadError("Model failed", details=details, error_code="MODEL_001")

        assert str(exc_info.value) == "Model failed"
        assert exc_info.value.details == details
        assert exc_info.value.error_code == "MODEL_001"

    def test_all_exception_types(self):
        """Test that all exception types can be raised properly."""
        exception_types = [
            (ModelLoadError, "Model error"),
            (GenerationError, "Generation failed"),
            (AudioProcessingError, "Audio error"),
            (ConfigurationError, "Config error"),
            (ValidationError, "Validation failed"),
            (StreamingError, "Stream error"),
            (CacheError, "Cache miss"),
            (TrainingError, "Training failed"),
            (DatasetError, "Dataset error"),
            (TokenizationError, "Token error"),
            (InferenceError, "Inference failed"),
            (ExportError, "Export failed"),
            (APIError, "API error"),
            (AuthenticationError, "Auth failed"),
            (RateLimitError, "Rate limited"),
            (NotFoundError, "Not found"),
            (MusicGenPermissionError, "Permission denied"),
            (ServiceUnavailableError, "Service down"),
            (InvalidRequestError, "Invalid request"),
            (MusicGenTimeoutError, "Timeout"),
            (ResourceExhaustedError, "Resources exhausted"),
            (DependencyError, "Dependency missing"),
        ]

        for exc_class, message in exception_types:
            with pytest.raises(exc_class) as exc_info:
                raise exc_class(message)
            assert str(exc_info.value) == message
            assert isinstance(exc_info.value, MusicGenException)


class TestExceptionFormatting:
    """Test exception formatting utilities."""

    def test_format_exception_message_simple(self):
        """Test formatting simple exception message."""
        try:
            raise ValueError("Test error")
        except Exception as e:
            formatted = format_exception_message(e)
            assert "ValueError: Test error" in formatted
            assert "Traceback" in formatted

    def test_format_exception_message_with_context(self):
        """Test formatting with context."""
        try:
            raise ModelLoadError("Failed to load", details={"model": "test"})
        except Exception as e:
            formatted = format_exception_message(e, include_traceback=True, context="Loading model")
            assert "Context: Loading model" in formatted
            assert "ModelLoadError: Failed to load" in formatted
            assert "Details: {'model': 'test'}" in formatted

    def test_format_exception_message_no_traceback(self):
        """Test formatting without traceback."""
        try:
            raise ValueError("Test error")
        except Exception as e:
            formatted = format_exception_message(e, include_traceback=False)
            assert "ValueError: Test error" in formatted
            assert "Traceback" not in formatted


class TestExceptionHandling:
    """Test exception handling decorators."""

    def test_handle_exception_decorator_success(self):
        """Test handle_exception decorator with successful function."""

        @handle_exception(default_return="default", log_errors=False)
        def successful_function():
            return "success"

        result = successful_function()
        assert result == "success"

    def test_handle_exception_decorator_failure(self):
        """Test handle_exception decorator with failing function."""

        @handle_exception(default_return="default", log_errors=False)
        def failing_function():
            raise ValueError("Test error")

        result = failing_function()
        assert result == "default"

    def test_handle_exception_decorator_reraise(self):
        """Test handle_exception decorator with reraise option."""

        @handle_exception(reraise=True, log_errors=False)
        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            failing_function()

    def test_handle_exception_with_handler(self):
        """Test handle_exception with custom handler."""
        handler_called = False

        def custom_handler(e):
            nonlocal handler_called
            handler_called = True
            return "handled"

        @handle_exception(handler=custom_handler, log_errors=False)
        def failing_function():
            raise ValueError("Test error")

        result = failing_function()
        assert result == "handled"
        assert handler_called


class TestRetryDecorator:
    """Test retry_on_exception decorator."""

    def test_retry_success_first_try(self):
        """Test retry decorator with immediate success."""
        call_count = 0

        @retry_on_exception(max_retries=3, delay=0.01)
        def successful_function():
            nonlocal call_count
            call_count += 1
            return "success"

        result = successful_function()
        assert result == "success"
        assert call_count == 1

    def test_retry_success_after_failures(self):
        """Test retry decorator with eventual success."""
        call_count = 0

        @retry_on_exception(max_retries=3, delay=0.01)
        def eventually_successful():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Not yet")
            return "success"

        result = eventually_successful()
        assert result == "success"
        assert call_count == 3

    def test_retry_max_retries_exceeded(self):
        """Test retry decorator when max retries exceeded."""
        call_count = 0

        @retry_on_exception(max_retries=2, delay=0.01)
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")

        with pytest.raises(ValueError):
            always_fails()

        assert call_count == 3  # Initial + 2 retries

    def test_retry_specific_exceptions(self):
        """Test retry decorator with specific exception types."""
        call_count = 0

        @retry_on_exception(max_retries=3, delay=0.01, exceptions=(ValueError,))
        def raises_different_exceptions():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("Retry this")
            else:
                raise TypeError("Don't retry this")

        with pytest.raises(TypeError):
            raises_different_exceptions()

        assert call_count == 2  # Should not retry TypeError

    def test_retry_with_backoff(self):
        """Test retry decorator with exponential backoff."""
        import time

        @retry_on_exception(max_retries=2, delay=0.01, backoff=2.0)
        def measure_delays():
            raise ValueError("Measure delay")

        start_time = time.time()

        with pytest.raises(ValueError):
            measure_delays()

        elapsed = time.time() - start_time
        # Should have delays of approximately 0.01 + 0.02 = 0.03 seconds
        assert elapsed >= 0.03
        assert elapsed < 0.1  # Should not take too long


class TestLogException:
    """Test log_exception function."""

    def test_log_exception_basic(self, caplog):
        """Test basic exception logging."""
        try:
            raise ValueError("Test error")
        except Exception as e:
            log_exception(e, "Test context")

        assert "Test context" in caplog.text
        assert "ValueError: Test error" in caplog.text

    def test_log_exception_with_level(self, caplog):
        """Test exception logging with different levels."""
        import logging

        try:
            raise ValueError("Test error")
        except Exception as e:
            log_exception(e, "Test context", level=logging.WARNING)

        assert "WARNING" in caplog.text
        assert "Test context" in caplog.text

    def test_log_exception_with_extra(self, caplog):
        """Test exception logging with extra data."""
        try:
            raise ModelLoadError("Load failed", details={"model": "test"})
        except Exception as e:
            log_exception(e, "Loading", extra_data={"attempt": 1})

        assert "Loading" in caplog.text
        assert "ModelLoadError: Load failed" in caplog.text
        # The extra data should be logged
        assert "attempt" in caplog.text or "1" in caplog.text


class TestExceptionUsage:
    """Test real-world usage patterns."""

    def test_api_error_chain(self):
        """Test chaining API-related exceptions."""
        try:
            try:
                raise AuthenticationError("Invalid token")
            except AuthenticationError as e:
                raise APIError("API request failed") from e
        except APIError as e:
            assert e.__cause__ is not None
            assert isinstance(e.__cause__, AuthenticationError)

    def test_model_loading_error_flow(self):
        """Test model loading error flow."""

        def load_model(path):
            if not path:
                raise ValidationError("Path cannot be empty")
            if path == "missing":
                raise NotFoundError(f"Model not found: {path}")
            if path == "corrupt":
                raise ModelLoadError("Model file corrupted", details={"path": path})
            return "model"

        # Test validation
        with pytest.raises(ValidationError):
            load_model("")

        # Test not found
        with pytest.raises(NotFoundError):
            load_model("missing")

        # Test corruption
        with pytest.raises(ModelLoadError) as exc_info:
            load_model("corrupt")
        assert exc_info.value.details["path"] == "corrupt"

        # Test success
        assert load_model("valid") == "model"

    def test_generation_pipeline_errors(self):
        """Test generation pipeline error handling."""

        @handle_exception(default_return=None, log_errors=False)
        def generate_audio(prompt, model=None):
            if not prompt:
                raise ValidationError("Prompt cannot be empty")
            if model is None:
                raise ConfigurationError("Model not configured")
            if prompt == "timeout":
                raise MusicGenTimeoutError("Generation timed out")
            if prompt == "oom":
                raise ResourceExhaustedError("Out of memory")
            return f"audio_{prompt}"

        # Test various error scenarios
        assert generate_audio("") is None  # Validation error
        assert generate_audio("test") is None  # Config error
        assert generate_audio("timeout", model="test") is None  # Timeout
        assert generate_audio("oom", model="test") is None  # OOM

        # Test success
        assert generate_audio("music", model="test") == "audio_music"
