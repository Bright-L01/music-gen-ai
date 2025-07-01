"""
Tests for music_gen.utils.exceptions module
"""

import pytest

from music_gen.utils.exceptions import (
    MusicGenError,
    ModelError,
    AudioProcessingError,
    DataLoadingError,
    GenerationError,
    ValidationError,
    ConfigurationError,
    ResourceError,
    APIError,
    handle_exceptions,
    validate_input,
    with_error_context,
    retry_on_error,
    safe_execute,
    ErrorRecovery,
    format_error_for_user,
    get_error_summary,
)


class TestExceptions:
    """Test custom exception classes."""

    def test_base_exception(self):
        """Test base MusicGenError."""
        with pytest.raises(MusicGenError) as exc_info:
            raise MusicGenError("Test error")

        assert str(exc_info.value) == "Test error"
        assert exc_info.value.details == {}
        assert exc_info.value.error_code == "MusicGenError"

    def test_exception_with_details(self):
        """Test exception with details and error code."""
        details = {"key": "value", "code": 123}

        with pytest.raises(ModelError) as exc_info:
            raise ModelError("Model failed", details=details, error_code="MODEL_001")

        assert str(exc_info.value) == "Model failed"
        assert exc_info.value.details == details
        assert exc_info.value.error_code == "MODEL_001"

    def test_all_exception_types(self):
        """Test that all exception types can be raised properly."""
        exception_types = [
            (ModelError, "Model error"),
            (GenerationError, "Generation failed"),
            (AudioProcessingError, "Audio error"),
            (ConfigurationError, "Config error"),
            (ValidationError, "Validation failed"),
            (DataLoadingError, "Data loading error"),
            (ResourceError, "Resource error"),
            (APIError, "API error"),
        ]

        for exc_class, message in exception_types:
            with pytest.raises(exc_class) as exc_info:
                raise exc_class(message)
            assert str(exc_info.value) == message
            assert isinstance(exc_info.value, MusicGenError)


class TestExceptionFormatting:
    """Test exception formatting utilities."""

    def test_format_error_for_user(self):
        """Test formatting error for user display."""
        # Test with MusicGenError
        error = MusicGenError("Failed to generate music", details={"prompt": "test"})
        formatted = format_error_for_user(error, include_details=False)
        assert formatted == "Failed to generate music"
        
        formatted_with_details = format_error_for_user(error, include_details=True)
        assert "Failed to generate music" in formatted_with_details
        assert "prompt: test" in formatted_with_details
    
    def test_format_generic_error(self):
        """Test formatting generic error."""
        error = ValueError("Invalid value")
        formatted = format_error_for_user(error)
        assert "An unexpected error occurred: Invalid value" in formatted
    
    def test_get_error_summary(self):
        """Test getting error summary."""
        try:
            raise ModelError("Model failed", details={"path": "/model/path"})
        except Exception as e:
            summary = get_error_summary(e)
            assert summary["error_type"] == "ModelError"
            assert summary["message"] == "Model failed"
            assert "traceback" in summary
            assert summary["details"]["path"] == "/model/path"


class TestExceptionHandling:
    """Test exception handling decorators."""

    def test_handle_exceptions_decorator_success(self):
        """Test handle_exceptions decorator with successful function."""

        @handle_exceptions(ValueError, reraise=False, default_return="default")
        def successful_function():
            return "success"

        result = successful_function()
        assert result == "success"

    def test_handle_exceptions_decorator_failure(self):
        """Test handle_exceptions decorator with failing function."""

        @handle_exceptions(ValueError, reraise=False, default_return="default")
        def failing_function():
            raise ValueError("Test error")

        result = failing_function()
        assert result == "default"

    def test_handle_exceptions_decorator_reraise(self):
        """Test handle_exceptions decorator with reraise option."""

        @handle_exceptions(ValueError, reraise=True)
        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            failing_function()
    
    def test_safe_execute(self):
        """Test safe_execute function."""
        # Test successful execution
        result = safe_execute(lambda x: x * 2, 5)
        assert result == 10
        
        # Test with error
        def failing_func():
            raise ValueError("Error")
        
        result = safe_execute(failing_func, default_return="failed", log_errors=False)
        assert result == "failed"


class TestRetryDecorator:
    """Test retry_on_error decorator."""

    def test_retry_success_first_try(self):
        """Test retry decorator with immediate success."""
        call_count = 0

        @retry_on_error(max_attempts=3, delay=0.01)
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

        @retry_on_error(max_attempts=3, delay=0.01)
        def eventually_successful():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Not yet")
            return "success"

        result = eventually_successful()
        assert result == "success"
        assert call_count == 3

    def test_retry_max_attempts_exceeded(self):
        """Test retry decorator when max attempts exceeded."""
        call_count = 0

        @retry_on_error(max_attempts=2, delay=0.01)
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")

        with pytest.raises(ValueError):
            always_fails()

        assert call_count == 2  # Total attempts

    def test_retry_specific_exceptions(self):
        """Test retry decorator with specific exception types."""
        call_count = 0

        @retry_on_error(max_attempts=3, delay=0.01, exceptions=(ValueError,))
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

        @retry_on_error(max_attempts=2, delay=0.01, backoff_factor=2.0)
        def measure_delays():
            raise ValueError("Measure delay")

        start_time = time.time()

        with pytest.raises(ValueError):
            measure_delays()

        elapsed = time.time() - start_time
        # Should have delay of approximately 0.01 seconds (only one retry)
        assert elapsed >= 0.01
        assert elapsed < 0.1  # Should not take too long


class TestValidateInput:
    """Test validate_input decorator."""

    def test_validate_input_success(self):
        """Test validation success."""
        @validate_input(lambda x: x > 0, "Value must be positive")
        def process_number(value):
            return value * 2
        
        result = process_number(5)
        assert result == 10
    
    def test_validate_input_failure(self):
        """Test validation failure."""
        @validate_input(lambda x: x > 0, "Value must be positive")
        def process_number(value):
            return value * 2
        
        with pytest.raises(ValidationError) as exc_info:
            process_number(-5)
        
        assert str(exc_info.value) == "Value must be positive"


class TestWithErrorContext:
    """Test with_error_context decorator."""

    def test_with_error_context_musicgen_error(self):
        """Test adding context to MusicGenError."""
        @with_error_context({"operation": "generation", "model": "musicgen-small"})
        def generate():
            raise GenerationError("Failed to generate", details={"prompt": "test"})
        
        with pytest.raises(GenerationError) as exc_info:
            generate()
        
        assert exc_info.value.details["operation"] == "generation"
        assert exc_info.value.details["model"] == "musicgen-small"
        assert exc_info.value.details["prompt"] == "test"
    
    def test_with_error_context_generic_error(self):
        """Test converting generic error to MusicGenError with context."""
        @with_error_context({"operation": "file_read"})
        def read_file():
            raise IOError("File not found")
        
        with pytest.raises(MusicGenError) as exc_info:
            read_file()
        
        assert "Unexpected error in read_file" in str(exc_info.value)
        assert exc_info.value.details["operation"] == "file_read"
        assert "File not found" in exc_info.value.details["original_exception"]


class TestErrorRecovery:
    """Test ErrorRecovery context manager."""

    def test_error_recovery_success(self):
        """Test error recovery when no error occurs."""
        cleanup_called = False
        
        def cleanup():
            nonlocal cleanup_called
            cleanup_called = True
        
        with ErrorRecovery(log_errors=False) as recovery:
            recovery.add_cleanup(cleanup)
            # No error occurs
        
        # Cleanup should not be called on success
        assert not cleanup_called
    
    def test_error_recovery_with_error(self):
        """Test error recovery when error occurs."""
        cleanup_called = False
        
        def cleanup():
            nonlocal cleanup_called
            cleanup_called = True
        
        with pytest.raises(ValueError):
            with ErrorRecovery(log_errors=False) as recovery:
                recovery.add_cleanup(cleanup)
                raise ValueError("Test error")
        
        # Cleanup should be called on error
        assert cleanup_called
    
    def test_error_recovery_multiple_cleanups(self):
        """Test multiple cleanup functions in reverse order."""
        cleanup_order = []
        
        def cleanup1():
            cleanup_order.append(1)
        
        def cleanup2():
            cleanup_order.append(2)
        
        def cleanup3():
            cleanup_order.append(3)
        
        with pytest.raises(ValueError):
            with ErrorRecovery(log_errors=False) as recovery:
                recovery.add_cleanup(cleanup1)
                recovery.add_cleanup(cleanup2)
                recovery.add_cleanup(cleanup3)
                raise ValueError("Test error")
        
        # Cleanups should be called in reverse order
        assert cleanup_order == [3, 2, 1]


class TestExceptionUsage:
    """Test real-world usage patterns."""

    def test_api_error_chain(self):
        """Test chaining API-related exceptions."""
        try:
            try:
                raise ValidationError("Invalid input")
            except ValidationError as e:
                raise APIError("API request failed") from e
        except APIError as e:
            assert e.__cause__ is not None
            assert isinstance(e.__cause__, ValidationError)

    def test_model_loading_error_flow(self):
        """Test model loading error flow."""

        def load_model(path):
            if not path:
                raise ValidationError("Path cannot be empty")
            if path == "missing":
                raise ModelError(f"Model not found: {path}")
            if path == "corrupt":
                raise ModelError("Model file corrupted", details={"path": path})
            return "model"

        # Test validation
        with pytest.raises(ValidationError):
            load_model("")

        # Test not found
        with pytest.raises(ModelError) as exc_info:
            load_model("missing")
        assert "Model not found" in str(exc_info.value)

        # Test corruption
        with pytest.raises(ModelError) as exc_info:
            load_model("corrupt")
        assert exc_info.value.details["path"] == "corrupt"

        # Test success
        assert load_model("valid") == "model"

    def test_generation_pipeline_errors(self):
        """Test generation pipeline error handling."""

        @handle_exceptions(MusicGenError, reraise=False, default_return=None)
        def generate_audio(prompt, model=None):
            if not prompt:
                raise ValidationError("Prompt cannot be empty")
            if model is None:
                raise ConfigurationError("Model not configured")
            if prompt == "oom":
                raise ResourceError("Out of memory")
            return f"audio_{prompt}"

        # Test various error scenarios
        assert generate_audio("") is None  # Validation error
        assert generate_audio("test") is None  # Config error
        assert generate_audio("oom", model="test") is None  # OOM

        # Test success
        assert generate_audio("music", model="test") == "audio_music"
