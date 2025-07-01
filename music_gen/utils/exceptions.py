"""
Custom exceptions and error handling utilities for MusicGen.
"""

import functools
import traceback
from typing import Any, Callable, Dict, Optional, Type, TypeVar, Union

from .logging import get_logger

F = TypeVar("F", bound=Callable[..., Any])

logger = get_logger(__name__)


class MusicGenError(Exception):
    """Base exception for all MusicGen errors."""

    def __init__(
        self, message: str, error_code: Optional[str] = None, details: Optional[Dict] = None
    ):
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/API responses."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
        }


class ModelError(MusicGenError):
    """Errors related to model loading, initialization, or execution."""



class AudioProcessingError(MusicGenError):
    """Errors related to audio processing operations."""



class DataLoadingError(MusicGenError):
    """Errors related to data loading and preprocessing."""



class GenerationError(MusicGenError):
    """Errors during music generation."""



class ValidationError(MusicGenError):
    """Errors related to input validation."""



class ConfigurationError(MusicGenError):
    """Errors related to configuration and setup."""



class ResourceError(MusicGenError):
    """Errors related to resource availability (memory, disk, etc.)."""



class APIError(MusicGenError):
    """Errors related to API operations."""



def handle_exceptions(
    *exception_types: Type[Exception],
    reraise: bool = True,
    default_return: Any = None,
    log_level: str = "error",
) -> Callable[[F], F]:
    """
    Decorator to handle specific exceptions with logging and optional reraising.

    Args:
        exception_types: Exception types to catch
        reraise: Whether to reraise the exception after logging
        default_return: Value to return if exception is caught and not reraised
        log_level: Log level for caught exceptions

    Example:
        @handle_exceptions(ValueError, TypeError, reraise=False, default_return=None)
        def risky_function():
            ...
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exception_types as e:
                error_info = {
                    "function": func.__name__,
                    "module": func.__module__,
                    "exception_type": type(e).__name__,
                    "args": str(args)[:100] + "..." if len(str(args)) > 100 else str(args),
                    "kwargs": list(kwargs.keys()),
                }

                log_func = getattr(logger, log_level)
                log_func(f"Exception in {func.__name__}: {e}", extra={"error_info": error_info})

                if reraise:
                    raise
                return default_return

        return wrapper

    return decorator


def validate_input(
    validation_func: Callable[[Any], bool],
    error_message: str,
    error_type: Type[MusicGenError] = ValidationError,
) -> Callable[[F], F]:
    """
    Decorator to validate function inputs.

    Args:
        validation_func: Function that takes the first argument and returns bool
        error_message: Error message if validation fails
        error_type: Exception type to raise

    Example:
        @validate_input(lambda x: x > 0, "Value must be positive")
        def process_positive_number(value):
            ...
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if args and not validation_func(args[0]):
                raise error_type(error_message)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def with_error_context(context: Dict[str, Any]) -> Callable[[F], F]:
    """
    Decorator to add context information to any exceptions raised.

    Args:
        context: Dictionary of context information to add to exceptions

    Example:
        @with_error_context({"model_name": "musicgen-small", "operation": "generation"})
        def generate_music():
            ...
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except MusicGenError as e:
                e.details.update(context)
                raise
            except Exception as e:
                # Convert to MusicGenError with context
                raise MusicGenError(
                    f"Unexpected error in {func.__name__}: {e}",
                    details={"original_exception": str(e), **context},
                )

        return wrapper

    return decorator


def retry_on_error(
    max_attempts: int = 3,
    exceptions: tuple = (Exception,),
    delay: float = 1.0,
    backoff_factor: float = 2.0,
) -> Callable[[F], F]:
    """
    Decorator to retry function calls on specific exceptions.

    Args:
        max_attempts: Maximum number of attempts
        exceptions: Tuple of exception types to retry on
        delay: Initial delay between retries in seconds
        backoff_factor: Factor to multiply delay by after each attempt

    Example:
        @retry_on_error(max_attempts=3, exceptions=(ConnectionError,))
        def download_model():
            ...
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts - 1:
                        break

                    logger.warning(
                        f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {e}. "
                        f"Retrying in {current_delay:.1f}s..."
                    )

                    import time

                    time.sleep(current_delay)
                    current_delay *= backoff_factor

            logger.error(f"All {max_attempts} attempts failed for {func.__name__}")
            raise last_exception

        return wrapper

    return decorator


def safe_execute(
    func: Callable, *args, default_return: Any = None, log_errors: bool = True, **kwargs
) -> Any:
    """
    Safely execute a function, returning default value on error.

    Args:
        func: Function to execute
        *args: Arguments to pass to function
        default_return: Value to return on error
        log_errors: Whether to log errors
        **kwargs: Keyword arguments to pass to function

    Returns:
        Function result or default_return on error
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            logger.error(f"Error in safe_execute: {e}")
        return default_return


class ErrorRecovery:
    """
    Context manager for error recovery with automatic cleanup.

    Example:
        with ErrorRecovery() as recovery:
            recovery.add_cleanup(lambda: model.cleanup())
            model.load()
            # If error occurs, cleanup will be called automatically
    """

    def __init__(self, log_errors: bool = True):
        self.cleanup_functions = []
        self.log_errors = log_errors
        self.logger = get_logger(f"{__name__}.ErrorRecovery")

    def add_cleanup(self, cleanup_func: Callable):
        """Add a cleanup function to be called on error."""
        self.cleanup_functions.append(cleanup_func)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            if self.log_errors:
                self.logger.error(f"Error occurred: {exc_val}")
                self.logger.debug(f"Traceback: {''.join(traceback.format_tb(exc_tb))}")

            # Execute cleanup functions in reverse order
            for cleanup_func in reversed(self.cleanup_functions):
                try:
                    cleanup_func()
                except Exception as cleanup_error:
                    if self.log_errors:
                        self.logger.error(f"Error during cleanup: {cleanup_error}")

        return False  # Don't suppress the exception


def format_error_for_user(
    error: Union[Exception, MusicGenError], include_details: bool = False
) -> str:
    """
    Format error for user-friendly display.

    Args:
        error: Exception to format
        include_details: Whether to include technical details

    Returns:
        Formatted error message
    """
    if isinstance(error, MusicGenError):
        message = error.message
        if include_details and error.details:
            details_str = ", ".join(f"{k}: {v}" for k, v in error.details.items())
            message += f" (Details: {details_str})"
        return message
    else:
        return f"An unexpected error occurred: {str(error)}"


def get_error_summary(error: Exception) -> Dict[str, Any]:
    """
    Get a comprehensive summary of an error for logging/debugging.

    Args:
        error: Exception to summarize

    Returns:
        Dictionary with error information
    """
    summary = {
        "error_type": type(error).__name__,
        "message": str(error),
        "traceback": traceback.format_exc(),
    }

    if isinstance(error, MusicGenError):
        summary.update(error.to_dict())

    return summary
