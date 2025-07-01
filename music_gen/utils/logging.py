"""
Standardized logging configuration for MusicGen.
"""

import logging
import logging.config
from pathlib import Path
from typing import Dict, Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_style: str = "detailed",
    enable_console: bool = True,
) -> None:
    """
    Setup standardized logging configuration for MusicGen.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file. If None, only console logging is used.
        format_style: Format style ('simple', 'detailed', 'json')
        enable_console: Whether to enable console output
    """

    # Define log formats
    formats = {
        "simple": "%(levelname)s - %(name)s - %(message)s",
        "detailed": "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        "json": "%(asctime)s %(name)s %(levelname)s %(filename)s %(lineno)d %(message)s",
    }

    log_format = formats.get(format_style, formats["detailed"])

    # Base configuration
    config: Dict = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": log_format,
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "json": {
                "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                "format": "%(asctime)s %(name)s %(levelname)s %(filename)s %(lineno)d %(message)s",
            },
        },
        "handlers": {},
        "loggers": {
            "music_gen": {
                "level": level,
                "handlers": [],
                "propagate": False,
            },
            "transformers": {
                "level": "WARNING",
                "handlers": [],
                "propagate": False,
            },
            "torch": {
                "level": "WARNING",
                "handlers": [],
                "propagate": False,
            },
        },
        "root": {
            "level": level,
            "handlers": [],
        },
    }

    handler_names = []

    # Console handler
    if enable_console:
        config["handlers"]["console"] = {
            "class": "logging.StreamHandler",
            "level": level,
            "formatter": "json" if format_style == "json" else "standard",
            "stream": "ext://sys.stdout",
        }
        handler_names.append("console")

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": level,
            "formatter": "json" if format_style == "json" else "standard",
            "filename": str(log_path),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf8",
        }
        handler_names.append("file")

    # Add error file handler for errors and above
    if log_file:
        error_log_path = log_path.parent / f"{log_path.stem}.error{log_path.suffix}"
        config["handlers"]["error_file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "ERROR",
            "formatter": "json" if format_style == "json" else "standard",
            "filename": str(error_log_path),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf8",
        }
        handler_names.append("error_file")

    # Assign handlers
    for logger_name in config["loggers"]:
        config["loggers"][logger_name]["handlers"] = handler_names
    config["root"]["handlers"] = handler_names

    # Apply configuration
    logging.config.dictConfig(config)

    # Log startup message
    logger = logging.getLogger("music_gen.logging")
    logger.info(f"Logging initialized - Level: {level}, Handlers: {handler_names}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with standardized naming.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


class LoggerMixin:
    """
    Mixin class that provides logging functionality to any class.
    """

    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        if not hasattr(self, "_logger"):
            self._logger = get_logger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        return self._logger


def log_function_call(func):
    """
    Decorator to log function calls with arguments and execution time.

    Usage:
        @log_function_call
        def my_function(arg1, arg2):
            pass
    """
    import functools
    import time

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = time.time()

        # Log function entry
        logger.debug(
            f"Calling {func.__name__} with args={args[:3]}{'...' if len(args) > 3 else ''}, kwargs={list(kwargs.keys())}"
        )

        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"{func.__name__} completed in {execution_time:.4f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.4f}s: {e}")
            raise

    return wrapper


def log_gpu_memory(logger: logging.Logger, operation: str = ""):
    """
    Log current GPU memory usage if CUDA is available.

    Args:
        logger: Logger instance
        operation: Description of the operation being performed
    """
    try:
        import torch

        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_cached = torch.cuda.memory_reserved() / 1024**3  # GB
            logger.debug(
                f"GPU Memory {operation}: {memory_allocated:.2f}GB allocated, {memory_cached:.2f}GB cached"
            )
    except ImportError:
        pass


# Configure default logging on import
if not logging.getLogger("music_gen").handlers:
    setup_logging()
