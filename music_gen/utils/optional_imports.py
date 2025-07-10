import torch

"""
Utility functions for handling optional dependencies gracefully.

This module provides decorators and context managers to handle optional
dependencies that may not be installed, allowing the system to degrade
gracefully when advanced features are not available.
"""

import functools
import logging
import warnings
from typing import Any, Callable, Dict, List, Optional, Type, Union

logger = logging.getLogger(__name__)


class OptionalDependencyError(ImportError):
    """Raised when an optional dependency is required but not available."""


class OptionalImport:
    """Context manager for optional imports."""

    def __init__(
        self,
        package_name: str,
        fallback_message: Optional[str] = None,
        install_command: Optional[str] = None,
    ):
        self.package_name = package_name
        self.fallback_message = fallback_message or f"{package_name} is not available"
        self.install_command = install_command or f"pip install {package_name}"
        self.module = None
        self.available = False

    def __enter__(self):
        try:
            # Try importing the package
            if "." in self.package_name:
                # Handle submodule imports like 'torch.nn'
                parts = self.package_name.split(".")
                self.module = __import__(parts[0])
                for part in parts[1:]:
                    self.module = getattr(self.module, part)
            else:
                self.module = __import__(self.package_name)

            self.available = True
            return self.module
        except ImportError as e:
            logger.debug(f"Optional dependency {self.package_name} not available: {e}")
            self.available = False
            return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def require(self) -> Any:
        """Require the dependency, raising an error if not available."""
        if not self.available:
            raise OptionalDependencyError(
                f"{self.fallback_message}. Install with: {self.install_command}"
            )
        return self.module


def optional_import(
    package_name: str, fallback_message: Optional[str] = None, install_command: Optional[str] = None
) -> OptionalImport:
    """Create an optional import context manager."""
    return OptionalImport(package_name, fallback_message, install_command)


def requires_optional(
    dependencies: Union[str, List[str]], fallback_return: Any = None, warn: bool = True
):
    """Decorator to mark functions that require optional dependencies."""

    if isinstance(dependencies, str):
        dependencies = [dependencies]

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            missing_deps = []

            for dep in dependencies:
                with optional_import(dep) as module:
                    if module is None:
                        missing_deps.append(dep)

            if missing_deps:
                message = f"Function {func.__name__} requires optional dependencies: {missing_deps}"
                if warn:
                    warnings.warn(message, UserWarning)
                logger.warning(message)
                return fallback_return

            return func(*args, **kwargs)

        return wrapper

    return decorator


def get_available_backends() -> Dict[str, bool]:
    """Get availability status of all optional backends."""
    backends = {
        # Audio processing
        "pedalboard": False,
        "pyrubberband": False,
        "librosa": False,
        "soundfile": False,
        # Database
        "aioredis": False,
        "asyncpg": False,
        "sqlalchemy": False,
        # ML/AI
        "transformers": False,
        "torch": False,
        "torchaudio": False,
        # Advanced features
        "wandb": False,
        "tensorboard": False,
        "pretty_midi": False,
    }

    for backend in backends:
        with optional_import(backend) as module:
            backends[backend] = module is not None

    return backends


def check_system_requirements() -> Dict[str, Any]:
    """Check system requirements and available features."""
    import platform
    import sys

    backends = get_available_backends()

    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "available_backends": backends,
        "missing_core_deps": [
            dep
            for dep, available in backends.items()
            if dep in ["torch", "torchaudio", "transformers"] and not available
        ],
        "missing_optional_deps": [
            dep
            for dep, available in backends.items()
            if dep not in ["torch", "torchaudio", "transformers"] and not available
        ],
    }


def suggest_installations(requirements: List[str]) -> str:
    """Suggest installation commands for missing requirements."""
    install_commands = {
        "pedalboard": "pip install pedalboard",
        "pyrubberband": "pip install pyrubberband",
        "librosa": "pip install librosa",
        "soundfile": "pip install soundfile",
        "aioredis": "pip install aioredis",
        "asyncpg": "pip install asyncpg",
        "sqlalchemy": "pip install sqlalchemy[asyncio]",
        "wandb": "pip install wandb",
        "tensorboard": "pip install tensorboard",
        "pretty_midi": "pip install pretty_midi",
    }

    suggestions = []
    for req in requirements:
        if req in install_commands:
            suggestions.append(install_commands[req])
        else:
            suggestions.append(f"pip install {req}")

    return "\n".join(suggestions)


# Pre-check common dependencies
_TORCH_AVAILABLE = None
_TORCHAUDIO_AVAILABLE = None
_TRANSFORMERS_AVAILABLE = None


def is_torch_available() -> bool:
    """Check if PyTorch is available."""
    global _TORCH_AVAILABLE
    if _TORCH_AVAILABLE is None:
        with optional_import("torch") as torch:
            _TORCH_AVAILABLE = torch is not None
    return _TORCH_AVAILABLE


def is_torchaudio_available() -> bool:
    """Check if torchaudio is available."""
    global _TORCHAUDIO_AVAILABLE
    if _TORCHAUDIO_AVAILABLE is None:
        with optional_import("torchaudio") as torchaudio:
            _TORCHAUDIO_AVAILABLE = torchaudio is not None
    return _TORCHAUDIO_AVAILABLE


def is_transformers_available() -> bool:
    """Check if transformers is available."""
    global _TRANSFORMERS_AVAILABLE
    if _TRANSFORMERS_AVAILABLE is None:
        with optional_import("transformers") as transformers:
            _TRANSFORMERS_AVAILABLE = transformers is not None
    return _TRANSFORMERS_AVAILABLE


def safe_import_or_mock(package_name: str, mock_class: Optional[Type] = None) -> Any:
    """Safely import a package or return a mock if not available."""
    with optional_import(package_name) as module:
        if module is not None:
            return module

        if mock_class is not None:
            logger.warning(f"Using mock for {package_name}")
            return mock_class()

        # Return a generic mock object
        class MockModule:
            def __getattr__(self, name):
                def mock_method(*args, **kwargs):
                    raise OptionalDependencyError(
                        f"Method {name} requires {package_name} which is not installed"
                    )

                return mock_method

        return MockModule()
