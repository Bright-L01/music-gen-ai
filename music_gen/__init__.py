"""Music Generation AI Package"""

__version__ = "0.1.0"

# Core imports
from .utils.exceptions import MusicGenException
from .utils.logging import get_logger, setup_logging

# Optional imports with graceful fallback
try:
    from .generation.generator import MusicGenerator
    from .models.musicgen import MusicGenModel
except ImportError as e:
    import warnings
    warnings.warn(f"Optional dependencies not installed: {e}")
    MusicGenModel = None
    MusicGenerator = None

__all__ = [
    "__version__",
    "MusicGenException",
    "setup_logging",
    "get_logger",
    "MusicGenModel",
    "MusicGenerator",
]
