"""Audio source separation module for track isolation."""

from .base import BaseSeparator, SeparationResult
from .demucs_separator import DemucsSeparator
from .hybrid_separator import HybridSeparator
from .spleeter_separator import SpleeterSeparator

__all__ = [
    "BaseSeparator",
    "SeparationResult",
    "DemucsSeparator",
    "SpleeterSeparator",
    "HybridSeparator",
]
