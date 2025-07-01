"""Multi-instrument generation module for MusicGen.

This module provides:
- Instrument-aware transformer models
- Multi-track generation capabilities
- Instrument conditioning systems
- Track separation and mixing
"""

from .conditioning import InstrumentConditioner
from .config import InstrumentConfig, MultiInstrumentConfig
from .generator import GenerationResult, MultiTrackGenerator, TrackGenerationConfig
from .model import MultiInstrumentMusicGen

__all__ = [
    "MultiInstrumentConfig",
    "InstrumentConfig",
    "MultiInstrumentMusicGen",
    "InstrumentConditioner",
    "MultiTrackGenerator",
    "TrackGenerationConfig",
    "GenerationResult",
]
