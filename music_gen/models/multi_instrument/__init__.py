"""Multi-instrument generation module for MusicGen.

This module provides:
- Instrument-aware transformer models
- Multi-track generation capabilities
- Instrument conditioning systems
- Track separation and mixing
"""

from .config import MultiInstrumentConfig, InstrumentConfig
from .model import MultiInstrumentMusicGen
from .conditioning import InstrumentConditioner
from .generator import MultiTrackGenerator

__all__ = [
    "MultiInstrumentConfig",
    "InstrumentConfig", 
    "MultiInstrumentMusicGen",
    "InstrumentConditioner",
    "MultiTrackGenerator",
]