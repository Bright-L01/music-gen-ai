"""Professional audio mixing engine for multi-track music."""

from .mixer import MixingEngine, MixingConfig, TrackConfig
from .effects import (
    EffectChain,
    EQ,
    Compressor,
    Reverb,
    Delay,
    Chorus,
    Limiter,
    Gate,
    Distortion,
)
from .automation import AutomationLane, AutomationPoint
from .mastering import MasteringChain

__all__ = [
    "MixingEngine",
    "MixingConfig",
    "TrackConfig",
    "EffectChain",
    "EQ",
    "Compressor",
    "Reverb",
    "Delay",
    "Chorus",
    "Limiter",
    "Gate",
    "Distortion",
    "AutomationLane",
    "AutomationPoint",
    "MasteringChain",
]