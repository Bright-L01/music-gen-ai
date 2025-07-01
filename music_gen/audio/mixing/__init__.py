"""Professional audio mixing engine for multi-track music."""

from .automation import AutomationLane, AutomationPoint
from .effects import EQ, Chorus, Compressor, Delay, Distortion, EffectChain, Gate, Limiter, Reverb
from .mastering import MasteringChain
from .mixer import MixingConfig, MixingEngine, TrackConfig

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
