"""
Configuration management for MusicGen using Hydra.
"""

from .config import (
    DataConfig,
    InferenceConfig,
    ModelConfig,
    MusicGenTrainingConfig,
    TrainingConfig,
    register_configs,
)

__all__ = [
    "MusicGenTrainingConfig",
    "DataConfig",
    "ModelConfig",
    "TrainingConfig",
    "InferenceConfig",
    "register_configs",
]
