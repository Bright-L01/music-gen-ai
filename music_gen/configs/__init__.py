"""
Configuration management for MusicGen using Hydra.
"""
from .config import (
    MusicGenTrainingConfig,
    DataConfig, 
    ModelConfig,
    TrainingConfig,
    InferenceConfig,
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