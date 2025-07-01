"""
Transformer architecture module for MusicGen.
"""

from .config import ConditioningConfig, EnCodecConfig, MusicGenConfig, T5Config, TransformerConfig
from .model import MultiHeadAttention, MusicGenTransformer, TransformerLayer

__all__ = [
    "TransformerConfig",
    "MusicGenConfig",
    "EnCodecConfig",
    "T5Config",
    "ConditioningConfig",
    "MusicGenTransformer",
    "MultiHeadAttention",
    "TransformerLayer",
]
