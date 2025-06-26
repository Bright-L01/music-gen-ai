"""
Transformer architecture module for MusicGen.
"""
from .config import TransformerConfig, MusicGenConfig, EnCodecConfig, T5Config, ConditioningConfig
from .model import MusicGenTransformer, MultiHeadAttention, TransformerLayer

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