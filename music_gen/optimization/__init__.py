"""
Performance optimization modules for MusicGen.
"""

from .model_cache import clear_cache, get_cache_stats, get_cached_model, model_cache, warmup_cache

__all__ = ["get_cached_model", "warmup_cache", "clear_cache", "get_cache_stats", "model_cache"]
