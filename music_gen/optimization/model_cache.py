"""
Model caching system for MusicGen to avoid reloading models.
"""

import logging
import threading
import time
from typing import Dict

import torch

logger = logging.getLogger(__name__)


class ModelCache:
    """
    Singleton model cache to store loaded MusicGen models and avoid reloading.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._models: Dict[str, object] = {}
            self._load_times: Dict[str, float] = {}
            self._access_counts: Dict[str, int] = {}
            self._max_models = 3  # Limit memory usage
            self._initialized = True
            logger.info("ModelCache initialized")

    def get_model(self, model_name: str, device: str = "cpu", **kwargs):
        """
        Get cached model or load if not cached.

        Args:
            model_name: Name of the model (e.g., "facebook/musicgen-small")
            device: Device to load model on
            **kwargs: Additional arguments for model loading

        Returns:
            Loaded model instance
        """
        cache_key = f"{model_name}_{device}"

        # Check if model is cached
        if cache_key in self._models:
            self._access_counts[cache_key] += 1
            logger.info(
                f"✓ Using cached model: {cache_key} (accessed {self._access_counts[cache_key]} times)"
            )
            return self._models[cache_key]

        # Need to load model
        logger.info(f"Loading model: {cache_key}")
        start_time = time.time()

        # Import here to avoid circular imports
        from ..inference.real_multi_instrument import RealMultiInstrumentGenerator

        model = RealMultiInstrumentGenerator(model_name=model_name, device=device, **kwargs)

        load_time = time.time() - start_time

        # Cache management - remove oldest if at limit
        if len(self._models) >= self._max_models:
            self._evict_oldest()

        # Cache the model
        self._models[cache_key] = model
        self._load_times[cache_key] = load_time
        self._access_counts[cache_key] = 1

        logger.info(f"✓ Model loaded and cached: {cache_key} (took {load_time:.2f}s)")
        return model

    def _evict_oldest(self):
        """Remove the least recently used model to free memory."""
        if not self._models:
            return

        # Find model with lowest access count (LRU approximation)
        oldest_key = min(self._access_counts, key=self._access_counts.get)

        logger.info(f"Evicting cached model: {oldest_key}")

        # Cleanup
        model = self._models.pop(oldest_key, None)
        if model and hasattr(model, "model"):
            # Clear GPU memory if using CUDA
            if hasattr(model.model, "cpu"):
                model.model.cpu()
            del model.model

        self._load_times.pop(oldest_key, None)
        self._access_counts.pop(oldest_key, None)

        # Force garbage collection
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def warmup(self, model_name: str = "facebook/musicgen-small", device: str = None):
        """
        Warmup the cache by preloading a model.

        Args:
            model_name: Model to preload
            device: Device to load on (auto-detect if None)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Warming up model cache: {model_name} on {device}")
        start_time = time.time()

        model = self.get_model(model_name, device)
        warmup_time = time.time() - start_time

        logger.info(f"✓ Model warmup complete in {warmup_time:.2f}s")
        return model

    def clear(self):
        """Clear all cached models."""
        logger.info("Clearing model cache")

        for key, model in self._models.items():
            if model and hasattr(model, "model"):
                if hasattr(model.model, "cpu"):
                    model.model.cpu()
                del model.model

        self._models.clear()
        self._load_times.clear()
        self._access_counts.clear()

        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total_accesses = sum(self._access_counts.values())
        avg_load_time = (
            sum(self._load_times.values()) / len(self._load_times) if self._load_times else 0
        )

        return {
            "cached_models": len(self._models),
            "total_accesses": total_accesses,
            "average_load_time": avg_load_time,
            "cache_keys": list(self._models.keys()),
            "access_counts": dict(self._access_counts),
        }


# Global cache instance
model_cache = ModelCache()


def get_cached_model(model_name: str = "facebook/musicgen-small", device: str = None, **kwargs):
    """
    Convenience function to get a cached model.

    Args:
        model_name: Model to load
        device: Device to use (auto-detect if None)
        **kwargs: Additional model loading arguments

    Returns:
        Cached model instance
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    return model_cache.get_model(model_name, device, **kwargs)


def warmup_cache(model_name: str = "facebook/musicgen-small", device: str = None):
    """
    Warmup the model cache by preloading a model.

    Args:
        model_name: Model to preload
        device: Device to use (auto-detect if None)
    """
    return model_cache.warmup(model_name, device)


def clear_cache():
    """Clear the model cache."""
    model_cache.clear()


def get_cache_stats():
    """Get model cache statistics."""
    return model_cache.get_stats()
