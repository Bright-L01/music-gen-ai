"""
Model manager for Music Gen AI.

Handles model loading, caching, and lifecycle management.
"""

import gc
import logging
from pathlib import Path
from typing import Dict, Optional, Any

import torch

from ..optimization.fast_generator import FastMusicGenerator
from ..optimization.fast_generator import FastMusicGenerator

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Singleton model manager for handling model lifecycle.
    """

    _instance = None
    _models: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._models = {}
        self._default_device = "cuda" if torch.cuda.is_available() else "cpu"
        self._cache_dir = Path.home() / ".cache" / "musicgen"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Model manager initialized with device: {self._default_device}")

    def get_model(
        self,
        model_name: str = "facebook/musicgen-small",
        model_type: str = "optimized",
        device: Optional[str] = None,
        **kwargs,
    ) -> Any:
        """
        Get or load a model.

        Args:
            model_name: Name of the model to load
            model_type: Type of model ("optimized" or "multi_instrument")
            device: Device to load model on
            **kwargs: Additional model configuration

        Returns:
            Loaded model instance
        """
        if device is None:
            device = self._default_device

        # Check if model is already loaded
        model_key = f"{model_name}_{model_type}_{device}"
        if model_key in self._models:
            logger.info(f"Using cached model: {model_key}")
            return self._models[model_key]

        # Load model based on type
        logger.info(f"Loading model: {model_key}")

        if model_type == "optimized":
            model = FastMusicGenerator(
                model_name=model_name, device=device, cache_dir=str(self._cache_dir), **kwargs
            )
        elif model_type == "multi_instrument":
            model = FastMusicGenerator(
                model_name=model_name, device=device, cache_dir=str(self._cache_dir), **kwargs
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Cache model
        self._models[model_key] = model
        logger.info(f"Model loaded and cached: {model_key}")

        return model

    def list_loaded_models(self) -> Dict[str, Dict[str, Any]]:
        """
        List all currently loaded models.

        Returns:
            Dictionary of loaded models with their info
        """
        models_info = {}

        for key, model in self._models.items():
            parts = key.split("_")
            model_name = parts[0]
            model_type = parts[1] if len(parts) > 1 else "unknown"
            device = parts[2] if len(parts) > 2 else "unknown"

            models_info[model_name] = {
                "type": model_type,
                "device": device,
                "loaded": True,
                "model_class": model.__class__.__name__,
            }

        return models_info

    def has_loaded_models(self) -> bool:
        """
        Check if any models are loaded.

        Returns:
            True if models are loaded
        """
        return len(self._models) > 0

    def unload_model(self, model_name: str) -> bool:
        """
        Unload a specific model from memory.

        Args:
            model_name: Name of the model to unload

        Returns:
            True if model was unloaded
        """
        unloaded = False

        # Find and remove matching models
        keys_to_remove = [key for key in self._models.keys() if key.startswith(model_name)]

        for key in keys_to_remove:
            del self._models[key]
            unloaded = True
            logger.info(f"Unloaded model: {key}")

        # Force garbage collection
        if unloaded:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return unloaded

    def clear_cache(self):
        """
        Clear all cached models.
        """
        self._models.clear()
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Model cache cleared")

    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about the model cache.

        Returns:
            Cache information
        """
        cache_info = {
            "loaded_models": len(self._models),
            "models": list(self._models.keys()),
            "cache_dir": str(self._cache_dir),
            "cache_size_mb": self._get_cache_size() / (1024 * 1024),
        }

        if torch.cuda.is_available():
            cache_info["gpu_memory"] = {
                "allocated_gb": torch.cuda.memory_allocated() / (1024**3),
                "cached_gb": torch.cuda.memory_reserved() / (1024**3),
                "max_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3),
            }

        return cache_info

    def _get_cache_size(self) -> int:
        """
        Get total size of cache directory.

        Returns:
            Size in bytes
        """
        total_size = 0

        if self._cache_dir.exists():
            for file in self._cache_dir.rglob("*"):
                if file.is_file():
                    total_size += file.stat().st_size

        return total_size

    @property
    def device(self) -> str:
        """Get default device."""
        return self._default_device
