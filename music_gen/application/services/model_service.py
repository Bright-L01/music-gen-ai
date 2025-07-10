"""
Model service implementation.

Handles model loading, caching, and management with proper error handling
and logging.
"""

import asyncio
import gc
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiofiles
import aiohttp
import torch

from music_gen.core.config import AppConfig
from music_gen.core.exceptions import InsufficientResourcesError, ModelLoadError
from music_gen.core.interfaces.repositories import ModelRepository
from music_gen.core.interfaces.services import ModelService
from music_gen.core.resource_manager import ResourceManager, resource_monitored
from music_gen.models.musicgen import MusicGenModel
from music_gen.utils.optional_imports import optional_import

logger = logging.getLogger(__name__)


# Model registry with download information
MODEL_REGISTRY = {
    "facebook/musicgen-small": {
        "description": "Small MusicGen model (300M parameters)",
        "parameters": "300M",
        "size_gb": 1.2,
        "huggingface_id": "facebook/musicgen-small",
        "files": {
            "pytorch_model.bin": "https://huggingface.co/facebook/musicgen-small/resolve/main/pytorch_model.bin",
            "config.json": "https://huggingface.co/facebook/musicgen-small/resolve/main/config.json",
        },
        "sha256": {
            "pytorch_model.bin": "abc123...",  # Would be actual checksums
            "config.json": "def456...",
        },
    },
    "facebook/musicgen-medium": {
        "description": "Medium MusicGen model (1.5B parameters)",
        "parameters": "1.5B",
        "size_gb": 6.0,
        "huggingface_id": "facebook/musicgen-medium",
        "files": {
            "pytorch_model.bin": "https://huggingface.co/facebook/musicgen-medium/resolve/main/pytorch_model.bin",
            "config.json": "https://huggingface.co/facebook/musicgen-medium/resolve/main/config.json",
        },
        "sha256": {
            "pytorch_model.bin": "ghi789...",
            "config.json": "jkl012...",
        },
    },
    "facebook/musicgen-large": {
        "description": "Large MusicGen model (3.3B parameters)",
        "parameters": "3.3B",
        "size_gb": 12.0,
        "huggingface_id": "facebook/musicgen-large",
        "files": {
            "pytorch_model.bin": "https://huggingface.co/facebook/musicgen-large/resolve/main/pytorch_model.bin",
            "config.json": "https://huggingface.co/facebook/musicgen-large/resolve/main/config.json",
        },
        "sha256": {
            "pytorch_model.bin": "mno345...",
            "config.json": "pqr678...",
        },
    },
}


class ModelServiceImpl(ModelService):
    """Implementation of model service with caching and management."""

    def __init__(self, model_repository: ModelRepository, config: AppConfig):
        """Initialize model service.

        Args:
            model_repository: Repository for model storage
            config: Application configuration
        """
        self._repository = model_repository
        self._config = config
        self._model_cache: Dict[str, Any] = {}
        self._cache_lock = asyncio.Lock()
        self._download_progress: Dict[str, Dict[str, Any]] = {}
        self._download_locks: Dict[str, asyncio.Lock] = {}

        # Create download directory
        self._download_dir = Path(config.model_cache_dir) / "downloads"
        self._download_dir.mkdir(parents=True, exist_ok=True)

        # Initialize resource manager
        self.resource_manager = ResourceManager(config)
        logger.info("Resource management system initialized")

    @resource_monitored
    async def load_model(self, model_id: str) -> Any:
        """Load a model with caching support and resource validation."""
        # Check cache first
        if model_id in self._model_cache:
            logger.info(f"Model loaded from cache: {model_id}")
            return self._model_cache[model_id]

        # Validate resources before loading
        logger.info(f"Validating resources for model: {model_id}")
        validation_result = self.resource_manager.validate_system_resources(model_id)

        if validation_result.get("warnings"):
            for warning in validation_result["warnings"]:
                logger.warning(f"Resource warning: {warning}")

        async with self._cache_lock:
            # Double-check after acquiring lock
            if model_id in self._model_cache:
                return self._model_cache[model_id]

            # Allocate resources for loading
            resource_allocation_id = f"model_load_{model_id}_{time.time()}"
            requirements = self.resource_manager.get_model_requirements(model_id)

            try:
                # Allocate resources
                self.resource_manager.allocate_resources(
                    resource_allocation_id,
                    cpu_memory_gb=requirements.cpu_memory_gb,
                    gpu_memory_gb=requirements.gpu_memory_gb,
                )

                # Check if it's a custom model in repository
                if await self._repository.exists(model_id):
                    logger.info(f"Loading custom model from repository: {model_id}")
                    model_state = await self._repository.load_model(model_id)
                    model = self._create_model_from_state(model_state)
                else:
                    # Load from Hugging Face or pre-trained models
                    logger.info(f"Loading pre-trained model: {model_id}")
                    model = await self._load_pretrained_model(model_id)

                # Manage cache size
                await self._manage_cache_size()

                # Add to cache and track for resource management
                self._model_cache[model_id] = model

                # Calculate model size for tracking
                model_size_gb = self._estimate_model_size(model)
                self.resource_manager.track_model_cache(model_id, model_size_gb)

                logger.info(f"Model loaded successfully: {model_id} ({model_size_gb:.2f}GB)")

                return model

            except InsufficientResourcesError:
                # Re-raise resource errors as-is
                raise
            except Exception as e:
                logger.error(f"Failed to load model {model_id}: {e}")
                raise ModelLoadError(f"Failed to load model: {e}")
            finally:
                # Always release the loading resources
                self.resource_manager.release_resources(resource_allocation_id)

    async def save_model(
        self, model: Any, model_id: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save a model to repository."""
        try:
            # Extract model state
            if hasattr(model, "state_dict"):
                model_state = {
                    "state_dict": model.state_dict(),
                    "config": model.config.__dict__ if hasattr(model, "config") else {},
                    "model_type": type(model).__name__,
                }
            else:
                raise ValueError("Model must have state_dict method")

            # Add metadata
            if metadata is None:
                metadata = {}

            metadata.update(
                {
                    "model_id": model_id,
                    "model_type": type(model).__name__,
                    "pytorch_version": torch.__version__,
                    "device": str(next(model.parameters()).device),
                }
            )

            # Save to repository
            await self._repository.save_model(model_id, model_state, metadata)

            # Update cache
            self._model_cache[model_id] = model

            logger.info(f"Model saved successfully: {model_id}")

        except Exception as e:
            logger.error(f"Failed to save model {model_id}: {e}")
            raise

    async def list_available_models(self) -> List[Dict[str, Any]]:
        """List all available models."""
        models = []

        # Get custom models from repository
        custom_models = await self._repository.list_models()
        for model_id in custom_models:
            metadata = await self._repository.get_metadata(model_id)
            models.append(
                {
                    "id": model_id,
                    "type": "custom",
                    "loaded": model_id in self._model_cache,
                    "metadata": metadata or {},
                }
            )

        # Add pre-trained models from registry
        for model_id, model_info in MODEL_REGISTRY.items():
            model_dir = self._download_dir / model_id.replace("/", "_")
            is_downloaded = await self._is_model_valid(model_dir, model_info)

            models.append(
                {
                    "id": model_id,
                    "type": "pretrained",
                    "loaded": model_id in self._model_cache,
                    "downloaded": is_downloaded,
                    "download_progress": self._download_progress.get(model_id),
                    "metadata": {
                        "source": "huggingface",
                        "description": model_info["description"],
                        "parameters": model_info["parameters"],
                        "size_gb": model_info["size_gb"],
                    },
                }
            )

        return models

    async def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get detailed information about a model."""
        info = {
            "id": model_id,
            "loaded": model_id in self._model_cache,
        }

        # Check if custom model
        if await self._repository.exists(model_id):
            metadata = await self._repository.get_metadata(model_id)
            info.update(
                {
                    "type": "custom",
                    "metadata": metadata or {},
                }
            )
        else:
            # Assume pre-trained model
            info.update(
                {
                    "type": "pretrained",
                    "metadata": {
                        "source": "huggingface",
                    },
                }
            )

        # Add model details if loaded
        if model_id in self._model_cache:
            model = self._model_cache[model_id]
            info["details"] = {
                "parameters": sum(p.numel() for p in model.parameters()),
                "device": str(next(model.parameters()).device),
                "dtype": str(next(model.parameters()).dtype),
            }

        return info

    async def delete_model(self, model_id: str) -> None:
        """Delete a model from storage."""
        # Remove from cache
        if model_id in self._model_cache:
            del self._model_cache[model_id]
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Delete from repository
        await self._repository.delete_model(model_id)

        logger.info(f"Model deleted: {model_id}")

    async def _load_pretrained_model(self, model_id: str) -> Any:
        """Load a pre-trained model with downloading and caching."""
        try:
            # Check if model is in registry
            if model_id not in MODEL_REGISTRY:
                # Try loading with transformers library as fallback
                return await self._load_with_transformers(model_id)

            # Get model metadata
            model_info = MODEL_REGISTRY[model_id]
            model_dir = self._download_dir / model_id.replace("/", "_")

            # Ensure model is downloaded
            await self._ensure_model_downloaded(model_id, model_info, model_dir)

            # Load model from downloaded files
            config_path = model_dir / "config.json"
            weights_path = model_dir / "pytorch_model.bin"

            # Load config
            with open(config_path, "r") as f:
                config_dict = json.load(f)

            # Create model instance
            from music_gen.models.transformer.config import MusicGenConfig

            config = MusicGenConfig(**config_dict)
            model = MusicGenModel(config)

            # Load weights
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict)

            # Move to configured device
            device = torch.device(self._config.model_device)
            model = model.to(device)

            logger.info(f"Successfully loaded pre-trained model: {model_id}")
            return model

        except Exception as e:
            logger.error(f"Failed to load pre-trained model {model_id}: {e}")
            raise ModelLoadError(f"Failed to load pre-trained model: {e}")

    async def _load_with_transformers(self, model_id: str) -> Any:
        """Fallback to load model using transformers library."""
        with optional_import("transformers") as transformers:
            if transformers is None:
                raise ModelLoadError(
                    f"Model {model_id} not found in registry and transformers library not available"
                )

            try:
                # Try to load with transformers (this is a simplified version)
                logger.info(f"Attempting to load {model_id} with transformers library")

                # For now, create a default model as transformers integration
                # would require more extensive changes
                from music_gen.models.transformer.config import MusicGenConfig

                config = MusicGenConfig()
                model = MusicGenModel(config)

                device = torch.device(self._config.model_device)
                model = model.to(device)

                logger.warning(
                    f"Created default model for {model_id} - full transformers integration needed"
                )
                return model

            except Exception as e:
                raise ModelLoadError(f"Failed to load {model_id} with transformers: {e}")

    async def _ensure_model_downloaded(
        self, model_id: str, model_info: Dict[str, Any], model_dir: Path
    ) -> None:
        """Ensure a model is downloaded and validated."""
        # Check if model is already downloaded and valid
        if await self._is_model_valid(model_dir, model_info):
            logger.info(f"Model {model_id} already cached and valid")
            return

        # Get or create download lock for this model
        if model_id not in self._download_locks:
            self._download_locks[model_id] = asyncio.Lock()

        async with self._download_locks[model_id]:
            # Double-check after acquiring lock
            if await self._is_model_valid(model_dir, model_info):
                return

            logger.info(f"Downloading model {model_id}...")

            # Create model directory
            model_dir.mkdir(parents=True, exist_ok=True)

            # Initialize progress tracking
            self._download_progress[model_id] = {
                "status": "downloading",
                "progress": 0.0,
                "downloaded_mb": 0,
                "total_mb": model_info["size_gb"] * 1024,
                "files_downloaded": 0,
                "total_files": len(model_info["files"]),
                "started_at": datetime.now(),
            }

            try:
                # Download each file
                async with aiohttp.ClientSession() as session:
                    for filename, url in model_info["files"].items():
                        await self._download_file(
                            session, url, model_dir / filename, model_id, filename
                        )

                        self._download_progress[model_id]["files_downloaded"] += 1

                # Validate downloaded files
                if await self._validate_model_files(model_dir, model_info):
                    self._download_progress[model_id]["status"] = "completed"
                    self._download_progress[model_id]["progress"] = 1.0
                    logger.info(f"Successfully downloaded and validated model {model_id}")
                else:
                    raise ModelLoadError(f"Model validation failed for {model_id}")

            except Exception as e:
                self._download_progress[model_id]["status"] = "failed"
                self._download_progress[model_id]["error"] = str(e)
                # Clean up partial download
                import shutil

                if model_dir.exists():
                    shutil.rmtree(model_dir)
                raise

    async def _download_file(
        self, session: aiohttp.ClientSession, url: str, filepath: Path, model_id: str, filename: str
    ) -> None:
        """Download a single file with progress tracking."""
        logger.info(f"Downloading {filename} from {url}")

        async with session.get(url) as response:
            if response.status != 200:
                raise ModelLoadError(f"Failed to download {filename}: HTTP {response.status}")

            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0

            async with aiofiles.open(filepath, "wb") as f:
                async for chunk in response.content.iter_chunked(8192):
                    await f.write(chunk)
                    downloaded += len(chunk)

                    # Update progress
                    if model_id in self._download_progress:
                        progress = self._download_progress[model_id]
                        progress["downloaded_mb"] += len(chunk) / (1024 * 1024)

                        # Calculate overall progress
                        file_progress = downloaded / total_size if total_size > 0 else 0
                        files_completed = progress["files_downloaded"]
                        total_files = progress["total_files"]
                        overall_progress = (files_completed + file_progress) / total_files
                        progress["progress"] = overall_progress

    async def _is_model_valid(self, model_dir: Path, model_info: Dict[str, Any]) -> bool:
        """Check if a downloaded model is valid."""
        if not model_dir.exists():
            return False

        # Check if all required files exist
        for filename in model_info["files"].keys():
            filepath = model_dir / filename
            if not filepath.exists():
                return False

        # TODO: Add checksum validation for security
        # This would verify SHA256 hashes from model_info["sha256"]

        return True

    async def _validate_model_files(self, model_dir: Path, model_info: Dict[str, Any]) -> bool:
        """Validate downloaded model files."""
        # Basic validation - check file existence and non-zero size
        for filename in model_info["files"].keys():
            filepath = model_dir / filename
            if not filepath.exists() or filepath.stat().st_size == 0:
                logger.error(f"Invalid model file: {filepath}")
                return False

        # TODO: Add SHA256 checksum validation
        # for filename, expected_hash in model_info["sha256"].items():
        #     if not await self._verify_checksum(model_dir / filename, expected_hash):
        #         return False

        return True

    async def get_download_progress(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get download progress for a model."""
        return self._download_progress.get(model_id)

    def _create_model_from_state(self, model_state: Dict[str, Any]) -> Any:
        """Create model instance from saved state."""
        try:
            # Get model type and config
            model_type = model_state.get("model_type", "MusicGenModel")
            config_dict = model_state.get("config", {})

            # Create model instance
            if model_type == "MusicGenModel":
                from music_gen.models.transformer.config import MusicGenConfig

                config = MusicGenConfig(**config_dict)
                model = MusicGenModel(config)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            # Load state dict
            model.load_state_dict(model_state["state_dict"])

            # Move to configured device
            device = torch.device(self._config.model_device)
            model = model.to(device)

            return model

        except Exception as e:
            logger.error(f"Failed to create model from state: {e}")
            raise ModelLoadError(f"Failed to create model from state: {e}")

    async def _manage_cache_size(self) -> None:
        """Manage model cache size by removing least recently used models."""
        if len(self._model_cache) >= self._config.model_cache_size:
            # Get current resource usage
            resource_report = self.resource_manager.get_resource_report()
            current_health = resource_report.get("health_status", "healthy")

            # More aggressive cleanup if resources are constrained
            models_to_remove = 1
            if current_health in ["warning", "critical"]:
                models_to_remove = max(2, len(self._model_cache) // 3)
                logger.warning(
                    f"Resource pressure detected, removing {models_to_remove} models from cache"
                )

            # Remove oldest models (simple LRU)
            # In production, use proper LRU cache with access tracking
            removed_models = []
            for _ in range(min(models_to_remove, len(self._model_cache) - 1)):
                if self._model_cache:
                    oldest_id = next(iter(self._model_cache))
                    del self._model_cache[oldest_id]
                    removed_models.append(oldest_id)

            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info(f"Removed {len(removed_models)} model(s) from cache: {removed_models}")

    def _estimate_model_size(self, model: Any) -> float:
        """Estimate model size in GB."""
        try:
            # Calculate parameter size
            total_params = 0
            param_size = 0

            if hasattr(model, "parameters"):
                for param in model.parameters():
                    total_params += param.numel()
                    # Assume float32 (4 bytes per parameter)
                    param_size += param.numel() * param.element_size()

            # Add some overhead for buffers and other data structures
            overhead_factor = 1.2
            total_size_bytes = param_size * overhead_factor

            # Convert to GB
            size_gb = total_size_bytes / (1024**3)

            logger.debug(f"Model has {total_params:,} parameters, estimated size: {size_gb:.2f}GB")
            return size_gb

        except Exception as e:
            logger.warning(f"Could not estimate model size: {e}")
            # Return conservative estimate
            return 2.0  # Default 2GB
