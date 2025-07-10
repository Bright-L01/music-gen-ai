"""
Model repository implementations.

Provides concrete implementations for model storage and retrieval.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from music_gen.core.exceptions import ModelNotFoundError, ModelSaveError
from music_gen.core.interfaces.repositories import ModelRepository

logger = logging.getLogger(__name__)


class FileSystemModelRepository(ModelRepository):
    """File system based model repository."""

    def __init__(self, base_path: Path):
        """Initialize repository with base path.

        Args:
            base_path: Base directory for model storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_model_path(self, model_id: str) -> Path:
        """Get path for a model."""
        # Replace special characters in model_id
        safe_id = model_id.replace("/", "_").replace(":", "_")
        return self.base_path / safe_id

    async def save_model(
        self, model_id: str, model_state: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save model to filesystem."""
        try:
            model_path = self._get_model_path(model_id)
            model_path.mkdir(parents=True, exist_ok=True)

            # Save model state
            model_file = model_path / "pytorch_model.bin"
            torch.save(model_state, model_file)

            # Save metadata
            if metadata:
                metadata_file = model_path / "metadata.json"
                with open(metadata_file, "w") as f:
                    json.dump(metadata, f, indent=2)

            logger.info(f"Model saved successfully: {model_id}")

        except Exception as e:
            logger.error(f"Failed to save model {model_id}: {e}")
            raise ModelSaveError(f"Failed to save model: {e}")

    async def load_model(self, model_id: str) -> Dict[str, Any]:
        """Load model from filesystem."""
        model_path = self._get_model_path(model_id)
        model_file = model_path / "pytorch_model.bin"

        if not model_file.exists():
            raise ModelNotFoundError(f"Model not found: {model_id}")

        try:
            model_state = torch.load(model_file, map_location="cpu")
            logger.info(f"Model loaded successfully: {model_id}")
            return model_state
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            raise ModelNotFoundError(f"Failed to load model: {e}")

    async def exists(self, model_id: str) -> bool:
        """Check if model exists."""
        model_path = self._get_model_path(model_id)
        model_file = model_path / "pytorch_model.bin"
        return model_file.exists()

    async def delete_model(self, model_id: str) -> None:
        """Delete model from filesystem."""
        model_path = self._get_model_path(model_id)

        if model_path.exists():
            shutil.rmtree(model_path)
            logger.info(f"Model deleted: {model_id}")
        else:
            logger.warning(f"Model not found for deletion: {model_id}")

    async def list_models(self) -> List[str]:
        """List all available models."""
        models = []

        for path in self.base_path.iterdir():
            if path.is_dir() and (path / "pytorch_model.bin").exists():
                # Convert back from safe ID to model ID
                model_id = path.name.replace("_", "/", 1)
                models.append(model_id)

        return sorted(models)

    async def get_metadata(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model metadata."""
        model_path = self._get_model_path(model_id)
        metadata_file = model_path / "metadata.json"

        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                return json.load(f)

        return None
