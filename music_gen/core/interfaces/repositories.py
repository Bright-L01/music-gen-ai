"""
Repository interfaces for data access layer.

These interfaces define contracts for data access, allowing for
different implementations (filesystem, database, cloud storage, etc.).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch


class ModelRepository(ABC):
    """Interface for model storage and retrieval."""

    @abstractmethod
    async def save_model(
        self, model_id: str, model_state: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save a model state to storage.

        Args:
            model_id: Unique identifier for the model
            model_state: Model state dictionary
            metadata: Optional metadata about the model
        """

    @abstractmethod
    async def load_model(self, model_id: str) -> Dict[str, Any]:
        """Load a model state from storage.

        Args:
            model_id: Unique identifier for the model

        Returns:
            Model state dictionary

        Raises:
            ModelNotFoundError: If model doesn't exist
        """

    @abstractmethod
    async def exists(self, model_id: str) -> bool:
        """Check if a model exists in storage."""

    @abstractmethod
    async def delete_model(self, model_id: str) -> None:
        """Delete a model from storage."""

    @abstractmethod
    async def list_models(self) -> List[str]:
        """List all available model IDs."""

    @abstractmethod
    async def get_metadata(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model metadata if available."""


class TaskRepository(ABC):
    """Interface for task storage and retrieval."""

    @abstractmethod
    async def create_task(self, task_id: str, task_data: Dict[str, Any]) -> None:
        """Create a new task.

        Args:
            task_id: Unique identifier for the task
            task_data: Task data including status, parameters, etc.
        """

    @abstractmethod
    async def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task by ID.

        Returns:
            Task data if found, None otherwise
        """

    @abstractmethod
    async def update_task(self, task_id: str, updates: Dict[str, Any]) -> None:
        """Update task data.

        Args:
            task_id: Task identifier
            updates: Fields to update

        Raises:
            TaskNotFoundError: If task doesn't exist
        """

    @abstractmethod
    async def delete_task(self, task_id: str) -> None:
        """Delete a task."""

    @abstractmethod
    async def list_tasks(
        self, status: Optional[str] = None, limit: int = 100, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """List tasks with optional filtering."""


class MetadataRepository(ABC):
    """Interface for metadata storage and retrieval."""

    @abstractmethod
    async def save_metadata(self, dataset_id: str, metadata: Dict[str, Any]) -> None:
        """Save dataset metadata."""

    @abstractmethod
    async def load_metadata(self, dataset_id: str) -> Dict[str, Any]:
        """Load dataset metadata.

        Raises:
            MetadataNotFoundError: If metadata doesn't exist
        """

    @abstractmethod
    async def update_metadata(self, dataset_id: str, updates: Dict[str, Any]) -> None:
        """Update existing metadata."""

    @abstractmethod
    async def search_metadata(
        self, query: Dict[str, Any], limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Search metadata by criteria."""

    @abstractmethod
    async def delete_metadata(self, dataset_id: str) -> None:
        """Delete metadata."""


class AudioRepository(ABC):
    """Interface for audio file storage and retrieval."""

    @abstractmethod
    async def save_audio(
        self, audio_id: str, audio_data: torch.Tensor, sample_rate: int, format: str = "wav"
    ) -> str:
        """Save audio data.

        Args:
            audio_id: Unique identifier for the audio
            audio_data: Audio tensor data
            sample_rate: Sample rate of the audio
            format: Audio format (wav, mp3, etc.)

        Returns:
            Path or URL to the saved audio
        """

    @abstractmethod
    async def load_audio(self, audio_id: str) -> tuple[torch.Tensor, int]:
        """Load audio data.

        Returns:
            Tuple of (audio_data, sample_rate)

        Raises:
            AudioNotFoundError: If audio doesn't exist
        """

    @abstractmethod
    async def exists(self, audio_id: str) -> bool:
        """Check if audio exists."""

    @abstractmethod
    async def delete_audio(self, audio_id: str) -> None:
        """Delete audio file."""

    @abstractmethod
    async def get_audio_url(self, audio_id: str) -> str:
        """Get URL for audio file access."""

    @abstractmethod
    async def list_audio(self, prefix: Optional[str] = None, limit: int = 100) -> List[str]:
        """List available audio IDs."""
