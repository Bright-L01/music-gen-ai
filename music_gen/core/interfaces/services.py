"""
Service interfaces for business logic layer.

These interfaces define contracts for business logic services,
ensuring clean separation between presentation, business logic, and data access.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch


@dataclass
class GenerationRequest:
    """Request for music generation."""

    prompt: str
    duration: float = 30.0
    temperature: float = 1.0
    top_k: Optional[int] = 50
    top_p: Optional[float] = 0.9
    seed: Optional[int] = None
    conditioning: Optional[Dict[str, Any]] = None


@dataclass
class GenerationResult:
    """Result of music generation."""

    audio: torch.Tensor
    sample_rate: int
    duration: float
    metadata: Dict[str, Any]


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 10
    warmup_steps: int = 1000
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = True
    checkpoint_interval: int = 1000


class ModelService(ABC):
    """Interface for model management service."""

    @abstractmethod
    async def load_model(self, model_id: str) -> Any:
        """Load a model by ID.

        Args:
            model_id: Model identifier (e.g., "facebook/musicgen-small")

        Returns:
            Loaded model instance

        Raises:
            ModelNotFoundError: If model doesn't exist
            ModelLoadError: If model fails to load
        """

    @abstractmethod
    async def save_model(
        self, model: Any, model_id: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save a model.

        Args:
            model: Model instance to save
            model_id: Identifier for the model
            metadata: Optional metadata about the model
        """

    @abstractmethod
    async def list_available_models(self) -> List[Dict[str, Any]]:
        """List all available models with metadata."""

    @abstractmethod
    async def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get detailed information about a model."""

    @abstractmethod
    async def delete_model(self, model_id: str) -> None:
        """Delete a model from storage."""


class GenerationService(ABC):
    """Interface for music generation service."""

    @abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate music from a text prompt.

        Args:
            request: Generation request parameters

        Returns:
            Generated audio with metadata

        Raises:
            GenerationError: If generation fails
        """

    @abstractmethod
    async def generate_with_conditioning(
        self, request: GenerationRequest, conditioning: Dict[str, Any]
    ) -> GenerationResult:
        """Generate music with specific conditioning."""

    @abstractmethod
    async def generate_continuation(
        self, audio_context: torch.Tensor, duration: float, **kwargs
    ) -> GenerationResult:
        """Generate continuation of existing audio."""

    @abstractmethod
    async def get_supported_models(self) -> List[str]:
        """Get list of supported model IDs for generation."""


class AudioProcessingService(ABC):
    """Interface for audio processing service."""

    @abstractmethod
    async def process_audio(
        self, audio: torch.Tensor, sample_rate: int, operations: List[Dict[str, Any]]
    ) -> torch.Tensor:
        """Process audio with a chain of operations.

        Args:
            audio: Input audio tensor
            sample_rate: Sample rate of the audio
            operations: List of operations to apply

        Returns:
            Processed audio tensor
        """

    @abstractmethod
    async def normalize_audio(self, audio: torch.Tensor, method: str = "peak") -> torch.Tensor:
        """Normalize audio using specified method."""

    @abstractmethod
    async def resample_audio(
        self, audio: torch.Tensor, orig_sr: int, target_sr: int
    ) -> torch.Tensor:
        """Resample audio to target sample rate."""

    @abstractmethod
    async def mix_tracks(
        self, tracks: List[torch.Tensor], weights: Optional[List[float]] = None
    ) -> torch.Tensor:
        """Mix multiple audio tracks."""

    @abstractmethod
    async def apply_effects(
        self, audio: torch.Tensor, effects: List[Dict[str, Any]]
    ) -> torch.Tensor:
        """Apply audio effects chain."""


class TrainingService(ABC):
    """Interface for model training service."""

    @abstractmethod
    async def train_model(
        self, model_id: str, dataset_id: str, config: TrainingConfig
    ) -> Dict[str, Any]:
        """Train a model on a dataset.

        Args:
            model_id: Model to train
            dataset_id: Dataset to use
            config: Training configuration

        Returns:
            Training results and metrics
        """

    @abstractmethod
    async def fine_tune_model(
        self, base_model_id: str, dataset_id: str, config: TrainingConfig
    ) -> str:
        """Fine-tune a pre-trained model.

        Returns:
            ID of the fine-tuned model
        """

    @abstractmethod
    async def get_training_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a training job."""

    @abstractmethod
    async def stop_training(self, job_id: str) -> None:
        """Stop an ongoing training job."""

    @abstractmethod
    async def list_training_jobs(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List training jobs with optional status filter."""
