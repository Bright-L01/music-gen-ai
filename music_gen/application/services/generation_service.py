"""
Generation service implementation.

Handles music generation orchestration with proper task management,
error handling, and monitoring.
"""

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, List

import torch

from music_gen.core.exceptions import GenerationError, InsufficientResourcesError
from music_gen.core.interfaces.repositories import TaskRepository
from music_gen.core.interfaces.services import (
    AudioProcessingService,
    GenerationRequest,
    GenerationResult,
    GenerationService,
    ModelService,
)
from music_gen.core.resource_manager import resource_monitored

logger = logging.getLogger(__name__)


class GenerationServiceImpl(GenerationService):
    """Implementation of generation service with full orchestration."""

    def __init__(
        self,
        model_service: ModelService,
        audio_service: AudioProcessingService,
        task_repository: TaskRepository,
    ):
        """Initialize generation service.

        Args:
            model_service: Service for model management
            audio_service: Service for audio processing
            task_repository: Repository for task tracking
        """
        self._model_service = model_service
        self._audio_service = audio_service
        self._task_repository = task_repository
        self._active_generations: Dict[str, asyncio.Task] = {}

    @resource_monitored
    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate music from a text prompt with resource monitoring."""
        # Create task ID
        task_id = str(uuid.uuid4())

        # Create task record
        task_data = {
            "id": task_id,
            "status": "pending",
            "request": {
                "prompt": request.prompt,
                "duration": request.duration,
                "temperature": request.temperature,
                "top_k": request.top_k,
                "top_p": request.top_p,
                "seed": request.seed,
            },
            "progress": 0.0,
        }

        await self._task_repository.create_task(task_id, task_data)

        try:
            # Validate resources before starting
            model_id = "facebook/musicgen-small"  # TODO: Make configurable
            if hasattr(self._model_service, "resource_manager"):
                resource_manager = self._model_service.resource_manager
                try:
                    validation_result = resource_manager.validate_system_resources(model_id)
                    if validation_result.get("warnings"):
                        for warning in validation_result["warnings"]:
                            logger.warning(f"Resource warning: {warning}")
                except InsufficientResourcesError as e:
                    logger.error(f"Insufficient resources: {e}")
                    await self._update_task_status(task_id, "failed", error=str(e))
                    raise GenerationError(f"Insufficient resources: {e}")

            # Update status
            await self._update_task_status(task_id, "loading_model", 0.1)

            # Load model with resource tracking
            model = await self._model_service.load_model(model_id)

            # Update status
            await self._update_task_status(task_id, "generating", 0.3)

            # Generate audio
            start_time = time.time()
            audio = await self._generate_audio(model, request)
            generation_time = time.time() - start_time

            # Update status
            await self._update_task_status(task_id, "post_processing", 0.8)

            # Post-process audio
            processed_audio = await self._audio_service.normalize_audio(audio, method="peak")

            # Create result
            result = GenerationResult(
                audio=processed_audio,
                sample_rate=model.config.sample_rate,
                duration=request.duration,
                metadata={
                    "task_id": task_id,
                    "prompt": request.prompt,
                    "generation_time": generation_time,
                    "model_id": "facebook/musicgen-small",
                    "parameters": {
                        "temperature": request.temperature,
                        "top_k": request.top_k,
                        "top_p": request.top_p,
                        "seed": request.seed,
                    },
                },
            )

            # Update task as completed
            await self._update_task_status(
                task_id,
                "completed",
                1.0,
                {
                    "result": {
                        "duration": result.duration,
                        "sample_rate": result.sample_rate,
                        "generation_time": generation_time,
                    },
                },
            )

            logger.info(f"Generation completed: {task_id}")
            return result

        except Exception as e:
            logger.error(f"Generation failed for task {task_id}: {e}")
            await self._update_task_status(task_id, "failed", error=str(e))
            raise GenerationError(f"Generation failed: {e}")

    async def generate_with_conditioning(
        self, request: GenerationRequest, conditioning: Dict[str, Any]
    ) -> GenerationResult:
        """Generate music with specific conditioning."""
        # Add conditioning to request
        request.conditioning = conditioning

        # Use standard generation with conditioning
        return await self.generate(request)

    async def generate_continuation(
        self, audio_context: torch.Tensor, duration: float, **kwargs
    ) -> GenerationResult:
        """Generate continuation of existing audio."""
        # Create task ID
        task_id = str(uuid.uuid4())

        try:
            # Load model
            model = await self._model_service.load_model("facebook/musicgen-small")

            # Encode context audio
            with torch.no_grad():
                # This is a simplified version - actual implementation would
                # properly encode the audio context
                context_tokens = model.encode_audio(audio_context)

            # Generate continuation
            generation_params = {
                "context_tokens": context_tokens,
                "max_length": int(duration * model.config.sample_rate / model.config.hop_length),
                **kwargs,
            }

            audio = await self._generate_audio(model, None, **generation_params)

            # Concatenate with context
            full_audio = torch.cat([audio_context, audio], dim=-1)

            # Create result
            result = GenerationResult(
                audio=full_audio,
                sample_rate=model.config.sample_rate,
                duration=full_audio.shape[-1] / model.config.sample_rate,
                metadata={
                    "task_id": task_id,
                    "type": "continuation",
                    "context_duration": audio_context.shape[-1] / model.config.sample_rate,
                    "generated_duration": duration,
                },
            )

            return result

        except Exception as e:
            logger.error(f"Continuation generation failed: {e}")
            raise GenerationError(f"Continuation generation failed: {e}")

    async def get_supported_models(self) -> List[str]:
        """Get list of supported model IDs."""
        models = await self._model_service.list_available_models()
        return [model["id"] for model in models]

    async def _generate_audio(
        self, model: Any, request: Optional[GenerationRequest], **kwargs
    ) -> torch.Tensor:
        """Generate audio using the model."""
        try:
            # Set random seed if provided
            if request and request.seed is not None:
                torch.manual_seed(request.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(request.seed)

            # Prepare generation parameters
            if request:
                generation_params = {
                    "texts": [request.prompt],
                    "max_length": int(
                        request.duration * model.config.sample_rate / model.config.hop_length
                    ),
                    "temperature": request.temperature,
                    "top_k": request.top_k,
                    "top_p": request.top_p,
                    "do_sample": True,
                }

                # Add conditioning if provided
                if request.conditioning:
                    generation_params["conditioning"] = request.conditioning
            else:
                generation_params = kwargs

            # Generate audio
            with torch.no_grad():
                audio = model.generate(**generation_params)

            # Remove batch dimension if present
            if audio.dim() == 3:
                audio = audio.squeeze(0)

            return audio

        except Exception as e:
            logger.error(f"Audio generation failed: {e}")
            raise

    async def _update_task_status(
        self,
        task_id: str,
        status: str,
        progress: float = None,
        additional_data: Dict[str, Any] = None,
        error: str = None,
    ) -> None:
        """Update task status in repository."""
        updates = {"status": status}

        if progress is not None:
            updates["progress"] = progress

        if error:
            updates["error"] = error

        if additional_data:
            updates.update(additional_data)

        await self._task_repository.update_task(task_id, updates)
