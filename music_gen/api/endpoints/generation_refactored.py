"""
Generation endpoints for Music Gen AI API - Refactored with DI.
"""

import asyncio
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import scipy.io.wavfile
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field

from music_gen.core.container import get_container
from music_gen.core.interfaces.repositories import AudioRepository, TaskRepository
from music_gen.core.interfaces.services import GenerationRequest as ServiceGenerationRequest
from music_gen.core.interfaces.services import (
    GenerationService,
)

router = APIRouter()

# Configuration
TEMP_DIR = Path("/tmp/musicgen")
TEMP_DIR.mkdir(exist_ok=True)
MAX_DURATION = 60.0
DEFAULT_DURATION = 10.0


# Pydantic models
class GenerationRequest(BaseModel):
    """Request model for music generation."""

    prompt: str = Field(..., description="Text description of the music to generate")
    duration: float = Field(
        DEFAULT_DURATION, ge=1.0, le=MAX_DURATION, description="Duration in seconds"
    )
    temperature: float = Field(1.0, ge=0.1, le=2.0, description="Sampling temperature")
    top_k: int = Field(50, ge=1, le=100, description="Top-k sampling parameter")
    top_p: float = Field(0.9, ge=0.1, le=1.0, description="Top-p (nucleus) sampling parameter")
    do_sample: bool = Field(True, description="Whether to use sampling")
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0, description="Repetition penalty")
    num_beams: int = Field(
        1, ge=1, le=8, description="Number of beams for beam search (1 = greedy/sampling)"
    )
    guidance_scale: float = Field(3.0, ge=1.0, le=5.0, description="Guidance scale")
    genre: Optional[str] = Field(None, description="Musical genre")
    mood: Optional[str] = Field(None, description="Musical mood")
    tempo: Optional[int] = Field(None, ge=60, le=200, description="Tempo in BPM")
    instruments: Optional[List[str]] = Field(None, description="Preferred instruments")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")


class GenerationResponse(BaseModel):
    """Response model for music generation."""

    task_id: str = Field(..., description="Unique task identifier")
    status: str = Field(..., description="Task status")
    audio_url: Optional[str] = Field(None, description="URL to download generated audio")
    duration: Optional[float] = Field(None, description="Actual audio duration")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Generation metadata")
    error: Optional[str] = Field(None, description="Error message if failed")


class BatchGenerationRequest(BaseModel):
    """Request model for batch generation."""

    requests: List[GenerationRequest] = Field(
        ..., description="List of generation requests", max_items=5
    )


# Dependency injection functions
def get_generation_service() -> GenerationService:
    """Get generation service from DI container."""
    container = get_container()
    return container.get(GenerationService)


def get_audio_repository() -> AudioRepository:
    """Get audio repository from DI container."""
    container = get_container()
    return container.get(AudioRepository)


def get_task_repository() -> TaskRepository:
    """Get task repository from DI container."""
    container = get_container()
    return container.get(TaskRepository)


@router.post("/", response_model=GenerationResponse)
async def generate_music(
    request: GenerationRequest,
    background_tasks: BackgroundTasks,
    generation_service: GenerationService = Depends(get_generation_service),
    task_repository: TaskRepository = Depends(get_task_repository),
):
    """Generate music from text prompt."""

    # Create task
    task_id = str(uuid.uuid4())
    await task_repository.create_task(
        task_id,
        {
            "status": "pending",
            "request": request.dict(),
        },
    )

    # Start background generation
    background_tasks.add_task(
        generate_music_task,
        task_id,
        request,
        generation_service,
        task_repository,
    )

    return GenerationResponse(
        task_id=task_id,
        status="pending",
    )


@router.post("/batch")
async def generate_music_batch(
    batch_request: BatchGenerationRequest,
    background_tasks: BackgroundTasks,
    generation_service: GenerationService = Depends(get_generation_service),
    task_repository: TaskRepository = Depends(get_task_repository),
):
    """Generate multiple music clips concurrently."""

    # Create batch task
    batch_id = str(uuid.uuid4())
    task_ids = []

    for i, req in enumerate(batch_request.requests):
        task_id = f"{batch_id}_{i}"
        await task_repository.create_task(
            task_id,
            {
                "status": "pending",
                "request": req.dict(),
                "batch_id": batch_id,
                "batch_index": i,
            },
        )
        task_ids.append(task_id)

    # Start batch generation
    background_tasks.add_task(
        generate_music_batch_task,
        batch_id,
        batch_request.requests,
        generation_service,
        task_repository,
    )

    return {
        "batch_id": batch_id,
        "task_ids": task_ids,
        "status": "pending",
        "total_requests": len(batch_request.requests),
    }


@router.get("/{task_id}", response_model=GenerationResponse)
async def get_generation_status(
    task_id: str,
    task_repository: TaskRepository = Depends(get_task_repository),
):
    """Get the status of a generation task."""

    task = await task_repository.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    response = GenerationResponse(
        task_id=task_id,
        status=task["status"],
    )

    if task["status"] == "completed":
        response.audio_url = f"/download/{task_id}"
        response.duration = task.get("duration")
        response.metadata = task.get("metadata")
    elif task["status"] == "failed":
        response.error = task.get("error")

    return response


@router.get("/batch/{batch_id}")
async def get_batch_status(
    batch_id: str,
    task_repository: TaskRepository = Depends(get_task_repository),
):
    """Get status of a batch generation."""

    # Get all tasks for this batch
    all_tasks = await task_repository.list_tasks()
    batch_tasks = {task["id"]: task for task in all_tasks if task.get("batch_id") == batch_id}

    if not batch_tasks:
        raise HTTPException(status_code=404, detail="Batch not found")

    completed = sum(1 for task in batch_tasks.values() if task["status"] == "completed")
    failed = sum(1 for task in batch_tasks.values() if task["status"] == "failed")
    pending = len(batch_tasks) - completed - failed

    overall_status = (
        "completed"
        if completed == len(batch_tasks)
        else "processing"
        if completed + failed < len(batch_tasks)
        else "failed"
    )

    return {
        "batch_id": batch_id,
        "status": overall_status,
        "total": len(batch_tasks),
        "completed": completed,
        "failed": failed,
        "pending": pending,
        "tasks": {
            k: {
                "status": v["status"],
                "audio_url": f"/download/{k}" if v["status"] == "completed" else None,
            }
            for k, v in batch_tasks.items()
        },
    }


async def generate_music_task(
    task_id: str,
    request: GenerationRequest,
    generation_service: GenerationService,
    task_repository: TaskRepository,
):
    """Background task for music generation."""

    try:
        await task_repository.update_task(task_id, {"status": "processing"})

        # Convert to service request
        service_request = ServiceGenerationRequest(
            prompt=request.prompt,
            duration=request.duration,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            seed=request.seed,
            conditioning={
                "genre": request.genre,
                "mood": request.mood,
                "tempo": request.tempo,
                "instruments": request.instruments,
            },
        )

        # Generate audio
        result = await generation_service.generate(service_request)

        # Save audio file
        audio_path = TEMP_DIR / f"{task_id}.wav"
        scipy.io.wavfile.write(
            str(audio_path),
            rate=result.sample_rate,
            data=(result.audio.cpu().numpy() * 32767).astype(np.int16),
        )

        # Update task status
        await task_repository.update_task(
            task_id,
            {
                "status": "completed",
                "audio_path": str(audio_path),
                "duration": result.duration,
                "metadata": result.metadata,
            },
        )

    except Exception as e:
        await task_repository.update_task(
            task_id,
            {
                "status": "failed",
                "error": str(e),
            },
        )


async def generate_music_batch_task(
    batch_id: str,
    requests: List[GenerationRequest],
    generation_service: GenerationService,
    task_repository: TaskRepository,
):
    """Background task for batch music generation."""

    # Process each request concurrently
    tasks = []
    for i, req in enumerate(requests):
        task_id = f"{batch_id}_{i}"
        task = generate_music_task(task_id, req, generation_service, task_repository)
        tasks.append(task)

    # Run all tasks concurrently
    await asyncio.gather(*tasks, return_exceptions=True)
