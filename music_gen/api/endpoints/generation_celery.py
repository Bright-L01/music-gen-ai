from datetime import datetime

"""
Generation endpoints integrated with Celery workers.

This module provides API endpoints that dispatch tasks to Celery workers
instead of using FastAPI background tasks.
"""

import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from music_gen.core.container import get_container
from music_gen.core.interfaces.repositories import TaskRepository
from music_gen.infrastructure.repositories.redis_task_repository_advanced import (
    TaskPriority,
    TaskStatus,
)
from music_gen.workers import celery_app, generate_batch_task, generate_music_task

router = APIRouter()

# Check if Celery workers are enabled
USE_CELERY_WORKERS = os.getenv("USE_CELERY_WORKERS", "false").lower() == "true"


# Request/Response models
class GenerationRequest(BaseModel):
    """Request model for music generation."""

    prompt: str = Field(..., description="Text description of the music to generate")
    duration: float = Field(10.0, ge=1.0, le=60.0, description="Duration in seconds")
    temperature: float = Field(1.0, ge=0.1, le=2.0, description="Sampling temperature")
    top_k: int = Field(50, ge=1, le=100, description="Top-k sampling parameter")
    top_p: float = Field(0.9, ge=0.1, le=1.0, description="Top-p (nucleus) sampling parameter")
    do_sample: bool = Field(True, description="Whether to use sampling")
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0, description="Repetition penalty")
    num_beams: int = Field(1, ge=1, le=8, description="Number of beams for beam search")
    guidance_scale: float = Field(3.0, ge=1.0, le=5.0, description="Guidance scale")
    genre: Optional[str] = Field(None, description="Musical genre")
    mood: Optional[str] = Field(None, description="Musical mood")
    tempo: Optional[int] = Field(None, ge=60, le=200, description="Tempo in BPM")
    instruments: Optional[List[str]] = Field(None, description="Preferred instruments")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    priority: Optional[int] = Field(None, description="Task priority (1-20)")
    ttl: Optional[int] = Field(None, description="Task TTL in seconds")


class GenerationResponse(BaseModel):
    """Response model for music generation."""

    task_id: str = Field(..., description="Unique task identifier")
    status: str = Field(..., description="Task status")
    queue_position: Optional[int] = Field(None, description="Position in queue")
    estimated_time: Optional[float] = Field(None, description="Estimated completion time")
    audio_url: Optional[str] = Field(None, description="URL to download generated audio")
    duration: Optional[float] = Field(None, description="Actual audio duration")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Generation metadata")
    error: Optional[str] = Field(None, description="Error message if failed")


class BatchGenerationRequest(BaseModel):
    """Request model for batch generation."""

    requests: List[GenerationRequest] = Field(
        ..., description="List of generation requests", max_items=10
    )
    priority: Optional[int] = Field(None, description="Batch priority")


# Dependency injection
def get_task_repository() -> TaskRepository:
    """Get task repository from DI container."""
    container = get_container()
    return container.get(TaskRepository)


@router.post("/", response_model=GenerationResponse)
async def generate_music(
    request: GenerationRequest,
    task_repository: TaskRepository = Depends(get_task_repository),
):
    """Generate music from text prompt using Celery workers."""

    # Create task ID
    task_id = str(uuid.uuid4())

    # Determine priority
    if request.priority is not None:
        priority = request.priority
    else:
        # Auto-assign priority based on duration
        if request.duration <= 10:
            priority = TaskPriority.HIGH.value
        elif request.duration <= 30:
            priority = TaskPriority.NORMAL.value
        else:
            priority = TaskPriority.LOW.value

    # Prepare task data
    task_data = {
        "request": request.dict(),
        "priority": priority,
        "ttl": request.ttl or 86400,  # Default 24 hours
    }

    # Create task in repository
    await task_repository.create_task(task_id, task_data)

    if USE_CELERY_WORKERS:
        # Dispatch to Celery
        routing_key = _get_routing_key(priority)
        queue = _get_queue_name(priority)

        result = generate_music_task.apply_async(
            args=[task_id, request.dict()],
            task_id=task_id,
            queue=queue,
            routing_key=routing_key,
            priority=priority,
        )

        # Get queue position estimate
        queue_position = await _estimate_queue_position(task_repository, priority)
        estimated_time = queue_position * 30  # Rough estimate: 30s per task

        return GenerationResponse(
            task_id=task_id,
            status="queued",
            queue_position=queue_position,
            estimated_time=estimated_time,
        )
    else:
        # Fallback to traditional background task
        pass

        # This would need to be injected properly
        raise HTTPException(
            status_code=503, detail="Celery workers not available, background tasks not configured"
        )


@router.post("/batch", response_model=Dict[str, Any])
async def generate_music_batch(
    batch_request: BatchGenerationRequest,
    task_repository: TaskRepository = Depends(get_task_repository),
):
    """Generate multiple music clips in batch using Celery workers."""

    # Create batch ID
    batch_id = str(uuid.uuid4())

    # Determine batch priority
    priority = batch_request.priority or TaskPriority.LOW.value

    if USE_CELERY_WORKERS:
        # Dispatch batch to Celery
        result = generate_batch_task.apply_async(
            args=[batch_id, [r.dict() for r in batch_request.requests]],
            task_id=batch_id,
            queue="batch",
            routing_key="batch.generation",
            priority=priority,
        )

        # Create individual task entries
        task_ids = []
        for i, req in enumerate(batch_request.requests):
            task_id = f"{batch_id}_{i}"
            task_ids.append(task_id)

        return {
            "batch_id": batch_id,
            "task_ids": task_ids,
            "status": "queued",
            "total_requests": len(batch_request.requests),
            "priority": priority,
        }
    else:
        raise HTTPException(status_code=503, detail="Batch generation requires Celery workers")


@router.get("/{task_id}", response_model=GenerationResponse)
async def get_generation_status(
    task_id: str,
    task_repository: TaskRepository = Depends(get_task_repository),
):
    """Get the status of a generation task."""

    # Get task from repository
    task = await task_repository.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    # Build response
    response = GenerationResponse(
        task_id=task_id,
        status=task["status"],
    )

    # Add details based on status
    if task["status"] == TaskStatus.COMPLETED.value:
        response.audio_url = f"/api/v1/generate/download/{task_id}"
        response.duration = task.get("duration")
        response.metadata = task.get("metadata")
    elif task["status"] == TaskStatus.FAILED.value:
        response.error = task.get("error")
    elif task["status"] in [TaskStatus.PENDING.value, TaskStatus.QUEUED.value]:
        # Get queue position
        priority = task.get("priority", TaskPriority.NORMAL.value)
        response.queue_position = await _estimate_queue_position(task_repository, priority)
        response.estimated_time = response.queue_position * 30

    return response


@router.get("/download/{task_id}")
async def download_audio(
    task_id: str,
    task_repository: TaskRepository = Depends(get_task_repository),
):
    """Download generated audio file."""

    # Get task from repository
    task = await task_repository.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task["status"] != TaskStatus.COMPLETED.value:
        raise HTTPException(status_code=400, detail="Generation not completed")

    # Get audio file path
    audio_path = task.get("audio_path")
    if not audio_path or not Path(audio_path).exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    # Return file
    return FileResponse(
        audio_path,
        media_type="audio/wav",
        filename=f"musicgen_{task_id}.wav",
    )


@router.delete("/{task_id}")
async def cancel_generation(
    task_id: str,
    task_repository: TaskRepository = Depends(get_task_repository),
):
    """Cancel a generation task."""

    # Get task from repository
    task = await task_repository.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    # Check if cancellable
    if task["status"] in [TaskStatus.COMPLETED.value, TaskStatus.FAILED.value]:
        raise HTTPException(status_code=400, detail=f"Cannot cancel task in {task['status']} state")

    if USE_CELERY_WORKERS:
        # Revoke Celery task
        celery_app.control.revoke(task_id, terminate=True)

    # Update task status
    await task_repository.update_task(
        task_id,
        {
            "status": TaskStatus.CANCELLED.value,
            "cancelled_at": datetime.utcnow().isoformat(),
        },
    )

    return {"message": "Task cancelled", "task_id": task_id}


@router.get("/queue/status")
async def get_queue_status(
    task_repository: TaskRepository = Depends(get_task_repository),
):
    """Get current queue status and wait times."""

    if not USE_CELERY_WORKERS:
        raise HTTPException(status_code=503, detail="Queue status requires Celery workers")

    # Get queue lengths from Celery
    inspector = celery_app.control.inspect()
    active = inspector.active() or {}
    reserved = inspector.reserved() or {}

    # Calculate queue metrics
    queue_metrics = {}
    for priority in TaskPriority:
        queue_name = _get_queue_name(priority.value)

        # Count tasks in queue
        active_count = sum(
            1
            for worker_tasks in active.values()
            for task in worker_tasks
            if task.get("queue") == queue_name
        )

        reserved_count = sum(
            1
            for worker_tasks in reserved.values()
            for task in worker_tasks
            if task.get("queue") == queue_name
        )

        queue_metrics[priority.name] = {
            "active": active_count,
            "reserved": reserved_count,
            "total": active_count + reserved_count,
            "estimated_wait_time": (active_count + reserved_count) * 30,
        }

    # Get worker stats
    stats = inspector.stats() or {}
    worker_count = len(stats)

    return {
        "workers": {
            "online": worker_count,
            "stats": stats,
        },
        "queues": queue_metrics,
        "timestamp": datetime.utcnow().isoformat(),
    }


# Helper functions
def _get_routing_key(priority: int) -> str:
    """Get Celery routing key for priority."""
    if priority >= TaskPriority.CRITICAL.value:
        return "generation.critical"
    elif priority >= TaskPriority.HIGH.value:
        return "generation.high.music"
    else:
        return "generation.music"


def _get_queue_name(priority: int) -> str:
    """Get Celery queue name for priority."""
    if priority >= TaskPriority.CRITICAL.value:
        return "critical"
    elif priority >= TaskPriority.HIGH.value:
        return "generation-high"
    else:
        return "generation"


async def _estimate_queue_position(task_repository: TaskRepository, priority: int) -> int:
    """Estimate position in queue based on priority."""

    # This is a simplified estimation
    # In production, would query Redis directly
    try:
        # Count pending/queued tasks with same or higher priority
        all_tasks = await task_repository.list_tasks(status=TaskStatus.QUEUED.value, limit=1000)

        position = sum(1 for t in all_tasks if t.get("priority", 0) >= priority)

        return position

    except:
        return 0
