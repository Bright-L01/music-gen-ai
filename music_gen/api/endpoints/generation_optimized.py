"""
Optimized generation endpoints with performance improvements.

This module implements all performance optimizations including:
- Query result caching
- Lazy loading
- Memory optimization
- File streaming
"""

import asyncio
import hashlib
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ...application.services import GenerationService
from ...core.cache import CacheKey, cache_result, get_cache_manager
from ...core.container import get_container
from ...core.memory_optimizer import memory_efficient_decorator, MemoryMonitor
from ...infrastructure.repositories import TaskRepository
from ...utils.file_optimizer import file_stream_handler, optimize_audio_file
from ..deps import get_current_user
from ..schemas import GenerationRequest, GenerationResponse, UserProfile

router = APIRouter()


class OptimizedGenerationRequest(GenerationRequest):
    """Extended generation request with optimization options."""

    enable_caching: bool = Field(True, description="Enable result caching")
    optimize_output: bool = Field(True, description="Optimize output file size")
    stream_response: bool = Field(False, description="Stream audio response")


@router.post("/generate", response_model=GenerationResponse)
@memory_efficient_decorator(max_memory_gb=4.0, monitor=True)
async def generate_music_optimized(
    request: OptimizedGenerationRequest,
    background_tasks: BackgroundTasks,
    req: Request,
    user: Optional[UserProfile] = Depends(get_current_user),
) -> GenerationResponse:
    """
    Optimized music generation endpoint with caching and memory efficiency.
    """

    # Get services
    container = get_container()
    generation_service: GenerationService = container.generation_service()
    task_repository: TaskRepository = container.task_repository()
    cache_manager = await get_cache_manager()

    # Generate cache key if caching is enabled
    cache_key = None
    if request.enable_caching:
        # Create deterministic cache key
        request_hash = hashlib.sha256(
            f"{request.prompt}:{request.duration}:{request.temperature}:{request.top_k}:{request.top_p}".encode()
        ).hexdigest()

        cache_key = CacheKey.model_inference_cache(
            model_id="musicgen", prompt_hash=request_hash[:16], params_hash=request_hash[16:32]
        )

        # Check cache
        cached_result = await cache_manager.get(cache_key)
        if cached_result:
            return GenerationResponse(**cached_result)

    # Use performance hints from middleware
    perf_hints = getattr(req.state, "perf_hints", {})

    # Create generation task with optimization flags
    task_id = await generation_service.create_generation_task(
        prompt=request.prompt,
        duration=request.duration,
        user_id=user.id if user else "anonymous",
        parameters={
            "temperature": request.temperature,
            "top_k": request.top_k,
            "top_p": request.top_p,
            "use_cache": perf_hints.get("use_cache", True),
            "optimize_memory": True,
            "stream_generation": request.stream_response,
        },
    )

    # Start generation in background with memory monitoring
    async def generate_with_monitoring():
        try:
            # Monitor memory during generation
            MemoryMonitor.log_memory_usage("Generation start")

            result = await generation_service.generate_music(
                task_id=task_id,
                prompt=request.prompt,
                duration=request.duration,
                parameters={
                    "temperature": request.temperature,
                    "top_k": request.top_k,
                    "top_p": request.top_p,
                },
            )

            # Optimize output file if requested
            if request.optimize_output and result.get("audio_path"):
                original_path = Path(result["audio_path"])
                optimized_path = await optimize_audio_file(
                    original_path, target_format="mp3", target_bitrate="192k"
                )
                result["audio_path"] = str(optimized_path)
                result["optimized"] = True

                # Clean up original file
                background_tasks.add_task(original_path.unlink)

            # Update task with result
            await task_repository.update_task_status(task_id, "completed", result=result)

            # Cache result if enabled
            if cache_key:
                await cache_manager.set(
                    cache_key,
                    result,
                    ttl=3600,  # 1 hour
                    tags=["generation", f"user:{user.id}" if user else "anonymous"],
                )

            MemoryMonitor.log_memory_usage("Generation complete")

        except Exception as e:
            await task_repository.update_task_status(task_id, "failed", error=str(e))
            raise

    # Start generation
    background_tasks.add_task(generate_with_monitoring)

    # Return immediate response
    return GenerationResponse(
        task_id=task_id,
        status="processing",
        message="Generation started with optimizations",
        created_at=int(time.time()),
        metadata={
            "cached": False,
            "optimizations": {
                "caching": request.enable_caching,
                "output_optimization": request.optimize_output,
                "streaming": request.stream_response,
            },
        },
    )


@router.get("/stream/{task_id}")
async def stream_generation_result(
    task_id: str, user: Optional[UserProfile] = Depends(get_current_user)
):
    """
    Stream generated audio file for memory efficiency.
    """

    container = get_container()
    task_repository: TaskRepository = container.task_repository()

    # Get task
    task = await task_repository.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    # Check ownership
    if user and task.user_id != user.id and task.user_id != "anonymous":
        raise HTTPException(status_code=403, detail="Access denied")

    # Check if completed
    if task.status != "completed":
        raise HTTPException(
            status_code=400, detail=f"Task not completed. Current status: {task.status}"
        )

    # Get audio file path
    audio_path = task.result.get("audio_path") if task.result else None
    if not audio_path or not Path(audio_path).exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    # Stream file
    async def audio_streamer():
        async for chunk in file_stream_handler.stream_file_download(Path(audio_path)):
            yield chunk

    # Determine media type
    media_type = "audio/mpeg" if audio_path.endswith(".mp3") else "audio/wav"

    return StreamingResponse(
        audio_streamer(),
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="generated_{task_id}.{Path(audio_path).suffix}"',
            "Cache-Control": "public, max-age=3600",  # Cache for 1 hour
        },
    )


@router.get("/history", response_model=List[GenerationResponse])
@cache_result(
    key_func=lambda user, page, limit: f"user:generations:{user.id}:page:{page}",
    ttl=300,  # 5 minutes
    tags=["user_data"],
)
async def get_generation_history_optimized(
    page: int = 1,
    limit: int = 10,
    user: UserProfile = Depends(get_current_user),
    req: Request = None,
) -> List[GenerationResponse]:
    """
    Get user's generation history with caching and lazy loading.
    """

    container = get_container()
    task_repository: TaskRepository = container.task_repository()

    # Use lazy loading hints from middleware
    lazy_config = getattr(req.state, "lazy_loading", {})

    # Apply limits
    limit = min(limit, lazy_config.get("max_limit", 100))
    offset = (page - 1) * limit

    # Get tasks with optimized query
    tasks = await task_repository.get_user_tasks(
        user_id=user.id,
        limit=limit,
        offset=offset,
        # Only fetch necessary fields
        fields=["id", "status", "created_at", "prompt", "result"],
    )

    # Convert to response format
    responses = []
    for task in tasks:
        response = GenerationResponse(
            task_id=task.id,
            status=task.status,
            created_at=int(task.created_at.timestamp()),
            prompt=task.metadata.get("prompt", "") if task.metadata else "",
            result=task.result,
            metadata={
                "duration": task.metadata.get("duration") if task.metadata else None,
                "cached": True,  # This response is cached
            },
        )
        responses.append(response)

    return responses


@router.post("/batch", response_model=List[GenerationResponse])
@memory_efficient_decorator(max_memory_gb=8.0, monitor=True)
async def generate_batch_optimized(
    requests: List[OptimizedGenerationRequest],
    background_tasks: BackgroundTasks,
    user: Optional[UserProfile] = Depends(get_current_user),
) -> List[GenerationResponse]:
    """
    Batch generation with memory optimization and parallel processing.
    """

    if len(requests) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 generations per batch")

    container = get_container()
    generation_service: GenerationService = container.generation_service()

    # Create tasks for each request
    tasks = []
    for req in requests:
        task_id = await generation_service.create_generation_task(
            prompt=req.prompt,
            duration=req.duration,
            user_id=user.id if user else "anonymous",
            parameters={
                "temperature": req.temperature,
                "top_k": req.top_k,
                "top_p": req.top_p,
                "batch_mode": True,
            },
        )
        tasks.append({"task_id": task_id, "request": req})

    # Process in parallel with concurrency limit
    async def process_batch():
        semaphore = asyncio.Semaphore(3)  # Limit concurrent generations

        async def generate_one(task_info):
            async with semaphore:
                task_id = task_info["task_id"]
                req = task_info["request"]

                try:
                    result = await generation_service.generate_music(
                        task_id=task_id,
                        prompt=req.prompt,
                        duration=req.duration,
                        parameters={
                            "temperature": req.temperature,
                            "top_k": req.top_k,
                            "top_p": req.top_p,
                        },
                    )

                    # Optimize if requested
                    if req.optimize_output and result.get("audio_path"):
                        original_path = Path(result["audio_path"])
                        optimized_path = await optimize_audio_file(original_path)
                        result["audio_path"] = str(optimized_path)

                        # Schedule cleanup
                        background_tasks.add_task(original_path.unlink)

                    return task_id, "completed", result

                except Exception as e:
                    return task_id, "failed", {"error": str(e)}

        # Process all tasks
        results = await asyncio.gather(
            *[generate_one(task) for task in tasks], return_exceptions=True
        )

        # Update task statuses
        for result in results:
            if isinstance(result, Exception):
                continue

            task_id, status, data = result
            await container.task_repository().update_task_status(task_id, status, result=data)

    # Start batch processing
    background_tasks.add_task(process_batch)

    # Return immediate responses
    return [
        GenerationResponse(
            task_id=task["task_id"],
            status="processing",
            message="Batch generation started",
            created_at=int(time.time()),
            metadata={"batch": True},
        )
        for task in tasks
    ]


@router.delete("/cache")
async def clear_generation_cache(user: UserProfile = Depends(get_current_user)):
    """
    Clear generation cache for the current user.
    """

    cache_manager = await get_cache_manager()

    # Invalidate user-specific cache
    deleted = await cache_manager.invalidate_by_tags([f"user:{user.id}"])

    return {"message": "Cache cleared successfully", "entries_deleted": deleted}


@router.get("/performance/stats")
async def get_performance_stats(user: Optional[UserProfile] = Depends(get_current_user)):
    """
    Get performance statistics for monitoring.
    """

    from ...core.memory_optimizer import get_memory_usage_summary

    cache_manager = await get_cache_manager()

    return {
        "cache": {
            "hit_rate": cache_manager.get_hit_rate(),
            "hits": cache_manager.hit_count,
            "misses": cache_manager.miss_count,
        },
        "memory": get_memory_usage_summary(),
        "timestamp": int(time.time()),
    }
