"""
Generation endpoints for Music Gen AI API.
"""

import asyncio
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import scipy.io.wavfile
import torch
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from ...core.model_manager import ModelManager
from ...optimization.fast_generator import GenerationRequest as OptRequest

router = APIRouter()

# Task storage (in production, use Redis or database)
tasks: Dict[str, Dict[str, Any]] = {}

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


@router.post("/", response_model=GenerationResponse)
async def generate_music(
    request: GenerationRequest,
    background_tasks: BackgroundTasks,
):
    """Generate music from text prompt."""
    
    model_manager = ModelManager()
    if not model_manager.has_loaded_models():
        raise HTTPException(status_code=503, detail="No models loaded")
    
    # Create task
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "status": "pending",
        "request": request.dict(),
        "created_at": asyncio.get_event_loop().time(),
    }
    
    # Start background generation
    background_tasks.add_task(
        generate_music_task,
        task_id,
        request,
    )
    
    return GenerationResponse(
        task_id=task_id,
        status="pending",
    )


@router.post("/batch")
async def generate_music_batch(
    batch_request: BatchGenerationRequest,
    background_tasks: BackgroundTasks,
):
    """Generate multiple music clips concurrently."""
    
    model_manager = ModelManager()
    if not model_manager.has_loaded_models():
        raise HTTPException(status_code=503, detail="No models loaded")
    
    # Create batch task
    batch_id = str(uuid.uuid4())
    task_ids = []
    
    for i, req in enumerate(batch_request.requests):
        task_id = f"{batch_id}_{i}"
        tasks[task_id] = {
            "status": "pending",
            "request": req.dict(),
            "created_at": asyncio.get_event_loop().time(),
            "batch_id": batch_id,
            "batch_index": i,
        }
        task_ids.append(task_id)
    
    # Start batch generation
    background_tasks.add_task(
        generate_music_batch_task,
        batch_id,
        batch_request.requests,
    )
    
    return {
        "batch_id": batch_id,
        "task_ids": task_ids,
        "status": "pending",
        "total_requests": len(batch_request.requests),
    }


@router.get("/{task_id}", response_model=GenerationResponse)
async def get_generation_status(task_id: str):
    """Get the status of a generation task."""
    
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    
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
async def get_batch_status(batch_id: str):
    """Get status of a batch generation."""
    
    batch_tasks = {k: v for k, v in tasks.items() if v.get("batch_id") == batch_id}
    
    if not batch_tasks:
        raise HTTPException(status_code=404, detail="Batch not found")
    
    completed = sum(1 for task in batch_tasks.values() if task["status"] == "completed")
    failed = sum(1 for task in batch_tasks.values() if task["status"] == "failed")
    pending = len(batch_tasks) - completed - failed
    
    overall_status = (
        "completed"
        if completed == len(batch_tasks)
        else "processing" if completed + failed < len(batch_tasks) else "failed"
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


async def generate_music_task(task_id: str, request: GenerationRequest):
    """Background task for music generation."""
    
    try:
        tasks[task_id]["status"] = "processing"
        
        # Get model from manager
        model_manager = ModelManager()
        model = model_manager.get_model("facebook/musicgen-small")
        
        # Set random seed if provided
        if request.seed is not None:
            torch.manual_seed(request.seed)
            np.random.seed(request.seed)
        
        # Generate audio using optimized pipeline
        result = model.generate_single(
            prompt=request.prompt,
            duration=request.duration,
            temperature=request.temperature,
            guidance_scale=request.guidance_scale,
        )
        
        # Save audio file
        audio_path = TEMP_DIR / f"{task_id}.wav"
        scipy.io.wavfile.write(
            str(audio_path),
            rate=result.sample_rate,
            data=(result.audio * 32767).astype(np.int16)
        )
        
        # Update task status
        tasks[task_id].update({
            "status": "completed",
            "audio_path": str(audio_path),
            "duration": request.duration,
            "metadata": {
                "prompt": request.prompt,
                "generation_params": {
                    "temperature": request.temperature,
                    "top_k": request.top_k,
                    "top_p": request.top_p,
                    "repetition_penalty": request.repetition_penalty,
                    "num_beams": request.num_beams,
                    "guidance_scale": request.guidance_scale,
                },
                "conditioning": {
                    "genre": request.genre,
                    "mood": request.mood,
                    "tempo": request.tempo,
                    "instruments": request.instruments,
                },
                "model_info": {
                    "sample_rate": result.sample_rate,
                    "device": str(model.device),
                    "generation_time": result.generation_time,
                },
            },
            "completed_at": asyncio.get_event_loop().time(),
        })
        
    except Exception as e:
        tasks[task_id].update({
            "status": "failed",
            "error": str(e),
            "failed_at": asyncio.get_event_loop().time(),
        })


async def generate_music_batch_task(batch_id: str, requests: List[GenerationRequest]):
    """Background task for batch music generation."""
    
    try:
        # Get model from manager
        model_manager = ModelManager()
        model = model_manager.get_model("facebook/musicgen-small")
        
        # Convert to optimization requests
        opt_requests = [
            OptRequest(
                prompt=req.prompt,
                duration=req.duration,
                temperature=req.temperature,
                guidance_scale=req.guidance_scale,
                request_id=f"{batch_id}_{i}",
            )
            for i, req in enumerate(requests)
        ]
        
        # Generate batch
        results = model.generate_batch(opt_requests)
        
        # Save results
        for i, result in enumerate(results):
            task_id = f"{batch_id}_{i}"
            
            if result.metadata and "error" in result.metadata:
                # Handle error
                tasks[task_id].update({
                    "status": "failed",
                    "error": result.metadata["error"],
                    "failed_at": asyncio.get_event_loop().time(),
                })
            else:
                # Save audio file
                audio_path = TEMP_DIR / f"{task_id}.wav"
                scipy.io.wavfile.write(
                    str(audio_path),
                    rate=result.sample_rate,
                    data=(result.audio * 32767).astype(np.int16),
                )
                
                # Update task status
                tasks[task_id].update({
                    "status": "completed",
                    "audio_path": str(audio_path),
                    "duration": result.duration,
                    "metadata": {
                        "prompt": result.metadata["prompt"],
                        "generation_time": result.generation_time,
                        "batch_info": {
                            "batch_id": batch_id,
                            "batch_size": len(requests),
                            "batch_index": i,
                        },
                    },
                    "completed_at": asyncio.get_event_loop().time(),
                })
        
    except Exception as e:
        # Mark all tasks as failed
        for i in range(len(requests)):
            task_id = f"{batch_id}_{i}"
            if task_id in tasks:
                tasks[task_id].update({
                    "status": "failed",
                    "error": f"Batch failed: {str(e)}",
                    "failed_at": asyncio.get_event_loop().time(),
                })
