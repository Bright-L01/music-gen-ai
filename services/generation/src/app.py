"""
Music Generation Microservice

Handles all music generation requests using MusicGen models.
Provides queue-based processing, caching, and GPU optimization.
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List

import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch

from .models import (
    GenerationRequest, 
    GenerationResponse, 
    GenerationStatus,
    SongStructure,
    AudioSection
)
from .service import GenerationService
from .queue import GenerationQueue
from .cache import GenerationCache
from .auth import verify_token, get_current_user


# Initialize FastAPI app
app = FastAPI(
    title="Music Generation Service",
    description="Core music generation microservice using MusicGen",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
generation_service = GenerationService()
generation_queue = GenerationQueue()
generation_cache = GenerationCache()


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    await generation_service.initialize()
    await generation_queue.connect()
    await generation_cache.connect()
    print("✅ Generation service started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    await generation_service.cleanup()
    await generation_queue.disconnect()
    await generation_cache.disconnect()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "generation",
        "models_loaded": generation_service.models_loaded,
        "gpu_available": torch.cuda.is_available(),
        "queue_size": await generation_queue.size()
    }


@app.post("/generate", response_model=GenerationResponse)
async def generate_music(
    request: GenerationRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict = Depends(get_current_user)
):
    """
    Generate music from text prompt
    
    Supports:
    - Simple generation from prompt
    - Structured composition with sections
    - Extended duration (up to 5 minutes)
    - Various musical parameters
    """
    # Check cache first
    cache_key = generation_cache.get_cache_key(request)
    if cached := await generation_cache.get(cache_key):
        return GenerationResponse(
            job_id=cached['job_id'],
            status=GenerationStatus.COMPLETED,
            audio_url=cached['audio_url'],
            cached=True
        )
    
    # Create job
    job_id = str(uuid.uuid4())
    job_data = {
        "job_id": job_id,
        "user_id": current_user["id"],
        "request": request.dict(),
        "status": GenerationStatus.QUEUED,
        "created_at": datetime.utcnow().isoformat()
    }
    
    # Add to queue based on user tier
    priority = "high" if current_user.get("tier") == "premium" else "normal"
    await generation_queue.enqueue(job_id, job_data, priority)
    
    # Process in background
    background_tasks.add_task(process_generation, job_id)
    
    return GenerationResponse(
        job_id=job_id,
        status=GenerationStatus.QUEUED,
        position_in_queue=await generation_queue.position(job_id)
    )


@app.get("/status/{job_id}", response_model=GenerationResponse)
async def get_generation_status(job_id: str):
    """Get status of a generation job"""
    job_data = await generation_queue.get_job(job_id)
    
    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return GenerationResponse(
        job_id=job_id,
        status=job_data["status"],
        audio_url=job_data.get("audio_url"),
        error=job_data.get("error"),
        progress=job_data.get("progress", 0),
        position_in_queue=await generation_queue.position(job_id) if job_data["status"] == GenerationStatus.QUEUED else None
    )


@app.post("/generate/structured", response_model=GenerationResponse)
async def generate_structured_music(
    prompt: str,
    structure: SongStructure,
    duration_minutes: float = Field(2.0, ge=0.5, le=5.0),
    background_tasks: BackgroundTasks,
    current_user: Dict = Depends(get_current_user)
):
    """
    Generate music with specific song structure
    
    Supports verse-chorus-bridge patterns and extended compositions
    """
    request = GenerationRequest(
        prompt=prompt,
        duration=duration_minutes * 60,
        structure=structure,
        advanced_features={
            "coherent_structure": True,
            "smooth_transitions": True,
            "consistent_theme": True
        }
    )
    
    return await generate_music(request, background_tasks, current_user)


@app.post("/generate/batch")
async def generate_batch(
    requests: List[GenerationRequest],
    background_tasks: BackgroundTasks,
    current_user: Dict = Depends(get_current_user)
):
    """Batch generation for multiple requests"""
    if len(requests) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 requests per batch")
    
    jobs = []
    for request in requests:
        response = await generate_music(request, background_tasks, current_user)
        jobs.append(response.dict())
    
    return {"jobs": jobs, "count": len(jobs)}


async def process_generation(job_id: str):
    """Background task to process generation"""
    try:
        # Get job from queue
        job_data = await generation_queue.get_job(job_id)
        if not job_data:
            return
        
        # Update status
        job_data["status"] = GenerationStatus.PROCESSING
        await generation_queue.update_job(job_id, job_data)
        
        # Get request
        request = GenerationRequest(**job_data["request"])
        
        # Generate audio
        if request.structure:
            # Structured generation
            audio_url = await generation_service.generate_structured(
                prompt=request.prompt,
                structure=request.structure,
                duration=request.duration,
                progress_callback=lambda p: update_progress(job_id, p)
            )
        else:
            # Simple generation
            audio_url = await generation_service.generate_simple(
                prompt=request.prompt,
                duration=request.duration,
                temperature=request.temperature,
                progress_callback=lambda p: update_progress(job_id, p)
            )
        
        # Update job with result
        job_data["status"] = GenerationStatus.COMPLETED
        job_data["audio_url"] = audio_url
        job_data["completed_at"] = datetime.utcnow().isoformat()
        await generation_queue.update_job(job_id, job_data)
        
        # Cache result
        cache_key = generation_cache.get_cache_key(request)
        await generation_cache.set(cache_key, {
            "job_id": job_id,
            "audio_url": audio_url
        })
        
    except Exception as e:
        # Update job with error
        job_data["status"] = GenerationStatus.FAILED
        job_data["error"] = str(e)
        await generation_queue.update_job(job_id, job_data)


async def update_progress(job_id: str, progress: float):
    """Update job progress"""
    job_data = await generation_queue.get_job(job_id)
    if job_data:
        job_data["progress"] = progress
        await generation_queue.update_job(job_id, job_data)


# API documentation
@app.get("/")
async def root():
    """Service information"""
    return {
        "service": "Music Generation Service",
        "version": "1.0.0",
        "endpoints": [
            "/generate",
            "/generate/structured",
            "/generate/batch",
            "/status/{job_id}",
            "/health"
        ],
        "capabilities": [
            "Text-to-music generation",
            "Structured composition (verse-chorus-bridge)",
            "Extended duration (up to 5 minutes)",
            "Batch processing",
            "GPU acceleration",
            "Result caching"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)