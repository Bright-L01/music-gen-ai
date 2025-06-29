"""
Audio Processing Service API

Handles audio file operations, format conversion, waveform generation,
and audio analysis for the music generation platform.
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from prometheus_client import Counter, Histogram, generate_latest

from .models import (
    ConversionRequest,
    ConversionResponse,
    AnalysisRequest,
    AnalysisResponse,
    WaveformRequest,
    WaveformResponse,
    MixRequest,
    MixResponse,
    ProcessingStatus
)
from .service import AudioProcessingService
from .storage import AudioStorage
from .auth import get_current_user, verify_service_token


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
request_counter = Counter(
    "audio_processing_requests_total",
    "Total number of audio processing requests",
    ["endpoint", "status"]
)
process_duration = Histogram(
    "audio_processing_duration_seconds",
    "Audio processing duration in seconds",
    ["operation"]
)

# Service instances
audio_service = AudioProcessingService()
storage = AudioStorage()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting Audio Processing Service...")
    await audio_service.initialize()
    await storage.initialize()
    logger.info("Audio Processing Service started successfully")
    
    yield
    
    logger.info("Shutting down Audio Processing Service...")
    await audio_service.cleanup()
    await storage.cleanup()
    logger.info("Audio Processing Service stopped")


# Create FastAPI app
app = FastAPI(
    title="Audio Processing Service",
    description="Microservice for audio file operations and analysis",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "audio-processing",
        "version": "1.0.0"
    }


@app.post("/convert", response_model=ConversionResponse)
async def convert_audio(
    request: ConversionRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """
    Convert audio between formats
    
    Supports: WAV, MP3, FLAC, OGG
    """
    request_counter.labels(endpoint="convert", status="started").inc()
    
    try:
        # Create job
        job_id = await audio_service.create_conversion_job(
            source_url=request.source_url,
            target_format=request.target_format,
            options=request.options,
            user_id=current_user["id"]
        )
        
        # Process in background
        background_tasks.add_task(
            audio_service.process_conversion,
            job_id
        )
        
        request_counter.labels(endpoint="convert", status="success").inc()
        
        return ConversionResponse(
            job_id=job_id,
            status=ProcessingStatus.QUEUED,
            progress=0.0
        )
        
    except Exception as e:
        logger.error(f"Conversion error: {e}")
        request_counter.labels(endpoint="convert", status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_audio(
    request: AnalysisRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Analyze audio file and extract features
    
    Features: tempo, key, energy, loudness, spectral features
    """
    request_counter.labels(endpoint="analyze", status="started").inc()
    
    try:
        with process_duration.labels(operation="analyze").time():
            features = await audio_service.analyze_audio(
                audio_url=request.audio_url,
                feature_types=request.feature_types
            )
        
        request_counter.labels(endpoint="analyze", status="success").inc()
        
        return AnalysisResponse(
            audio_url=request.audio_url,
            features=features,
            duration=features.get("duration", 0),
            sample_rate=features.get("sample_rate", 0)
        )
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        request_counter.labels(endpoint="analyze", status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/waveform", response_model=WaveformResponse)
async def generate_waveform(
    request: WaveformRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Generate waveform visualization data
    
    Returns time-series amplitude data for visualization
    """
    request_counter.labels(endpoint="waveform", status="started").inc()
    
    try:
        with process_duration.labels(operation="waveform").time():
            waveform_data = await audio_service.generate_waveform(
                audio_url=request.audio_url,
                width=request.width,
                height=request.height,
                color_scheme=request.color_scheme
            )
        
        request_counter.labels(endpoint="waveform", status="success").inc()
        
        return WaveformResponse(
            waveform_url=waveform_data["url"],
            peaks=waveform_data["peaks"],
            duration=waveform_data["duration"]
        )
        
    except Exception as e:
        logger.error(f"Waveform generation error: {e}")
        request_counter.labels(endpoint="waveform", status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/mix", response_model=MixResponse)
async def mix_audio(
    request: MixRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """
    Mix multiple audio tracks
    
    Supports volume adjustment, panning, effects
    """
    request_counter.labels(endpoint="mix", status="started").inc()
    
    try:
        # Validate tracks
        if len(request.tracks) < 2:
            raise HTTPException(
                status_code=400,
                detail="At least 2 tracks required for mixing"
            )
        
        # Create mixing job
        job_id = await audio_service.create_mix_job(
            tracks=request.tracks,
            output_format=request.output_format,
            master_volume=request.master_volume,
            user_id=current_user["id"]
        )
        
        # Process in background
        background_tasks.add_task(
            audio_service.process_mix,
            job_id
        )
        
        request_counter.labels(endpoint="mix", status="success").inc()
        
        return MixResponse(
            job_id=job_id,
            status=ProcessingStatus.QUEUED,
            progress=0.0
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Mixing error: {e}")
        request_counter.labels(endpoint="mix", status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_audio(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """Upload audio file to storage"""
    request_counter.labels(endpoint="upload", status="started").inc()
    
    try:
        # Validate file type
        allowed_types = ["audio/wav", "audio/mpeg", "audio/mp3", "audio/flac", "audio/ogg"]
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file.content_type}"
            )
        
        # Upload to storage
        file_url = await storage.upload_file(
            file=file,
            user_id=current_user["id"]
        )
        
        request_counter.labels(endpoint="upload", status="success").inc()
        
        return {
            "url": file_url,
            "filename": file.filename,
            "size": file.size,
            "content_type": file.content_type
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        request_counter.labels(endpoint="upload", status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download/{file_id}")
async def download_audio(
    file_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Download audio file"""
    try:
        file_path = await storage.get_file_path(file_id, current_user["id"])
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(
            path=file_path,
            media_type="audio/wav",
            filename=f"{file_id}.wav"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/job/{job_id}")
async def get_job_status(
    job_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get processing job status"""
    try:
        job_data = await audio_service.get_job_status(job_id, current_user["id"])
        
        if not job_data:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return job_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Job status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/trim")
async def trim_audio(
    audio_url: str,
    start_time: float,
    end_time: float,
    current_user: dict = Depends(get_current_user)
):
    """Trim audio file to specified time range"""
    request_counter.labels(endpoint="trim", status="started").inc()
    
    try:
        with process_duration.labels(operation="trim").time():
            trimmed_url = await audio_service.trim_audio(
                audio_url=audio_url,
                start_time=start_time,
                end_time=end_time,
                user_id=current_user["id"]
            )
        
        request_counter.labels(endpoint="trim", status="success").inc()
        
        return {
            "url": trimmed_url,
            "start_time": start_time,
            "end_time": end_time,
            "duration": end_time - start_time
        }
        
    except Exception as e:
        logger.error(f"Trim error: {e}")
        request_counter.labels(endpoint="trim", status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/normalize")
async def normalize_audio(
    audio_url: str,
    target_loudness: float = -14.0,  # LUFS
    current_user: dict = Depends(get_current_user)
):
    """Normalize audio loudness"""
    request_counter.labels(endpoint="normalize", status="started").inc()
    
    try:
        with process_duration.labels(operation="normalize").time():
            normalized_url = await audio_service.normalize_audio(
                audio_url=audio_url,
                target_loudness=target_loudness,
                user_id=current_user["id"]
            )
        
        request_counter.labels(endpoint="normalize", status="success").inc()
        
        return {
            "url": normalized_url,
            "target_loudness": target_loudness
        }
        
    except Exception as e:
        logger.error(f"Normalization error: {e}")
        request_counter.labels(endpoint="normalize", status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return StreamingResponse(
        generate_latest(),
        media_type="text/plain"
    )


# Service-to-service endpoints (no user auth required)
@app.post("/internal/process-generated")
async def process_generated_audio(
    audio_url: str,
    job_id: str,
    service_auth: dict = Depends(verify_service_token)
):
    """
    Internal endpoint for post-processing generated audio
    Called by generation service after audio creation
    """
    try:
        # Apply default processing (normalization, format conversion)
        processed_url = await audio_service.post_process_generated(
            audio_url=audio_url,
            job_id=job_id
        )
        
        return {
            "processed_url": processed_url,
            "job_id": job_id
        }
        
    except Exception as e:
        logger.error(f"Post-processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)