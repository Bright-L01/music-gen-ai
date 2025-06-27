"""
FastAPI server for MusicGen inference.
"""
import os
import asyncio
import tempfile
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

import torch
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from ..models.musicgen import MusicGenModel, create_musicgen_model
from ..models.multi_instrument import MultiInstrumentMusicGen
from ..utils.audio import save_audio_file, load_audio_file
from ..evaluation.metrics import AudioQualityMetrics
from ..streaming import SessionManager
from .streaming_api import setup_streaming_routes
from .multi_instrument_api import setup_multi_instrument_routes
from ..web.app import setup_web_routes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance
model: Optional[MusicGenModel] = None
audio_metrics: Optional[AudioQualityMetrics] = None
session_manager: Optional[SessionManager] = None
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "models/musicgen-base")
MAX_DURATION = float(os.getenv("MAX_DURATION", "60.0"))
DEFAULT_DURATION = float(os.getenv("DEFAULT_DURATION", "10.0"))
TEMP_DIR = Path(os.getenv("TEMP_DIR", "/tmp/musicgen"))
TEMP_DIR.mkdir(exist_ok=True)

# Pydantic models
class GenerationRequest(BaseModel):
    """Request model for music generation."""
    
    prompt: str = Field(..., description="Text description of the music to generate")
    duration: float = Field(DEFAULT_DURATION, ge=1.0, le=MAX_DURATION, description="Duration in seconds")
    temperature: float = Field(1.0, ge=0.1, le=2.0, description="Sampling temperature")
    top_k: int = Field(50, ge=1, le=100, description="Top-k sampling parameter")
    top_p: float = Field(0.9, ge=0.1, le=1.0, description="Top-p (nucleus) sampling parameter")
    do_sample: bool = Field(True, description="Whether to use sampling")
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0, description="Repetition penalty")
    num_beams: int = Field(1, ge=1, le=8, description="Number of beams for beam search (1 = greedy/sampling)")
    length_penalty: float = Field(1.0, ge=0.0, le=2.0, description="Length penalty for beam search")
    early_stopping: bool = Field(True, description="Whether to stop early in beam search")
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


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    device: str = Field(..., description="Device being used")
    memory_usage: Optional[Dict[str, float]] = Field(None, description="Memory usage statistics")


class EvaluationRequest(BaseModel):
    """Request model for audio evaluation."""
    
    evaluate_quality: bool = Field(True, description="Evaluate audio quality metrics")
    reference_audio_url: Optional[str] = Field(None, description="Reference audio for comparison")


class EvaluationResponse(BaseModel):
    """Response model for audio evaluation."""
    
    metrics: Dict[str, float] = Field(..., description="Evaluation metrics")
    quality_score: float = Field(..., description="Overall quality score")


# Task storage (in production, use Redis or database)
tasks: Dict[str, Dict[str, Any]] = {}

# Create FastAPI app
app = FastAPI(
    title="MusicGen API",
    description="Production-ready text-to-music generation API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup."""
    global model, audio_metrics, session_manager
    
    logger.info("Starting MusicGen API server...")
    logger.info(f"Using device: {device}")
    
    try:
        # Load model
        if os.path.exists(MODEL_PATH):
            logger.info(f"Loading model from {MODEL_PATH}")
            model = MusicGenModel.from_pretrained(MODEL_PATH)
        else:
            logger.info("Creating default model")
            model = create_musicgen_model("base")
        
        model = model.to(device)
        model.eval()
        
        # Initialize metrics
        audio_metrics = AudioQualityMetrics()
        
        # Initialize session manager for streaming
        max_concurrent_sessions = int(os.getenv("MAX_CONCURRENT_SESSIONS", "5"))
        session_manager = SessionManager(model, max_concurrent_sessions)
        
        # Setup streaming routes
        setup_streaming_routes(app, model, session_manager)
        
        # Setup multi-instrument routes if model supports it
        if hasattr(model, 'multi_config') or isinstance(model, MultiInstrumentMusicGen):
            setup_multi_instrument_routes(app, model, TEMP_DIR)
            logger.info("Multi-instrument API routes enabled")
        
        # Setup web UI routes
        setup_web_routes(app)
        
        logger.info("Model, streaming services, and web UI loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global session_manager
    
    logger.info("Shutting down MusicGen API server...")
    
    # Shutdown session manager
    if session_manager:
        await session_manager.shutdown()
    
    # Cleanup temporary files
    for file_path in TEMP_DIR.glob("*.wav"):
        try:
            file_path.unlink()
        except Exception as e:
            logger.warning(f"Failed to cleanup {file_path}: {e}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    
    memory_usage = None
    if torch.cuda.is_available():
        memory_usage = {
            "allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
            "cached": torch.cuda.memory_reserved() / 1024**3,      # GB
            "max_allocated": torch.cuda.max_memory_allocated() / 1024**3,
        }
    
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        device=str(device),
        memory_usage=memory_usage,
    )


@app.post("/generate", response_model=GenerationResponse)
async def generate_music(
    request: GenerationRequest,
    background_tasks: BackgroundTasks,
):
    """Generate music from text prompt."""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
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


@app.get("/generate/{task_id}", response_model=GenerationResponse)
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


@app.get("/download/{task_id}")
async def download_audio(task_id: str):
    """Download generated audio file."""
    
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="Generation not completed")
    
    audio_path = task.get("audio_path")
    if not audio_path or not Path(audio_path).exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(
        audio_path,
        media_type="audio/wav",
        filename=f"generated_music_{task_id}.wav",
    )


@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_audio(
    audio_file: UploadFile = File(...),
    request: EvaluationRequest = EvaluationRequest(),
):
    """Evaluate audio quality metrics."""
    
    if audio_metrics is None:
        raise HTTPException(status_code=503, detail="Evaluation metrics not available")
    
    try:
        # Save uploaded file temporarily
        temp_path = TEMP_DIR / f"eval_{uuid.uuid4()}.wav"
        
        with open(temp_path, "wb") as f:
            content = await audio_file.read()
            f.write(content)
        
        # Load and evaluate audio
        audio, sample_rate = load_audio_file(str(temp_path))
        audio_np = audio.numpy()
        
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=0)  # Convert to mono
        
        # Compute metrics
        metrics = audio_metrics.evaluate_audio_quality([audio_np])
        
        # Compute overall quality score
        quality_score = _compute_quality_score(metrics)
        
        # Cleanup
        temp_path.unlink()
        
        return EvaluationResponse(
            metrics=metrics,
            quality_score=quality_score,
        )
        
    except Exception as e:
        logger.error(f"Failed to evaluate audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
async def list_models():
    """List available models."""
    
    return {
        "current_model": MODEL_PATH,
        "available_models": [
            "musicgen-small",
            "musicgen-base", 
            "musicgen-large",
        ],
        "model_info": {
            "parameters": "1.5B" if "base" in MODEL_PATH else "unknown",
            "sample_rate": 24000,
            "max_duration": MAX_DURATION,
        }
    }


@app.get("/genres")
async def list_genres():
    """List supported musical genres."""
    
    return {
        "genres": [
            "jazz", "classical", "rock", "electronic", "ambient", "folk",
            "blues", "country", "reggae", "hip-hop", "pop", "metal",
            "orchestral", "piano", "acoustic", "instrumental",
        ]
    }


@app.get("/moods") 
async def list_moods():
    """List supported musical moods."""
    
    return {
        "moods": [
            "happy", "sad", "energetic", "calm", "dramatic", "peaceful",
            "melancholic", "uplifting", "mysterious", "romantic", "epic",
            "nostalgic", "playful", "intense", "serene", "triumphant",
        ]
    }


async def generate_music_task(task_id: str, request: GenerationRequest):
    """Background task for music generation."""
    
    try:
        tasks[task_id]["status"] = "processing"
        
        # Set random seed if provided
        if request.seed is not None:
            torch.manual_seed(request.seed)
            np.random.seed(request.seed)
        
        # Prepare conditioning
        conditioning = {}
        
        if request.genre:
            # Map genre to ID (would need proper vocab mapping)
            genre_vocab = {"jazz": 0, "classical": 1, "rock": 2, "electronic": 3}
            genre_id = genre_vocab.get(request.genre.lower(), 0)
            conditioning["genre_ids"] = torch.tensor([genre_id], device=device)
        
        if request.mood:
            # Map mood to ID
            mood_vocab = {"happy": 0, "sad": 1, "energetic": 2, "calm": 3}
            mood_id = mood_vocab.get(request.mood.lower(), 0)
            conditioning["mood_ids"] = torch.tensor([mood_id], device=device)
        
        if request.tempo:
            conditioning["tempo"] = torch.tensor([float(request.tempo)], device=device)
        
        # Generate audio
        with torch.no_grad():
            if request.num_beams > 1:
                # Use beam search
                audio_tensor = model.generate_audio_with_beam_search(
                    texts=[request.prompt],
                    duration=request.duration,
                    num_beams=request.num_beams,
                    temperature=request.temperature,
                    top_k=request.top_k,
                    top_p=request.top_p,
                    repetition_penalty=request.repetition_penalty,
                    length_penalty=request.length_penalty,
                    early_stopping=request.early_stopping,
                    **conditioning,
                )
            else:
                # Use standard generation
                audio_tensor = model.generate_audio(
                    texts=[request.prompt],
                    duration=request.duration,
                    temperature=request.temperature,
                    top_k=request.top_k,
                    top_p=request.top_p,
                    do_sample=request.do_sample,
                    repetition_penalty=request.repetition_penalty,
                    **conditioning,
                )
        
        # Save audio file
        audio_path = TEMP_DIR / f"{task_id}.wav"
        audio_np = audio_tensor.cpu().numpy()
        
        if audio_np.ndim > 2:
            audio_np = audio_np[0]  # Take first batch item
        
        save_audio_file(
            torch.from_numpy(audio_np),
            str(audio_path),
            sample_rate=model.audio_tokenizer.sample_rate,
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
                    "length_penalty": request.length_penalty,
                    "early_stopping": request.early_stopping,
                    "method": "beam_search" if request.num_beams > 1 else "sampling" if request.do_sample else "greedy",
                },
                "conditioning": {
                    "genre": request.genre,
                    "mood": request.mood,
                    "tempo": request.tempo,
                },
                "model_info": {
                    "sample_rate": model.audio_tokenizer.sample_rate,
                    "device": str(device),
                }
            },
            "completed_at": asyncio.get_event_loop().time(),
        })
        
        logger.info(f"Successfully generated audio for task {task_id}")
        
    except Exception as e:
        logger.error(f"Failed to generate audio for task {task_id}: {e}")
        tasks[task_id].update({
            "status": "failed",
            "error": str(e),
            "failed_at": asyncio.get_event_loop().time(),
        })


def _compute_quality_score(metrics: Dict[str, float]) -> float:
    """Compute overall quality score from metrics."""
    
    # Simple weighted average of normalized metrics
    weights = {
        "snr_mean": 0.3,
        "harmonic_percussive_ratio": 0.2,
        "tempo_stability": 0.2,
        "pitch_stability": 0.2,
        "diversity": 0.1,
    }
    
    score = 0.0
    total_weight = 0.0
    
    for metric, weight in weights.items():
        if metric in metrics:
            # Normalize metric to 0-1 range (approximate)
            normalized = min(max(metrics[metric] / 10.0, 0.0), 1.0)
            score += normalized * weight
            total_weight += weight
    
    if total_weight > 0:
        score /= total_weight
    
    return score


def start_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    workers: int = 1,
    reload: bool = False,
):
    """Start the FastAPI server."""
    
    uvicorn.run(
        "music_gen.api.main:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    start_server()