"""
Real MusicGen API implementation using pre-trained models.

This module provides FastAPI endpoints for the production-ready
MusicGen implementation.
"""

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..inference.real_multi_instrument import InstrumentConfig, RealMultiInstrumentGenerator


# Pydantic models for API
class SimpleGenerationRequest(BaseModel):
    """Request for simple single-prompt generation."""

    prompt: str = Field(..., description="Text description of the music")
    duration: float = Field(10.0, ge=1.0, le=60.0, description="Duration in seconds")
    temperature: float = Field(0.8, ge=0.1, le=2.0, description="Sampling temperature")
    guidance_scale: float = Field(3.0, ge=1.0, le=5.0, description="Guidance scale")


class InstrumentRequest(BaseModel):
    """Configuration for a single instrument in multi-track generation."""

    name: str = Field(..., description="Instrument name")
    volume: float = Field(0.7, ge=0.0, le=1.0, description="Volume level")
    pan: float = Field(0.0, ge=-1.0, le=1.0, description="Stereo pan")
    custom_prompt: Optional[str] = Field(None, description="Override default prompt template")


class MultiTrackGenerationRequest(BaseModel):
    """Request for multi-instrument generation."""

    style: str = Field(..., description="Musical style (jazz, rock, classical, etc.)")
    mood: str = Field(..., description="Mood description")
    instruments: List[InstrumentRequest] = Field(..., description="List of instruments")
    duration: float = Field(15.0, ge=1.0, le=60.0, description="Duration in seconds")
    temperature: float = Field(0.8, ge=0.1, le=2.0, description="Sampling temperature")
    guidance_scale: float = Field(3.0, ge=1.0, le=5.0, description="Guidance scale")
    master_volume: float = Field(0.8, ge=0.0, le=1.0, description="Master volume")
    save_stems: bool = Field(True, description="Save individual instrument tracks")


class GenerationStatus(BaseModel):
    """Status of a generation task."""

    task_id: str
    status: str  # "pending", "processing", "completed", "failed"
    progress: float  # 0.0 to 1.0
    message: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# Global storage for tasks (in production, use Redis or database)
tasks_db: Dict[str, GenerationStatus] = {}


def setup_musicgen_api(app: FastAPI, output_dir: str = "api_outputs"):
    """Setup MusicGen API routes."""

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Initialize generator (shared across requests)
    generator = RealMultiInstrumentGenerator(
        model_name="facebook/musicgen-small", cache_dir=str(output_path / "model_cache")
    )

    @app.get("/")
    async def root():
        """API root endpoint."""
        return {
            "service": "MusicGen API",
            "version": "1.0.0",
            "endpoints": {
                "simple_generation": "/generate/simple",
                "multi_track_generation": "/generate/multi-track",
                "task_status": "/status/{task_id}",
                "list_instruments": "/instruments",
                "health": "/health",
            },
        }

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "model_loaded": generator.model is not None,
            "device": generator.device,
        }

    @app.get("/instruments")
    async def list_instruments():
        """List available instruments."""
        return {
            "instruments": list(generator.instrument_configs.keys()),
            "categories": {
                "keyboards": ["piano", "electric_piano"],
                "guitars": ["acoustic_guitar", "electric_guitar", "bass"],
                "drums": ["drums"],
                "strings": ["violin", "cello"],
                "brass": ["trumpet", "saxophone"],
                "synth": ["synthesizer"],
            },
        }

    @app.post("/generate/simple")
    async def generate_simple(request: SimpleGenerationRequest, background_tasks: BackgroundTasks):
        """Generate music from a simple text prompt."""
        task_id = str(uuid.uuid4())

        # Initialize task status
        tasks_db[task_id] = GenerationStatus(
            task_id=task_id, status="pending", progress=0.0, message="Task queued"
        )

        # Start background task
        background_tasks.add_task(generate_simple_task, task_id, request, generator, output_path)

        return {"task_id": task_id, "status": "processing"}

    @app.post("/generate/multi-track")
    async def generate_multi_track(
        request: MultiTrackGenerationRequest, background_tasks: BackgroundTasks
    ):
        """Generate multi-instrument music."""
        task_id = str(uuid.uuid4())

        # Initialize task status
        tasks_db[task_id] = GenerationStatus(
            task_id=task_id, status="pending", progress=0.0, message="Task queued"
        )

        # Start background task
        background_tasks.add_task(
            generate_multi_track_task, task_id, request, generator, output_path
        )

        return {"task_id": task_id, "status": "processing"}

    @app.get("/status/{task_id}")
    async def get_status(task_id: str):
        """Get status of a generation task."""
        if task_id not in tasks_db:
            raise HTTPException(status_code=404, detail="Task not found")

        return tasks_db[task_id]

    @app.get("/download/{task_id}/{file_type}")
    async def download_result(task_id: str, file_type: str):
        """Download generated files."""
        if task_id not in tasks_db:
            raise HTTPException(status_code=404, detail="Task not found")

        task = tasks_db[task_id]
        if task.status != "completed":
            raise HTTPException(status_code=400, detail="Task not completed")

        if not task.result or "files" not in task.result:
            raise HTTPException(status_code=404, detail="No files available")

        files = task.result["files"]

        if file_type == "mixed":
            file_path = files.get("mixed")
        elif file_type == "metadata":
            file_path = files.get("metadata")
        elif file_type.startswith("stem_"):
            instrument = file_type.replace("stem_", "")
            file_path = files.get(f"stem_{instrument}")
        else:
            raise HTTPException(status_code=400, detail="Invalid file type")

        if not file_path or not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        return FileResponse(
            file_path,
            media_type="audio/wav" if file_path.endswith(".wav") else "application/json",
            filename=os.path.basename(file_path),
        )


async def generate_simple_task(
    task_id: str,
    request: SimpleGenerationRequest,
    generator: RealMultiInstrumentGenerator,
    output_dir: Path,
):
    """Background task for simple generation."""
    try:
        # Update status
        tasks_db[task_id].status = "processing"
        tasks_db[task_id].message = "Generating audio..."
        tasks_db[task_id].progress = 0.1

        # Generate audio
        time.time()
        audio, gen_time = generator.generate_single_track(
            prompt=request.prompt,
            duration=request.duration,
            temperature=request.temperature,
            guidance_scale=request.guidance_scale,
        )

        tasks_db[task_id].progress = 0.8

        # Save audio
        task_dir = output_dir / task_id
        task_dir.mkdir(exist_ok=True)

        audio_path = task_dir / "output.wav"
        import scipy.io.wavfile

        audio_int16 = (audio * 32767).astype("int16")
        scipy.io.wavfile.write(str(audio_path), 32000, audio_int16)

        # Save metadata
        metadata = {
            "prompt": request.prompt,
            "duration": request.duration,
            "temperature": request.temperature,
            "guidance_scale": request.guidance_scale,
            "generation_time": gen_time,
            "audio_shape": audio.shape,
            "sample_rate": 32000,
        }

        metadata_path = task_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Update task status
        tasks_db[task_id].status = "completed"
        tasks_db[task_id].progress = 1.0
        tasks_db[task_id].message = "Generation completed"
        tasks_db[task_id].result = {
            "files": {"mixed": str(audio_path), "metadata": str(metadata_path)},
            "duration": len(audio) / 32000,
            "generation_time": gen_time,
        }

    except Exception as e:
        tasks_db[task_id].status = "failed"
        tasks_db[task_id].error = str(e)
        tasks_db[task_id].message = "Generation failed"


async def generate_multi_track_task(
    task_id: str,
    request: MultiTrackGenerationRequest,
    generator: RealMultiInstrumentGenerator,
    output_dir: Path,
):
    """Background task for multi-track generation."""
    try:
        # Update status
        tasks_db[task_id].status = "processing"
        tasks_db[task_id].message = "Starting multi-track generation..."
        tasks_db[task_id].progress = 0.1

        # Create instrument configs
        instrument_configs = {}
        instrument_names = []

        for inst_req in request.instruments:
            instrument_names.append(inst_req.name)

            if inst_req.custom_prompt:
                # Use custom prompt template
                instrument_configs[inst_req.name] = InstrumentConfig(
                    name=inst_req.name,
                    prompt_template=inst_req.custom_prompt,
                    volume=inst_req.volume,
                    pan=inst_req.pan,
                )

        # Generate multi-track
        total_instruments = len(instrument_names)

        def update_progress(current: int):
            progress = 0.1 + (0.8 * current / total_instruments)
            tasks_db[task_id].progress = progress
            tasks_db[task_id].message = f"Generating instrument {current}/{total_instruments}"

        # Generate
        result = generator.generate_multi_track(
            style=request.style,
            mood=request.mood,
            instruments=instrument_names,
            duration=request.duration,
            instrument_configs=instrument_configs if instrument_configs else None,
            temperature=request.temperature,
            guidance_scale=request.guidance_scale,
            master_volume=request.master_volume,
        )

        tasks_db[task_id].progress = 0.9
        tasks_db[task_id].message = "Saving files..."

        # Save results
        task_dir = output_dir / task_id
        saved_files = generator.save_result(result, task_dir, save_stems=request.save_stems)

        # Update task status
        tasks_db[task_id].status = "completed"
        tasks_db[task_id].progress = 1.0
        tasks_db[task_id].message = "Multi-track generation completed"
        tasks_db[task_id].result = {
            "files": saved_files,
            "instruments": instrument_names,
            "duration": result.duration,
            "total_generation_time": sum(result.generation_times.values()),
            "prompts_used": result.prompts_used,
        }

    except Exception as e:
        tasks_db[task_id].status = "failed"
        tasks_db[task_id].error = str(e)
        tasks_db[task_id].message = "Generation failed"


# Example usage
if __name__ == "__main__":
    import uvicorn

    app = FastAPI(title="MusicGen API", version="1.0.0")
    setup_musicgen_api(app)

    print("Starting MusicGen API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
