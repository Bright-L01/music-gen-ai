"""
Minimal API for musicgen-minimal.
Simple FastAPI server with just the essentials - no authentication, no complex middleware.
"""

import os
import tempfile
import time
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import uvicorn

from .generator import MusicGenerator

# Global generator instance (loaded once)
generator = None


# Request/Response models
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Text description of music to generate")
    duration: float = Field(10.0, ge=0.1, le=120.0, description="Duration in seconds")
    temperature: float = Field(1.0, ge=0.1, le=2.0, description="Sampling temperature")
    guidance_scale: float = Field(3.0, ge=1.0, le=10.0, description="Guidance scale")
    model: str = Field("small", description="Model size (small, medium, large)")


class GenerateResponse(BaseModel):
    success: bool
    audio_url: str
    duration: float
    sample_rate: int
    generation_time: float
    model_used: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    version: str


# Create FastAPI app
app = FastAPI(
    title="MusicGen Minimal API",
    description="Simple text-to-music generation API that actually works",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


def get_generator(model_name: str = "facebook/musicgen-small") -> MusicGenerator:
    """Get or create generator instance."""
    global generator

    if generator is None or generator.model_name != model_name:
        generator = MusicGenerator(model_name)

    return generator


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=generator is not None,
        device=generator.device if generator else "unknown",
        version="0.1.0",
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate_music(request: GenerateRequest, background_tasks: BackgroundTasks):
    """Generate music from text prompt."""

    # Model name mapping
    model_names = {
        "small": "facebook/musicgen-small",
        "medium": "facebook/musicgen-medium",
        "large": "facebook/musicgen-large",
    }

    if request.model not in model_names:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model. Must be one of: {', '.join(model_names.keys())}",
        )

    model_name = model_names[request.model]

    try:
        # Get generator
        gen = get_generator(model_name)

        # Generate music
        start_time = time.time()
        audio, sample_rate = gen.generate(
            request.prompt,
            duration=request.duration,
            temperature=request.temperature,
            guidance_scale=request.guidance_scale,
        )
        generation_time = time.time() - start_time

        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False, dir=tempfile.gettempdir()
        )
        temp_path = temp_file.name
        temp_file.close()

        gen.save_audio(audio, sample_rate, temp_path)

        # Schedule cleanup
        def cleanup():
            try:
                os.unlink(temp_path)
            except:
                pass

        # Clean up after 1 hour
        background_tasks.add_task(cleanup)

        # Return download URL
        filename = os.path.basename(temp_path)
        actual_duration = len(audio) / sample_rate

        return GenerateResponse(
            success=True,
            audio_url=f"/download/{filename}",
            duration=actual_duration,
            sample_rate=sample_rate,
            generation_time=generation_time,
            model_used=request.model,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.get("/download/{filename}")
async def download_audio(filename: str):
    """Download generated audio file."""

    # Security: only allow downloading from temp directory
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, filename)

    # Validate filename (basic security)
    if not filename.endswith(".wav") or "/" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        file_path,
        media_type="audio/wav",
        filename=f"generated_music_{filename}",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/models")
async def list_models():
    """List available models."""
    return {
        "models": [
            {
                "name": "small",
                "full_name": "facebook/musicgen-small",
                "description": "300M parameters, fastest generation",
                "memory_gb": 2,
            },
            {
                "name": "medium",
                "full_name": "facebook/musicgen-medium",
                "description": "1.5B parameters, better quality",
                "memory_gb": 6,
            },
            {
                "name": "large",
                "full_name": "facebook/musicgen-large",
                "description": "3.3B parameters, best quality",
                "memory_gb": 12,
            },
        ]
    }


@app.get("/")
async def root():
    """Root endpoint with basic info."""
    return {
        "name": "MusicGen Minimal API",
        "version": "0.1.0",
        "description": "Simple text-to-music generation API",
        "endpoints": {
            "POST /generate": "Generate music from text",
            "GET /health": "Health check",
            "GET /models": "List available models",
            "GET /docs": "API documentation",
        },
        "example": {
            "curl": 'curl -X POST http://localhost:8000/generate -H \'Content-Type: application/json\' -d \'{"prompt": "upbeat jazz piano", "duration": 10}\''
        },
    }


def main(host: str = "0.0.0.0", port: int = 8000, workers: int = 1):
    """Run the API server."""
    uvicorn.run("musicgen.api:app", host=host, port=port, workers=workers, reload=False)


if __name__ == "__main__":
    import typer

    typer.run(main)
