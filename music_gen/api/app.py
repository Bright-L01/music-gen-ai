"""
Unified Music Gen AI API - Consolidated from multiple API implementations.
"""

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .endpoints import generation, health, models, streaming
from .middleware import monitoring, rate_limiting


def create_app(
    title: str = "Music Gen AI",
    version: str = "1.0.0",
    static_dir: Optional[Path] = None,
) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        title: API title
        version: API version
        static_dir: Directory for static files (web UI)

    Returns:
        Configured FastAPI application
    """

    # Create FastAPI instance
    app = FastAPI(
        title=title,
        description="Production-ready AI music generation API",
        version=version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add custom middleware
    app.add_middleware(monitoring.MetricsMiddleware)
    app.add_middleware(rate_limiting.RateLimitMiddleware)

    # Include routers
    app.include_router(health.router, prefix="/health", tags=["health"])
    app.include_router(generation.router, prefix="/api/v1/generate", tags=["generation"])
    app.include_router(streaming.router, prefix="/api/v1/stream", tags=["streaming"])
    app.include_router(models.router, prefix="/api/v1/models", tags=["models"])

    # Mount static files for web UI if directory provided
    if static_dir and static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        # Serve index.html at root
        @app.get("/")
        async def serve_ui():
            """Serve the web UI."""
            index_path = static_dir / "index.html"
            if index_path.exists():
                return FileResponse(str(index_path))
            return {"message": "Music Gen AI API", "docs": "/docs"}

    else:

        @app.get("/")
        async def root():
            """API root endpoint."""
            return {
                "message": "Music Gen AI API",
                "version": version,
                "docs": "/docs",
                "health": "/health",
            }

    # Add download endpoint
    @app.get("/download/{task_id}")
    async def download_audio(task_id: str):
        """Download generated audio file."""
        from .endpoints.generation import tasks

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

    # Startup event
    @app.on_event("startup")
    async def startup_event():
        """Initialize services on startup."""
        from ..core.model_manager import ModelManager

        # Initialize model manager
        model_manager = ModelManager()

        # Pre-load default model
        default_model = os.getenv("DEFAULT_MODEL", "facebook/musicgen-small")
        try:
            model_manager.get_model(default_model)
            print(f"✓ Pre-loaded model: {default_model}")
        except Exception as e:
            print(f"⚠ Failed to pre-load model: {e}")

    # Shutdown event
    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on shutdown."""
        from ..core.model_manager import ModelManager

        # Clear model cache
        model_manager = ModelManager()
        model_manager.clear_cache()

        # Cleanup temp files
        temp_dir = Path("/tmp/musicgen")
        if temp_dir.exists():
            for file in temp_dir.glob("*.wav"):
                try:
                    file.unlink()
                except:
                    pass

    return app


# Create default app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    # Run development server
    uvicorn.run(
        "music_gen.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
