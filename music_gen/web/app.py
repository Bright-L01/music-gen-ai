"""
Web UI server for MusicGen.
"""

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

# Get the directory of this file
WEB_DIR = Path(__file__).parent
STATIC_DIR = WEB_DIR / "static"


def create_web_app():
    """Create the web UI FastAPI app."""

    app = FastAPI(
        title="MusicGen Web UI",
        description="Web interface for MusicGen AI",
        version="1.0.0",
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount static files
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def root():
        """Serve the main web UI."""
        index_path = STATIC_DIR / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
        else:
            return HTMLResponse(content="<h1>MusicGen Web UI files not found</h1>", status_code=404)

    @app.get("/health")
    async def health_check():
        """Health check for web UI."""
        return {"status": "healthy", "service": "web_ui"}

    return app


def setup_web_routes(main_app):
    """
    Setup web UI routes on the main API app.
    This allows serving the web UI from the same server as the API.
    """

    # Mount static files
    if STATIC_DIR.exists():
        main_app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
        logger.info(f"Mounted static files from {STATIC_DIR}")
    else:
        logger.warning(f"Static directory not found: {STATIC_DIR}")

    # Add root route for web UI
    @main_app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def serve_web_ui():
        """Serve the main web UI."""
        index_path = STATIC_DIR / "index.html"
        if index_path.exists():
            with open(index_path, "r") as f:
                content = f.read()
            return HTMLResponse(content=content)
        else:
            # Fallback to API documentation
            return HTMLResponse(
                content="""
                <html>
                <head>
                    <title>MusicGen API</title>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 40px; }
                        h1 { color: #333; }
                        .links { margin-top: 20px; }
                        .links a { margin-right: 20px; }
                    </style>
                </head>
                <body>
                    <h1>MusicGen API Server</h1>
                    <p>Welcome to the MusicGen API server. The web UI is currently not available.</p>
                    <div class="links">
                        <a href="/docs">API Documentation</a>
                        <a href="/redoc">ReDoc</a>
                        <a href="/stream/demo">Streaming Demo</a>
                        <a href="/studio">Multi-Track Studio</a>
                    </div>
                </body>
                </html>
                """
            )

    # Add multi-track studio route
    @main_app.get("/studio", response_class=HTMLResponse, include_in_schema=False)
    async def serve_multi_track_studio():
        """Serve the multi-track studio UI."""
        studio_path = STATIC_DIR / "multi_track.html"
        if studio_path.exists():
            with open(studio_path, "r") as f:
                content = f.read()
            return HTMLResponse(content=content)
        else:
            return HTMLResponse(content="<h1>Multi-Track Studio not found</h1>", status_code=404)

    # Add convenience route
    @main_app.get("/index.html", response_class=HTMLResponse, include_in_schema=False)
    async def serve_index():
        """Serve index.html directly."""
        return await serve_web_ui()

    logger.info("Web UI routes configured")


# Standalone web server (optional)
if __name__ == "__main__":
    import uvicorn

    # Create standalone web app
    app = create_web_app()

    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info",
    )
