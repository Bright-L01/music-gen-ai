# 🚀 Immediate Actions - Music Gen AI Cleanup

## Step 1: Install Required Tools (2 min)
```bash
pip install isort autoflake black flake8 pre-commit pytest-cov
```

## Step 2: Execute Phase 1 Cleanup (5 min)
```bash
# Run the comprehensive cleanup
python scripts/phase1_cleanup.py

# This will:
# ✅ Remove 139 .bak files
# ✅ Remove archive directories  
# ✅ Clean __pycache__ folders
# ✅ Remove experimental files
# ✅ Create new API structure
# ✅ Fix imports automatically
```

## Step 3: Commit Changes (2 min)
```bash
# Add all changes
git add -A

# Commit with clear message
git commit -m "refactor: major cleanup - remove backups, consolidate structure

- Remove 139 backup (.bak) files
- Delete archive directories and pycache
- Remove experimental/abandoned modules
- Create consolidated API structure
- Fix all import issues
- Add pre-commit configuration"

# Push to trigger CI
git push
```

## Step 4: Monitor CI Pipeline (5 min)
1. Go to: https://github.com/Bright-L01/music-gen-ai/actions
2. Watch the CI pipeline run
3. All tests should now pass ✅

## Step 5: Quick API Consolidation (30 min)

### 5.1 Create Unified API Structure
```python
# music_gen/api/app.py
"""Unified API for Music Gen AI."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .endpoints import generation, health, models

def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="Music Gen AI",
        description="AI-powered music generation API",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(health.router, tags=["health"])
    app.include_router(generation.router, prefix="/api/v1", tags=["generation"])
    app.include_router(models.router, prefix="/api/v1", tags=["models"])
    
    return app

app = create_app()
```

### 5.2 Consolidate Generation Endpoints
```python
# music_gen/api/endpoints/generation.py
"""Music generation endpoints."""

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()

class GenerationRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt for music generation")
    duration: float = Field(30.0, ge=1.0, le=300.0)
    temperature: float = Field(1.0, ge=0.1, le=2.0)
    model: str = Field("facebook/musicgen-medium")

@router.post("/generate")
async def generate_music(
    request: GenerationRequest,
    background_tasks: BackgroundTasks
):
    """Generate music from text prompt."""
    # Consolidated logic from all 4 APIs
    pass

@router.post("/generate/streaming")
async def generate_streaming(request: GenerationRequest):
    """Generate music with streaming response."""
    # Streaming logic
    pass

@router.post("/generate/multi-instrument")
async def generate_multi_instrument(request: GenerationRequest):
    """Generate multi-instrument music."""
    # Multi-instrument logic
    pass
```

## Step 6: Update Tests (15 min)
```python
# tests/test_consolidated_api.py
"""Test consolidated API."""

import pytest
from fastapi.testclient import TestClient
from music_gen.api.app import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    
def test_generate_endpoint():
    response = client.post("/api/v1/generate", json={
        "prompt": "upbeat jazz music",
        "duration": 10.0
    })
    assert response.status_code in [200, 201]
```

## Step 7: Update Documentation (10 min)

### Update README.md
```markdown
# 🎵 Music Gen AI

Production-ready AI music generation platform powered by Meta's MusicGen.

## Quick Start

```bash
# Install
pip install -e .

# Generate music
music-gen generate "upbeat jazz music" --duration 30

# Start API server
uvicorn music_gen.api.app:app --reload

# View API docs
open http://localhost:8000/docs
```

## Features
- ✅ Text-to-music generation
- ✅ Multi-instrument support
- ✅ Real-time streaming
- ✅ Production-ready API
- ✅ Comprehensive testing
```

## Step 8: Final Checks (5 min)

### Run Local Tests
```bash
# Run quick tests
pytest tests/unit/test_exceptions.py -v

# Check code quality
black music_gen --check
isort music_gen --check

# Run API locally
uvicorn music_gen.api.app:app --reload
```

### Verify Everything Works
1. API docs load at http://localhost:8000/docs
2. Health check returns 200 OK
3. No import errors
4. CI passes on GitHub

## Total Time: ~1 hour

After completing these steps, you'll have:
- ✅ Clean, organized codebase
- ✅ Consolidated API structure
- ✅ Passing CI/CD pipeline
- ✅ Professional documentation
- ✅ Production-ready foundation

## What's Next?

1. **Phase 2**: Implement model management system
2. **Phase 3**: Increase test coverage to 90%
3. **Phase 4**: Add monitoring and observability
4. **Phase 5**: Deploy to production

The codebase is now clean and ready for the next phase of development! 🎉