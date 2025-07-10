"""
Unified API endpoints that match frontend expectations.

This module provides all endpoints that the frontend expects, with proper
error handling, request/response logging, and OpenAPI documentation.
"""

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from music_gen.core.container import get_container
from music_gen.core.interfaces.repositories import TaskRepository

logger = logging.getLogger(__name__)

# Create main router
router = APIRouter()

# ============================================================================
# Request/Response Models
# ============================================================================


class ApiResponse(BaseModel):
    """Standard API response wrapper."""

    success: bool = Field(..., description="Whether the request was successful")
    data: Any = Field(..., description="Response data")
    message: Optional[str] = Field(None, description="Optional message")
    error: Optional[str] = Field(None, description="Error message if unsuccessful")


class PaginatedResponse(BaseModel):
    """Paginated response model."""

    items: List[Any] = Field(..., description="List of items")
    total: int = Field(..., description="Total number of items")
    page: int = Field(..., description="Current page number")
    limit: int = Field(..., description="Items per page")
    totalPages: int = Field(..., description="Total number of pages")


class GenerationRequest(BaseModel):
    """Music generation request."""

    prompt: str = Field(..., description="Text prompt for music generation")
    genre: Optional[str] = Field(None, description="Musical genre")
    mood: Optional[str] = Field(None, description="Musical mood")
    duration: float = Field(10.0, ge=1.0, le=60.0, description="Duration in seconds")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Additional parameters")

    # Advanced parameters
    temperature: float = Field(1.0, ge=0.1, le=2.0)
    top_k: int = Field(50, ge=1, le=100)
    top_p: float = Field(0.9, ge=0.1, le=1.0)
    seed: Optional[int] = Field(None)


class GenerationResponse(BaseModel):
    """Music generation response."""

    id: str = Field(..., description="Generation task ID")
    status: str = Field(..., description="Task status: pending, processing, completed, failed")
    progress: float = Field(0.0, ge=0.0, le=1.0, description="Progress (0-1)")
    audioUrl: Optional[str] = Field(None, description="URL to download audio when completed")
    waveformData: Optional[List[float]] = Field(None, description="Waveform visualization data")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Generation metadata")
    error: Optional[str] = Field(None, description="Error message if failed")


class UserProfile(BaseModel):
    """User profile model."""

    id: str
    email: str
    name: str
    username: str
    avatar: Optional[str] = None
    bio: Optional[str] = None
    isFollowing: bool = False
    stats: Dict[str, int] = Field(
        default_factory=lambda: {"followers": 0, "following": 0, "tracks": 0, "likes": 0}
    )
    createdAt: str
    preferences: Dict[str, Any] = Field(default_factory=dict)
    subscription: Dict[str, Any] = Field(
        default_factory=lambda: {
            "tier": "free",
            "expiresAt": None,
            "features": ["basic_generation", "community_access"],
        }
    )


class Track(BaseModel):
    """Track model."""

    id: str
    title: str
    description: Optional[str] = None
    genre: str
    duration: float
    audioUrl: str
    waveformData: List[float]
    isPublic: bool = True
    tags: List[str] = Field(default_factory=list)
    user: UserProfile
    stats: Dict[str, int] = Field(
        default_factory=lambda: {"plays": 0, "likes": 0, "comments": 0, "shares": 0}
    )
    isLiked: bool = False
    createdAt: str
    updatedAt: str


# ============================================================================
# Middleware for Request/Response Logging
# ============================================================================


@router.middleware("http")
async def log_requests(request, call_next):
    """Log all API requests and responses."""
    start_time = datetime.utcnow()

    # Log request
    logger.info(f"API Request: {request.method} {request.url.path}")

    # Process request
    response = await call_next(request)

    # Calculate duration
    duration = (datetime.utcnow() - start_time).total_seconds() * 1000

    # Log response
    logger.info(
        f"API Response: {request.method} {request.url.path} - "
        f"Status: {response.status_code} - Duration: {duration:.2f}ms"
    )

    return response


# ============================================================================
# Generation Endpoints
# ============================================================================


@router.post("/api/v1/generate", response_model=ApiResponse)
async def generate_music(
    request: GenerationRequest,
    task_repository: TaskRepository = Depends(lambda: get_container().get(TaskRepository)),
):
    """
    Start music generation from text prompt.

    Returns a task ID that can be used to track progress.
    """
    try:
        # Create task
        task_id = str(uuid.uuid4())

        task_data = {
            "status": "pending",
            "request": request.dict(),
            "created_at": datetime.utcnow().isoformat(),
        }

        await task_repository.create_task(task_id, task_data)

        # TODO: Dispatch to Celery if enabled
        # For now, return immediately

        response = GenerationResponse(
            id=task_id,
            status="pending",
            progress=0.0,
        )

        return ApiResponse(
            success=True, data=response.dict(), message="Generation started successfully"
        )

    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/v1/generate/{task_id}", response_model=ApiResponse)
async def get_generation_status(
    task_id: str,
    task_repository: TaskRepository = Depends(lambda: get_container().get(TaskRepository)),
):
    """Get the status of a generation task."""
    try:
        task = await task_repository.get_task(task_id)

        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        # Map task data to response
        response = GenerationResponse(
            id=task_id,
            status=task.get("status", "unknown"),
            progress=task.get("progress", 0.0),
            audioUrl=(
                f"/api/v1/generate/{task_id}/download"
                if task.get("status") == "completed"
                else None
            ),
            waveformData=task.get("waveform_data"),
            metadata=task.get("metadata"),
            error=task.get("error"),
        )

        return ApiResponse(success=True, data=response.dict())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get task status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/v1/generate/{task_id}", response_model=ApiResponse)
async def cancel_generation(
    task_id: str,
    task_repository: TaskRepository = Depends(lambda: get_container().get(TaskRepository)),
):
    """Cancel a generation task."""
    try:
        task = await task_repository.get_task(task_id)

        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        if task.get("status") in ["completed", "failed"]:
            raise HTTPException(
                status_code=400, detail=f"Cannot cancel task in {task['status']} status"
            )

        # Update task status
        await task_repository.update_task(
            task_id,
            {
                "status": "cancelled",
                "cancelled_at": datetime.utcnow().isoformat(),
            },
        )

        return ApiResponse(
            success=True,
            data={"id": task_id, "status": "cancelled"},
            message="Task cancelled successfully",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel task: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/v1/generate/history", response_model=ApiResponse)
async def get_generation_history(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    task_repository: TaskRepository = Depends(lambda: get_container().get(TaskRepository)),
):
    """Get user's generation history."""
    try:
        # Get tasks from repository
        offset = (page - 1) * limit
        tasks = await task_repository.list_tasks(status="completed", limit=limit, offset=offset)

        # Count total
        all_tasks = await task_repository.list_tasks(status="completed", limit=10000)
        total = len(all_tasks)

        # Map to response format
        items = []
        for task in tasks:
            task_id = task.get("id", str(uuid.uuid4()))
            items.append(
                GenerationResponse(
                    id=task_id,
                    status="completed",
                    progress=1.0,
                    audioUrl=f"/api/v1/generate/{task_id}/download",
                    waveformData=task.get("waveform_data"),
                    metadata=task.get("metadata"),
                ).dict()
            )

        response = PaginatedResponse(
            items=items,
            total=total,
            page=page,
            limit=limit,
            totalPages=(total + limit - 1) // limit,
        )

        return ApiResponse(success=True, data=response.dict())

    except Exception as e:
        logger.error(f"Failed to get history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/v1/generate/{task_id}/save", response_model=ApiResponse)
async def save_generation(
    task_id: str,
    metadata: Dict[str, Any] = Body(default={}),
    task_repository: TaskRepository = Depends(lambda: get_container().get(TaskRepository)),
):
    """Save generation metadata."""
    try:
        task = await task_repository.get_task(task_id)

        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        # Update task with metadata
        await task_repository.update_task(
            task_id,
            {
                "saved": True,
                "saved_at": datetime.utcnow().isoformat(),
                "user_metadata": metadata,
            },
        )

        return ApiResponse(
            success=True,
            data={"id": task_id, "saved": True},
            message="Generation saved successfully",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to save generation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/v1/generate/{task_id}/download")
async def download_generated_audio(
    task_id: str,
    task_repository: TaskRepository = Depends(lambda: get_container().get(TaskRepository)),
):
    """Download generated audio file."""
    try:
        task = await task_repository.get_task(task_id)

        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        if task.get("status") != "completed":
            raise HTTPException(status_code=400, detail="Generation not completed")

        # Get audio file path
        audio_path = task.get("audio_path")
        if not audio_path:
            # For testing, return a placeholder
            raise HTTPException(status_code=404, detail="Audio file not found")

        # Check if file exists
        if not Path(audio_path).exists():
            raise HTTPException(status_code=404, detail="Audio file not found")

        return FileResponse(audio_path, media_type="audio/wav", filename=f"musicgen_{task_id}.wav")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download audio: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# User Endpoints
# ============================================================================


@router.get("/api/v1/user/profile", response_model=ApiResponse)
async def get_user_profile():
    """Get current user's profile."""
    # Mock user profile for testing
    profile = UserProfile(
        id="user-123",
        email="user@example.com",
        name="Test User",
        username="testuser",
        createdAt=datetime.utcnow().isoformat(),
    )

    return ApiResponse(success=True, data=profile.dict())


@router.put("/api/v1/user/profile", response_model=ApiResponse)
async def update_user_profile(updates: Dict[str, Any]):
    """Update user profile."""
    # Mock update
    profile = UserProfile(
        id="user-123",
        email="user@example.com",
        name=updates.get("name", "Test User"),
        username=updates.get("username", "testuser"),
        bio=updates.get("bio"),
        createdAt=datetime.utcnow().isoformat(),
    )

    return ApiResponse(success=True, data=profile.dict(), message="Profile updated successfully")


# ============================================================================
# Track Endpoints
# ============================================================================


@router.get("/api/v1/tracks/trending", response_model=ApiResponse)
async def get_trending_tracks(limit: int = Query(20, ge=1, le=100)):
    """Get trending tracks."""
    # Mock trending tracks
    tracks = []
    for i in range(min(limit, 5)):
        track = Track(
            id=f"track-{i}",
            title=f"Trending Track {i + 1}",
            genre="Electronic",
            duration=180.0,
            audioUrl=f"/audio/track-{i}.wav",
            waveformData=[0.5] * 100,
            user=UserProfile(
                id=f"user-{i}",
                email=f"artist{i}@example.com",
                name=f"Artist {i}",
                username=f"artist{i}",
                createdAt=datetime.utcnow().isoformat(),
            ),
            createdAt=datetime.utcnow().isoformat(),
            updatedAt=datetime.utcnow().isoformat(),
        )
        tracks.append(track.dict())

    return ApiResponse(success=True, data=tracks)


@router.get("/api/v1/tracks/recent", response_model=ApiResponse)
async def get_recent_tracks(page: int = Query(1, ge=1), limit: int = Query(20, ge=1, le=100)):
    """Get recent tracks."""
    # Mock recent tracks
    items = []
    total = 50  # Mock total

    for i in range(min(limit, 10)):
        track = Track(
            id=f"track-recent-{i}",
            title=f"Recent Track {i + 1}",
            genre="Various",
            duration=240.0,
            audioUrl=f"/audio/track-recent-{i}.wav",
            waveformData=[0.3] * 100,
            user=UserProfile(
                id=f"user-{i}",
                email=f"creator{i}@example.com",
                name=f"Creator {i}",
                username=f"creator{i}",
                createdAt=datetime.utcnow().isoformat(),
            ),
            createdAt=datetime.utcnow().isoformat(),
            updatedAt=datetime.utcnow().isoformat(),
        )
        items.append(track.dict())

    response = PaginatedResponse(
        items=items, total=total, page=page, limit=limit, totalPages=(total + limit - 1) // limit
    )

    return ApiResponse(success=True, data=response.dict())


@router.get("/api/v1/tracks/search", response_model=ApiResponse)
async def search_tracks(
    q: str = Query(..., description="Search query"),
    genre: Optional[str] = Query(None),
    user: Optional[str] = Query(None),
    tags: Optional[List[str]] = Query(None),
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
):
    """Search tracks."""
    # Mock search results
    items = []
    total = 15  # Mock total

    for i in range(min(limit, 5)):
        track = Track(
            id=f"track-search-{i}",
            title=f"Search Result: {q} - Track {i + 1}",
            genre=genre or "Mixed",
            duration=200.0,
            audioUrl=f"/audio/track-search-{i}.wav",
            waveformData=[0.4] * 100,
            tags=tags or ["search", "result"],
            user=UserProfile(
                id=f"user-search-{i}",
                email=f"searcher{i}@example.com",
                name=f"User {i}",
                username=f"user{i}",
                createdAt=datetime.utcnow().isoformat(),
            ),
            createdAt=datetime.utcnow().isoformat(),
            updatedAt=datetime.utcnow().isoformat(),
        )
        items.append(track.dict())

    response = PaginatedResponse(
        items=items, total=total, page=page, limit=limit, totalPages=(total + limit - 1) // limit
    )

    return ApiResponse(success=True, data=response.dict())


# ============================================================================
# Community Endpoints
# ============================================================================


@router.get("/api/v1/community/stats", response_model=ApiResponse)
async def get_community_stats():
    """Get community statistics."""
    stats = {
        "totalUsers": 15234,
        "totalTracks": 48291,
        "totalPlays": 1829472,
        "totalMinutes": 3847293,
    }

    return ApiResponse(success=True, data=stats)


@router.get("/api/v1/community/featured-users", response_model=ApiResponse)
async def get_featured_users(limit: int = Query(10, ge=1, le=50)):
    """Get featured users."""
    users = []

    for i in range(min(limit, 5)):
        user = UserProfile(
            id=f"featured-user-{i}",
            email=f"featured{i}@example.com",
            name=f"Featured Artist {i}",
            username=f"featured{i}",
            stats={
                "followers": 1000 + i * 500,
                "following": 50 + i * 10,
                "tracks": 20 + i * 5,
                "likes": 500 + i * 100,
            },
            createdAt=datetime.utcnow().isoformat(),
        )
        users.append(user.dict())

    return ApiResponse(success=True, data=users)


@router.get("/api/v1/community/trending-topics", response_model=ApiResponse)
async def get_trending_topics():
    """Get trending topics/tags."""
    topics = [
        {"tag": "lofi", "count": 4821},
        {"tag": "ambient", "count": 3472},
        {"tag": "electronic", "count": 2981},
        {"tag": "chillout", "count": 2156},
        {"tag": "experimental", "count": 1893},
    ]

    return ApiResponse(success=True, data=topics)


# ============================================================================
# Analytics Endpoints
# ============================================================================


@router.post("/api/v1/analytics/track", response_model=ApiResponse)
async def track_analytics_event(
    event: str = Body(...),
    properties: Optional[Dict[str, Any]] = Body(None),
    timestamp: Optional[int] = Body(None),
):
    """Track analytics event."""
    logger.info(f"Analytics event: {event} - {properties}")

    return ApiResponse(success=True, data={"tracked": True}, message="Event tracked successfully")


# ============================================================================
# Models Endpoint
# ============================================================================


@router.get("/api/v1/models", response_model=ApiResponse)
async def list_available_models():
    """List available music generation models."""
    models = [
        {
            "id": "musicgen-small",
            "name": "MusicGen Small",
            "description": "Fast generation, good quality",
            "parameters": 300_000_000,
            "maxDuration": 30,
        },
        {
            "id": "musicgen-medium",
            "name": "MusicGen Medium",
            "description": "Balanced speed and quality",
            "parameters": 1_500_000_000,
            "maxDuration": 60,
        },
        {
            "id": "musicgen-large",
            "name": "MusicGen Large",
            "description": "Best quality, slower generation",
            "parameters": 3_300_000_000,
            "maxDuration": 120,
        },
    ]

    return ApiResponse(success=True, data=models)


# ============================================================================
# Error Handling
# ============================================================================


@router.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Handle HTTP exceptions with consistent format."""
    return ApiResponse(
        success=False, data=None, error=exc.detail, message=f"Request failed: {exc.detail}"
    )


@router.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)

    return ApiResponse(
        success=False,
        data=None,
        error="Internal server error",
        message="An unexpected error occurred",
    )
