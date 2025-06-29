"""
API Gateway Service

Central routing hub for all microservices in the music generation platform.
Handles authentication, rate limiting, request routing, and response aggregation.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any

import httpx
from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from prometheus_client import Counter, Histogram, generate_latest

from .models import (
    GenerationRequest,
    BatchGenerationRequest,
    ConversionRequest,
    AnalysisRequest,
    UserCreate,
    UserLogin,
    PlaylistCreate
)
from .auth import get_current_user, verify_token
from .router import ServiceRouter
from .middleware import RateLimitMiddleware, LoggingMiddleware
from .health import HealthChecker


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
request_counter = Counter(
    "gateway_requests_total",
    "Total gateway requests",
    ["method", "endpoint", "service", "status"]
)
request_duration = Histogram(
    "gateway_request_duration_seconds",
    "Gateway request duration",
    ["service", "endpoint"]
)

# Service instances
service_router = ServiceRouter()
health_checker = HealthChecker()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting API Gateway...")
    await service_router.initialize()
    await health_checker.initialize()
    logger.info("API Gateway started successfully")
    
    yield
    
    logger.info("Shutting down API Gateway...")
    await service_router.cleanup()
    await health_checker.cleanup()
    logger.info("API Gateway stopped")


# Create FastAPI app
app = FastAPI(
    title="Music Generation API Gateway",
    description="Central API gateway for the music generation platform",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(RateLimitMiddleware)
app.add_middleware(LoggingMiddleware)


@app.get("/health")
async def health_check():
    """Gateway health check"""
    return {
        "status": "healthy",
        "service": "api-gateway",
        "version": "1.0.0",
        "timestamp": time.time()
    }


@app.get("/health/services")
async def services_health():
    """Check health of all services"""
    try:
        health_status = await health_checker.check_all_services()
        return health_status
        
    except Exception as e:
        logger.error(f"Service health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Health check failed"
        )


# =============================================================================
# AUTH ENDPOINTS (User Management Service)
# =============================================================================

@app.post("/auth/register")
async def register_user(user: UserCreate):
    """Register a new user"""
    request_counter.labels(
        method="POST", endpoint="/auth/register", service="user-management", status="started"
    ).inc()
    
    try:
        with request_duration.labels(service="user-management", endpoint="/auth/register").time():
            response = await service_router.route_to_user_service(
                method="POST",
                path="/register",
                json_data=user.dict()
            )
        
        request_counter.labels(
            method="POST", endpoint="/auth/register", service="user-management", status="success"
        ).inc()
        
        return response
        
    except Exception as e:
        request_counter.labels(
            method="POST", endpoint="/auth/register", service="user-management", status="error"
        ).inc()
        raise


@app.post("/auth/login")
async def login_user(request: Request):
    """Login user"""
    request_counter.labels(
        method="POST", endpoint="/auth/login", service="user-management", status="started"
    ).inc()
    
    try:
        # Get form data
        form_data = await request.form()
        
        with request_duration.labels(service="user-management", endpoint="/auth/login").time():
            response = await service_router.route_to_user_service(
                method="POST",
                path="/login",
                form_data=dict(form_data)
            )
        
        request_counter.labels(
            method="POST", endpoint="/auth/login", service="user-management", status="success"
        ).inc()
        
        return response
        
    except Exception as e:
        request_counter.labels(
            method="POST", endpoint="/auth/login", service="user-management", status="error"
        ).inc()
        raise


@app.get("/auth/me")
async def get_current_user_profile(current_user: dict = Depends(get_current_user)):
    """Get current user profile"""
    try:
        with request_duration.labels(service="user-management", endpoint="/me").time():
            response = await service_router.route_to_user_service(
                method="GET",
                path="/me",
                headers={"Authorization": f"Bearer {current_user.get('token')}"}
            )
        
        return response
        
    except Exception as e:
        logger.error(f"Get user profile error: {e}")
        raise


# =============================================================================
# GENERATION ENDPOINTS (Generation Service)
# =============================================================================

@app.post("/generate")
async def generate_music(
    request: GenerationRequest,
    current_user: dict = Depends(get_current_user)
):
    """Generate music from text prompt"""
    request_counter.labels(
        method="POST", endpoint="/generate", service="generation", status="started"
    ).inc()
    
    try:
        # Add user context to request
        request_data = request.dict()
        request_data["user_id"] = current_user["id"]
        
        with request_duration.labels(service="generation", endpoint="/generate").time():
            response = await service_router.route_to_generation_service(
                method="POST",
                path="/generate",
                json_data=request_data,
                headers={"Authorization": f"Bearer {current_user.get('token')}"}
            )
        
        request_counter.labels(
            method="POST", endpoint="/generate", service="generation", status="success"
        ).inc()
        
        return response
        
    except Exception as e:
        request_counter.labels(
            method="POST", endpoint="/generate", service="generation", status="error"
        ).inc()
        raise


@app.post("/generate/batch")
async def generate_batch(
    request: BatchGenerationRequest,
    current_user: dict = Depends(get_current_user)
):
    """Generate multiple tracks in batch"""
    try:
        # Add user context to each request
        batch_data = request.dict()
        for req in batch_data["requests"]:
            req["user_id"] = current_user["id"]
        
        with request_duration.labels(service="generation", endpoint="/generate/batch").time():
            response = await service_router.route_to_generation_service(
                method="POST",
                path="/generate/batch",
                json_data=batch_data,
                headers={"Authorization": f"Bearer {current_user.get('token')}"}
            )
        
        return response
        
    except Exception as e:
        logger.error(f"Batch generation error: {e}")
        raise


@app.get("/generate/job/{job_id}")
async def get_generation_job(
    job_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get generation job status"""
    try:
        with request_duration.labels(service="generation", endpoint="/job").time():
            response = await service_router.route_to_generation_service(
                method="GET",
                path=f"/job/{job_id}",
                headers={"Authorization": f"Bearer {current_user.get('token')}"}
            )
        
        return response
        
    except Exception as e:
        logger.error(f"Get job status error: {e}")
        raise


# =============================================================================
# AUDIO PROCESSING ENDPOINTS (Audio Processing Service)
# =============================================================================

@app.post("/audio/convert")
async def convert_audio(
    request: ConversionRequest,
    current_user: dict = Depends(get_current_user)
):
    """Convert audio between formats"""
    request_counter.labels(
        method="POST", endpoint="/audio/convert", service="audio-processing", status="started"
    ).inc()
    
    try:
        with request_duration.labels(service="audio-processing", endpoint="/convert").time():
            response = await service_router.route_to_audio_service(
                method="POST",
                path="/convert",
                json_data=request.dict(),
                headers={"Authorization": f"Bearer {current_user.get('token')}"}
            )
        
        request_counter.labels(
            method="POST", endpoint="/audio/convert", service="audio-processing", status="success"
        ).inc()
        
        return response
        
    except Exception as e:
        request_counter.labels(
            method="POST", endpoint="/audio/convert", service="audio-processing", status="error"
        ).inc()
        raise


@app.post("/audio/analyze")
async def analyze_audio(
    request: AnalysisRequest,
    current_user: dict = Depends(get_current_user)
):
    """Analyze audio features"""
    try:
        with request_duration.labels(service="audio-processing", endpoint="/analyze").time():
            response = await service_router.route_to_audio_service(
                method="POST",
                path="/analyze",
                json_data=request.dict(),
                headers={"Authorization": f"Bearer {current_user.get('token')}"}
            )
        
        return response
        
    except Exception as e:
        logger.error(f"Audio analysis error: {e}")
        raise


@app.post("/audio/waveform")
async def generate_waveform(
    audio_url: str,
    width: int = 1920,
    height: int = 200,
    current_user: dict = Depends(get_current_user)
):
    """Generate waveform visualization"""
    try:
        request_data = {
            "audio_url": audio_url,
            "width": width,
            "height": height
        }
        
        response = await service_router.route_to_audio_service(
            method="POST",
            path="/waveform",
            json_data=request_data,
            headers={"Authorization": f"Bearer {current_user.get('token')}"}
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Waveform generation error: {e}")
        raise


@app.post("/audio/upload")
async def upload_audio(request: Request, current_user: dict = Depends(get_current_user)):
    """Upload audio file"""
    try:
        # Stream upload to audio service
        response = await service_router.stream_to_audio_service(
            method="POST",
            path="/upload",
            request=request,
            headers={"Authorization": f"Bearer {current_user.get('token')}"}
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Audio upload error: {e}")
        raise


# =============================================================================
# SOCIAL & PLAYLIST ENDPOINTS (User Management Service)
# =============================================================================

@app.get("/social/profile")
async def get_social_profile(current_user: dict = Depends(get_current_user)):
    """Get user's social profile"""
    try:
        response = await service_router.route_to_user_service(
            method="GET",
            path="/social/profile",
            headers={"Authorization": f"Bearer {current_user.get('token')}"}
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Get social profile error: {e}")
        raise


@app.post("/social/follow")
async def follow_user(
    user_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Follow another user"""
    try:
        response = await service_router.route_to_user_service(
            method="POST",
            path="/social/follow",
            json_data={"user_id": user_id},
            headers={"Authorization": f"Bearer {current_user.get('token')}"}
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Follow user error: {e}")
        raise


@app.post("/playlists")
async def create_playlist(
    playlist: PlaylistCreate,
    current_user: dict = Depends(get_current_user)
):
    """Create a new playlist"""
    try:
        response = await service_router.route_to_user_service(
            method="POST",
            path="/playlists",
            json_data=playlist.dict(),
            headers={"Authorization": f"Bearer {current_user.get('token')}"}
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Create playlist error: {e}")
        raise


@app.get("/playlists")
async def get_user_playlists(current_user: dict = Depends(get_current_user)):
    """Get user's playlists"""
    try:
        response = await service_router.route_to_user_service(
            method="GET",
            path="/playlists",
            headers={"Authorization": f"Bearer {current_user.get('token')}"}
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Get playlists error: {e}")
        raise


# =============================================================================
# AGGREGATED ENDPOINTS (Multiple Services)
# =============================================================================

@app.get("/dashboard")
async def get_dashboard_data(current_user: dict = Depends(get_current_user)):
    """Get aggregated dashboard data from multiple services"""
    try:
        # Fetch data from multiple services concurrently
        auth_header = {"Authorization": f"Bearer {current_user.get('token')}"}
        
        tasks = [
            service_router.route_to_user_service("GET", "/me", headers=auth_header),
            service_router.route_to_user_service("GET", "/social/profile", headers=auth_header),
            service_router.route_to_user_service("GET", "/playlists", headers=auth_header),
            # Could also fetch recent generations, etc.
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Aggregate results
        dashboard_data = {
            "user_profile": results[0] if not isinstance(results[0], Exception) else None,
            "social_profile": results[1] if not isinstance(results[1], Exception) else None,
            "playlists": results[2] if not isinstance(results[2], Exception) else None,
        }
        
        return dashboard_data
        
    except Exception as e:
        logger.error(f"Dashboard data error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not fetch dashboard data"
        )


@app.get("/search")
async def search_content(
    query: str,
    type: str = "all",  # users, playlists, tracks
    limit: int = 20,
    current_user: dict = Depends(get_current_user)
):
    """Search across all content types"""
    try:
        auth_header = {"Authorization": f"Bearer {current_user.get('token')}"}
        
        search_results = {
            "query": query,
            "results": {}
        }
        
        # Search users (if type includes users)
        if type in ["all", "users"]:
            # Would implement user search in user service
            search_results["results"]["users"] = []
            
        # Search playlists
        if type in ["all", "playlists"]:
            # Would implement playlist search
            search_results["results"]["playlists"] = []
            
        # Search tracks (would be in a separate content service)
        if type in ["all", "tracks"]:
            search_results["results"]["tracks"] = []
            
        return search_results
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise


# =============================================================================
# ADMIN & MONITORING ENDPOINTS
# =============================================================================

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return StreamingResponse(
        iter([generate_latest().decode()]),
        media_type="text/plain"
    )


@app.get("/admin/services/stats")
async def get_services_stats(current_user: dict = Depends(get_current_user)):
    """Get statistics from all services (admin only)"""
    # TODO: Add admin role check
    try:
        auth_header = {"Authorization": f"Bearer {current_user.get('token')}"}
        
        # Fetch stats from all services
        tasks = [
            service_router.route_to_generation_service("GET", "/stats", headers=auth_header),
            service_router.route_to_audio_service("GET", "/stats", headers=auth_header),
            service_router.route_to_user_service("GET", "/stats", headers=auth_header),
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            "generation_service": results[0] if not isinstance(results[0], Exception) else None,
            "audio_service": results[1] if not isinstance(results[1], Exception) else None,
            "user_service": results[2] if not isinstance(results[2], Exception) else None,
        }
        
    except Exception as e:
        logger.error(f"Services stats error: {e}")
        raise


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)