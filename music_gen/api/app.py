"""
Unified Music Gen AI API - Consolidated from multiple API implementations.
"""

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .cors_config import cors_config, get_cors_config
from .endpoints import auth, generation, health, models, streaming, unified
from .endpoints import generation_optimized  # Import optimized endpoints
from .middleware import monitoring, rate_limiting
from .middleware.csrf import CSRFProtectionMiddleware
from .middleware.performance import (
    PerformanceMiddleware,
    LazyLoadingMiddleware,
    DatabaseOptimizationMiddleware,
    MemoryOptimizationMiddleware,
)
from .middleware.security_headers import create_security_headers_middleware, csp_router
from .middleware.request_size_limit import RequestSizeLimitMiddleware, RequestSizeLimitConfig
from .middleware.correlation_id import CorrelationIdMiddleware
from .middleware.performance_logging import PerformanceLoggingMiddleware
from .middleware.audit_logging import AuditLoggingMiddleware
from .routes import monitoring as resource_monitoring
from .routes import task_monitoring
from .routes import logging as logging_routes
from .websocket.routes import router as websocket_router


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

    # Add CORS middleware with secure configuration
    cors_options = get_cors_config()
    app.add_middleware(CORSMiddleware, **cors_options)

    # Add custom CORS validation for additional security
    @app.middleware("http")
    async def validate_cors_origin(request: Request, call_next):
        """Additional CORS origin validation middleware."""
        origin = request.headers.get("origin")

        # For preflight requests, validate thoroughly
        if request.method == "OPTIONS":
            if origin and not cors_config.validate_origin_header(origin):
                return Response(
                    content="CORS origin not allowed", status_code=403, headers={"Vary": "Origin"}
                )

        # For regular requests, add appropriate CORS headers
        response = await call_next(request)

        if origin:
            cors_headers = cors_config.get_response_headers(origin)
            for header, value in cors_headers.items():
                response.headers[header] = value

        return response

    # Add structured logging middleware stack (early in pipeline)
    # 1. Correlation ID middleware - must be first to ensure all logs have correlation IDs
    app.add_middleware(CorrelationIdMiddleware)

    # 2. Performance logging middleware - tracks detailed metrics
    performance_config = {
        "log_slow_requests_only": os.getenv("LOG_SLOW_REQUESTS_ONLY", "false").lower() == "true",
        "slow_request_threshold_ms": float(os.getenv("SLOW_REQUEST_THRESHOLD_MS", "1000")),
        "include_memory_metrics": os.getenv("INCLUDE_MEMORY_METRICS", "true").lower() == "true",
        "include_database_metrics": os.getenv("INCLUDE_DATABASE_METRICS", "true").lower() == "true",
    }
    app.add_middleware(PerformanceLoggingMiddleware, **performance_config)

    # 3. Audit logging middleware - logs security-sensitive operations
    audit_config = {
        "log_all_requests": os.getenv("AUDIT_LOG_ALL_REQUESTS", "false").lower() == "true",
        "log_request_body": os.getenv("AUDIT_LOG_REQUEST_BODY", "false").lower() == "true",
        "log_response_body": os.getenv("AUDIT_LOG_RESPONSE_BODY", "false").lower() == "true",
    }
    app.add_middleware(AuditLoggingMiddleware, **audit_config)

    # Add security headers middleware (should be early for security)
    app = create_security_headers_middleware(app)

    # Add request size limiting middleware
    request_size_config = RequestSizeLimitConfig()
    app.add_middleware(RequestSizeLimitMiddleware, config=request_size_config)

    # Add performance optimization middleware stack
    app.add_middleware(MemoryOptimizationMiddleware)
    app.add_middleware(DatabaseOptimizationMiddleware)
    app.add_middleware(LazyLoadingMiddleware)
    app.add_middleware(PerformanceMiddleware)

    # Add custom middleware
    app.add_middleware(monitoring.MetricsMiddleware)

    # Add rate limiting middleware with proxy support
    # Use environment variables for configuration
    trusted_proxies = (
        os.getenv("TRUSTED_PROXIES", "").split(",") if os.getenv("TRUSTED_PROXIES") else None
    )
    internal_keys = (
        set(key.strip() for key in os.getenv("INTERNAL_API_KEYS", "").split(",") if key.strip())
        if os.getenv("INTERNAL_API_KEYS")
        else None
    )

    app.add_middleware(
        rate_limiting.RateLimitMiddleware,
        redis_client=rate_limiting.redis_client,
        trusted_proxies=trusted_proxies,
        enable_proxy_headers=os.getenv("ENABLE_PROXY_HEADERS", "true").lower() == "true",
        internal_service_keys=internal_keys,
        default_tier=rate_limiting.RateLimitTier(os.getenv("DEFAULT_RATE_LIMIT_TIER", "free")),
    )

    # Add CSRF protection middleware with appropriate configuration
    csrf_middleware = CSRFProtectionMiddleware(
        app,
        cookie_secure=os.getenv("ENVIRONMENT", "development") == "production",
        cookie_domain=os.getenv("COOKIE_DOMAIN", None),
    )

    # Include routers
    app.include_router(auth.router, tags=["authentication"])  # Auth routes at /api/auth
    app.include_router(health.router, prefix="/health", tags=["health"])
    app.include_router(unified.router, tags=["api"])  # Unified API that matches frontend
    app.include_router(generation.router, prefix="/api/v1/generate", tags=["generation"])
    app.include_router(
        generation_optimized.router, prefix="/api/v2/generate", tags=["generation-optimized"]
    )  # Optimized generation endpoints
    app.include_router(streaming.router, prefix="/api/v1/stream", tags=["streaming"])
    app.include_router(models.router, prefix="/api/v1/models", tags=["models"])
    app.include_router(resource_monitoring.router, prefix="/api/v1", tags=["monitoring"])
    app.include_router(task_monitoring.router, prefix="/api/v1", tags=["task-monitoring"])
    app.include_router(logging_routes.router, prefix="/api/v1", tags=["logging"])
    app.include_router(websocket_router, prefix="/api/v1", tags=["websocket"])
    app.include_router(csp_router, prefix="/api/v1/security", tags=["security"])  # CSP reporting

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
        from ..core.cache import get_cache_manager, CacheKey
        from ..core.logging_config import setup_logging
        from ..infrastructure.database.connection_pool import get_database_pool
        from .websocket.secure_auth import get_websocket_authenticator

        # Initialize structured logging first
        try:
            setup_logging()
            print("✓ Structured logging initialized")
        except Exception as e:
            print(f"⚠ Logging initialization failed: {e}")

        # Initialize database connection pool
        try:
            db_pool = await get_database_pool()
            health_status = await db_pool._perform_health_check()
            print("✓ Database connection pool initialized")
        except Exception as e:
            print(f"⚠ Database pool initialization failed: {e}")

        # Initialize WebSocket authenticator
        try:
            ws_auth = await get_websocket_authenticator()
            print("✓ WebSocket authenticator initialized")
        except Exception as e:
            print(f"⚠ WebSocket authenticator initialization failed: {e}")

        # Initialize model manager
        model_manager = ModelManager()

        # Pre-load default model
        default_model = os.getenv("DEFAULT_MODEL", "facebook/musicgen-small")
        try:
            model_manager.get_model(default_model)
            print(f"✓ Pre-loaded model: {default_model}")
        except Exception as e:
            print(f"⚠ Failed to pre-load model: {e}")

        # Initialize cache manager
        try:
            cache_manager = await get_cache_manager()
            print("✓ Cache manager initialized")

            # Pre-warm critical caches
            # Note: In production, you'd fetch actual trending data
            await cache_manager.set(
                CacheKey.trending_tracks(10), [], ttl=300  # Empty list for now  # 5 minutes
            )
        except Exception as e:
            print(f"⚠ Cache initialization failed: {e}")

    # Shutdown event
    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on shutdown."""
        from ..core.model_manager import ModelManager
        from ..core.cache import get_cache_manager
        from ..utils.file_optimizer import cleanup_temp_files
        from ..core.memory_optimizer import clear_memory_caches
        from ..infrastructure.database.connection_pool import close_database_pool
        from .websocket.secure_auth import cleanup_websocket_authenticator

        # Close database connection pool
        try:
            await close_database_pool()
            print("✓ Database connection pool closed")
        except Exception as e:
            print(f"⚠ Database pool cleanup failed: {e}")

        # Cleanup WebSocket authenticator
        try:
            await cleanup_websocket_authenticator()
            print("✓ WebSocket authenticator cleaned up")
        except Exception as e:
            print(f"⚠ WebSocket authenticator cleanup failed: {e}")

        # Clear model cache
        model_manager = ModelManager()
        model_manager.clear_cache()

        # Clear memory caches
        clear_memory_caches()

        # Clean up temporary files
        await cleanup_temp_files(max_age_hours=0)  # Clean all temp files

        # Cleanup legacy temp files
        temp_dir = Path("/tmp/musicgen")
        if temp_dir.exists():
            for file in temp_dir.glob("*.wav"):
                try:
                    file.unlink()
                except:
                    pass

        # Disconnect cache backend
        try:
            cache_manager = await get_cache_manager()
            if hasattr(cache_manager.backend, "disconnect"):
                await cache_manager.backend.disconnect()
                print("✓ Cache backend disconnected")
        except Exception as e:
            print(f"⚠ Cache disconnect failed: {e}")

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
