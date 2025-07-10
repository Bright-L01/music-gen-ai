"""
Health check endpoints for Music Gen AI API.
"""

from typing import Any, Dict

import torch
from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def health_check() -> Dict[str, Any]:
    """
    Basic health check endpoint.

    Returns:
        Health status information
    """
    memory_usage = None
    if torch.cuda.is_available():
        memory_usage = {
            "allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
            "cached": torch.cuda.memory_reserved() / 1024**3,  # GB
            "max_allocated": torch.cuda.max_memory_allocated() / 1024**3,
        }

    return {
        "status": "healthy",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "memory_usage": memory_usage,
    }


@router.get("/ready")
async def readiness_check() -> Dict[str, Any]:
    """
    Readiness check endpoint for Kubernetes.

    Returns:
        Readiness status
    """
    from ...core.model_manager import ModelManager

    try:
        model_manager = ModelManager()
        has_models = model_manager.has_loaded_models()

        return {
            "ready": has_models,
            "models_loaded": has_models,
            "status": "ready" if has_models else "not_ready",
        }
    except Exception as e:
        return {
            "ready": False,
            "status": "error",
            "error": str(e),
        }


@router.get("/live")
async def liveness_check() -> Dict[str, str]:
    """
    Liveness check endpoint for Kubernetes.

    Returns:
        Liveness status
    """
    return {"status": "alive"}
