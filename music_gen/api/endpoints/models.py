"""
Model management endpoints for Music Gen AI API.
"""

from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ...core.model_manager import ModelManager

router = APIRouter()


class ModelInfo(BaseModel):
    """Model information."""

    name: str = Field(..., description="Model name")
    size: str = Field(..., description="Model size (small, medium, large)")
    parameters: Optional[str] = Field(None, description="Number of parameters")
    loaded: bool = Field(..., description="Whether model is currently loaded")
    device: Optional[str] = Field(None, description="Device model is loaded on")
    sample_rate: int = Field(32000, description="Model sample rate")


class LoadModelRequest(BaseModel):
    """Request to load a model."""

    model_name: str = Field(..., description="Model name to load")
    device: Optional[str] = Field(None, description="Device to load model on")


@router.get("/", response_model=List[ModelInfo])
async def list_models():
    """List available models."""

    available_models = [
        {
            "name": "facebook/musicgen-small",
            "size": "small",
            "parameters": "300M",
            "sample_rate": 32000,
        },
        {
            "name": "facebook/musicgen-medium",
            "size": "medium",
            "parameters": "1.5B",
            "sample_rate": 32000,
        },
        {
            "name": "facebook/musicgen-large",
            "size": "large",
            "parameters": "3.3B",
            "sample_rate": 32000,
        },
        {
            "name": "facebook/musicgen-melody",
            "size": "medium",
            "parameters": "1.5B",
            "sample_rate": 32000,
        },
    ]

    # Check which models are loaded
    model_manager = ModelManager()
    loaded_models = model_manager.list_loaded_models()

    models = []
    for model_info in available_models:
        model = ModelInfo(
            name=model_info["name"],
            size=model_info["size"],
            parameters=model_info["parameters"],
            loaded=model_info["name"] in loaded_models,
            device=loaded_models.get(model_info["name"], {}).get("device"),
            sample_rate=model_info["sample_rate"],
        )
        models.append(model)

    return models


@router.get("/loaded")
async def list_loaded_models():
    """List currently loaded models."""

    model_manager = ModelManager()
    loaded_models = model_manager.list_loaded_models()

    return {
        "models": loaded_models,
        "count": len(loaded_models),
    }


@router.post("/load")
async def load_model(request: LoadModelRequest):
    """Load a model into memory."""

    model_manager = ModelManager()

    try:
        # Load model
        model = model_manager.get_model(
            model_name=request.model_name,
            device=request.device,
        )

        return {
            "message": f"Model {request.model_name} loaded successfully",
            "model_name": request.model_name,
            "device": str(model.device),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@router.delete("/{model_name}")
async def unload_model(model_name: str):
    """Unload a model from memory."""

    model_manager = ModelManager()

    if not model_manager.unload_model(model_name):
        raise HTTPException(
            status_code=404, detail=f"Model {model_name} not found in loaded models"
        )

    return {
        "message": f"Model {model_name} unloaded successfully",
        "model_name": model_name,
    }


@router.get("/cache")
async def get_cache_info():
    """Get model cache information."""

    model_manager = ModelManager()
    cache_info = model_manager.get_cache_info()

    return cache_info


@router.delete("/cache")
async def clear_cache():
    """Clear model cache."""

    model_manager = ModelManager()
    model_manager.clear_cache()

    return {"message": "Model cache cleared successfully"}


@router.get("/performance")
async def get_model_performance():
    """Get performance statistics for loaded models."""

    model_manager = ModelManager()
    loaded_models = model_manager.list_loaded_models()

    if not loaded_models:
        return {
            "message": "No models currently loaded",
            "models": {},
        }

    performance_stats = {}

    for model_name in loaded_models:
        model = model_manager.get_model(model_name)
        if hasattr(model, "get_performance_stats"):
            performance_stats[model_name] = model.get_performance_stats()
        else:
            performance_stats[model_name] = {
                "status": "No performance stats available",
                "device": str(model.device),
            }

    return {
        "models": performance_stats,
        "optimization_features": [
            "Model caching",
            "Concurrent generation",
            "Memory management",
            "GPU optimization",
            "Torch compile (if available)",
        ],
    }


@router.get("/genres")
async def list_genres():
    """List supported musical genres."""

    return {
        "genres": [
            "jazz",
            "classical",
            "rock",
            "electronic",
            "ambient",
            "folk",
            "blues",
            "country",
            "reggae",
            "hip-hop",
            "pop",
            "metal",
            "orchestral",
            "piano",
            "acoustic",
            "instrumental",
        ]
    }


@router.get("/moods")
async def list_moods():
    """List supported musical moods."""

    return {
        "moods": [
            "happy",
            "sad",
            "energetic",
            "calm",
            "dramatic",
            "peaceful",
            "melancholic",
            "uplifting",
            "mysterious",
            "romantic",
            "epic",
            "nostalgic",
            "playful",
            "intense",
            "serene",
            "triumphant",
        ]
    }


@router.get("/instruments")
async def list_instruments():
    """List supported instruments."""

    return {
        "instruments": {
            "keyboards": ["piano", "electric_piano", "organ", "synthesizer"],
            "strings": ["violin", "viola", "cello", "double_bass", "harp"],
            "guitars": ["acoustic_guitar", "electric_guitar", "bass_guitar"],
            "brass": ["trumpet", "trombone", "french_horn", "tuba"],
            "woodwinds": ["flute", "clarinet", "saxophone", "oboe"],
            "percussion": ["drums", "timpani", "xylophone", "tambourine"],
            "electronic": ["synthesizer", "drum_machine", "sampler"],
        },
        "total": 25,
    }
