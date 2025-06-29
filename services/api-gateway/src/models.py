"""
Data models for API Gateway

Shared models for request/response schemas across services.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# Re-export models from services for gateway use
class GenerationRequest(BaseModel):
    """Request for music generation"""
    prompt: str = Field(..., description="Text prompt for generation")
    duration: float = Field(30.0, gt=0, le=300, description="Duration in seconds")
    temperature: float = Field(1.0, gt=0, le=2.0, description="Sampling temperature")
    top_k: Optional[int] = Field(250, ge=0, description="Top-k sampling")
    top_p: Optional[float] = Field(0.0, ge=0, le=1, description="Top-p sampling")
    
    # Advanced options
    genre: Optional[str] = Field(None, description="Target genre")
    mood: Optional[str] = Field(None, description="Target mood")
    instruments: Optional[List[str]] = Field(None, description="Specific instruments")
    
    # Output options
    output_format: str = Field("wav", description="Output format")
    sample_rate: int = Field(32000, description="Sample rate")
    
    class Config:
        schema_extra = {
            "example": {
                "prompt": "Upbeat jazz piano with saxophone solo",
                "duration": 30,
                "genre": "jazz",
                "mood": "upbeat",
                "instruments": ["piano", "saxophone"]
            }
        }


class BatchGenerationRequest(BaseModel):
    """Request for batch generation"""
    requests: List[GenerationRequest] = Field(..., max_items=10)
    sequential: bool = Field(False, description="Process sequentially")


class ConversionRequest(BaseModel):
    """Request for audio conversion"""
    source_url: str = Field(..., description="Source audio URL")
    target_format: str = Field(..., description="Target format")
    options: Optional[Dict[str, Any]] = Field(None, description="Conversion options")


class AnalysisRequest(BaseModel):
    """Request for audio analysis"""
    audio_url: str = Field(..., description="Audio URL to analyze")
    feature_types: List[str] = Field(["all"], description="Features to extract")


class UserCreate(BaseModel):
    """User registration request"""
    username: str = Field(..., min_length=3, max_length=30)
    email: str = Field(..., description="Email address")
    password: str = Field(..., min_length=8, description="Password")
    full_name: Optional[str] = Field(None, description="Full name")


class UserLogin(BaseModel):
    """User login request"""
    email: str = Field(..., description="Email address")
    password: str = Field(..., description="Password")


class PlaylistCreate(BaseModel):
    """Playlist creation request"""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    is_public: bool = Field(True, description="Public visibility")
    tags: List[str] = Field([], description="Tags")


class APIResponse(BaseModel):
    """Standard API response format"""
    success: bool = Field(True, description="Request success status")
    data: Optional[Any] = Field(None, description="Response data")
    message: Optional[str] = Field(None, description="Response message")
    error: Optional[str] = Field(None, description="Error message")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ServiceStatus(BaseModel):
    """Service health status"""
    name: str
    status: str  # healthy, unhealthy, unknown
    response_time: Optional[float] = None
    last_check: datetime
    error: Optional[str] = None