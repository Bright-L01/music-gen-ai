"""
Data models for the Generation Service

Defines request/response schemas and data structures for music generation.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator


class GenerationStatus(str, Enum):
    """Status of a generation job"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AudioFormat(str, Enum):
    """Supported audio output formats"""
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    OGG = "ogg"


class SectionType(str, Enum):
    """Types of song sections"""
    INTRO = "intro"
    VERSE = "verse"
    CHORUS = "chorus"
    BRIDGE = "bridge"
    SOLO = "solo"
    OUTRO = "outro"
    TRANSITION = "transition"


class AudioSection(BaseModel):
    """Represents a section of a song"""
    type: SectionType
    duration: float = Field(gt=0, le=300, description="Duration in seconds")
    prompt_modifier: Optional[str] = Field(None, description="Additional prompt context for this section")
    energy: float = Field(0.5, ge=0, le=1, description="Energy level (0-1)")
    instruments: Optional[List[str]] = Field(None, description="Specific instruments for this section")
    transition_type: Optional[str] = Field(None, description="Type of transition to next section")


class SongStructure(BaseModel):
    """Defines the structure of a song"""
    sections: List[AudioSection]
    tempo: Optional[int] = Field(None, ge=40, le=200, description="BPM")
    key: Optional[str] = Field(None, description="Musical key (e.g., 'C major', 'A minor')")
    time_signature: Optional[str] = Field("4/4", description="Time signature")
    
    @validator('sections')
    def validate_sections(cls, v):
        if not v:
            raise ValueError("Song structure must have at least one section")
        total_duration = sum(s.duration for s in v)
        if total_duration > 300:  # 5 minutes max
            raise ValueError("Total song duration cannot exceed 5 minutes")
        return v


class GenerationRequest(BaseModel):
    """Request schema for music generation"""
    prompt: str = Field(..., min_length=1, max_length=1000, description="Text prompt for generation")
    duration: float = Field(30.0, gt=0, le=300, description="Duration in seconds")
    temperature: float = Field(1.0, gt=0, le=2.0, description="Sampling temperature")
    top_k: Optional[int] = Field(250, ge=0, description="Top-k sampling")
    top_p: Optional[float] = Field(0.0, ge=0, le=1, description="Top-p sampling")
    
    # Advanced features
    structure: Optional[SongStructure] = Field(None, description="Song structure definition")
    genre: Optional[str] = Field(None, description="Target genre")
    mood: Optional[str] = Field(None, description="Target mood/emotion")
    instruments: Optional[List[str]] = Field(None, description="Specific instruments to include")
    reference_audio_url: Optional[str] = Field(None, description="Reference audio for style")
    
    # Output options
    output_format: AudioFormat = Field(AudioFormat.WAV, description="Output audio format")
    sample_rate: int = Field(32000, description="Sample rate in Hz")
    bitrate: Optional[int] = Field(None, description="Bitrate for compressed formats")
    
    # User/project context
    user_id: Optional[str] = Field(None, description="User ID for tracking")
    project_id: Optional[str] = Field(None, description="Project ID for organization")
    
    # Advanced options
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    guidance_scale: float = Field(3.0, gt=0, le=20, description="Classifier-free guidance scale")
    use_cache: bool = Field(True, description="Whether to use cached results")
    priority: Optional[str] = Field("normal", description="Job priority")
    
    # Feature flags
    advanced_features: Optional[Dict[str, Any]] = Field(
        None,
        description="Advanced feature configuration",
        example={
            "use_multiband_diffusion": True,
            "coherent_structure": True,
            "smooth_transitions": True
        }
    )
    
    class Config:
        schema_extra = {
            "example": {
                "prompt": "Upbeat jazz piano with saxophone solo",
                "duration": 30,
                "temperature": 1.0,
                "structure": {
                    "sections": [
                        {"type": "intro", "duration": 4, "energy": 0.3},
                        {"type": "verse", "duration": 8, "energy": 0.5},
                        {"type": "chorus", "duration": 8, "energy": 0.8},
                        {"type": "solo", "duration": 8, "energy": 0.7, "instruments": ["saxophone"]},
                        {"type": "outro", "duration": 2, "energy": 0.2}
                    ],
                    "tempo": 120,
                    "key": "F major"
                }
            }
        }


class GenerationResponse(BaseModel):
    """Response schema for music generation"""
    job_id: str = Field(..., description="Unique job identifier")
    status: GenerationStatus = Field(..., description="Current job status")
    audio_url: Optional[str] = Field(None, description="URL to generated audio file")
    waveform_url: Optional[str] = Field(None, description="URL to waveform visualization")
    
    # Progress tracking
    progress: float = Field(0.0, ge=0, le=100, description="Progress percentage")
    position_in_queue: Optional[int] = Field(None, description="Position in processing queue")
    estimated_time_remaining: Optional[float] = Field(None, description="Estimated seconds until completion")
    
    # Metadata
    created_at: Optional[datetime] = Field(None, description="Job creation time")
    started_at: Optional[datetime] = Field(None, description="Processing start time")
    completed_at: Optional[datetime] = Field(None, description="Completion time")
    
    # Results
    duration_generated: Optional[float] = Field(None, description="Actual duration of generated audio")
    file_size_bytes: Optional[int] = Field(None, description="Size of generated file")
    
    # Additional data
    error: Optional[str] = Field(None, description="Error message if failed")
    warnings: Optional[List[str]] = Field(None, description="Any warnings during generation")
    cached: bool = Field(False, description="Whether result was served from cache")
    
    # Audio analysis (optional)
    audio_features: Optional[Dict[str, Any]] = Field(
        None,
        description="Extracted audio features",
        example={
            "tempo": 120,
            "key": "C major",
            "energy": 0.75,
            "danceability": 0.8
        }
    )
    
    class Config:
        schema_extra = {
            "example": {
                "job_id": "123e4567-e89b-12d3-a456-426614174000",
                "status": "completed",
                "audio_url": "https://storage.example.com/audio/123e4567.wav",
                "progress": 100.0,
                "duration_generated": 30.0,
                "cached": False
            }
        }


class BatchGenerationRequest(BaseModel):
    """Request for batch generation"""
    requests: List[GenerationRequest] = Field(..., max_items=10, description="Batch of generation requests")
    sequential: bool = Field(False, description="Process sequentially vs parallel")
    
    class Config:
        schema_extra = {
            "example": {
                "requests": [
                    {"prompt": "Peaceful piano melody", "duration": 20},
                    {"prompt": "Energetic rock guitar", "duration": 30}
                ],
                "sequential": False
            }
        }


class BatchGenerationResponse(BaseModel):
    """Response for batch generation"""
    batch_id: str = Field(..., description="Batch job identifier")
    jobs: List[GenerationResponse] = Field(..., description="Individual job responses")
    total_jobs: int = Field(..., description="Total number of jobs")
    completed_jobs: int = Field(0, description="Number of completed jobs")
    failed_jobs: int = Field(0, description="Number of failed jobs")
    
    class Config:
        schema_extra = {
            "example": {
                "batch_id": "batch_123",
                "jobs": [
                    {
                        "job_id": "job_1",
                        "status": "queued",
                        "progress": 0.0
                    }
                ],
                "total_jobs": 2,
                "completed_jobs": 0,
                "failed_jobs": 0
            }
        }


class JobUpdate(BaseModel):
    """WebSocket update for job progress"""
    job_id: str
    status: GenerationStatus
    progress: float
    message: Optional[str] = None
    audio_url: Optional[str] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class GenerationStats(BaseModel):
    """Service statistics"""
    total_generations: int
    active_jobs: int
    queued_jobs: int
    cache_hit_rate: float
    average_generation_time: float
    models_loaded: List[str]
    gpu_available: bool
    gpu_memory_used: Optional[float] = None
    cpu_usage: float
    memory_usage: float