"""
Data models for Audio Processing Service

Defines schemas for audio operations like conversion, analysis, and mixing.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator


class ProcessingStatus(str, Enum):
    """Status of processing jobs"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AudioFormat(str, Enum):
    """Supported audio formats"""
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    OGG = "ogg"
    M4A = "m4a"
    OPUS = "opus"


class ColorScheme(str, Enum):
    """Waveform color schemes"""
    BLUE = "blue"
    GREEN = "green"
    RED = "red"
    PURPLE = "purple"
    GRADIENT = "gradient"
    MONOCHROME = "monochrome"


class ConversionOptions(BaseModel):
    """Options for audio conversion"""
    sample_rate: Optional[int] = Field(None, ge=8000, le=192000, description="Target sample rate")
    bit_depth: Optional[int] = Field(None, description="Bit depth (16, 24, 32)")
    channels: Optional[int] = Field(None, ge=1, le=2, description="Number of channels")
    bitrate: Optional[int] = Field(None, description="Bitrate for lossy formats (kbps)")
    quality: Optional[str] = Field(None, description="Quality preset (low, medium, high)")
    normalize: bool = Field(False, description="Apply normalization")
    
    @validator('bit_depth')
    def validate_bit_depth(cls, v):
        if v and v not in [16, 24, 32]:
            raise ValueError("Bit depth must be 16, 24, or 32")
        return v


class ConversionRequest(BaseModel):
    """Request for audio format conversion"""
    source_url: str = Field(..., description="URL of source audio file")
    target_format: AudioFormat = Field(..., description="Target audio format")
    options: Optional[ConversionOptions] = Field(None, description="Conversion options")
    
    class Config:
        schema_extra = {
            "example": {
                "source_url": "https://storage.example.com/audio/song.wav",
                "target_format": "mp3",
                "options": {
                    "bitrate": 320,
                    "normalize": True
                }
            }
        }


class ConversionResponse(BaseModel):
    """Response for conversion request"""
    job_id: str = Field(..., description="Conversion job ID")
    status: ProcessingStatus = Field(..., description="Job status")
    progress: float = Field(0.0, ge=0, le=100, description="Progress percentage")
    output_url: Optional[str] = Field(None, description="URL of converted file")
    error: Optional[str] = Field(None, description="Error message if failed")
    
    class Config:
        schema_extra = {
            "example": {
                "job_id": "conv_123e4567",
                "status": "processing",
                "progress": 45.5
            }
        }


class FeatureType(str, Enum):
    """Types of audio features to analyze"""
    TEMPO = "tempo"
    KEY = "key"
    ENERGY = "energy"
    LOUDNESS = "loudness"
    SPECTRAL = "spectral"
    RHYTHM = "rhythm"
    TIMBRE = "timbre"
    PITCH = "pitch"
    ALL = "all"


class AnalysisRequest(BaseModel):
    """Request for audio analysis"""
    audio_url: str = Field(..., description="URL of audio file to analyze")
    feature_types: List[FeatureType] = Field(
        [FeatureType.ALL],
        description="Features to extract"
    )
    detailed: bool = Field(False, description="Return detailed analysis")
    
    class Config:
        schema_extra = {
            "example": {
                "audio_url": "https://storage.example.com/audio/song.wav",
                "feature_types": ["tempo", "key", "energy"],
                "detailed": True
            }
        }


class AnalysisResponse(BaseModel):
    """Response with audio analysis results"""
    audio_url: str = Field(..., description="Analyzed audio URL")
    duration: float = Field(..., description="Duration in seconds")
    sample_rate: int = Field(..., description="Sample rate in Hz")
    features: Dict[str, Any] = Field(..., description="Extracted features")
    
    class Config:
        schema_extra = {
            "example": {
                "audio_url": "https://storage.example.com/audio/song.wav",
                "duration": 180.5,
                "sample_rate": 44100,
                "features": {
                    "tempo": 120,
                    "key": "C major",
                    "energy": 0.75,
                    "loudness": -14.2,
                    "spectral_centroid": 2500.5
                }
            }
        }


class WaveformRequest(BaseModel):
    """Request for waveform generation"""
    audio_url: str = Field(..., description="URL of audio file")
    width: int = Field(1920, ge=100, le=4096, description="Image width in pixels")
    height: int = Field(200, ge=50, le=1024, description="Image height in pixels")
    color_scheme: ColorScheme = Field(ColorScheme.BLUE, description="Color scheme")
    style: str = Field("bars", description="Visualization style (bars, line, mirror)")
    
    class Config:
        schema_extra = {
            "example": {
                "audio_url": "https://storage.example.com/audio/song.wav",
                "width": 1920,
                "height": 200,
                "color_scheme": "gradient",
                "style": "mirror"
            }
        }


class WaveformResponse(BaseModel):
    """Response with waveform data"""
    waveform_url: str = Field(..., description="URL of waveform image")
    peaks: List[float] = Field(..., description="Peak amplitude values")
    duration: float = Field(..., description="Audio duration in seconds")
    
    class Config:
        schema_extra = {
            "example": {
                "waveform_url": "https://storage.example.com/waveforms/123.png",
                "peaks": [0.1, 0.3, 0.5, 0.8, 0.6, 0.4, 0.2],
                "duration": 180.5
            }
        }


class MixTrack(BaseModel):
    """Individual track for mixing"""
    audio_url: str = Field(..., description="URL of audio track")
    volume: float = Field(1.0, ge=0, le=2, description="Volume multiplier")
    pan: float = Field(0.0, ge=-1, le=1, description="Pan position (-1=left, 1=right)")
    start_time: float = Field(0.0, ge=0, description="Start time in mix")
    effects: Optional[Dict[str, Any]] = Field(None, description="Audio effects to apply")
    
    class Config:
        schema_extra = {
            "example": {
                "audio_url": "https://storage.example.com/audio/drums.wav",
                "volume": 0.8,
                "pan": -0.3,
                "start_time": 0.0,
                "effects": {
                    "reverb": 0.3,
                    "compression": True
                }
            }
        }


class MixRequest(BaseModel):
    """Request for audio mixing"""
    tracks: List[MixTrack] = Field(..., min_items=2, max_items=16, description="Tracks to mix")
    output_format: AudioFormat = Field(AudioFormat.WAV, description="Output format")
    master_volume: float = Field(1.0, ge=0, le=2, description="Master volume")
    normalize: bool = Field(True, description="Normalize output")
    
    @validator('tracks')
    def validate_tracks(cls, v):
        if len(v) < 2:
            raise ValueError("At least 2 tracks required for mixing")
        if len(v) > 16:
            raise ValueError("Maximum 16 tracks allowed")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "tracks": [
                    {
                        "audio_url": "https://storage.example.com/audio/drums.wav",
                        "volume": 0.8,
                        "pan": -0.3
                    },
                    {
                        "audio_url": "https://storage.example.com/audio/bass.wav",
                        "volume": 0.9,
                        "pan": 0.1
                    }
                ],
                "output_format": "wav",
                "master_volume": 0.95,
                "normalize": True
            }
        }


class MixResponse(BaseModel):
    """Response for mix request"""
    job_id: str = Field(..., description="Mix job ID")
    status: ProcessingStatus = Field(..., description="Job status")
    progress: float = Field(0.0, ge=0, le=100, description="Progress percentage")
    output_url: Optional[str] = Field(None, description="URL of mixed audio")
    duration: Optional[float] = Field(None, description="Duration of mixed audio")
    error: Optional[str] = Field(None, description="Error message if failed")
    
    class Config:
        schema_extra = {
            "example": {
                "job_id": "mix_123e4567",
                "status": "completed",
                "progress": 100.0,
                "output_url": "https://storage.example.com/audio/mixed.wav",
                "duration": 240.5
            }
        }


class ProcessingJob(BaseModel):
    """Generic processing job information"""
    job_id: str
    job_type: str
    status: ProcessingStatus
    progress: float
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    user_id: str
    input_data: Dict[str, Any]
    output_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "job_id": "job_123",
                "job_type": "conversion",
                "status": "processing",
                "progress": 50.0,
                "created_at": "2024-01-01T12:00:00Z",
                "user_id": "user_456",
                "input_data": {
                    "source_url": "https://example.com/audio.wav",
                    "target_format": "mp3"
                }
            }
        }


class AudioMetadata(BaseModel):
    """Audio file metadata"""
    format: str
    duration: float
    sample_rate: int
    channels: int
    bit_depth: Optional[int] = None
    bitrate: Optional[int] = None
    codec: Optional[str] = None
    file_size: int
    
    class Config:
        schema_extra = {
            "example": {
                "format": "wav",
                "duration": 180.5,
                "sample_rate": 44100,
                "channels": 2,
                "bit_depth": 16,
                "file_size": 31752000
            }
        }