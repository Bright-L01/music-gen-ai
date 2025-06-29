"""
Audio Processing Service Implementation

Core service for audio file operations, analysis, and processing.
"""

import asyncio
import hashlib
import json
import logging
import os
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
from PIL import Image, ImageDraw

from .models import (
    ProcessingStatus,
    AudioFormat,
    ConversionOptions,
    MixTrack,
    FeatureType,
    ProcessingJob,
    AudioMetadata
)


logger = logging.getLogger(__name__)


class AudioProcessingService:
    """Main service for audio processing operations"""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.temp_path = Path(tempfile.gettempdir()) / "audio_processing"
        self.temp_path.mkdir(exist_ok=True)
        self.jobs = {}  # In production, use Redis
        self._initialized = False
        
    async def initialize(self):
        """Initialize the service"""
        if self._initialized:
            return
            
        logger.info("Initializing Audio Processing Service...")
        self._initialized = True
        logger.info("Audio Processing Service initialized")
        
    async def cleanup(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)
        
    async def create_conversion_job(
        self,
        source_url: str,
        target_format: str,
        options: Optional[ConversionOptions],
        user_id: str
    ) -> str:
        """Create a conversion job"""
        job_id = f"conv_{uuid.uuid4().hex[:8]}"
        
        job = ProcessingJob(
            job_id=job_id,
            job_type="conversion",
            status=ProcessingStatus.QUEUED,
            progress=0.0,
            created_at=datetime.utcnow(),
            user_id=user_id,
            input_data={
                "source_url": source_url,
                "target_format": target_format,
                "options": options.dict() if options else {}
            }
        )
        
        self.jobs[job_id] = job
        return job_id
        
    async def process_conversion(self, job_id: str):
        """Process audio conversion job"""
        job = self.jobs.get(job_id)
        if not job:
            logger.error(f"Job {job_id} not found")
            return
            
        try:
            job.status = ProcessingStatus.PROCESSING
            job.started_at = datetime.utcnow()
            job.progress = 10.0
            
            # Download source audio
            source_path = await self._download_audio(job.input_data["source_url"])
            job.progress = 30.0
            
            # Convert audio
            loop = asyncio.get_event_loop()
            output_path = await loop.run_in_executor(
                self.executor,
                self._convert_audio,
                source_path,
                job.input_data["target_format"],
                job.input_data["options"]
            )
            job.progress = 80.0
            
            # Upload result (in production, to S3)
            output_url = f"/converted/{os.path.basename(output_path)}"
            
            # Update job
            job.status = ProcessingStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            job.progress = 100.0
            job.output_data = {"output_url": output_url}
            
            logger.info(f"Conversion job {job_id} completed")
            
        except Exception as e:
            logger.error(f"Conversion job {job_id} failed: {e}")
            job.status = ProcessingStatus.FAILED
            job.error = str(e)
            job.completed_at = datetime.utcnow()
            
    def _convert_audio(
        self,
        source_path: str,
        target_format: str,
        options: Dict[str, Any]
    ) -> str:
        """Convert audio file format"""
        # Load audio
        audio = AudioSegment.from_file(source_path)
        
        # Apply options
        if options.get("sample_rate"):
            audio = audio.set_frame_rate(options["sample_rate"])
            
        if options.get("channels"):
            if options["channels"] == 1:
                audio = audio.set_channels(1)
            elif options["channels"] == 2:
                audio = audio.set_channels(2)
                
        if options.get("normalize"):
            # Normalize to -14 LUFS
            audio = self._normalize_audio(audio)
            
        # Convert format
        output_path = str(self.temp_path / f"{uuid.uuid4().hex}.{target_format}")
        
        export_params = {}
        if target_format == "mp3" and options.get("bitrate"):
            export_params["bitrate"] = f"{options['bitrate']}k"
            
        audio.export(output_path, format=target_format, **export_params)
        
        return output_path
        
    async def analyze_audio(
        self,
        audio_url: str,
        feature_types: List[FeatureType]
    ) -> Dict[str, Any]:
        """Analyze audio and extract features"""
        # Download audio
        audio_path = await self._download_audio(audio_url)
        
        # Run analysis in thread pool
        loop = asyncio.get_event_loop()
        features = await loop.run_in_executor(
            self.executor,
            self._analyze_audio_sync,
            audio_path,
            feature_types
        )
        
        return features
        
    def _analyze_audio_sync(
        self,
        audio_path: str,
        feature_types: List[FeatureType]
    ) -> Dict[str, Any]:
        """Synchronous audio analysis"""
        # Load audio
        y, sr = librosa.load(audio_path, sr=None)
        duration = len(y) / sr
        
        features = {
            "duration": duration,
            "sample_rate": sr
        }
        
        # Extract requested features
        if FeatureType.ALL in feature_types or FeatureType.TEMPO in feature_types:
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            features["tempo"] = float(tempo)
            features["beat_times"] = beats.tolist()
            
        if FeatureType.ALL in feature_types or FeatureType.KEY in feature_types:
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            key = self._estimate_key(chroma)
            features["key"] = key
            
        if FeatureType.ALL in feature_types or FeatureType.ENERGY in feature_types:
            rms = librosa.feature.rms(y=y)[0]
            features["energy"] = float(np.mean(rms))
            features["energy_curve"] = rms.tolist()
            
        if FeatureType.ALL in feature_types or FeatureType.LOUDNESS in feature_types:
            # Estimate loudness in LUFS
            features["loudness"] = self._estimate_loudness(y, sr)
            
        if FeatureType.ALL in feature_types or FeatureType.SPECTRAL in feature_types:
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            features["spectral_centroid"] = float(np.mean(spectral_centroid))
            features["spectral_rolloff"] = float(np.mean(spectral_rolloff))
            
        if FeatureType.ALL in feature_types or FeatureType.TIMBRE in feature_types:
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features["mfcc"] = mfcc.mean(axis=1).tolist()
            
        return features
        
    def _estimate_key(self, chroma: np.ndarray) -> str:
        """Estimate musical key from chroma features"""
        # Simplified key estimation
        chroma_mean = np.mean(chroma, axis=1)
        pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        key_idx = np.argmax(chroma_mean)
        
        # Determine major/minor (simplified)
        major_profile = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
        minor_profile = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
        
        major_corr = np.corrcoef(chroma_mean, np.roll(major_profile, key_idx))[0, 1]
        minor_corr = np.corrcoef(chroma_mean, np.roll(minor_profile, key_idx))[0, 1]
        
        mode = "major" if major_corr > minor_corr else "minor"
        return f"{pitch_classes[key_idx]} {mode}"
        
    def _estimate_loudness(self, y: np.ndarray, sr: int) -> float:
        """Estimate loudness in LUFS"""
        # Simplified LUFS estimation
        rms = np.sqrt(np.mean(y ** 2))
        db = 20 * np.log10(rms + 1e-10)
        lufs = db - 0.691  # Rough approximation
        return float(lufs)
        
    async def generate_waveform(
        self,
        audio_url: str,
        width: int,
        height: int,
        color_scheme: str
    ) -> Dict[str, Any]:
        """Generate waveform visualization"""
        # Download audio
        audio_path = await self._download_audio(audio_url)
        
        # Generate waveform in thread pool
        loop = asyncio.get_event_loop()
        waveform_data = await loop.run_in_executor(
            self.executor,
            self._generate_waveform_sync,
            audio_path,
            width,
            height,
            color_scheme
        )
        
        return waveform_data
        
    def _generate_waveform_sync(
        self,
        audio_path: str,
        width: int,
        height: int,
        color_scheme: str
    ) -> Dict[str, Any]:
        """Generate waveform image"""
        # Load audio
        y, sr = librosa.load(audio_path, sr=None)
        duration = len(y) / sr
        
        # Downsample to fit width
        hop_length = len(y) // width
        if hop_length == 0:
            hop_length = 1
            
        # Calculate peaks
        peaks = []
        for i in range(0, len(y), hop_length):
            chunk = y[i:i + hop_length]
            if len(chunk) > 0:
                peaks.append(float(np.max(np.abs(chunk))))
                
        # Normalize peaks
        max_peak = max(peaks) if peaks else 1.0
        peaks = [p / max_peak for p in peaks]
        
        # Create image
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Get color
        colors = {
            "blue": (70, 130, 180),
            "green": (50, 205, 50),
            "red": (220, 20, 60),
            "purple": (138, 43, 226),
            "monochrome": (128, 128, 128)
        }
        color = colors.get(color_scheme, colors["blue"])
        
        # Draw waveform
        for i, peak in enumerate(peaks):
            x = i
            peak_height = int(peak * height / 2)
            
            # Draw from center
            y_center = height // 2
            draw.line(
                [(x, y_center - peak_height), (x, y_center + peak_height)],
                fill=color + (255,),
                width=1
            )
            
        # Save image
        output_path = str(self.temp_path / f"waveform_{uuid.uuid4().hex}.png")
        img.save(output_path)
        
        # In production, upload to S3
        waveform_url = f"/waveforms/{os.path.basename(output_path)}"
        
        return {
            "url": waveform_url,
            "peaks": peaks[:1000],  # Limit data size
            "duration": duration
        }
        
    async def create_mix_job(
        self,
        tracks: List[MixTrack],
        output_format: str,
        master_volume: float,
        user_id: str
    ) -> str:
        """Create audio mixing job"""
        job_id = f"mix_{uuid.uuid4().hex[:8]}"
        
        job = ProcessingJob(
            job_id=job_id,
            job_type="mix",
            status=ProcessingStatus.QUEUED,
            progress=0.0,
            created_at=datetime.utcnow(),
            user_id=user_id,
            input_data={
                "tracks": [t.dict() for t in tracks],
                "output_format": output_format,
                "master_volume": master_volume
            }
        )
        
        self.jobs[job_id] = job
        return job_id
        
    async def process_mix(self, job_id: str):
        """Process audio mixing job"""
        job = self.jobs.get(job_id)
        if not job:
            return
            
        try:
            job.status = ProcessingStatus.PROCESSING
            job.started_at = datetime.utcnow()
            
            # Mix tracks
            loop = asyncio.get_event_loop()
            output_path = await loop.run_in_executor(
                self.executor,
                self._mix_tracks,
                job.input_data["tracks"],
                job.input_data["master_volume"]
            )
            
            # Convert to target format if needed
            if job.input_data["output_format"] != "wav":
                output_path = await loop.run_in_executor(
                    self.executor,
                    self._convert_audio,
                    output_path,
                    job.input_data["output_format"],
                    {}
                )
                
            # Upload result
            output_url = f"/mixed/{os.path.basename(output_path)}"
            
            # Get duration
            audio = AudioSegment.from_file(output_path)
            duration = len(audio) / 1000.0
            
            # Update job
            job.status = ProcessingStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            job.progress = 100.0
            job.output_data = {
                "output_url": output_url,
                "duration": duration
            }
            
        except Exception as e:
            logger.error(f"Mix job {job_id} failed: {e}")
            job.status = ProcessingStatus.FAILED
            job.error = str(e)
            
    def _mix_tracks(
        self,
        tracks_data: List[Dict[str, Any]],
        master_volume: float
    ) -> str:
        """Mix multiple audio tracks"""
        mixed = None
        
        for track_data in tracks_data:
            # Load track (in production, download from URL)
            track = AudioSegment.from_file(track_data["audio_url"])
            
            # Apply volume
            track = track + (20 * np.log10(track_data.get("volume", 1.0)))
            
            # Apply pan
            pan = track_data.get("pan", 0.0)
            if pan != 0:
                track = track.pan(pan)
                
            # Apply start time
            start_ms = int(track_data.get("start_time", 0) * 1000)
            if start_ms > 0:
                silence = AudioSegment.silent(duration=start_ms)
                track = silence + track
                
            # Mix
            if mixed is None:
                mixed = track
            else:
                mixed = mixed.overlay(track)
                
        # Apply master volume
        if mixed and master_volume != 1.0:
            mixed = mixed + (20 * np.log10(master_volume))
            
        # Save
        output_path = str(self.temp_path / f"mixed_{uuid.uuid4().hex}.wav")
        mixed.export(output_path, format="wav")
        
        return output_path
        
    async def trim_audio(
        self,
        audio_url: str,
        start_time: float,
        end_time: float,
        user_id: str
    ) -> str:
        """Trim audio to specified time range"""
        # Download audio
        audio_path = await self._download_audio(audio_url)
        
        # Trim in thread pool
        loop = asyncio.get_event_loop()
        output_path = await loop.run_in_executor(
            self.executor,
            self._trim_audio_sync,
            audio_path,
            start_time,
            end_time
        )
        
        # Upload result
        output_url = f"/trimmed/{os.path.basename(output_path)}"
        return output_url
        
    def _trim_audio_sync(
        self,
        audio_path: str,
        start_time: float,
        end_time: float
    ) -> str:
        """Trim audio file"""
        audio = AudioSegment.from_file(audio_path)
        
        start_ms = int(start_time * 1000)
        end_ms = int(end_time * 1000)
        
        trimmed = audio[start_ms:end_ms]
        
        output_path = str(self.temp_path / f"trimmed_{uuid.uuid4().hex}.wav")
        trimmed.export(output_path, format="wav")
        
        return output_path
        
    async def normalize_audio(
        self,
        audio_url: str,
        target_loudness: float,
        user_id: str
    ) -> str:
        """Normalize audio loudness"""
        # Download audio
        audio_path = await self._download_audio(audio_url)
        
        # Normalize in thread pool
        loop = asyncio.get_event_loop()
        output_path = await loop.run_in_executor(
            self.executor,
            self._normalize_audio_sync,
            audio_path,
            target_loudness
        )
        
        # Upload result
        output_url = f"/normalized/{os.path.basename(output_path)}"
        return output_url
        
    def _normalize_audio_sync(
        self,
        audio_path: str,
        target_loudness: float
    ) -> str:
        """Normalize audio to target loudness"""
        audio = AudioSegment.from_file(audio_path)
        
        # Calculate current loudness
        current_loudness = audio.dBFS
        
        # Calculate adjustment
        adjustment = target_loudness - current_loudness
        
        # Apply adjustment
        normalized = audio + adjustment
        
        output_path = str(self.temp_path / f"normalized_{uuid.uuid4().hex}.wav")
        normalized.export(output_path, format="wav")
        
        return output_path
        
    def _normalize_audio(self, audio: AudioSegment) -> AudioSegment:
        """Normalize audio segment"""
        # Target -14 LUFS (simplified)
        target_dBFS = -14
        change_in_dBFS = target_dBFS - audio.dBFS
        return audio.apply_gain(change_in_dBFS)
        
    async def post_process_generated(
        self,
        audio_url: str,
        job_id: str
    ) -> str:
        """Post-process generated audio"""
        # Download audio
        audio_path = await self._download_audio(audio_url)
        
        # Apply standard post-processing
        loop = asyncio.get_event_loop()
        output_path = await loop.run_in_executor(
            self.executor,
            self._post_process_sync,
            audio_path
        )
        
        # Upload result
        output_url = f"/processed/{os.path.basename(output_path)}"
        return output_url
        
    def _post_process_sync(self, audio_path: str) -> str:
        """Apply standard post-processing"""
        audio = AudioSegment.from_file(audio_path)
        
        # Normalize
        audio = self._normalize_audio(audio)
        
        # Add fade in/out
        audio = audio.fade_in(100).fade_out(100)
        
        output_path = str(self.temp_path / f"processed_{uuid.uuid4().hex}.wav")
        audio.export(output_path, format="wav")
        
        return output_path
        
    async def get_job_status(self, job_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Get job status"""
        job = self.jobs.get(job_id)
        
        if job and job.user_id == user_id:
            return job.dict()
            
        return None
        
    async def _download_audio(self, audio_url: str) -> str:
        """Download audio file from URL"""
        # In production, download from S3/CDN
        # For now, assume local file
        return audio_url