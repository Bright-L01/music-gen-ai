"""
Music Generation Service

Core service for handling music generation using MusicGen models.
Provides both simple and structured generation capabilities.
"""

import asyncio
import hashlib
import json
import logging
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
from concurrent.futures import ThreadPoolExecutor

import torch
import torchaudio
from transformers import AutoProcessor, MusicgenForConditionalGeneration

from .models import (
    GenerationRequest,
    GenerationResponse,
    GenerationStatus,
    SongStructure,
    AudioSection,
    SectionType
)


logger = logging.getLogger(__name__)


class ModelPool:
    """Manages a pool of loaded models for concurrent generation"""
    
    def __init__(self, max_models: int = 3):
        self.max_models = max_models
        self.models = {}
        self.processors = {}
        self.locks = {}
        self._lock = asyncio.Lock()
        
    async def get_model(self, model_name: str = "facebook/musicgen-small"):
        """Get or load a model from the pool"""
        async with self._lock:
            if model_name not in self.models:
                if len(self.models) >= self.max_models:
                    # Evict least recently used
                    oldest = min(self.models.keys(), key=lambda k: self.models[k]['last_used'])
                    del self.models[oldest]
                    del self.processors[oldest]
                    del self.locks[oldest]
                
                # Load new model
                logger.info(f"Loading model: {model_name}")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                
                processor = AutoProcessor.from_pretrained(model_name)
                model = MusicgenForConditionalGeneration.from_pretrained(model_name)
                model = model.to(device)
                
                self.models[model_name] = {
                    'model': model,
                    'device': device,
                    'last_used': time.time()
                }
                self.processors[model_name] = processor
                self.locks[model_name] = asyncio.Lock()
                
                logger.info(f"Model loaded successfully on {device}")
            
            self.models[model_name]['last_used'] = time.time()
            
        return (
            self.models[model_name]['model'],
            self.processors[model_name],
            self.models[model_name]['device'],
            self.locks[model_name]
        )


class GenerationService:
    """Main service for music generation"""
    
    def __init__(self):
        self.model_pool = ModelPool(max_models=3)
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.storage_path = Path(os.getenv("STORAGE_PATH", "/tmp/music_gen/audio"))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._initialized = False
        
    async def initialize(self):
        """Initialize the service and preload models"""
        if self._initialized:
            return
            
        logger.info("Initializing Generation Service...")
        
        # Preload default model
        await self.model_pool.get_model("facebook/musicgen-small")
        
        self._initialized = True
        logger.info("Generation Service initialized successfully")
        
    async def cleanup(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)
        
    @property
    def models_loaded(self) -> List[str]:
        """Get list of currently loaded models"""
        return list(self.model_pool.models.keys())
        
    async def generate_simple(
        self,
        prompt: str,
        duration: float = 30.0,
        temperature: float = 1.0,
        model_name: str = "facebook/musicgen-small",
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> str:
        """Generate music from a simple text prompt"""
        
        # Get model from pool
        model, processor, device, lock = await self.model_pool.get_model(model_name)
        
        async with lock:
            # Run generation in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            audio_url = await loop.run_in_executor(
                self.executor,
                self._generate_audio,
                model,
                processor,
                device,
                prompt,
                duration,
                temperature,
                progress_callback
            )
            
        return audio_url
        
    def _generate_audio(
        self,
        model,
        processor,
        device: str,
        prompt: str,
        duration: float,
        temperature: float,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> str:
        """Synchronous audio generation (runs in thread pool)"""
        
        try:
            # Update progress
            if progress_callback:
                progress_callback(10.0)
            
            # Process inputs
            inputs = processor(
                text=[prompt],
                padding=True,
                return_tensors="pt"
            ).to(device)
            
            if progress_callback:
                progress_callback(20.0)
            
            # Calculate tokens needed
            sample_rate = model.config.audio_encoder.sampling_rate
            tokens_per_second = model.config.audio_encoder.frame_rate
            max_new_tokens = int(duration * tokens_per_second)
            
            # Generate audio
            with torch.no_grad():
                audio_values = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature
                )
                
            if progress_callback:
                progress_callback(80.0)
            
            # Save audio
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{hashlib.md5(prompt.encode()).hexdigest()[:8]}.wav"
            filepath = self.storage_path / filename
            
            torchaudio.save(
                str(filepath),
                audio_values[0].cpu(),
                sample_rate
            )
            
            if progress_callback:
                progress_callback(100.0)
            
            # Return URL (in production, this would be S3/CDN URL)
            return f"/audio/{filename}"
            
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            raise
            
    async def generate_structured(
        self,
        prompt: str,
        structure: SongStructure,
        duration: float,
        model_name: str = "facebook/musicgen-small",
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> str:
        """Generate music with specific song structure"""
        
        # Get model from pool
        model, processor, device, lock = await self.model_pool.get_model(model_name)
        
        sections_audio = []
        total_sections = len(structure.sections)
        
        async with lock:
            for idx, section in enumerate(structure.sections):
                # Update progress
                section_progress = (idx / total_sections) * 100
                if progress_callback:
                    progress_callback(section_progress)
                
                # Create section-specific prompt
                section_prompt = self._create_section_prompt(
                    prompt, section, structure
                )
                
                # Generate section audio
                loop = asyncio.get_event_loop()
                section_audio = await loop.run_in_executor(
                    self.executor,
                    self._generate_audio,
                    model,
                    processor,
                    device,
                    section_prompt,
                    section.duration,
                    1.0,  # temperature
                    None  # no sub-progress for sections
                )
                
                sections_audio.append(section_audio)
        
        # Combine sections with smooth transitions
        final_audio_url = await self._combine_sections(
            sections_audio, structure, progress_callback
        )
        
        return final_audio_url
        
    def _create_section_prompt(
        self,
        base_prompt: str,
        section: AudioSection,
        structure: SongStructure
    ) -> str:
        """Create a prompt for a specific section"""
        
        # Add section-specific context
        section_context = {
            SectionType.INTRO: "intro beginning opening",
            SectionType.VERSE: "verse calm steady narrative",
            SectionType.CHORUS: "chorus catchy memorable hook energetic",
            SectionType.BRIDGE: "bridge transition different contrasting",
            SectionType.SOLO: "solo instrumental virtuoso showcase",
            SectionType.OUTRO: "outro ending conclusion fade"
        }
        
        # Build enhanced prompt
        parts = [base_prompt]
        
        # Add section type
        if section.type in section_context:
            parts.append(section_context[section.type])
            
        # Add energy level
        energy_map = {
            (0.0, 0.3): "calm quiet gentle soft",
            (0.3, 0.6): "moderate steady balanced",
            (0.6, 0.8): "energetic upbeat dynamic",
            (0.8, 1.0): "intense powerful climactic"
        }
        
        for (low, high), energy_desc in energy_map.items():
            if low <= section.energy < high:
                parts.append(energy_desc)
                break
                
        # Add tempo if specified
        if structure.tempo:
            parts.append(f"{structure.tempo} bpm")
            
        # Add key if specified
        if structure.key:
            parts.append(f"in {structure.key}")
            
        # Add instruments if specified
        if section.instruments:
            parts.append(f"featuring {', '.join(section.instruments)}")
            
        # Add custom modifier
        if section.prompt_modifier:
            parts.append(section.prompt_modifier)
            
        return " ".join(parts)
        
    async def _combine_sections(
        self,
        section_urls: List[str],
        structure: SongStructure,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> str:
        """Combine multiple sections into a single track with transitions"""
        
        # For now, just return the first section
        # In production, this would use audio processing service
        # to properly mix and transition between sections
        
        if progress_callback:
            progress_callback(100.0)
            
        return section_urls[0] if section_urls else ""
        
    async def get_generation_stats(self) -> Dict[str, Any]:
        """Get current service statistics"""
        
        # Get system stats
        cpu_percent = 0.0
        memory_percent = 0.0
        
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
        except ImportError:
            pass
            
        # GPU stats
        gpu_available = torch.cuda.is_available()
        gpu_memory_used = None
        
        if gpu_available:
            gpu_memory_used = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            
        return {
            "models_loaded": self.models_loaded,
            "gpu_available": gpu_available,
            "gpu_memory_used": gpu_memory_used,
            "cpu_usage": cpu_percent,
            "memory_usage": memory_percent
        }