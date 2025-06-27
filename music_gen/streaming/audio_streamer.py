"""
Audio streaming utilities for smooth real-time audio delivery.
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
import logging
import time
from dataclasses import dataclass
from queue import Queue, Empty
import threading

logger = logging.getLogger(__name__)


@dataclass
class AudioChunk:
    """Represents a chunk of audio with metadata."""
    
    chunk_id: int
    audio: torch.Tensor  # Shape: (batch, channels, samples)
    sample_rate: int
    duration: float
    timestamp: float
    overlap_samples: int = 0
    is_final: bool = False
    
    def __post_init__(self):
        if self.audio.dim() == 1:
            # Convert to (1, 1, samples) format
            self.audio = self.audio.unsqueeze(0).unsqueeze(0)
        elif self.audio.dim() == 2:
            # Convert to (1, channels, samples) format
            self.audio = self.audio.unsqueeze(0)
    
    @property
    def num_samples(self) -> int:
        return self.audio.shape[-1]
    
    @property
    def num_channels(self) -> int:
        return self.audio.shape[-2]
    
    def to_numpy(self) -> np.ndarray:
        """Convert audio to numpy array."""
        return self.audio.detach().cpu().numpy()


class CrossfadeProcessor:
    """Handles crossfading between audio chunks for smooth transitions."""
    
    def __init__(self, fade_duration: float = 0.1, sample_rate: int = 24000):
        self.fade_duration = fade_duration
        self.sample_rate = sample_rate
        self.fade_samples = int(fade_duration * sample_rate)
        
        # Pre-compute fade curves
        self.fade_in = self._create_fade_curve("in")
        self.fade_out = self._create_fade_curve("out")
        
        logger.info(f"Crossfade processor initialized with {fade_duration}s fade ({self.fade_samples} samples)")
    
    def _create_fade_curve(self, direction: str) -> torch.Tensor:
        """Create fade curve (cosine-based for smooth transitions)."""
        t = torch.linspace(0, 1, self.fade_samples)
        
        if direction == "in":
            # Fade in: 0 -> 1
            curve = 0.5 * (1 - torch.cos(torch.pi * t))
        else:  # fade out
            # Fade out: 1 -> 0  
            curve = 0.5 * (1 + torch.cos(torch.pi * t))
        
        return curve
    
    def crossfade_chunks(
        self,
        chunk1: AudioChunk,
        chunk2: AudioChunk,
        overlap_samples: Optional[int] = None,
    ) -> AudioChunk:
        """
        Crossfade two audio chunks for smooth transition.
        
        Args:
            chunk1: First chunk (will be faded out at the end)
            chunk2: Second chunk (will be faded in at the beginning)
            overlap_samples: Number of samples to overlap (default: fade_samples)
            
        Returns:
            Combined audio chunk
        """
        
        if overlap_samples is None:
            overlap_samples = min(self.fade_samples, chunk1.num_samples, chunk2.num_samples)
        
        if overlap_samples <= 0:
            # No overlap, just concatenate
            return self._concatenate_chunks(chunk1, chunk2)
        
        # Ensure both chunks have same shape
        audio1, audio2 = self._match_audio_shapes(chunk1.audio, chunk2.audio)
        
        # Extract overlap regions
        chunk1_overlap = audio1[..., -overlap_samples:]  # End of chunk1
        chunk2_overlap = audio2[..., :overlap_samples]   # Start of chunk2
        
        # Create fade curves for this overlap length
        if overlap_samples != self.fade_samples:
            fade_out = self._create_fade_curve_length("out", overlap_samples)
            fade_in = self._create_fade_curve_length("in", overlap_samples)
        else:
            fade_out = self.fade_out
            fade_in = self.fade_in
        
        # Apply crossfade
        fade_out_expanded = fade_out.view(1, 1, -1).expand_as(chunk1_overlap)
        fade_in_expanded = fade_in.view(1, 1, -1).expand_as(chunk2_overlap)
        
        faded_chunk1 = chunk1_overlap * fade_out_expanded
        faded_chunk2 = chunk2_overlap * fade_in_expanded
        
        # Combine overlapped region
        crossfaded_overlap = faded_chunk1 + faded_chunk2
        
        # Construct final audio
        chunk1_before = audio1[..., :-overlap_samples]
        chunk2_after = audio2[..., overlap_samples:]
        
        combined_audio = torch.cat([
            chunk1_before,
            crossfaded_overlap,
            chunk2_after
        ], dim=-1)
        
        # Create new chunk
        combined_chunk = AudioChunk(
            chunk_id=chunk2.chunk_id,
            audio=combined_audio,
            sample_rate=chunk2.sample_rate,
            duration=chunk1.duration + chunk2.duration - (overlap_samples / chunk2.sample_rate),
            timestamp=chunk1.timestamp,
            overlap_samples=overlap_samples,
            is_final=chunk2.is_final,
        )
        
        return combined_chunk
    
    def _create_fade_curve_length(self, direction: str, length: int) -> torch.Tensor:
        """Create fade curve with specific length."""
        t = torch.linspace(0, 1, length)
        
        if direction == "in":
            curve = 0.5 * (1 - torch.cos(torch.pi * t))
        else:
            curve = 0.5 * (1 + torch.cos(torch.pi * t))
        
        return curve
    
    def _match_audio_shapes(
        self,
        audio1: torch.Tensor,
        audio2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Ensure two audio tensors have matching shapes."""
        
        # Match batch dimensions
        if audio1.shape[0] != audio2.shape[0]:
            target_batch = max(audio1.shape[0], audio2.shape[0])
            if audio1.shape[0] == 1:
                audio1 = audio1.expand(target_batch, -1, -1)
            if audio2.shape[0] == 1:
                audio2 = audio2.expand(target_batch, -1, -1)
        
        # Match channel dimensions
        if audio1.shape[1] != audio2.shape[1]:
            target_channels = max(audio1.shape[1], audio2.shape[1])
            
            if audio1.shape[1] == 1 and target_channels > 1:
                audio1 = audio1.expand(-1, target_channels, -1)
            elif audio1.shape[1] > 1 and target_channels == 1:
                audio1 = audio1.mean(dim=1, keepdim=True)
            
            if audio2.shape[1] == 1 and target_channels > 1:
                audio2 = audio2.expand(-1, target_channels, -1)
            elif audio2.shape[1] > 1 and target_channels == 1:
                audio2 = audio2.mean(dim=1, keepdim=True)
        
        return audio1, audio2
    
    def _concatenate_chunks(self, chunk1: AudioChunk, chunk2: AudioChunk) -> AudioChunk:
        """Concatenate chunks without crossfading."""
        audio1, audio2 = self._match_audio_shapes(chunk1.audio, chunk2.audio)
        combined_audio = torch.cat([audio1, audio2], dim=-1)
        
        return AudioChunk(
            chunk_id=chunk2.chunk_id,
            audio=combined_audio,
            sample_rate=chunk2.sample_rate,
            duration=chunk1.duration + chunk2.duration,
            timestamp=chunk1.timestamp,
            is_final=chunk2.is_final,
        )


class StreamingBuffer:
    """Manages buffering and flow control for audio streaming."""
    
    def __init__(
        self,
        buffer_size: int = 8,
        min_buffer_size: int = 2,
        sample_rate: int = 24000,
    ):
        self.buffer_size = buffer_size
        self.min_buffer_size = min_buffer_size
        self.sample_rate = sample_rate
        
        self.chunks: List[AudioChunk] = []
        self.is_buffering = True
        self.total_buffered_duration = 0.0
        self.playback_position = 0.0
        self.last_access_time = time.time()
        
        self._lock = threading.Lock()
        
    def add_chunk(self, chunk: AudioChunk) -> bool:
        """
        Add chunk to buffer.
        
        Returns:
            True if chunk was added, False if buffer is full
        """
        with self._lock:
            if len(self.chunks) >= self.buffer_size:
                # Buffer full, drop oldest chunk
                dropped = self.chunks.pop(0)
                self.total_buffered_duration -= dropped.duration
                logger.debug(f"Dropped chunk {dropped.chunk_id} due to full buffer")
            
            self.chunks.append(chunk)
            self.total_buffered_duration += chunk.duration
            
            # Check if we have enough to start playback
            if self.is_buffering and len(self.chunks) >= self.min_buffer_size:
                self.is_buffering = False
                logger.info(f"Buffer ready for playback with {len(self.chunks)} chunks")
            
            return True
    
    def get_next_chunk(self) -> Optional[AudioChunk]:
        """Get the next chunk for playback."""
        with self._lock:
            if not self.chunks:
                # Buffer empty
                if not self.is_buffering:
                    self.is_buffering = True
                    logger.warning("Buffer underrun - rebuffering")
                return None
            
            if self.is_buffering:
                return None
            
            chunk = self.chunks.pop(0)
            self.total_buffered_duration -= chunk.duration
            self.playback_position += chunk.duration
            self.last_access_time = time.time()
            
            return chunk
    
    def peek_next_chunk(self) -> Optional[AudioChunk]:
        """Peek at the next chunk without removing it."""
        with self._lock:
            if not self.chunks or self.is_buffering:
                return None
            return self.chunks[0]
    
    def get_buffer_status(self) -> Dict[str, Any]:
        """Get current buffer status."""
        with self._lock:
            return {
                "chunk_count": len(self.chunks),
                "buffered_duration": self.total_buffered_duration,
                "is_buffering": self.is_buffering,
                "playback_position": self.playback_position,
                "buffer_utilization": len(self.chunks) / self.buffer_size,
                "last_access": time.time() - self.last_access_time,
            }
    
    def clear(self):
        """Clear the buffer."""
        with self._lock:
            self.chunks.clear()
            self.total_buffered_duration = 0.0
            self.is_buffering = True


class AudioStreamer:
    """Manages real-time audio streaming with smooth transitions."""
    
    def __init__(
        self,
        sample_rate: int = 24000,
        crossfade_duration: float = 0.1,
        buffer_size: int = 8,
        min_buffer_size: int = 2,
    ):
        self.sample_rate = sample_rate
        
        # Initialize components
        self.crossfade_processor = CrossfadeProcessor(crossfade_duration, sample_rate)
        self.buffer = StreamingBuffer(buffer_size, min_buffer_size, sample_rate)
        
        # State management
        self.is_streaming = False
        self.current_chunk = None
        self.chunk_counter = 0
        
        # Performance tracking
        self.stats = {
            "chunks_processed": 0,
            "total_duration": 0.0,
            "crossfades_applied": 0,
            "buffer_underruns": 0,
            "average_latency": 0.0,
        }
        
        logger.info(f"Audio streamer initialized: {sample_rate}Hz, "
                   f"{crossfade_duration}s crossfade, buffer={buffer_size}")
    
    def start_streaming(self):
        """Start audio streaming."""
        self.is_streaming = True
        self.buffer.clear()
        self.current_chunk = None
        self.chunk_counter = 0
        logger.info("Audio streaming started")
    
    def stop_streaming(self):
        """Stop audio streaming."""
        self.is_streaming = False
        self.buffer.clear()
        logger.info("Audio streaming stopped")
    
    def add_audio_chunk(self, audio: torch.Tensor, duration: float) -> int:
        """
        Add raw audio chunk to the streaming pipeline.
        
        Args:
            audio: Audio tensor
            duration: Duration in seconds
            
        Returns:
            Chunk ID
        """
        if not self.is_streaming:
            raise RuntimeError("Streamer not started")
        
        chunk = AudioChunk(
            chunk_id=self.chunk_counter,
            audio=audio,
            sample_rate=self.sample_rate,
            duration=duration,
            timestamp=time.time(),
        )
        
        self.chunk_counter += 1
        self.buffer.add_chunk(chunk)
        
        return chunk.chunk_id
    
    def get_next_audio_segment(self) -> Optional[AudioChunk]:
        """
        Get the next smoothly crossfaded audio segment.
        
        Returns:
            AudioChunk with smooth transitions, or None if buffering
        """
        if not self.is_streaming:
            return None
        
        next_chunk = self.buffer.get_next_chunk()
        
        if next_chunk is None:
            # Buffering or no more chunks
            if self.buffer.is_buffering:
                self.stats["buffer_underruns"] += 1
            return None
        
        # Apply crossfading if we have a previous chunk
        if self.current_chunk is not None:
            try:
                # Calculate overlap based on chunk characteristics
                overlap_samples = min(
                    self.crossfade_processor.fade_samples,
                    self.current_chunk.num_samples // 4,  # Max 25% of chunk
                    next_chunk.num_samples // 4,
                )
                
                if overlap_samples > 0:
                    crossfaded_chunk = self.crossfade_processor.crossfade_chunks(
                        self.current_chunk, next_chunk, overlap_samples
                    )
                    
                    # Use the crossfaded version
                    final_chunk = crossfaded_chunk
                    self.stats["crossfades_applied"] += 1
                else:
                    # No overlap possible, just use next chunk
                    final_chunk = next_chunk
                
            except Exception as e:
                logger.warning(f"Crossfade failed, using raw chunk: {e}")
                final_chunk = next_chunk
        else:
            # First chunk, no crossfading needed
            final_chunk = next_chunk
        
        # Update state
        self.current_chunk = next_chunk  # Keep original for next crossfade
        self.stats["chunks_processed"] += 1
        self.stats["total_duration"] += final_chunk.duration
        
        return final_chunk
    
    def get_buffer_info(self) -> Dict[str, Any]:
        """Get comprehensive streaming information."""
        buffer_status = self.buffer.get_buffer_status()
        
        info = {
            "is_streaming": self.is_streaming,
            "buffer_status": buffer_status,
            "stats": self.stats.copy(),
            "crossfade_settings": {
                "duration": self.crossfade_processor.fade_duration,
                "samples": self.crossfade_processor.fade_samples,
            },
        }
        
        # Calculate real-time performance metrics
        if self.stats["chunks_processed"] > 0:
            info["average_chunk_duration"] = (
                self.stats["total_duration"] / self.stats["chunks_processed"]
            )
        
        return info
    
    def adjust_buffer_size(self, new_size: int, new_min_size: Optional[int] = None):
        """Dynamically adjust buffer size."""
        self.buffer.buffer_size = new_size
        if new_min_size is not None:
            self.buffer.min_buffer_size = new_min_size
        
        logger.info(f"Buffer size adjusted to {new_size} (min: {self.buffer.min_buffer_size})")
    
    def set_crossfade_duration(self, duration: float):
        """Adjust crossfade duration."""
        self.crossfade_processor = CrossfadeProcessor(duration, self.sample_rate)
        logger.info(f"Crossfade duration set to {duration}s")


class AdaptiveStreamer(AudioStreamer):
    """Audio streamer with adaptive quality based on network conditions."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Adaptive parameters
        self.target_latency = 0.5  # Target 500ms latency
        self.latency_history = []
        self.adaptation_threshold = 0.1  # 100ms threshold for adaptation
        
    def add_audio_chunk(self, audio: torch.Tensor, duration: float, latency: Optional[float] = None) -> int:
        """Add chunk with latency tracking for adaptation."""
        
        chunk_id = super().add_audio_chunk(audio, duration)
        
        # Track latency for adaptation
        if latency is not None:
            self.latency_history.append(latency)
            if len(self.latency_history) > 10:
                self.latency_history.pop(0)  # Keep only recent history
            
            self._adapt_to_latency()
        
        return chunk_id
    
    def _adapt_to_latency(self):
        """Adapt streaming parameters based on observed latency."""
        if len(self.latency_history) < 3:
            return
        
        avg_latency = sum(self.latency_history) / len(self.latency_history)
        
        if avg_latency > self.target_latency + self.adaptation_threshold:
            # High latency - increase buffering, reduce crossfade quality
            new_min_buffer = min(self.buffer.min_buffer_size + 1, self.buffer.buffer_size)
            self.adjust_buffer_size(self.buffer.buffer_size, new_min_buffer)
            
            # Reduce crossfade duration for lower latency
            new_crossfade = max(0.05, self.crossfade_processor.fade_duration * 0.9)
            self.set_crossfade_duration(new_crossfade)
            
            logger.info(f"Adapted for high latency ({avg_latency:.3f}s): "
                       f"buffer={new_min_buffer}, crossfade={new_crossfade:.3f}s")
        
        elif avg_latency < self.target_latency - self.adaptation_threshold:
            # Low latency - can improve quality
            new_crossfade = min(0.2, self.crossfade_processor.fade_duration * 1.1)
            self.set_crossfade_duration(new_crossfade)
            
            logger.debug(f"Adapted for low latency ({avg_latency:.3f}s): "
                        f"crossfade={new_crossfade:.3f}s")