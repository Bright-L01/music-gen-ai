"""
Utilities for streaming audio generation.
"""

import base64
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def audio_to_base64(audio: torch.Tensor, sample_rate: int) -> str:
    """
    Convert audio tensor to base64 encoded string for transmission.

    Args:
        audio: Audio tensor of shape (batch, channels, samples) or (channels, samples)
        sample_rate: Sample rate of the audio

    Returns:
        Base64 encoded audio data
    """
    # Ensure audio is in the right format
    if audio.dim() == 3:
        audio = audio[0]  # Take first batch
    if audio.dim() == 2 and audio.shape[0] > 1:
        audio = audio.mean(dim=0)  # Convert to mono
    elif audio.dim() == 2:
        audio = audio[0]  # Take first channel

    # Convert to numpy and normalize
    audio_np = audio.detach().cpu().numpy()

    # Normalize to [-1, 1] range
    audio_np = audio_np / (np.abs(audio_np).max() + 1e-8)

    # Convert to 16-bit PCM
    audio_int16 = (audio_np * 32767).astype(np.int16)

    # Convert to bytes and base64 encode
    audio_bytes = audio_int16.tobytes()
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    return audio_b64


def base64_to_audio(audio_b64: str, sample_rate: int) -> torch.Tensor:
    """
    Convert base64 encoded audio back to tensor.

    Args:
        audio_b64: Base64 encoded audio data
        sample_rate: Sample rate of the audio

    Returns:
        Audio tensor of shape (1, samples)
    """
    # Decode base64
    audio_bytes = base64.b64decode(audio_b64)

    # Convert to numpy array
    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)

    # Convert back to float32 in [-1, 1] range
    audio_np = audio_int16.astype(np.float32) / 32767.0

    # Convert to tensor
    audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)  # Add channel dimension

    return audio_tensor


@dataclass
class StreamingMetrics:
    """Metrics for streaming performance monitoring."""

    total_chunks: int = 0
    total_duration: float = 0.0
    total_generation_time: float = 0.0
    buffer_underruns: int = 0
    crossfades_applied: int = 0
    network_errors: int = 0
    start_time: float = 0.0

    def __post_init__(self):
        if self.start_time == 0.0:
            self.start_time = time.time()

    @property
    def average_chunk_time(self) -> float:
        """Average time per chunk generation."""
        if self.total_chunks == 0:
            return 0.0
        return self.total_generation_time / self.total_chunks

    @property
    def real_time_factor(self) -> float:
        """Real-time factor (generated duration / wall clock time)."""
        elapsed = time.time() - self.start_time
        if elapsed == 0:
            return 0.0
        return self.total_duration / elapsed

    @property
    def buffer_underrun_rate(self) -> float:
        """Buffer underrun rate (underruns / total chunks)."""
        if self.total_chunks == 0:
            return 0.0
        return self.buffer_underruns / self.total_chunks

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_chunks": self.total_chunks,
            "total_duration": self.total_duration,
            "total_generation_time": self.total_generation_time,
            "buffer_underruns": self.buffer_underruns,
            "crossfades_applied": self.crossfades_applied,
            "network_errors": self.network_errors,
            "start_time": self.start_time,
            "average_chunk_time": self.average_chunk_time,
            "real_time_factor": self.real_time_factor,
            "buffer_underrun_rate": self.buffer_underrun_rate,
            "uptime": time.time() - self.start_time,
        }


class LatencyTracker:
    """Tracks latency for adaptive streaming."""

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.latencies: List[float] = []
        self.timestamps: List[float] = []

    def add_measurement(self, latency: float):
        """Add a latency measurement."""
        self.latencies.append(latency)
        self.timestamps.append(time.time())

        # Keep only recent measurements
        if len(self.latencies) > self.window_size:
            self.latencies.pop(0)
            self.timestamps.pop(0)

    @property
    def average_latency(self) -> float:
        """Get average latency."""
        if not self.latencies:
            return 0.0
        return sum(self.latencies) / len(self.latencies)

    @property
    def p95_latency(self) -> float:
        """Get 95th percentile latency."""
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        idx = int(0.95 * len(sorted_latencies))
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]

    @property
    def trend(self) -> str:
        """Get latency trend (improving, stable, degrading)."""
        if len(self.latencies) < 3:
            return "unknown"

        recent = self.latencies[-3:]
        if recent[-1] < recent[0] * 0.9:
            return "improving"
        elif recent[-1] > recent[0] * 1.1:
            return "degrading"
        else:
            return "stable"


class NetworkConditionMonitor:
    """Monitors network conditions for adaptive streaming."""

    def __init__(self):
        self.latency_tracker = LatencyTracker()
        self.throughput_history: List[float] = []
        self.error_count = 0
        self.total_requests = 0

    def record_request(self, latency: float, success: bool, bytes_sent: int = 0):
        """Record a network request."""
        self.total_requests += 1

        if success:
            self.latency_tracker.add_measurement(latency)
            if bytes_sent > 0 and latency > 0:
                throughput = bytes_sent / latency  # bytes per second
                self.throughput_history.append(throughput)
                if len(self.throughput_history) > 10:
                    self.throughput_history.pop(0)
        else:
            self.error_count += 1

    @property
    def error_rate(self) -> float:
        """Get error rate."""
        if self.total_requests == 0:
            return 0.0
        return self.error_count / self.total_requests

    @property
    def network_quality(self) -> str:
        """Assess overall network quality."""
        avg_latency = self.latency_tracker.average_latency
        error_rate = self.error_rate

        if error_rate > 0.1 or avg_latency > 2.0:
            return "poor"
        elif error_rate > 0.05 or avg_latency > 1.0:
            return "fair"
        elif avg_latency > 0.5:
            return "good"
        else:
            return "excellent"

    def get_adaptive_settings(self) -> Dict[str, Any]:
        """Get recommended adaptive settings based on network conditions."""
        quality = self.network_quality

        if quality == "poor":
            return {
                "chunk_duration": 2.0,  # Larger chunks
                "buffer_size": 6,  # More buffering
                "crossfade_duration": 0.05,  # Minimal crossfade
                "quality_mode": "fast",
            }
        elif quality == "fair":
            return {
                "chunk_duration": 1.5,
                "buffer_size": 4,
                "crossfade_duration": 0.08,
                "quality_mode": "balanced",
            }
        elif quality == "good":
            return {
                "chunk_duration": 1.0,
                "buffer_size": 3,
                "crossfade_duration": 0.1,
                "quality_mode": "balanced",
            }
        else:  # excellent
            return {
                "chunk_duration": 0.5,
                "buffer_size": 2,
                "crossfade_duration": 0.15,
                "quality_mode": "quality",
            }


class AudioAnalyzer:
    """Analyzes audio content for streaming optimization."""

    @staticmethod
    def analyze_audio_properties(audio: torch.Tensor, sample_rate: int) -> Dict[str, float]:
        """
        Analyze audio properties for optimization.

        Args:
            audio: Audio tensor
            sample_rate: Sample rate

        Returns:
            Dictionary of audio properties
        """
        # Ensure mono audio for analysis
        if audio.dim() > 1:
            audio = audio.mean(dim=0) if audio.dim() == 2 else audio[0, 0]

        audio_np = audio.detach().cpu().numpy()

        # Basic properties
        rms = np.sqrt(np.mean(audio_np**2))
        peak = np.abs(audio_np).max()

        # Dynamic range
        dynamic_range = 20 * np.log10(peak / (rms + 1e-8))

        # Zero crossing rate (indicates speech-like vs music-like content)
        zero_crossings = np.sum(np.diff(np.signbit(audio_np)))
        zcr = zero_crossings / len(audio_np)

        # Spectral analysis
        fft = np.fft.fft(audio_np)
        magnitude = np.abs(fft[: len(fft) // 2])
        freqs = np.fft.fftfreq(len(audio_np), 1 / sample_rate)[: len(fft) // 2]

        # Spectral centroid (brightness)
        spectral_centroid = np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-8)

        # High frequency content
        high_freq_energy = np.sum(magnitude[freqs > sample_rate / 4]) / (np.sum(magnitude) + 1e-8)

        return {
            "rms": float(rms),
            "peak": float(peak),
            "dynamic_range": float(dynamic_range),
            "zero_crossing_rate": float(zcr),
            "spectral_centroid": float(spectral_centroid),
            "high_frequency_energy": float(high_freq_energy),
        }

    @staticmethod
    def suggest_crossfade_duration(
        audio1_props: Dict[str, float],
        audio2_props: Dict[str, float],
        default_duration: float = 0.1,
    ) -> float:
        """
        Suggest optimal crossfade duration based on audio properties.

        Args:
            audio1_props: Properties of first audio chunk
            audio2_props: Properties of second audio chunk
            default_duration: Default crossfade duration

        Returns:
            Suggested crossfade duration
        """

        # Longer crossfades for:
        # - High dynamic range content (classical music)
        # - Different spectral characteristics
        # - Low zero crossing rate (sustained tones)

        dr_diff = abs(audio1_props["dynamic_range"] - audio2_props["dynamic_range"])
        spectral_diff = abs(audio1_props["spectral_centroid"] - audio2_props["spectral_centroid"])

        # Normalize spectral difference
        spectral_diff_norm = spectral_diff / 10000.0  # Rough normalization

        # Calculate adjustment factor
        adjustment = 1.0
        adjustment += dr_diff * 0.01  # Dynamic range difference
        adjustment += spectral_diff_norm * 0.5  # Spectral difference

        # Reduce for percussive content (high ZCR)
        avg_zcr = (audio1_props["zero_crossing_rate"] + audio2_props["zero_crossing_rate"]) / 2
        if avg_zcr > 0.1:  # Percussive content
            adjustment *= 0.7

        # Clamp to reasonable range
        suggested_duration = default_duration * adjustment
        return max(0.05, min(0.3, suggested_duration))


class StreamingProtocol:
    """Defines protocol for streaming communication."""

    MESSAGE_TYPES = {
        "CHUNK": "audio_chunk",
        "STATUS": "status_update",
        "ERROR": "error",
        "CONTROL": "control_message",
        "HEARTBEAT": "heartbeat",
    }

    @staticmethod
    def create_chunk_message(
        session_id: str,
        chunk_id: int,
        audio_data: str,  # base64 encoded
        sample_rate: int,
        duration: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create audio chunk message."""
        return {
            "type": StreamingProtocol.MESSAGE_TYPES["CHUNK"],
            "session_id": session_id,
            "chunk_id": chunk_id,
            "audio_data": audio_data,
            "sample_rate": sample_rate,
            "duration": duration,
            "timestamp": time.time(),
            "metadata": metadata or {},
        }

    @staticmethod
    def create_status_message(
        session_id: str, status: str, details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create status update message."""
        return {
            "type": StreamingProtocol.MESSAGE_TYPES["STATUS"],
            "session_id": session_id,
            "status": status,
            "timestamp": time.time(),
            "details": details or {},
        }

    @staticmethod
    def create_error_message(
        session_id: str,
        error_code: str,
        error_message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create error message."""
        return {
            "type": StreamingProtocol.MESSAGE_TYPES["ERROR"],
            "session_id": session_id,
            "error_code": error_code,
            "error_message": error_message,
            "timestamp": time.time(),
            "details": details or {},
        }

    @staticmethod
    def create_control_message(
        session_id: str, command: str, parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create control message."""
        return {
            "type": StreamingProtocol.MESSAGE_TYPES["CONTROL"],
            "session_id": session_id,
            "command": command,
            "parameters": parameters or {},
            "timestamp": time.time(),
        }

    @staticmethod
    def create_heartbeat_message(session_id: str) -> Dict[str, Any]:
        """Create heartbeat message."""
        return {
            "type": StreamingProtocol.MESSAGE_TYPES["HEARTBEAT"],
            "session_id": session_id,
            "timestamp": time.time(),
        }


async def cleanup_old_sessions(session_manager, max_age_hours: float = 24.0):
    """Clean up old streaming sessions."""
    cutoff_time = time.time() - (max_age_hours * 3600)

    sessions_to_remove = []
    for session_id, session in session_manager.sessions.items():
        if session.info.created_at < cutoff_time:
            sessions_to_remove.append(session_id)

    for session_id in sessions_to_remove:
        await session_manager.remove_session(session_id)
        logger.info(f"Cleaned up old session {session_id}")

    return len(sessions_to_remove)


def validate_streaming_request(request_data: Dict[str, Any]) -> List[str]:
    """
    Validate streaming request data.

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Required fields
    if "prompt" not in request_data or not request_data["prompt"].strip():
        errors.append("Prompt is required and cannot be empty")

    # Optional numeric fields with ranges
    numeric_fields = {
        "chunk_duration": (0.1, 5.0),
        "temperature": (0.1, 2.0),
        "top_k": (1, 100),
        "top_p": (0.1, 1.0),
        "repetition_penalty": (1.0, 2.0),
        "crossfade_duration": (0.01, 0.5),
        "tempo": (60, 200),
    }

    for field, (min_val, max_val) in numeric_fields.items():
        if field in request_data:
            try:
                value = float(request_data[field])
                if not min_val <= value <= max_val:
                    errors.append(f"{field} must be between {min_val} and {max_val}")
            except (ValueError, TypeError):
                errors.append(f"{field} must be a valid number")

    # String fields with allowed values
    if "quality_mode" in request_data:
        allowed_modes = {"fast", "balanced", "quality"}
        if request_data["quality_mode"] not in allowed_modes:
            errors.append(f"quality_mode must be one of: {', '.join(allowed_modes)}")

    if "genre" in request_data:
        allowed_genres = {"jazz", "classical", "rock", "electronic", "ambient", "folk"}
        if request_data["genre"].lower() not in allowed_genres:
            errors.append(f"genre must be one of: {', '.join(allowed_genres)}")

    if "mood" in request_data:
        allowed_moods = {"happy", "sad", "energetic", "calm", "dramatic", "peaceful"}
        if request_data["mood"].lower() not in allowed_moods:
            errors.append(f"mood must be one of: {', '.join(allowed_moods)}")

    return errors
