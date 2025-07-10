"""
Audio processing service implementation.

Handles audio processing operations with proper error handling
and performance optimization.
"""

import logging
from typing import Any, Dict, List, Optional

import torch

from music_gen.core.config import AppConfig
from music_gen.core.exceptions import AudioProcessingError
from music_gen.core.interfaces.repositories import AudioRepository
from music_gen.core.interfaces.services import AudioProcessingService
from music_gen.utils.optional_imports import optional_import

logger = logging.getLogger(__name__)


class AudioProcessingServiceImpl(AudioProcessingService):
    """Implementation of audio processing service."""

    def __init__(self, audio_repository: AudioRepository, config: AppConfig):
        """Initialize audio processing service.

        Args:
            audio_repository: Repository for audio storage
            config: Application configuration
        """
        self._repository = audio_repository
        self._config = config

    async def process_audio(
        self, audio: torch.Tensor, sample_rate: int, operations: List[Dict[str, Any]]
    ) -> torch.Tensor:
        """Process audio with a chain of operations."""
        try:
            processed = audio.clone()

            for op in operations:
                op_type = op.get("type")
                params = op.get("params", {})

                if op_type == "normalize":
                    processed = await self.normalize_audio(
                        processed, method=params.get("method", "peak")
                    )
                elif op_type == "resample":
                    processed = await self.resample_audio(
                        processed, sample_rate, params.get("target_sr", sample_rate)
                    )
                    sample_rate = params.get("target_sr", sample_rate)
                elif op_type == "fade":
                    processed = self._apply_fade(
                        processed, sample_rate, params.get("fade_in", 0), params.get("fade_out", 0)
                    )
                elif op_type == "trim_silence":
                    processed = self._trim_silence(
                        processed, sample_rate, params.get("threshold", -40)
                    )
                elif op_type == "effects":
                    processed = await self.apply_effects(processed, params.get("effects", []))
                else:
                    logger.warning(f"Unknown operation type: {op_type}")

            return processed

        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            raise AudioProcessingError(f"Audio processing failed: {e}")

    async def normalize_audio(self, audio: torch.Tensor, method: str = "peak") -> torch.Tensor:
        """Normalize audio using specified method."""
        try:
            if method == "peak":
                # Peak normalization
                max_val = audio.abs().max()
                if max_val > 0:
                    return audio / max_val
                return audio

            elif method == "rms":
                # RMS normalization
                rms = torch.sqrt(torch.mean(audio**2))
                if rms > 0:
                    return audio / rms
                return audio

            elif method == "lufs":
                # Simplified LUFS normalization
                # In production, use proper LUFS calculation
                target_lufs = -14.0  # Standard for streaming
                current_lufs = self._estimate_lufs(audio)
                gain_db = target_lufs - current_lufs
                gain_linear = 10 ** (gain_db / 20)
                return audio * gain_linear

            else:
                raise ValueError(f"Unknown normalization method: {method}")

        except Exception as e:
            logger.error(f"Normalization failed: {e}")
            raise AudioProcessingError(f"Normalization failed: {e}")

    async def resample_audio(
        self, audio: torch.Tensor, orig_sr: int, target_sr: int
    ) -> torch.Tensor:
        """Resample audio to target sample rate."""
        if orig_sr == target_sr:
            return audio

        try:
            # Try using torchaudio first
            with optional_import("torchaudio") as torchaudio:
                if torchaudio is not None:
                    resampler = torchaudio.transforms.Resample(
                        orig_freq=orig_sr, new_freq=target_sr, resampling_method="kaiser_window"
                    )
                    # Move to same device as audio
                    resampler = resampler.to(audio.device)
                    return resampler(audio)

            # Fallback to simple interpolation
            logger.warning("torchaudio not available, using simple interpolation for resampling")
            return self._simple_resample(audio, orig_sr, target_sr)

        except Exception as e:
            logger.error(f"Resampling failed: {e}")
            raise AudioProcessingError(f"Resampling failed: {e}")

    def _simple_resample(self, audio: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
        """Simple resampling fallback using linear interpolation."""
        ratio = target_sr / orig_sr
        new_length = int(audio.shape[-1] * ratio)

        # Simple linear interpolation
        old_indices = torch.linspace(0, audio.shape[-1] - 1, new_length, device=audio.device)
        old_indices_floor = torch.floor(old_indices).long()
        old_indices_ceil = torch.ceil(old_indices).long()

        # Clamp to valid range
        old_indices_floor = torch.clamp(old_indices_floor, 0, audio.shape[-1] - 1)
        old_indices_ceil = torch.clamp(old_indices_ceil, 0, audio.shape[-1] - 1)

        # Interpolation weights
        weights = old_indices - old_indices_floor.float()

        if audio.dim() > 1:
            # Multi-channel
            resampled = torch.zeros(
                audio.shape[0], new_length, device=audio.device, dtype=audio.dtype
            )
            for ch in range(audio.shape[0]):
                floor_values = audio[ch, old_indices_floor]
                ceil_values = audio[ch, old_indices_ceil]
                resampled[ch] = floor_values * (1 - weights) + ceil_values * weights
        else:
            # Mono
            floor_values = audio[old_indices_floor]
            ceil_values = audio[old_indices_ceil]
            resampled = floor_values * (1 - weights) + ceil_values * weights

        return resampled

    async def mix_tracks(
        self, tracks: List[torch.Tensor], weights: Optional[List[float]] = None
    ) -> torch.Tensor:
        """Mix multiple audio tracks."""
        if not tracks:
            raise ValueError("No tracks to mix")

        try:
            # Use equal weights if not provided
            if weights is None:
                weights = [1.0 / len(tracks)] * len(tracks)

            # Ensure all tracks have same length
            max_length = max(track.shape[-1] for track in tracks)

            # Pad tracks to same length
            padded_tracks = []
            for track in tracks:
                if track.shape[-1] < max_length:
                    pad_length = max_length - track.shape[-1]
                    track = torch.nn.functional.pad(track, (0, pad_length))
                padded_tracks.append(track)

            # Mix tracks
            mixed = torch.zeros_like(padded_tracks[0])
            for track, weight in zip(padded_tracks, weights):
                mixed += track * weight

            # Normalize to prevent clipping
            return await self.normalize_audio(mixed, method="peak")

        except Exception as e:
            logger.error(f"Track mixing failed: {e}")
            raise AudioProcessingError(f"Track mixing failed: {e}")

    async def apply_effects(
        self, audio: torch.Tensor, effects: List[Dict[str, Any]]
    ) -> torch.Tensor:
        """Apply audio effects chain."""
        try:
            processed = audio.clone()

            for effect in effects:
                effect_type = effect.get("type")
                params = effect.get("params", {})

                if effect_type == "reverb":
                    processed = self._apply_reverb(
                        processed,
                        params.get("room_size", 0.5),
                        params.get("damping", 0.5),
                        params.get("wet_mix", 0.3),
                    )
                elif effect_type == "delay":
                    processed = self._apply_delay(
                        processed,
                        params.get("delay_time", 0.5),
                        params.get("feedback", 0.5),
                        params.get("wet_mix", 0.5),
                    )
                elif effect_type == "eq":
                    processed = self._apply_eq(
                        processed,
                        params.get("low_gain", 0),
                        params.get("mid_gain", 0),
                        params.get("high_gain", 0),
                    )
                elif effect_type == "compression":
                    processed = self._apply_compression(
                        processed,
                        params.get("threshold", -10),
                        params.get("ratio", 4),
                        params.get("attack", 0.005),
                        params.get("release", 0.1),
                    )
                else:
                    logger.warning(f"Unknown effect type: {effect_type}")

            return processed

        except Exception as e:
            logger.error(f"Effects processing failed: {e}")
            raise AudioProcessingError(f"Effects processing failed: {e}")

    def _apply_fade(
        self, audio: torch.Tensor, sample_rate: int, fade_in: float, fade_out: float
    ) -> torch.Tensor:
        """Apply fade in/out to audio."""
        if fade_in <= 0 and fade_out <= 0:
            return audio

        result = audio.clone()

        # Apply fade in
        if fade_in > 0:
            fade_in_samples = int(fade_in * sample_rate)
            fade_in_curve = torch.linspace(0, 1, fade_in_samples, device=audio.device)
            result[..., :fade_in_samples] *= fade_in_curve

        # Apply fade out
        if fade_out > 0:
            fade_out_samples = int(fade_out * sample_rate)
            fade_out_curve = torch.linspace(1, 0, fade_out_samples, device=audio.device)
            result[..., -fade_out_samples:] *= fade_out_curve

        return result

    def _trim_silence(
        self, audio: torch.Tensor, sample_rate: int, threshold_db: float
    ) -> torch.Tensor:
        """Trim silence from beginning and end of audio."""
        # Convert to amplitude threshold
        threshold = 10 ** (threshold_db / 20)

        # Find non-silent samples
        non_silent = audio.abs().max(dim=0)[0] > threshold
        indices = non_silent.nonzero(as_tuple=True)[0]

        if len(indices) == 0:
            return audio

        # Trim audio
        start_idx = indices[0].item()
        end_idx = indices[-1].item() + 1

        return audio[..., start_idx:end_idx]

    def _estimate_lufs(self, audio: torch.Tensor) -> float:
        """Estimate LUFS (simplified version)."""
        # This is a simplified estimation
        # In production, use proper LUFS measurement
        rms = torch.sqrt(torch.mean(audio**2))
        lufs = 20 * torch.log10(rms + 1e-8) - 0.691
        return lufs.item()

    def _apply_reverb(
        self, audio: torch.Tensor, room_size: float, damping: float, wet_mix: float
    ) -> torch.Tensor:
        """Apply reverb effect using impulse response convolution."""
        # Create a simple impulse response based on room size
        ir_length = int(
            room_size * self._config.get("default_sample_rate", 32000) * 0.5
        )  # Up to 0.5 seconds
        ir_length = max(1024, min(ir_length, 16384))  # Reasonable bounds

        # Generate exponentially decaying impulse response
        t = torch.arange(ir_length, dtype=audio.dtype, device=audio.device)
        decay_rate = 10.0 * (1.0 - room_size)  # Higher room_size = slower decay
        impulse_response = torch.exp(-decay_rate * t / ir_length)

        # Apply damping (high frequency rolloff)
        if damping > 0:
            # Simple low-pass filter for damping
            cutoff = 1.0 - damping * 0.8  # Higher damping = lower cutoff
            for i in range(1, len(impulse_response)):
                impulse_response[i] = impulse_response[i] * cutoff + impulse_response[i - 1] * (
                    1 - cutoff
                )

        # Add some random reflections for realism
        random_reflections = torch.randn(ir_length, dtype=audio.dtype, device=audio.device) * 0.1
        impulse_response = impulse_response + random_reflections * impulse_response

        # Normalize impulse response
        impulse_response = impulse_response / impulse_response.abs().max()

        # Apply convolution reverb
        # Handle multi-channel audio
        if audio.dim() > 1:
            reverb_audio = torch.zeros_like(audio)
            for ch in range(audio.shape[0]):
                # Use valid convolution to avoid length changes
                conv_result = torch.nn.functional.conv1d(
                    audio[ch : ch + 1].unsqueeze(0),
                    impulse_response.unsqueeze(0).unsqueeze(0),
                    padding=ir_length // 2,
                )
                reverb_audio[ch] = conv_result.squeeze()[: audio.shape[-1]]
        else:
            conv_result = torch.nn.functional.conv1d(
                audio.unsqueeze(0).unsqueeze(0),
                impulse_response.unsqueeze(0).unsqueeze(0),
                padding=ir_length // 2,
            )
            reverb_audio = conv_result.squeeze()[: audio.shape[-1]]

        # Mix wet and dry signals
        return audio * (1.0 - wet_mix) + reverb_audio * wet_mix

    def _apply_delay(
        self, audio: torch.Tensor, delay_time: float, feedback: float, wet_mix: float
    ) -> torch.Tensor:
        """Apply delay effect with feedback."""
        sample_rate = self._config.get("default_sample_rate", 32000)
        delay_samples = int(delay_time * sample_rate)

        if delay_samples <= 0:
            return audio

        # Clamp feedback to prevent instability
        feedback = max(0.0, min(feedback, 0.95))

        # Create delayed version
        if audio.dim() > 1:
            # Multi-channel
            delayed_audio = torch.zeros_like(audio)
            for ch in range(audio.shape[0]):
                channel_audio = audio[ch]
                delayed_channel = torch.zeros_like(channel_audio)

                # Simple delay line with feedback
                for i in range(len(channel_audio)):
                    delayed_channel[i] = channel_audio[i]
                    if i >= delay_samples:
                        delayed_channel[i] += delayed_channel[i - delay_samples] * feedback

                delayed_audio[ch] = delayed_channel
        else:
            # Mono
            delayed_audio = torch.zeros_like(audio)
            for i in range(len(audio)):
                delayed_audio[i] = audio[i]
                if i >= delay_samples:
                    delayed_audio[i] += delayed_audio[i - delay_samples] * feedback

        # Mix wet and dry signals
        return audio * (1.0 - wet_mix) + delayed_audio * wet_mix

    def _apply_eq(
        self, audio: torch.Tensor, low_gain: float, mid_gain: float, high_gain: float
    ) -> torch.Tensor:
        """Apply 3-band EQ using frequency domain filtering."""
        # Convert gains from dB to linear
        low_gain_linear = 10 ** (low_gain / 20)
        mid_gain_linear = 10 ** (mid_gain / 20)
        high_gain_linear = 10 ** (high_gain / 20)

        # Define frequency bands (normalized frequencies)
        sample_rate = self._config.get("default_sample_rate", 32000)
        low_cutoff = 300  # Hz
        high_cutoff = 3000  # Hz

        # Convert to normalized frequencies
        low_norm = low_cutoff / (sample_rate / 2)
        high_norm = high_cutoff / (sample_rate / 2)

        # Apply EQ using FFT
        if audio.dim() > 1:
            # Multi-channel
            eq_audio = torch.zeros_like(audio)
            for ch in range(audio.shape[0]):
                eq_audio[ch] = self._apply_eq_channel(
                    audio[ch],
                    low_gain_linear,
                    mid_gain_linear,
                    high_gain_linear,
                    low_norm,
                    high_norm,
                )
        else:
            # Mono
            eq_audio = self._apply_eq_channel(
                audio, low_gain_linear, mid_gain_linear, high_gain_linear, low_norm, high_norm
            )

        return eq_audio

    def _apply_eq_channel(
        self,
        audio: torch.Tensor,
        low_gain: float,
        mid_gain: float,
        high_gain: float,
        low_norm: float,
        high_norm: float,
    ) -> torch.Tensor:
        """Apply EQ to a single channel."""
        # FFT
        audio_fft = torch.fft.fft(audio)
        freqs = torch.fft.fftfreq(len(audio), device=audio.device)

        # Create gain curve
        gain_curve = torch.ones_like(freqs, dtype=torch.float32)

        # Normalize frequencies to [0, 1]
        freqs_norm = torch.abs(freqs) * 2  # *2 because fftfreq goes to 0.5

        # Apply gains to frequency bands
        low_mask = freqs_norm <= low_norm
        high_mask = freqs_norm >= high_norm
        mid_mask = (~low_mask) & (~high_mask)

        gain_curve[low_mask] *= low_gain
        gain_curve[mid_mask] *= mid_gain
        gain_curve[high_mask] *= high_gain

        # Apply smooth transitions between bands
        transition_width = 0.1
        for i in range(len(freqs_norm)):
            f = freqs_norm[i]

            # Low to mid transition
            if abs(f - low_norm) < transition_width:
                blend = 0.5 + 0.5 * torch.cos(torch.pi * (f - low_norm) / transition_width)
                gain_curve[i] = low_gain * blend + mid_gain * (1 - blend)

            # Mid to high transition
            elif abs(f - high_norm) < transition_width:
                blend = 0.5 + 0.5 * torch.cos(torch.pi * (f - high_norm) / transition_width)
                gain_curve[i] = mid_gain * blend + high_gain * (1 - blend)

        # Apply gains and return to time domain
        audio_fft_eq = audio_fft * gain_curve.to(audio_fft.dtype)
        return torch.fft.ifft(audio_fft_eq).real

    def _apply_compression(
        self, audio: torch.Tensor, threshold: float, ratio: float, attack: float, release: float
    ) -> torch.Tensor:
        """Apply dynamic range compression."""
        # Convert threshold from dB to linear
        threshold_linear = 10 ** (threshold / 20)

        # Clamp ratio to reasonable values
        ratio = max(1.0, min(ratio, 20.0))

        # Convert attack/release times to coefficients
        sample_rate = self._config.get("default_sample_rate", 32000)
        attack_coeff = torch.exp(-1.0 / (attack * sample_rate)) if attack > 0 else 0.0
        release_coeff = torch.exp(-1.0 / (release * sample_rate)) if release > 0 else 0.0

        if audio.dim() > 1:
            # Multi-channel - apply compression to each channel
            compressed_audio = torch.zeros_like(audio)
            for ch in range(audio.shape[0]):
                compressed_audio[ch] = self._apply_compression_channel(
                    audio[ch], threshold_linear, ratio, attack_coeff, release_coeff
                )
        else:
            # Mono
            compressed_audio = self._apply_compression_channel(
                audio, threshold_linear, ratio, attack_coeff, release_coeff
            )

        return compressed_audio

    def _apply_compression_channel(
        self,
        audio: torch.Tensor,
        threshold: float,
        ratio: float,
        attack_coeff: float,
        release_coeff: float,
    ) -> torch.Tensor:
        """Apply compression to a single channel."""
        # Initialize gain reduction envelope
        gain_reduction = torch.zeros_like(audio)
        envelope = 0.0

        for i in range(len(audio)):
            # Get current sample magnitude
            current_level = abs(audio[i].item())

            # Update envelope follower
            if current_level > envelope:
                # Attack
                envelope = current_level + (envelope - current_level) * attack_coeff
            else:
                # Release
                envelope = current_level + (envelope - current_level) * release_coeff

            # Calculate gain reduction
            if envelope > threshold:
                # Above threshold - apply compression
                excess = envelope - threshold
                compressed_excess = excess / ratio
                target_level = threshold + compressed_excess
                gain_reduction[i] = target_level / envelope if envelope > 0 else 1.0
            else:
                # Below threshold - no gain reduction
                gain_reduction[i] = 1.0

        # Apply gain reduction with makeup gain
        makeup_gain = ratio**0.5  # Simple makeup gain calculation
        return audio * gain_reduction * makeup_gain
