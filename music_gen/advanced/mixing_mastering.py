"""
Automatic Mixing and Mastering System

Implements intelligent audio processing pipeline for professional-quality
music production including EQ, compression, reverb, stereo imaging,
and mastering chain automation.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..utils.audio.effects import AudioEffectsProcessor
from ..utils.audio.processing import AudioProcessor


class ProcessingQuality(Enum):
    """Audio processing quality levels."""

    DRAFT = "draft"
    STANDARD = "standard"
    HIGH = "high"
    MASTERING = "mastering"


class LoudnessStandard(Enum):
    """Industry loudness standards."""

    SPOTIFY = -14.0  # LUFS
    APPLE_MUSIC = -16.0  # LUFS
    YOUTUBE = -14.0  # LUFS
    BROADCAST = -23.0  # LUFS
    STREAMING = -14.0  # LUFS
    CD = -9.0  # LUFS (RMS approximation)


@dataclass
class MixingPreset:
    """Preset configuration for mixing parameters."""

    name: str
    genre: str
    eq_curve: Dict[int, float]  # Frequency (Hz) -> Gain (dB)
    compression: Dict[str, float]
    reverb: Dict[str, float]
    stereo_width: float = 1.0
    harmonic_excitement: float = 0.0
    tape_saturation: float = 0.0
    description: str = ""


@dataclass
class MasteringChain:
    """Mastering chain configuration."""

    eq_settings: Dict[int, float]
    multiband_compression: Dict[str, Dict[str, float]]
    stereo_enhancement: Dict[str, float]
    harmonic_enhancement: Dict[str, float]
    limiting: Dict[str, float]
    target_loudness: float = -14.0  # LUFS
    true_peak_limit: float = -1.0  # dBTP


class IntelligentEQ(nn.Module):
    """Neural network-based intelligent EQ system."""

    def __init__(self, num_bands: int = 31, sample_rate: int = 44100):
        super().__init__()

        self.num_bands = num_bands
        self.sample_rate = sample_rate

        # Generate frequency bands (31-band graphic EQ)
        self.frequencies = self._generate_frequency_bands()

        # EQ prediction network
        self.eq_predictor = nn.Sequential(
            nn.Linear(512, 256),  # Audio feature input
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_bands),
            nn.Tanh(),  # Output range [-1, 1] for ±12dB
        )

        # Genre-specific EQ embeddings
        self.genre_embeddings = nn.Embedding(20, 64)  # 20 genres

        # Spectral analysis network
        self.spectral_analyzer = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=1024, stride=512),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=64, stride=32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(128),
            nn.Flatten(),
            nn.Linear(64 * 128, 512),
            nn.ReLU(),
        )

    def _generate_frequency_bands(self) -> np.ndarray:
        """Generate ISO 31-band graphic EQ frequencies."""
        # Standard 31-band frequencies from 20 Hz to 20 kHz
        frequencies = np.array(
            [
                20,
                25,
                31.5,
                40,
                50,
                63,
                80,
                100,
                125,
                160,
                200,
                250,
                315,
                400,
                500,
                630,
                800,
                1000,
                1250,
                1600,
                2000,
                2500,
                3150,
                4000,
                5000,
                6300,
                8000,
                10000,
                12500,
                16000,
                20000,
            ]
        )
        return frequencies

    def analyze_spectrum(self, audio: torch.Tensor) -> torch.Tensor:
        """Analyze audio spectrum for EQ prediction."""
        # Ensure audio is the right shape for conv1d
        if audio.dim() == 2:
            audio = audio.mean(dim=0, keepdim=True)  # Convert stereo to mono
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        # Spectral analysis
        features = self.spectral_analyzer(audio)
        return features

    def predict_eq_curve(
        self, audio: torch.Tensor, genre: str = "pop", target_loudness: float = -14.0
    ) -> Dict[int, float]:
        """Predict optimal EQ curve for given audio."""

        # Analyze spectrum
        spectral_features = self.analyze_spectrum(audio)

        # Get genre embedding
        genre_map = {
            "pop": 0,
            "rock": 1,
            "jazz": 2,
            "electronic": 3,
            "classical": 4,
            "hip-hop": 5,
            "country": 6,
            "r&b": 7,
            "folk": 8,
            "blues": 9,
            "reggae": 10,
            "funk": 11,
            "soul": 12,
            "punk": 13,
            "metal": 14,
            "indie": 15,
            "ambient": 16,
            "world": 17,
            "experimental": 18,
            "other": 19,
        }
        genre_idx = genre_map.get(genre, 19)
        genre_emb = self.genre_embeddings(torch.tensor([genre_idx], device=audio.device))

        # Combine features
        combined_features = torch.cat([spectral_features, genre_emb.flatten()], dim=-1)

        # Predict EQ gains
        eq_gains = self.eq_predictor(combined_features)
        eq_gains = eq_gains * 12.0  # Scale to ±12dB range

        # Convert to frequency dict
        eq_curve = {}
        for i, freq in enumerate(self.frequencies):
            if i < len(eq_gains[0]):
                eq_curve[int(freq)] = float(eq_gains[0][i].item())

        return eq_curve


class SmartCompressor:
    """Intelligent multiband compression system."""

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate

        # Frequency bands for multiband compression
        self.bands = {
            "sub": (20, 60),
            "low": (60, 200),
            "low_mid": (200, 800),
            "mid": (800, 3200),
            "high_mid": (3200, 8000),
            "high": (8000, 20000),
        }

    def analyze_dynamics(self, audio: torch.Tensor) -> Dict[str, float]:
        """Analyze audio dynamics for compression settings."""

        # Calculate RMS and peak levels
        rms = torch.sqrt(torch.mean(audio**2))
        peak = torch.max(torch.abs(audio))

        # Calculate crest factor
        crest_factor = 20 * torch.log10(peak / (rms + 1e-8))

        # Calculate dynamic range
        percentile_95 = torch.quantile(torch.abs(audio), 0.95)
        percentile_5 = torch.quantile(torch.abs(audio), 0.05)
        dynamic_range = 20 * torch.log10(percentile_95 / (percentile_5 + 1e-8))

        return {
            "rms_db": 20 * torch.log10(rms + 1e-8).item(),
            "peak_db": 20 * torch.log10(peak + 1e-8).item(),
            "crest_factor": crest_factor.item(),
            "dynamic_range": dynamic_range.item(),
        }

    def predict_compression_settings(
        self, audio: torch.Tensor, genre: str = "pop", target_dynamics: str = "balanced"
    ) -> Dict[str, Dict[str, float]]:
        """Predict optimal compression settings for each band."""

        dynamics = self.analyze_dynamics(audio)

        # Base settings by genre
        genre_settings = {
            "pop": {"ratio": 4.0, "attack": 3.0, "release": 100.0, "threshold": -12.0},
            "rock": {"ratio": 6.0, "attack": 1.0, "release": 50.0, "threshold": -8.0},
            "jazz": {"ratio": 2.5, "attack": 10.0, "release": 200.0, "threshold": -16.0},
            "electronic": {"ratio": 8.0, "attack": 0.5, "release": 30.0, "threshold": -6.0},
            "classical": {"ratio": 2.0, "attack": 20.0, "release": 300.0, "threshold": -20.0},
        }

        base_settings = genre_settings.get(genre, genre_settings["pop"])

        # Adjust based on dynamics analysis
        if dynamics["dynamic_range"] > 20:
            # High dynamic range - lighter compression
            base_settings["ratio"] *= 0.8
            base_settings["threshold"] -= 3.0
        elif dynamics["dynamic_range"] < 10:
            # Low dynamic range - heavier compression
            base_settings["ratio"] *= 1.2
            base_settings["threshold"] += 2.0

        # Per-band settings
        band_settings = {}
        for band_name, (low_freq, high_freq) in self.bands.items():
            settings = base_settings.copy()

            # Frequency-specific adjustments
            if band_name == "sub":
                settings["ratio"] *= 1.5  # More compression on sub bass
                settings["attack"] *= 2.0  # Slower attack for bass
            elif band_name == "high":
                settings["ratio"] *= 0.8  # Less compression on highs
                settings["attack"] *= 0.5  # Faster attack for highs

            band_settings[band_name] = settings

        return band_settings


class StereoProcessor:
    """Advanced stereo imaging and spatial processing."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def analyze_stereo_image(self, audio: torch.Tensor) -> Dict[str, float]:
        """Analyze stereo imaging characteristics."""

        if audio.shape[0] != 2:
            return {"mono_compatibility": 1.0, "stereo_width": 0.0, "phase_correlation": 1.0}

        left = audio[0]
        right = audio[1]

        # Calculate mid/side
        mid = (left + right) / 2
        side = (left - right) / 2

        # Stereo width calculation
        mid_energy = torch.mean(mid**2)
        side_energy = torch.mean(side**2)
        stereo_width = side_energy / (mid_energy + side_energy + 1e-8)

        # Phase correlation
        correlation = torch.corrcoef(torch.stack([left, right]))[0, 1]

        # Mono compatibility (how much is lost when summed to mono)
        mono_sum = left + right
        mono_energy = torch.mean(mono_sum**2)
        stereo_energy = torch.mean(left**2) + torch.mean(right**2)
        mono_compatibility = mono_energy / (stereo_energy + 1e-8)

        return {
            "stereo_width": float(stereo_width.item()),
            "phase_correlation": float(correlation.item()),
            "mono_compatibility": float(mono_compatibility.item()),
        }

    def enhance_stereo_image(
        self, audio: torch.Tensor, width_factor: float = 1.2, bass_mono_freq: float = 120.0
    ) -> torch.Tensor:
        """Enhance stereo image while maintaining mono compatibility."""

        if audio.shape[0] != 2:
            return audio

        left = audio[0]
        right = audio[1]

        # Convert to mid/side
        mid = (left + right) / 2
        side = (left - right) / 2

        # Apply high-pass filter to side channel to keep bass centered
        # (Simplified implementation - in practice would use proper filters)
        nyquist = 22050  # Assuming 44.1kHz sample rate
        cutoff_norm = bass_mono_freq / nyquist

        # Simple high-pass approximation
        if cutoff_norm < 0.5:
            alpha = np.exp(-2 * np.pi * cutoff_norm)
            side_filtered = torch.zeros_like(side)
            side_filtered[0] = side[0]
            for i in range(1, len(side)):
                side_filtered[i] = alpha * side_filtered[i - 1] + (1 - alpha) * side[i]
            side = side_filtered

        # Enhance stereo width
        side_enhanced = side * width_factor

        # Convert back to left/right
        left_enhanced = mid + side_enhanced
        right_enhanced = mid - side_enhanced

        return torch.stack([left_enhanced, right_enhanced])


class AutoMixingEngine:
    """Intelligent automatic mixing engine."""

    def __init__(
        self,
        audio_processor: AudioProcessor,
        effects_processor: AudioEffectsProcessor,
        sample_rate: int = 44100,
    ):
        self.audio_processor = audio_processor
        self.effects_processor = effects_processor
        self.sample_rate = sample_rate

        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.intelligent_eq = IntelligentEQ(sample_rate=sample_rate)
        self.smart_compressor = SmartCompressor(sample_rate=sample_rate)
        self.stereo_processor = StereoProcessor()

        # Load mixing presets
        self.mixing_presets = self._load_mixing_presets()

    def _load_mixing_presets(self) -> Dict[str, MixingPreset]:
        """Load genre-specific mixing presets."""

        presets = {
            "pop_modern": MixingPreset(
                name="Modern Pop",
                genre="pop",
                eq_curve={60: 1.0, 200: -0.5, 800: 0.5, 2000: 1.5, 5000: 2.0, 10000: 1.0},
                compression={
                    "ratio": 4.0,
                    "attack": 3.0,
                    "release": 100.0,
                    "threshold": -12.0,
                    "knee": 2.0,
                    "makeup_gain": 3.0,
                },
                reverb={"room_size": 0.3, "damping": 0.5, "wet_level": 0.15, "dry_level": 0.85},
                stereo_width=1.2,
                harmonic_excitement=0.1,
                description="Modern pop mixing with enhanced presence and punch",
            ),
            "rock_classic": MixingPreset(
                name="Classic Rock",
                genre="rock",
                eq_curve={100: 2.0, 400: -1.0, 1000: 0.0, 3000: 2.5, 8000: 1.5},
                compression={
                    "ratio": 6.0,
                    "attack": 1.0,
                    "release": 50.0,
                    "threshold": -8.0,
                    "knee": 1.0,
                    "makeup_gain": 4.0,
                },
                reverb={"room_size": 0.5, "damping": 0.3, "wet_level": 0.25, "dry_level": 0.75},
                stereo_width=1.0,
                harmonic_excitement=0.3,
                tape_saturation=0.2,
                description="Classic rock mix with warmth and aggression",
            ),
            "jazz_acoustic": MixingPreset(
                name="Acoustic Jazz",
                genre="jazz",
                eq_curve={80: 0.5, 300: 0.0, 1000: 0.5, 4000: 1.0, 12000: 0.5},
                compression={
                    "ratio": 2.5,
                    "attack": 10.0,
                    "release": 200.0,
                    "threshold": -16.0,
                    "knee": 3.0,
                    "makeup_gain": 2.0,
                },
                reverb={"room_size": 0.7, "damping": 0.6, "wet_level": 0.3, "dry_level": 0.7},
                stereo_width=1.1,
                description="Natural jazz mix with spacious reverb",
            ),
            "electronic_dance": MixingPreset(
                name="Electronic Dance",
                genre="electronic",
                eq_curve={50: 3.0, 100: 1.0, 500: -1.0, 2000: 1.0, 8000: 2.5, 16000: 1.5},
                compression={
                    "ratio": 8.0,
                    "attack": 0.5,
                    "release": 30.0,
                    "threshold": -6.0,
                    "knee": 1.0,
                    "makeup_gain": 5.0,
                },
                reverb={"room_size": 0.2, "damping": 0.2, "wet_level": 0.1, "dry_level": 0.9},
                stereo_width=1.4,
                harmonic_excitement=0.2,
                description="Punchy electronic mix with enhanced stereo width",
            ),
        }

        return presets

    def auto_mix(
        self,
        audio: torch.Tensor,
        genre: str = "pop",
        mix_style: str = "balanced",
        target_loudness: float = -14.0,
        quality: ProcessingQuality = ProcessingQuality.STANDARD,
    ) -> torch.Tensor:
        """
        Automatically mix audio with intelligent processing.

        Args:
            audio: Input audio tensor
            genre: Musical genre for style selection
            mix_style: Mixing style preference
            target_loudness: Target loudness in LUFS
            quality: Processing quality level

        Returns:
            Mixed audio tensor
        """

        self.logger.info(f"Auto-mixing {genre} track with {mix_style} style")

        # Select appropriate preset
        preset_key = f"{genre}_{mix_style}"
        if preset_key not in self.mixing_presets:
            # Fall back to genre-based preset
            available_presets = [k for k in self.mixing_presets.keys() if k.startswith(genre)]
            if available_presets:
                preset_key = available_presets[0]
            else:
                preset_key = "pop_modern"  # Ultimate fallback

        preset = self.mixing_presets[preset_key]

        # Start with original audio
        processed_audio = audio.clone()

        # 1. Intelligent EQ
        self.logger.debug("Applying intelligent EQ")
        eq_curve = self.intelligent_eq.predict_eq_curve(audio, genre, target_loudness)

        # Blend AI prediction with preset curve
        blended_curve = {}
        for freq in set(list(eq_curve.keys()) + list(preset.eq_curve.keys())):
            ai_gain = eq_curve.get(freq, 0.0)
            preset_gain = preset.eq_curve.get(freq, 0.0)
            # 70% AI, 30% preset
            blended_curve[freq] = 0.7 * ai_gain + 0.3 * preset_gain

        processed_audio = self.effects_processor.apply_parametric_eq(processed_audio, blended_curve)

        # 2. Smart Compression
        self.logger.debug("Applying smart compression")
        compression_settings = self.smart_compressor.predict_compression_settings(
            processed_audio, genre, mix_style
        )

        # Apply multiband compression
        processed_audio = self.effects_processor.apply_multiband_compression(
            processed_audio, compression_settings
        )

        # 3. Harmonic Enhancement
        if preset.harmonic_excitement > 0:
            self.logger.debug("Applying harmonic enhancement")
            processed_audio = self.effects_processor.apply_harmonic_enhancement(
                processed_audio, amount=preset.harmonic_excitement
            )

        # 4. Tape Saturation
        if preset.tape_saturation > 0:
            self.logger.debug("Applying tape saturation")
            processed_audio = self.effects_processor.apply_tape_saturation(
                processed_audio, amount=preset.tape_saturation
            )

        # 5. Reverb
        self.logger.debug("Applying reverb")
        processed_audio = self.effects_processor.apply_reverb(processed_audio, **preset.reverb)

        # 6. Stereo Enhancement
        if processed_audio.dim() == 2 and processed_audio.shape[0] == 2:
            self.logger.debug("Enhancing stereo image")
            processed_audio = self.stereo_processor.enhance_stereo_image(
                processed_audio, width_factor=preset.stereo_width
            )

        # 7. Final Level Adjustment
        processed_audio = self.audio_processor.normalize_loudness(
            processed_audio, target_lufs=target_loudness
        )

        self.logger.info("Auto-mixing completed")
        return processed_audio


class MasteringEngine:
    """Professional mastering engine with intelligent processing."""

    def __init__(
        self,
        audio_processor: AudioProcessor,
        effects_processor: AudioEffectsProcessor,
        sample_rate: int = 44100,
    ):
        self.audio_processor = audio_processor
        self.effects_processor = effects_processor
        self.sample_rate = sample_rate

        self.logger = logging.getLogger(__name__)

        # Load mastering chains
        self.mastering_chains = self._load_mastering_chains()

    def _load_mastering_chains(self) -> Dict[str, MasteringChain]:
        """Load mastering chain presets."""

        chains = {
            "streaming": MasteringChain(
                eq_settings={30: -0.5, 80: 0.5, 200: 0.0, 1000: 0.3, 5000: 0.8, 12000: 0.5},
                multiband_compression={
                    "low": {"ratio": 3.0, "threshold": -18.0, "attack": 10.0, "release": 100.0},
                    "mid": {"ratio": 2.5, "threshold": -15.0, "attack": 5.0, "release": 80.0},
                    "high": {"ratio": 2.0, "threshold": -12.0, "attack": 2.0, "release": 50.0},
                },
                stereo_enhancement={"width": 1.05, "bass_mono_freq": 120.0},
                harmonic_enhancement={"amount": 0.05, "frequency": 2000.0},
                limiting={"threshold": -1.0, "release": 5.0, "lookahead": 10.0},
                target_loudness=-14.0,
                true_peak_limit=-1.0,
            ),
            "cd": MasteringChain(
                eq_settings={40: 0.0, 100: 0.3, 300: 0.0, 1500: 0.5, 6000: 1.0, 15000: 0.3},
                multiband_compression={
                    "low": {"ratio": 2.5, "threshold": -20.0, "attack": 15.0, "release": 120.0},
                    "mid": {"ratio": 2.0, "threshold": -18.0, "attack": 8.0, "release": 100.0},
                    "high": {"ratio": 1.8, "threshold": -15.0, "attack": 3.0, "release": 60.0},
                },
                stereo_enhancement={"width": 1.0, "bass_mono_freq": 100.0},
                harmonic_enhancement={"amount": 0.08, "frequency": 3000.0},
                limiting={"threshold": -0.3, "release": 3.0, "lookahead": 8.0},
                target_loudness=-9.0,
                true_peak_limit=-0.1,
            ),
            "vinyl": MasteringChain(
                eq_settings={
                    20: -2.0,
                    50: 0.0,
                    150: 0.5,
                    800: 0.0,
                    3000: 0.3,
                    10000: -0.5,
                    16000: -1.0,
                },
                multiband_compression={
                    "low": {"ratio": 2.0, "threshold": -22.0, "attack": 20.0, "release": 150.0},
                    "mid": {"ratio": 1.8, "threshold": -20.0, "attack": 12.0, "release": 120.0},
                    "high": {"ratio": 1.5, "threshold": -18.0, "attack": 5.0, "release": 80.0},
                },
                stereo_enhancement={"width": 0.9, "bass_mono_freq": 150.0},
                harmonic_enhancement={"amount": 0.12, "frequency": 1000.0},
                limiting={"threshold": -2.0, "release": 10.0, "lookahead": 5.0},
                target_loudness=-16.0,
                true_peak_limit=-2.0,
            ),
        }

        return chains

    def master_track(
        self,
        audio: torch.Tensor,
        target_format: str = "streaming",
        genre: str = "pop",
        reference_track: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Master audio track for target format.

        Args:
            audio: Input mixed audio
            target_format: Target format (streaming, cd, vinyl)
            genre: Musical genre for optimization
            reference_track: Optional reference for matching

        Returns:
            Mastered audio tensor
        """

        self.logger.info(f"Mastering {genre} track for {target_format}")

        # Get mastering chain
        chain = self.mastering_chains.get(target_format, self.mastering_chains["streaming"])

        # Start with original audio
        processed_audio = audio.clone()

        # 1. Pre-mastering EQ
        self.logger.debug("Applying mastering EQ")
        processed_audio = self.effects_processor.apply_parametric_eq(
            processed_audio, chain.eq_settings
        )

        # 2. Multiband Compression
        self.logger.debug("Applying multiband compression")
        processed_audio = self.effects_processor.apply_multiband_compression(
            processed_audio, chain.multiband_compression
        )

        # 3. Stereo Enhancement
        if processed_audio.dim() == 2 and processed_audio.shape[0] == 2:
            self.logger.debug("Applying stereo enhancement")
            processed_audio = self.effects_processor.enhance_stereo_width(
                processed_audio, **chain.stereo_enhancement
            )

        # 4. Harmonic Enhancement
        self.logger.debug("Applying harmonic enhancement")
        processed_audio = self.effects_processor.apply_harmonic_enhancement(
            processed_audio, **chain.harmonic_enhancement
        )

        # 5. Loudness Normalization
        self.logger.debug("Normalizing loudness")
        processed_audio = self.audio_processor.normalize_loudness(
            processed_audio, target_lufs=chain.target_loudness
        )

        # 6. True Peak Limiting
        self.logger.debug("Applying peak limiting")
        processed_audio = self.effects_processor.apply_limiter(
            processed_audio,
            threshold=chain.limiting["threshold"],
            release=chain.limiting["release"],
        )

        # 7. Final True Peak Check
        peak_level = torch.max(torch.abs(processed_audio))
        if 20 * torch.log10(peak_level) > chain.true_peak_limit:
            # Apply gentle limiting to meet true peak requirement
            reduction_needed = 20 * torch.log10(peak_level) - chain.true_peak_limit
            gain_reduction = 10 ** (-reduction_needed / 20)
            processed_audio = processed_audio * gain_reduction

        self.logger.info("Mastering completed")
        return processed_audio

    def analyze_master(self, audio: torch.Tensor) -> Dict[str, Any]:
        """Analyze mastered audio for quality metrics."""

        analysis = {}

        # Loudness analysis
        lufs = self.audio_processor.calculate_lufs(audio)
        analysis["loudness_lufs"] = lufs

        # True peak analysis
        true_peak = torch.max(torch.abs(audio))
        analysis["true_peak_dbtp"] = 20 * torch.log10(true_peak).item()

        # Dynamic range
        peak = torch.max(torch.abs(audio))
        rms = torch.sqrt(torch.mean(audio**2))
        analysis["peak_to_rms_db"] = 20 * torch.log10(peak / (rms + 1e-8)).item()

        # Frequency balance
        spectrum = torch.fft.rfft(audio.mean(dim=0) if audio.dim() == 2 else audio)
        spectrum_mag = torch.abs(spectrum)

        # Define frequency bands
        nyquist = self.sample_rate / 2
        freqs = torch.linspace(0, nyquist, len(spectrum_mag))

        low_band = spectrum_mag[(freqs >= 20) & (freqs <= 200)].mean()
        mid_band = spectrum_mag[(freqs >= 200) & (freqs <= 2000)].mean()
        high_band = spectrum_mag[(freqs >= 2000) & (freqs <= nyquist)].mean()

        total_energy = low_band + mid_band + high_band
        analysis["frequency_balance"] = {
            "low_percent": (low_band / total_energy * 100).item(),
            "mid_percent": (mid_band / total_energy * 100).item(),
            "high_percent": (high_band / total_energy * 100).item(),
        }

        # Stereo analysis
        if audio.dim() == 2 and audio.shape[0] == 2:
            stereo_analysis = {
                "is_stereo": True,
                "correlation": torch.corrcoef(audio)[0, 1].item(),
                "width": self._calculate_stereo_width(audio),
            }
        else:
            stereo_analysis = {"is_stereo": False}

        analysis["stereo"] = stereo_analysis

        return analysis

    def _calculate_stereo_width(self, audio: torch.Tensor) -> float:
        """Calculate stereo width metric."""
        if audio.shape[0] != 2:
            return 0.0

        left, right = audio[0], audio[1]
        mid = (left + right) / 2
        side = (left - right) / 2

        mid_energy = torch.mean(mid**2)
        side_energy = torch.mean(side**2)

        return (side_energy / (mid_energy + side_energy + 1e-8)).item()


class AutoMasteringManager:
    """High-level manager for automatic mixing and mastering workflows."""

    def __init__(self, auto_mixer: AutoMixingEngine, mastering_engine: MasteringEngine):
        self.auto_mixer = auto_mixer
        self.mastering_engine = mastering_engine
        self.logger = logging.getLogger(__name__)

    def full_production_chain(
        self,
        audio: torch.Tensor,
        genre: str = "pop",
        target_format: str = "streaming",
        target_loudness: Optional[float] = None,
        quality: ProcessingQuality = ProcessingQuality.STANDARD,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Complete mixing and mastering chain.

        Returns:
            Tuple of (processed audio, analysis report)
        """

        self.logger.info(f"Starting full production chain for {genre}")

        # Set target loudness based on format
        if target_loudness is None:
            loudness_targets = {"streaming": -14.0, "cd": -9.0, "vinyl": -16.0, "broadcast": -23.0}
            target_loudness = loudness_targets.get(target_format, -14.0)

        # 1. Mixing
        self.logger.info("Auto-mixing...")
        mixed_audio = self.auto_mixer.auto_mix(
            audio=audio,
            genre=genre,
            mix_style="balanced",
            target_loudness=target_loudness - 3.0,  # Leave headroom for mastering
            quality=quality,
        )

        # 2. Mastering
        self.logger.info("Mastering...")
        mastered_audio = self.mastering_engine.master_track(
            audio=mixed_audio, target_format=target_format, genre=genre
        )

        # 3. Analysis
        self.logger.info("Analyzing final master...")
        analysis = self.mastering_engine.analyze_master(mastered_audio)

        # Add processing report
        analysis["processing_report"] = {
            "genre": genre,
            "target_format": target_format,
            "target_loudness": target_loudness,
            "quality_level": quality.value,
            "chain_applied": ["auto_mix", "master"],
        }

        self.logger.info("Production chain completed")
        return mastered_audio, analysis
