"""
Audio enhancement pipeline for improving MusicGen output quality.
Uses stem separation, EQ, compression, and other effects.
"""

import logging
import numpy as np
from typing import Tuple, Optional, List, Dict
import warnings

# Core dependencies
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Demucs for stem separation
try:
    import demucs.api

    DEMUCS_AVAILABLE = True
except ImportError:
    DEMUCS_AVAILABLE = False

# Pedalboard for audio effects
try:
    import pedalboard
    from pedalboard import (
        Pedalboard,
        Compressor,
        Gain,
        Limiter,
        Reverb,
        HighpassFilter,
        LowpassFilter,
        HighShelfFilter,
        Chorus,
        Delay,
        Distortion,
    )

    PEDALBOARD_AVAILABLE = True
except ImportError:
    PEDALBOARD_AVAILABLE = False

# Noise reduction
try:
    import noisereduce as nr

    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False

# Audio analysis
try:
    import librosa

    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

logger = logging.getLogger(__name__)


class AudioEnhancer:
    """
    Enhances audio quality through various processing techniques.
    Designed to fix common MusicGen quality issues.
    """

    def __init__(self, device: Optional[str] = None):
        """
        Initialize audio enhancer.

        Args:
            device: Device for neural models ('cuda' or 'cpu')
        """
        self.device = (
            device or ("cuda" if torch.cuda.is_available() else "cpu")
            if TORCH_AVAILABLE
            else "cpu"
        )
        self.stem_separator = None
        self._initialize_models()

    def _initialize_models(self):
        """Initialize enhancement models."""
        # Initialize Demucs for stem separation
        if DEMUCS_AVAILABLE and TORCH_AVAILABLE:
            try:
                logger.info("Loading Demucs stem separator...")
                self.stem_separator = demucs.api.Separator(
                    model="htdemucs",  # Best quality model
                    device=self.device,
                    shifts=1,  # Reduce for faster processing
                )
                logger.info("✓ Demucs loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load Demucs: {e}")
                self.stem_separator = None
        else:
            logger.warning("Demucs not available. Install with: pip install demucs")

    def separate_stems(
        self, audio: np.ndarray, sample_rate: int
    ) -> Dict[str, Tuple[np.ndarray, int]]:
        """
        Separate audio into stems (drums, bass, vocals, other).

        Args:
            audio: Input audio array
            sample_rate: Sample rate

        Returns:
            Dictionary of stem_name -> (audio, sample_rate)
        """
        if not self.stem_separator:
            logger.warning("Stem separation not available, returning original audio")
            return {"mixture": (audio, sample_rate)}

        try:
            # Ensure audio is in the right format
            if audio.ndim == 1:
                audio = audio[np.newaxis, :]  # Add channel dimension

            # Separate
            logger.info("Separating stems...")
            origin, separated = self.stem_separator.separate_tensor(
                torch.from_numpy(audio).float().to(self.device), sample_rate=sample_rate
            )

            # Convert back to numpy and organize
            stems = {}
            stem_names = ["drums", "bass", "other", "vocals"]

            for i, name in enumerate(stem_names):
                stem_audio = separated[i].cpu().numpy()
                if stem_audio.ndim > 1:
                    stem_audio = stem_audio.mean(axis=0)  # Convert to mono
                stems[name] = (stem_audio, sample_rate)

            logger.info(f"✓ Separated into {len(stems)} stems")
            return stems

        except Exception as e:
            logger.error(f"Stem separation failed: {e}")
            return {"mixture": (audio, sample_rate)}

    def create_mastering_chain(self, style: str = "default") -> "Pedalboard":
        """
        Create a mastering effects chain based on style.

        Args:
            style: Music style for tailored processing

        Returns:
            Pedalboard effects chain
        """
        if not PEDALBOARD_AVAILABLE:
            logger.warning("Pedalboard not available for effects")
            return None

        # Base mastering chain
        effects = [
            # Remove rumble and DC offset
            HighpassFilter(cutoff_frequency_hz=20),
            # Gentle compression for dynamics
            Compressor(threshold_db=-15, ratio=3, attack_ms=10, release_ms=100),
        ]

        # Style-specific processing
        if style == "electronic":
            effects.extend(
                [
                    # Enhance bass
                    HighShelfFilter(cutoff_frequency_hz=8000, gain_db=2),
                    # Tighten low end
                    Compressor(threshold_db=-10, ratio=4, attack_ms=5),
                ]
            )
        elif style == "rock":
            effects.extend(
                [
                    # Add presence
                    HighShelfFilter(cutoff_frequency_hz=3000, gain_db=3),
                    # More aggressive compression
                    Compressor(threshold_db=-12, ratio=5),
                ]
            )
        elif style == "jazz":
            effects.extend(
                [
                    # Warm tone
                    HighShelfFilter(cutoff_frequency_hz=10000, gain_db=-1),
                    # Gentle compression
                    Compressor(threshold_db=-18, ratio=2),
                ]
            )
        else:  # Default "pop" style
            effects.extend(
                [
                    # Brightness
                    HighShelfFilter(cutoff_frequency_hz=5000, gain_db=2),
                    # Controlled dynamics
                    Compressor(threshold_db=-15, ratio=3.5),
                ]
            )

        # Final limiting to prevent clipping
        effects.append(Limiter(threshold_db=-0.5, release_ms=50))

        return Pedalboard(effects)

    def reduce_artifacts(
        self, audio: np.ndarray, sample_rate: int, noise_profile_duration: float = 0.1
    ) -> np.ndarray:
        """
        Reduce noise and artifacts common in AI-generated audio.

        Args:
            audio: Input audio
            sample_rate: Sample rate
            noise_profile_duration: Duration of noise profile in seconds

        Returns:
            Cleaned audio
        """
        if not NOISEREDUCE_AVAILABLE:
            logger.warning("Noise reduction not available")
            return audio

        try:
            # Stationary noise reduction
            reduced = nr.reduce_noise(
                y=audio,
                sr=sample_rate,
                stationary=True,
                prop_decrease=0.8,  # Moderate reduction to avoid artifacts
            )

            # Additional spectral gating for "digital haze"
            if LIBROSA_AVAILABLE:
                # Simple spectral gating
                D = librosa.stft(reduced)
                magnitude = np.abs(D)

                # Gate threshold (adaptive)
                threshold = np.percentile(magnitude, 15)
                mask = magnitude > threshold

                # Apply gate
                D_gated = D * mask
                reduced = librosa.istft(D_gated, length=len(reduced))

            return reduced

        except Exception as e:
            logger.error(f"Artifact reduction failed: {e}")
            return audio

    def enhance_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
        style: str = "auto",
        enhance_level: str = "moderate",
    ) -> Tuple[np.ndarray, int]:
        """
        Full enhancement pipeline for audio.

        Args:
            audio: Input audio array
            sample_rate: Sample rate
            style: Music style or "auto" to detect
            enhance_level: Enhancement level (light, moderate, heavy)

        Returns:
            Enhanced audio and sample rate
        """
        logger.info(f"Enhancing audio (style: {style}, level: {enhance_level})")

        # Step 1: Reduce artifacts and noise
        if enhance_level in ["moderate", "heavy"]:
            audio = self.reduce_artifacts(audio, sample_rate)

        # Step 2: Apply mastering chain
        if PEDALBOARD_AVAILABLE:
            mastering = self.create_mastering_chain(style)
            if mastering:
                # Ensure audio is float32
                if audio.dtype != np.float32:
                    audio = audio.astype(np.float32)

                # Apply effects
                audio = mastering(audio, sample_rate)

        # Step 3: Normalize without clipping
        peak = np.abs(audio).max()
        if peak > 0:
            target_peak = 0.95 if enhance_level == "heavy" else 0.9
            audio = audio * (target_peak / peak)

        logger.info("✓ Enhancement complete")
        return audio, sample_rate

    def enhance_stems_separately(
        self, audio: np.ndarray, sample_rate: int, style: str = "auto"
    ) -> Tuple[np.ndarray, int]:
        """
        Enhance audio by processing stems separately for better quality.

        Args:
            audio: Input audio
            sample_rate: Sample rate
            style: Music style

        Returns:
            Enhanced audio and sample rate
        """
        # Separate stems
        stems = self.separate_stems(audio, sample_rate)

        if len(stems) == 1:  # Separation failed
            return self.enhance_audio(audio, sample_rate, style)

        # Process each stem with appropriate settings
        processed_stems = []

        for stem_name, (stem_audio, sr) in stems.items():
            logger.info(f"Processing {stem_name} stem...")

            if stem_name == "drums":
                # Punchy compression for drums
                if PEDALBOARD_AVAILABLE:
                    drum_chain = Pedalboard(
                        [
                            HighpassFilter(cutoff_frequency_hz=50),
                            Compressor(threshold_db=-10, ratio=6, attack_ms=2),
                            Limiter(threshold_db=-1),
                        ]
                    )
                    stem_audio = drum_chain(stem_audio.astype(np.float32), sr)

            elif stem_name == "bass":
                # Tight bass processing
                if PEDALBOARD_AVAILABLE:
                    bass_chain = Pedalboard(
                        [
                            HighpassFilter(cutoff_frequency_hz=30),
                            LowpassFilter(cutoff_frequency_hz=500),
                            Compressor(threshold_db=-12, ratio=4),
                        ]
                    )
                    stem_audio = bass_chain(stem_audio.astype(np.float32), sr)

            elif stem_name == "vocals":
                # Vocal clarity
                if PEDALBOARD_AVAILABLE:
                    vocal_chain = Pedalboard(
                        [
                            HighpassFilter(cutoff_frequency_hz=80),
                            Compressor(threshold_db=-15, ratio=3),
                            HighShelfFilter(cutoff_frequency_hz=8000, gain_db=2),
                        ]
                    )
                    stem_audio = vocal_chain(stem_audio.astype(np.float32), sr)

            else:  # "other" - instruments
                # General instrument processing
                if PEDALBOARD_AVAILABLE:
                    inst_chain = Pedalboard(
                        [
                            HighpassFilter(cutoff_frequency_hz=40),
                            Compressor(threshold_db=-18, ratio=2.5),
                        ]
                    )
                    stem_audio = inst_chain(stem_audio.astype(np.float32), sr)

            processed_stems.append(stem_audio)

        # Mix stems back together
        mixed = np.zeros_like(processed_stems[0])

        # Mixing ratios (can be adjusted)
        mix_ratios = {
            "drums": 0.9,
            "bass": 0.85,
            "other": 1.0,
            "vocals": 1.1,  # Slightly boost vocals
        }

        for i, (stem_name, _) in enumerate(stems.items()):
            ratio = mix_ratios.get(stem_name, 1.0)
            mixed += processed_stems[i] * ratio

        # Final mastering
        mixed = self.enhance_audio(mixed, sample_rate, style, "light")[0]

        return mixed, sample_rate


class AudioMixer:
    """
    Handles mixing of multiple audio sources (instrumental + vocals).
    """

    @staticmethod
    def mix_tracks(
        instrumental: Tuple[np.ndarray, int],
        vocals: Tuple[np.ndarray, int],
        vocal_level: float = 0.8,
        reverb_amount: float = 0.2,
        style: str = "pop",
    ) -> Tuple[np.ndarray, int]:
        """
        Mix instrumental and vocal tracks intelligently.

        Args:
            instrumental: (audio, sample_rate) for instrumental
            vocals: (audio, sample_rate) for vocals
            vocal_level: Relative vocal level (0-1)
            reverb_amount: Reverb amount for vocals (0-1)
            style: Music style for mixing decisions

        Returns:
            Mixed audio and sample rate
        """
        inst_audio, inst_sr = instrumental
        vocal_audio, vocal_sr = vocals

        # Resample if necessary
        if inst_sr != vocal_sr:
            if LIBROSA_AVAILABLE:
                vocal_audio = librosa.resample(
                    vocal_audio, orig_sr=vocal_sr, target_sr=inst_sr
                )
                vocal_sr = inst_sr
            else:
                logger.warning("Cannot resample - sample rates must match")
                return instrumental

        # Match lengths
        target_length = max(len(inst_audio), len(vocal_audio))

        if len(inst_audio) < target_length:
            inst_audio = np.pad(inst_audio, (0, target_length - len(inst_audio)))
        else:
            inst_audio = inst_audio[:target_length]

        if len(vocal_audio) < target_length:
            vocal_audio = np.pad(vocal_audio, (0, target_length - len(vocal_audio)))
        else:
            vocal_audio = vocal_audio[:target_length]

        # Process vocals before mixing
        if PEDALBOARD_AVAILABLE and reverb_amount > 0:
            vocal_effects = Pedalboard(
                [
                    # Vocal presence
                    HighpassFilter(cutoff_frequency_hz=100),
                    # Add space
                    Reverb(
                        room_size=reverb_amount,
                        damping=0.5,
                        wet_level=reverb_amount * 0.3,
                        dry_level=1.0 - (reverb_amount * 0.3),
                    ),
                    # Control dynamics
                    Compressor(threshold_db=-18, ratio=3),
                ]
            )
            vocal_audio = vocal_effects(vocal_audio.astype(np.float32), inst_sr)

        # Style-based mixing
        if style == "rock":
            # Vocals more prominent in rock
            vocal_level *= 1.1
        elif style == "electronic":
            # Vocals often more processed/blended
            vocal_level *= 0.9

        # Mix with automatic gain compensation
        mixed = inst_audio + (vocal_audio * vocal_level)

        # Prevent clipping
        peak = np.abs(mixed).max()
        if peak > 0.95:
            mixed = mixed * (0.95 / peak)

        return mixed, inst_sr

    @staticmethod
    def sidechain_compress(
        audio: np.ndarray,
        trigger: np.ndarray,
        sample_rate: int,
        threshold: float = -20,
        ratio: float = 4,
        attack_ms: float = 5,
        release_ms: float = 50,
    ) -> np.ndarray:
        """
        Apply sidechain compression (e.g., duck instruments when vocals play).

        Args:
            audio: Audio to compress
            trigger: Trigger signal (e.g., vocals)
            sample_rate: Sample rate
            threshold: Threshold in dB
            ratio: Compression ratio
            attack_ms: Attack time in milliseconds
            release_ms: Release time in milliseconds

        Returns:
            Compressed audio
        """
        # This is a simplified implementation
        # In production, we'd use a proper sidechain compressor

        # Calculate envelope of trigger signal
        envelope = np.abs(trigger)

        # Smooth envelope
        attack_samples = int(attack_ms * sample_rate / 1000)
        release_samples = int(release_ms * sample_rate / 1000)

        smoothed = np.zeros_like(envelope)

        for i in range(1, len(envelope)):
            if envelope[i] > smoothed[i - 1]:
                # Attack
                smoothed[i] = (
                    smoothed[i - 1] + (envelope[i] - smoothed[i - 1]) / attack_samples
                )
            else:
                # Release
                smoothed[i] = (
                    smoothed[i - 1] + (envelope[i] - smoothed[i - 1]) / release_samples
                )

        # Convert threshold to linear
        threshold_linear = 10 ** (threshold / 20)

        # Calculate gain reduction
        gain = np.ones_like(smoothed)
        above_threshold = smoothed > threshold_linear

        if np.any(above_threshold):
            # Calculate compression
            excess = smoothed[above_threshold] / threshold_linear
            gain[above_threshold] = 1 / (1 + (excess - 1) / ratio)

        # Apply gain
        return audio * gain
