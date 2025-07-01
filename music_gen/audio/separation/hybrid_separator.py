"""Hybrid separator combining multiple separation methods."""

import time
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from music_gen.audio.separation.base import BaseSeparator, SeparationResult
from music_gen.audio.separation.demucs_separator import DemucsSeparator
from music_gen.audio.separation.spleeter_separator import SpleeterSeparator


class HybridSeparator(BaseSeparator):
    """Combines multiple separation methods for better results."""

    def __init__(
        self,
        primary_method: str = "demucs",
        secondary_method: str = "spleeter",
        blend_mode: str = "weighted",
        device: Optional[str] = None,
    ):
        """Initialize hybrid separator.

        Args:
            primary_method: Primary separation method
            secondary_method: Secondary separation method
            blend_mode: How to combine results ("weighted", "selective", "ensemble")
            device: Device for processing
        """
        super().__init__(device)

        self.primary_method = primary_method
        self.secondary_method = secondary_method
        self.blend_mode = blend_mode

        # Initialize separators
        self.primary_separator = None
        self.secondary_separator = None

        # Blending weights (learned or heuristic)
        self.blend_weights = {
            "vocals": {"demucs": 0.7, "spleeter": 0.3},
            "drums": {"demucs": 0.8, "spleeter": 0.2},
            "bass": {"demucs": 0.6, "spleeter": 0.4},
            "other": {"demucs": 0.5, "spleeter": 0.5},
        }

    def load_model(self, model_path: Optional[str] = None):
        """Load separation models."""
        # Load primary separator
        if self.primary_method == "demucs":
            self.primary_separator = DemucsSeparator(device=str(self.device))
        elif self.primary_method == "spleeter":
            self.primary_separator = SpleeterSeparator(device=str(self.device))
        else:
            raise ValueError(f"Unknown primary method: {self.primary_method}")

        self.primary_separator.load_model()

        # Load secondary separator
        if self.secondary_method == "demucs":
            self.secondary_separator = DemucsSeparator(device=str(self.device))
        elif self.secondary_method == "spleeter":
            self.secondary_separator = SpleeterSeparator(device=str(self.device))

        if self.secondary_separator:
            self.secondary_separator.load_model()

        print(f"Loaded hybrid separator: {self.primary_method} + {self.secondary_method}")

    @property
    def available_stems(self) -> List[str]:
        """Get available stems from both separators."""
        stems = set()

        if self.primary_separator:
            stems.update(self.primary_separator.available_stems)
        if self.secondary_separator:
            stems.update(self.secondary_separator.available_stems)

        return list(stems)

    def separate(
        self,
        audio: Union[torch.Tensor, np.ndarray],
        sample_rate: int,
        targets: Optional[List[str]] = None,
    ) -> SeparationResult:
        """Perform hybrid separation.

        Args:
            audio: Input audio
            sample_rate: Sample rate
            targets: Target stems

        Returns:
            Combined separation result
        """
        start_time = time.time()

        # Validate targets
        valid_targets = self.validate_targets(targets)

        # Get results from primary separator
        primary_result = self.primary_separator.separate(audio, sample_rate, valid_targets)

        # Get results from secondary separator if available
        secondary_result = None
        if self.secondary_separator:
            secondary_result = self.secondary_separator.separate(audio, sample_rate, valid_targets)

        # Combine results based on blend mode
        if self.blend_mode == "weighted":
            combined_stems = self._weighted_blend(
                primary_result.stems,
                secondary_result.stems if secondary_result else {},
                valid_targets,
            )
        elif self.blend_mode == "selective":
            combined_stems = self._selective_blend(
                primary_result.stems,
                secondary_result.stems if secondary_result else {},
                valid_targets,
            )
        elif self.blend_mode == "ensemble":
            combined_stems = self._ensemble_blend(
                primary_result.stems,
                secondary_result.stems if secondary_result else {},
                valid_targets,
            )
        else:
            combined_stems = primary_result.stems

        # Calculate combined confidence
        confidence_scores = self._calculate_combined_confidence(
            primary_result.confidence_scores,
            secondary_result.confidence_scores if secondary_result else None,
        )

        processing_time = time.time() - start_time

        return SeparationResult(
            stems=combined_stems,
            sample_rate=primary_result.sample_rate,
            original_audio=primary_result.original_audio,
            confidence_scores=confidence_scores,
            processing_time=processing_time,
        )

    def _weighted_blend(
        self,
        primary_stems: Dict[str, torch.Tensor],
        secondary_stems: Dict[str, torch.Tensor],
        targets: List[str],
    ) -> Dict[str, torch.Tensor]:
        """Blend stems using predetermined weights."""
        blended = {}

        for target in targets:
            if target in primary_stems and target in secondary_stems:
                # Get weights
                primary_weight = self.blend_weights.get(target, {}).get(self.primary_method, 0.5)
                secondary_weight = self.blend_weights.get(target, {}).get(
                    self.secondary_method, 0.5
                )

                # Normalize weights
                total_weight = primary_weight + secondary_weight
                primary_weight /= total_weight
                secondary_weight /= total_weight

                # Blend
                blended[target] = (
                    primary_stems[target] * primary_weight
                    + secondary_stems[target] * secondary_weight
                )
            elif target in primary_stems:
                blended[target] = primary_stems[target]
            elif target in secondary_stems:
                blended[target] = secondary_stems[target]

        return blended

    def _selective_blend(
        self,
        primary_stems: Dict[str, torch.Tensor],
        secondary_stems: Dict[str, torch.Tensor],
        targets: List[str],
    ) -> Dict[str, torch.Tensor]:
        """Select best stem from each separator based on quality metrics."""
        blended = {}

        for target in targets:
            if target in primary_stems and target in secondary_stems:
                # Compare quality metrics
                primary_quality = self._calculate_stem_quality(primary_stems[target])
                secondary_quality = self._calculate_stem_quality(secondary_stems[target])

                # Select better quality stem
                if primary_quality > secondary_quality:
                    blended[target] = primary_stems[target]
                else:
                    blended[target] = secondary_stems[target]
            elif target in primary_stems:
                blended[target] = primary_stems[target]
            elif target in secondary_stems:
                blended[target] = secondary_stems[target]

        return blended

    def _ensemble_blend(
        self,
        primary_stems: Dict[str, torch.Tensor],
        secondary_stems: Dict[str, torch.Tensor],
        targets: List[str],
    ) -> Dict[str, torch.Tensor]:
        """Advanced ensemble blending using frequency-domain analysis."""
        blended = {}

        for target in targets:
            if target in primary_stems and target in secondary_stems:
                # Perform frequency-domain blending
                blended[target] = self._frequency_domain_blend(
                    primary_stems[target], secondary_stems[target], target
                )
            elif target in primary_stems:
                blended[target] = primary_stems[target]
            elif target in secondary_stems:
                blended[target] = secondary_stems[target]

        return blended

    def _frequency_domain_blend(
        self, stem1: torch.Tensor, stem2: torch.Tensor, stem_type: str
    ) -> torch.Tensor:
        """Blend stems in frequency domain based on stem characteristics."""
        # Compute STFT
        n_fft = 2048
        hop_length = 512

        stft1 = torch.stft(
            stem1.squeeze(0) if stem1.dim() > 1 else stem1,
            n_fft=n_fft,
            hop_length=hop_length,
            return_complex=True,
        )

        stft2 = torch.stft(
            stem2.squeeze(0) if stem2.dim() > 1 else stem2,
            n_fft=n_fft,
            hop_length=hop_length,
            return_complex=True,
        )

        # Frequency-dependent blending weights
        freqs = torch.linspace(0, 1, stft1.size(0), device=stem1.device)

        if stem_type == "bass":
            # Prefer low frequencies from primary
            weights = torch.exp(-5 * freqs).unsqueeze(-1)
        elif stem_type == "vocals":
            # Prefer mid frequencies
            weights = torch.exp(-10 * (freqs - 0.3).abs()).unsqueeze(-1)
        elif stem_type == "drums":
            # Prefer high frequencies and transients
            weights = (0.3 + 0.7 * freqs).unsqueeze(-1)
        else:
            # Equal weighting
            weights = torch.ones_like(freqs).unsqueeze(-1) * 0.5

        # Blend in frequency domain
        blended_stft = stft1 * weights + stft2 * (1 - weights)

        # Convert back to time domain
        blended = torch.istft(
            blended_stft, n_fft=n_fft, hop_length=hop_length, length=stem1.shape[-1]
        )

        if stem1.dim() > 1:
            blended = blended.unsqueeze(0)

        return blended

    def _calculate_stem_quality(self, stem: torch.Tensor) -> float:
        """Calculate quality metric for a stem."""
        # Simple quality metric based on:
        # 1. Signal-to-noise ratio estimate
        # 2. Spectral clarity
        # 3. Dynamic range

        # Estimate SNR
        signal_power = (stem**2).mean()
        noise_estimate = (stem**2).min() * 10  # Simple noise floor estimate
        snr = 10 * torch.log10(signal_power / (noise_estimate + 1e-8))

        # Spectral clarity (high frequency content)
        fft = torch.fft.rfft(stem.squeeze())
        spectral_clarity = fft.abs()[len(fft) // 2 :].mean() / (fft.abs().mean() + 1e-8)

        # Dynamic range
        dynamic_range = stem.abs().max() / (stem.abs().mean() + 1e-8)

        # Combine metrics
        quality = (
            snr.item() * 0.5 + spectral_clarity.item() * 100 * 0.3 + dynamic_range.item() * 0.2
        )

        return quality

    def _calculate_combined_confidence(
        self,
        primary_confidence: Optional[Dict[str, float]],
        secondary_confidence: Optional[Dict[str, float]],
    ) -> Dict[str, float]:
        """Calculate combined confidence scores."""
        if not primary_confidence:
            return {}

        if not secondary_confidence:
            return primary_confidence

        combined = {}
        all_stems = set(primary_confidence.keys()) | set(secondary_confidence.keys())

        for stem in all_stems:
            scores = []
            if stem in primary_confidence:
                scores.append(primary_confidence[stem])
            if stem in secondary_confidence:
                scores.append(secondary_confidence[stem])

            # Average confidence
            combined[stem] = sum(scores) / len(scores)

        return combined

    def adaptive_separate(
        self,
        audio: Union[torch.Tensor, np.ndarray],
        sample_rate: int,
        quality_threshold: float = 0.7,
    ) -> SeparationResult:
        """Adaptively choose separation method based on audio characteristics.

        Args:
            audio: Input audio
            sample_rate: Sample rate
            quality_threshold: Quality threshold for method selection

        Returns:
            Separation result using best method
        """
        # Analyze audio characteristics
        characteristics = self._analyze_audio(audio, sample_rate)

        # Choose best method based on characteristics
        if characteristics["has_vocals"] and characteristics["vocal_prominence"] > 0.5:
            # Use method better for vocals
            self.blend_weights["vocals"][self.primary_method] = 0.8
            self.blend_weights["vocals"][self.secondary_method] = 0.2

        if characteristics["percussive_ratio"] > 0.6:
            # Use method better for drums
            self.blend_weights["drums"][self.primary_method] = 0.9
            self.blend_weights["drums"][self.secondary_method] = 0.1

        # Perform separation with adapted weights
        return self.separate(audio, sample_rate)

    def _analyze_audio(
        self, audio: Union[torch.Tensor, np.ndarray], sample_rate: int
    ) -> Dict[str, float]:
        """Analyze audio characteristics for method selection."""
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)

        # Simple audio analysis
        characteristics = {}

        # Estimate vocal presence (mid-frequency energy)
        fft = torch.fft.rfft(audio.squeeze())
        freqs = torch.fft.rfftfreq(audio.shape[-1], 1 / sample_rate)

        mid_freq_mask = (freqs > 200) & (freqs < 4000)
        mid_energy = fft[mid_freq_mask].abs().mean()
        total_energy = fft.abs().mean()

        characteristics["has_vocals"] = mid_energy / total_energy > 0.3
        characteristics["vocal_prominence"] = float(mid_energy / total_energy)

        # Estimate percussive content
        onset_strength = torch.diff(audio.abs(), dim=-1).mean()
        characteristics["percussive_ratio"] = float(onset_strength * 100)

        return characteristics
