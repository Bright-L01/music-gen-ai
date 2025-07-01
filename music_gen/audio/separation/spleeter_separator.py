"""Spleeter-based source separation implementation."""

import time
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from music_gen.audio.separation.base import BaseSeparator, SeparationResult


class SpleeterSeparator(BaseSeparator):
    """Source separation using Spleeter models."""

    # Available Spleeter configurations
    CONFIGS = {
        "2stems": ["vocals", "accompaniment"],
        "4stems": ["vocals", "drums", "bass", "other"],
        "5stems": ["vocals", "drums", "bass", "piano", "other"],
    }

    # Instrument mapping
    INSTRUMENT_MAPPING = {
        "vocals": ["soprano", "alto", "tenor", "bass_voice", "choir"],
        "drums": ["drums", "percussion", "timpani"],
        "bass": ["bass_guitar", "double_bass", "tuba"],
        "piano": ["piano", "electric_piano", "harpsichord"],
        "other": ["guitar", "violin", "saxophone", "trumpet", "synthesizer"],
        "accompaniment": ["piano", "guitar", "bass", "drums", "strings", "brass"],
    }

    def __init__(
        self,
        config_name: str = "4stems",
        device: Optional[str] = None,
        bitrate: str = "320k",
        codec: str = "wav",
    ):
        """Initialize Spleeter separator.

        Args:
            config_name: Spleeter configuration to use
            device: Device for processing
            bitrate: Output bitrate
            codec: Output codec
        """
        super().__init__(device)

        self.config_name = config_name
        self.bitrate = bitrate
        self.codec = codec
        self.sample_rate = 44100  # Spleeter standard

        if config_name not in self.CONFIGS:
            raise ValueError(
                f"Unknown config: {config_name}. Available: {list(self.CONFIGS.keys())}"
            )

    def load_model(self, model_path: Optional[str] = None):
        """Load Spleeter model.

        Args:
            model_path: Optional path to custom model
        """
        try:
            # Try to import spleeter
            from spleeter.separator import Separator

            # Create separator with specified configuration
            self.separator = Separator(f"spleeter:{self.config_name}")

            # Load model to device
            self.separator.model.to(self.device)

            print(f"Loaded Spleeter model: {self.config_name}")

        except ImportError:
            print("Spleeter not installed. Creating mock separator for development.")
            self.separator = None

    @property
    def available_stems(self) -> List[str]:
        """Get available stem types for current configuration."""
        if self.config_name in self.CONFIGS:
            return self.CONFIGS[self.config_name]
        return ["vocals", "accompaniment"]

    def separate(
        self,
        audio: Union[torch.Tensor, np.ndarray],
        sample_rate: int,
        targets: Optional[List[str]] = None,
    ) -> SeparationResult:
        """Separate audio into stems using Spleeter.

        Args:
            audio: Input audio
            sample_rate: Sample rate
            targets: Target stems

        Returns:
            SeparationResult with separated stems
        """
        start_time = time.time()

        # Preprocess audio
        audio_tensor = self.preprocess_audio(audio, sample_rate, self.sample_rate)

        # Validate targets
        valid_targets = self.validate_targets(targets)

        if self.separator is not None:
            # Use real Spleeter
            stems = self._separate_with_spleeter(audio_tensor, valid_targets)
        else:
            # Use mock separation
            stems = self._mock_separate(audio_tensor, valid_targets)

        # Postprocess
        stems = self.postprocess_stems(stems)

        # Calculate confidence
        confidence_scores = self._calculate_confidence(stems, audio_tensor)

        processing_time = time.time() - start_time

        return SeparationResult(
            stems=stems,
            sample_rate=self.sample_rate,
            original_audio=audio_tensor.cpu(),
            confidence_scores=confidence_scores,
            processing_time=processing_time,
        )

    def _separate_with_spleeter(
        self, audio: torch.Tensor, targets: List[str]
    ) -> Dict[str, torch.Tensor]:
        """Perform separation using Spleeter."""
        # Convert to numpy for Spleeter
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        audio_np = audio.cpu().numpy().T  # Spleeter expects (samples, channels)

        # Perform separation
        prediction = self.separator.separate(audio_np)

        # Convert back to torch tensors
        stems = {}
        for stem_name in targets:
            if stem_name in prediction:
                stem_audio = prediction[stem_name].T  # Back to (channels, samples)
                stems[stem_name] = torch.from_numpy(stem_audio).to(self.device)

        return stems

    def _mock_separate(self, audio: torch.Tensor, targets: List[str]) -> Dict[str, torch.Tensor]:
        """Mock separation for development."""
        stems = {}

        # Simple separation based on configuration
        if self.config_name == "2stems":
            if "vocals" in targets:
                # Simulate vocal extraction (mid frequencies)
                stems["vocals"] = audio * 0.4
            if "accompaniment" in targets:
                # Everything else
                stems["accompaniment"] = audio * 0.6

        elif self.config_name in ["4stems", "5stems"]:
            # More detailed separation
            for target in targets:
                if target == "bass":
                    stems[target] = audio * 0.2  # Low frequencies
                elif target == "drums":
                    # Emphasize transients
                    diff = torch.diff(audio, dim=-1)
                    stems[target] = torch.nn.functional.pad(diff, (0, 1)) * 0.3
                elif target == "vocals":
                    stems[target] = audio * 0.3  # Mid frequencies
                elif target == "piano" and self.config_name == "5stems":
                    stems[target] = audio * 0.2  # Mid-high frequencies
                else:  # "other"
                    stems[target] = audio * 0.3

        return stems

    def _calculate_confidence(
        self, stems: Dict[str, torch.Tensor], original: torch.Tensor
    ) -> Dict[str, float]:
        """Calculate confidence scores."""
        confidence = {}

        # Energy-based confidence
        original_energy = (original**2).mean().item()

        for name, stem in stems.items():
            stem_energy = (stem**2).mean().item()
            ratio = stem_energy / (original_energy + 1e-8)

            # Adjust confidence based on stem type
            if name == "vocals":
                # Vocals typically have lower energy
                confidence[name] = min(ratio * 3, 1.0)
            elif name == "bass":
                confidence[name] = min(ratio * 2.5, 1.0)
            else:
                confidence[name] = min(ratio * 2, 1.0)

        return confidence

    def separate_with_filtering(
        self,
        audio: Union[torch.Tensor, np.ndarray],
        sample_rate: int,
        targets: Optional[List[str]] = None,
        use_wiener_filter: bool = True,
    ) -> SeparationResult:
        """Separate with additional filtering for quality improvement.

        Args:
            audio: Input audio
            sample_rate: Sample rate
            targets: Target stems
            use_wiener_filter: Whether to apply Wiener filtering

        Returns:
            Enhanced separation result
        """
        # Perform standard separation
        result = self.separate(audio, sample_rate, targets)

        if use_wiener_filter:
            # Apply Wiener-like filtering (simplified)
            result.stems = self._apply_wiener_filter(result.stems, result.original_audio)

        return result

    def _apply_wiener_filter(
        self, stems: Dict[str, torch.Tensor], original: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Apply simplified Wiener filter to reduce artifacts."""
        filtered_stems = {}

        # Calculate total reconstructed signal
        total_recon = torch.zeros_like(original)
        for stem in stems.values():
            total_recon += stem

        # Apply gain correction
        for name, stem in stems.items():
            # Calculate Wiener gain
            stem_power = stem**2 + 1e-8
            total_power = total_recon**2 + 1e-8
            wiener_gain = stem_power / total_power

            # Apply gain
            filtered_stems[name] = stem * wiener_gain

        return filtered_stems

    def batch_separate(
        self,
        audio_batch: List[Union[torch.Tensor, np.ndarray]],
        sample_rates: List[int],
        targets: Optional[List[str]] = None,
    ) -> List[SeparationResult]:
        """Separate multiple audio files in batch.

        Args:
            audio_batch: List of audio tensors/arrays
            sample_rates: List of sample rates
            targets: Target stems

        Returns:
            List of separation results
        """
        results = []

        for audio, sr in zip(audio_batch, sample_rates):
            result = self.separate(audio, sr, targets)
            results.append(result)

        return results
