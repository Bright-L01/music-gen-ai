"""Base classes for audio source separation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import torch


@dataclass
class SeparationResult:
    """Result of source separation."""

    stems: Dict[str, torch.Tensor]  # stem_name -> audio tensor
    sample_rate: int
    original_audio: Optional[torch.Tensor] = None
    confidence_scores: Optional[Dict[str, float]] = None
    processing_time: Optional[float] = None


class BaseSeparator(ABC):
    """Abstract base class for audio source separators."""

    def __init__(self, device: Optional[str] = None):
        """Initialize separator.

        Args:
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = None
        self.sample_rate = None

    @abstractmethod
    def load_model(self, model_path: Optional[str] = None):
        """Load the separation model.

        Args:
            model_path: Path to model checkpoint (optional)
        """

    @abstractmethod
    def separate(
        self,
        audio: Union[torch.Tensor, np.ndarray],
        sample_rate: int,
        targets: Optional[List[str]] = None,
    ) -> SeparationResult:
        """Separate audio into stems.

        Args:
            audio: Input audio tensor or numpy array
            sample_rate: Sample rate of input audio
            targets: Optional list of target stems to extract

        Returns:
            SeparationResult with separated stems
        """

    def preprocess_audio(
        self,
        audio: Union[torch.Tensor, np.ndarray],
        sample_rate: int,
        target_sr: Optional[int] = None,
    ) -> torch.Tensor:
        """Preprocess audio for separation.

        Args:
            audio: Input audio
            sample_rate: Current sample rate
            target_sr: Target sample rate (if resampling needed)

        Returns:
            Preprocessed audio tensor
        """
        # Convert to tensor if needed
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()

        # Ensure float32
        audio = audio.float()

        # Move to device
        audio = audio.to(self.device)

        # Resample if needed
        if target_sr is not None and sample_rate != target_sr:
            audio = self._resample(audio, sample_rate, target_sr)

        # Normalize
        if audio.abs().max() > 1.0:
            audio = audio / audio.abs().max()

        return audio

    def _resample(self, audio: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
        """Resample audio to target sample rate."""
        import torchaudio.transforms as T

        resampler = T.Resample(orig_sr, target_sr).to(self.device)
        return resampler(audio)

    def postprocess_stems(
        self, stems: Dict[str, torch.Tensor], normalize: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Postprocess separated stems.

        Args:
            stems: Dictionary of stem tensors
            normalize: Whether to normalize stems

        Returns:
            Postprocessed stems
        """
        processed = {}

        for name, stem in stems.items():
            # Move to CPU for output
            stem = stem.cpu()

            # Normalize if requested
            if normalize and stem.abs().max() > 0:
                stem = stem / stem.abs().max() * 0.95

            processed[name] = stem

        return processed

    @property
    def available_stems(self) -> List[str]:
        """Get list of available stem types."""
        return []

    def validate_targets(self, targets: Optional[List[str]] = None) -> List[str]:
        """Validate and return target stems.

        Args:
            targets: Requested target stems

        Returns:
            Valid target stems
        """
        available = self.available_stems

        if targets is None:
            return available

        # Validate each target
        valid_targets = []
        for target in targets:
            if target in available:
                valid_targets.append(target)
            else:
                print(f"Warning: '{target}' is not available. Available stems: {available}")

        return valid_targets if valid_targets else available
