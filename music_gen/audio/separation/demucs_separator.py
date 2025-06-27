"""DEMUCS-based source separation implementation."""

import torch
import torchaudio
from typing import Dict, List, Optional, Union
import numpy as np
import time
from pathlib import Path

from music_gen.audio.separation.base import BaseSeparator, SeparationResult


class DemucsSeparator(BaseSeparator):
    """Source separation using DEMUCS models."""
    
    # Available DEMUCS models and their stems
    MODELS = {
        "htdemucs": ["drums", "bass", "other", "vocals"],
        "htdemucs_ft": ["drums", "bass", "other", "vocals"],  # Fine-tuned version
        "mdx": ["drums", "bass", "other", "vocals"],
        "mdx_extra": ["drums", "bass", "other", "vocals"],
        "hdemucs_mmi": ["drums", "bass", "other", "vocals"],
    }
    
    # Instrument mapping for our system
    INSTRUMENT_MAPPING = {
        "drums": ["drums", "percussion", "timpani"],
        "bass": ["bass_guitar", "double_bass", "tuba"],
        "other": ["piano", "guitar", "synthesizer", "strings", "brass", "woodwinds"],
        "vocals": ["soprano", "alto", "tenor", "bass_voice", "choir"],
    }
    
    def __init__(
        self,
        model_name: str = "htdemucs",
        device: Optional[str] = None,
        segment_length: float = 10.0,  # Process in segments for memory efficiency
        overlap: float = 0.25,
    ):
        """Initialize DEMUCS separator.
        
        Args:
            model_name: Name of DEMUCS model to use
            device: Device to use for processing
            segment_length: Length of segments to process (seconds)
            overlap: Overlap between segments (fraction)
        """
        super().__init__(device)
        
        self.model_name = model_name
        self.segment_length = segment_length
        self.overlap = overlap
        self.sample_rate = 44100  # DEMUCS standard
        
        if model_name not in self.MODELS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.MODELS.keys())}")
    
    def load_model(self, model_path: Optional[str] = None):
        """Load DEMUCS model.
        
        Args:
            model_path: Optional path to custom model checkpoint
        """
        try:
            # Try to import demucs
            import demucs.api
            
            # Load model
            self.separator = demucs.api.Separator(
                model=self.model_name,
                device=str(self.device),
                progress=False,
            )
            
            # Set processing parameters
            self.separator.segment = self.segment_length
            self.separator.overlap = self.overlap
            
            print(f"Loaded DEMUCS model: {self.model_name}")
            
        except ImportError:
            print("DEMUCS not installed. Creating mock separator for development.")
            self.separator = None
    
    @property
    def available_stems(self) -> List[str]:
        """Get available stem types for current model."""
        if self.model_name in self.MODELS:
            return self.MODELS[self.model_name]
        return ["drums", "bass", "other", "vocals"]
    
    def separate(
        self,
        audio: Union[torch.Tensor, np.ndarray],
        sample_rate: int,
        targets: Optional[List[str]] = None
    ) -> SeparationResult:
        """Separate audio into stems using DEMUCS.
        
        Args:
            audio: Input audio tensor or numpy array
            sample_rate: Sample rate of input audio
            targets: Optional list of target stems
            
        Returns:
            SeparationResult with separated stems
        """
        start_time = time.time()
        
        # Preprocess audio
        audio_tensor = self.preprocess_audio(audio, sample_rate, self.sample_rate)
        
        # Validate targets
        valid_targets = self.validate_targets(targets)
        
        if self.separator is not None:
            # Use real DEMUCS
            stems = self._separate_with_demucs(audio_tensor, valid_targets)
        else:
            # Use mock separation for development
            stems = self._mock_separate(audio_tensor, valid_targets)
        
        # Postprocess
        stems = self.postprocess_stems(stems)
        
        # Calculate confidence scores
        confidence_scores = self._calculate_confidence(stems, audio_tensor)
        
        processing_time = time.time() - start_time
        
        return SeparationResult(
            stems=stems,
            sample_rate=self.sample_rate,
            original_audio=audio_tensor.cpu(),
            confidence_scores=confidence_scores,
            processing_time=processing_time
        )
    
    def _separate_with_demucs(
        self,
        audio: torch.Tensor,
        targets: List[str]
    ) -> Dict[str, torch.Tensor]:
        """Perform separation using DEMUCS model."""
        import demucs.api
        
        # Ensure correct shape (channels, samples)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        # Move to CPU for DEMUCS (it handles device placement)
        audio_np = audio.cpu().numpy()
        
        # Apply separation
        _, separated = self.separator.separate_tensor(
            audio_np,
            sample_rate=self.sample_rate
        )
        
        # Convert to dict
        stems = {}
        stem_names = self.available_stems
        
        for idx, stem_name in enumerate(stem_names):
            if stem_name in targets:
                stems[stem_name] = torch.from_numpy(separated[idx]).to(self.device)
        
        return stems
    
    def _mock_separate(
        self,
        audio: torch.Tensor,
        targets: List[str]
    ) -> Dict[str, torch.Tensor]:
        """Mock separation for development without DEMUCS."""
        stems = {}
        
        # Simple frequency-based separation
        for target in targets:
            if target == "bass":
                # Low-pass filter simulation
                stems[target] = audio * 0.3
            elif target == "drums":
                # Transient emphasis simulation
                diff = torch.diff(audio, dim=-1)
                stems[target] = torch.nn.functional.pad(diff, (0, 1)) * 0.5
            elif target == "vocals":
                # Mid-frequency emphasis
                stems[target] = audio * 0.4
            else:  # "other"
                # Remaining signal
                stems[target] = audio * 0.5
        
        return stems
    
    def _calculate_confidence(
        self,
        stems: Dict[str, torch.Tensor],
        original: torch.Tensor
    ) -> Dict[str, float]:
        """Calculate confidence scores for separation quality."""
        confidence = {}
        
        # Calculate energy ratio for each stem
        original_energy = (original ** 2).mean().item()
        
        for name, stem in stems.items():
            stem_energy = (stem ** 2).mean().item()
            ratio = stem_energy / (original_energy + 1e-8)
            
            # Simple confidence based on energy ratio
            confidence[name] = min(ratio * 2, 1.0)
        
        return confidence
    
    def separate_to_instruments(
        self,
        audio: Union[torch.Tensor, np.ndarray],
        sample_rate: int,
        target_instruments: List[str]
    ) -> Dict[str, torch.Tensor]:
        """Separate audio and map to specific instruments.
        
        Args:
            audio: Input audio
            sample_rate: Sample rate
            target_instruments: List of target instrument names
            
        Returns:
            Dictionary mapping instrument names to audio
        """
        # First, do standard separation
        result = self.separate(audio, sample_rate)
        
        # Map stems to instruments
        instrument_audio = {}
        
        for instrument in target_instruments:
            # Find which stem contains this instrument
            stem_found = False
            
            for stem_name, instruments_in_stem in self.INSTRUMENT_MAPPING.items():
                if instrument.lower() in instruments_in_stem and stem_name in result.stems:
                    instrument_audio[instrument] = result.stems[stem_name]
                    stem_found = True
                    break
            
            if not stem_found:
                # Default to "other" stem if available
                if "other" in result.stems:
                    instrument_audio[instrument] = result.stems["other"]
                else:
                    print(f"Warning: Could not map instrument '{instrument}' to any stem")
        
        return instrument_audio
    
    def enhance_separation(
        self,
        stems: Dict[str, torch.Tensor],
        enhance_vocals: bool = True,
        reduce_bleed: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Apply post-processing to enhance separation quality.
        
        Args:
            stems: Separated stems
            enhance_vocals: Whether to enhance vocal clarity
            reduce_bleed: Whether to reduce cross-talk between stems
            
        Returns:
            Enhanced stems
        """
        enhanced = {}
        
        for name, stem in stems.items():
            enhanced_stem = stem.clone()
            
            if enhance_vocals and name == "vocals":
                # Simple vocal enhancement (placeholder for real processing)
                enhanced_stem = enhanced_stem * 1.2
            
            if reduce_bleed:
                # Simple noise gate simulation
                threshold = enhanced_stem.abs().max() * 0.1
                mask = enhanced_stem.abs() > threshold
                enhanced_stem = enhanced_stem * mask.float()
            
            enhanced[name] = enhanced_stem
        
        return enhanced