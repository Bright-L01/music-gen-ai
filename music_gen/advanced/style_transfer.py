"""
Style Transfer and Interpolation System

Implements advanced style transfer between different musical styles,
artists, and compositions, along with smooth interpolation capabilities
for commercial-grade music generation.
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.conditioning.audio_encoder import CLAPAudioEncoder
from ..utils.audio.processing import AudioProcessor
from .conditioning import AdvancedConditioningSystem


class StyleAttribute(Enum):
    """Musical style attributes for transfer."""

    RHYTHM = "rhythm"
    HARMONY = "harmony"
    MELODY = "melody"
    TIMBRE = "timbre"
    DYNAMICS = "dynamics"
    ARTICULATION = "articulation"
    TEXTURE = "texture"
    FORM = "form"


class InterpolationMode(Enum):
    """Interpolation modes for style blending."""

    LINEAR = "linear"
    SPHERICAL = "spherical"
    CUBIC = "cubic"
    SMOOTH_STEP = "smooth_step"
    PERLIN = "perlin"


@dataclass
class StyleProfile:
    """Profile representing a musical style."""

    name: str
    genre: str
    artist: Optional[str] = None
    era: Optional[str] = None
    features: Dict[str, float] = field(default_factory=dict)
    conditioning_vector: Optional[torch.Tensor] = None
    audio_examples: List[torch.Tensor] = field(default_factory=list)
    description: str = ""

    def __post_init__(self):
        """Initialize default features if not provided."""
        if not self.features:
            self.features = {
                "tempo": 0.5,
                "energy": 0.5,
                "complexity": 0.5,
                "brightness": 0.5,
                "warmth": 0.5,
                "aggression": 0.5,
                "spaciousness": 0.5,
                "density": 0.5,
            }


@dataclass
class StyleTransferConfig:
    """Configuration for style transfer operations."""

    source_weight: float = 0.3
    target_weight: float = 0.7
    preserve_attributes: List[StyleAttribute] = field(default_factory=list)
    transfer_attributes: List[StyleAttribute] = field(default_factory=lambda: list(StyleAttribute))
    interpolation_steps: int = 10
    smoothing_factor: float = 0.1
    feature_scaling: Dict[str, float] = field(default_factory=dict)


class StyleEncoder(nn.Module):
    """Neural network for encoding musical styles."""

    def __init__(self, input_dim: int = 1024, style_dim: int = 512):
        super().__init__()

        self.input_dim = input_dim
        self.style_dim = style_dim

        # Style encoding network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, style_dim),
            nn.Tanh(),
        )

        # Style attribute decomposition
        self.attribute_heads = nn.ModuleDict(
            {
                attr.value: nn.Sequential(
                    nn.Linear(style_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.Sigmoid()
                )
                for attr in StyleAttribute
            }
        )

        # Style reconstruction network
        self.decoder = nn.Sequential(
            nn.Linear(style_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, input_dim),
            nn.Tanh(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to style space."""
        return self.encoder(x)

    def decode(self, style_vector: torch.Tensor) -> torch.Tensor:
        """Decode style vector back to conditioning space."""
        return self.decoder(style_vector)

    def extract_attributes(self, style_vector: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract individual style attributes."""
        attributes = {}
        for attr_name, head in self.attribute_heads.items():
            attributes[attr_name] = head(style_vector)
        return attributes

    def combine_attributes(self, attributes: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Combine attributes back to style vector."""
        # Simple concatenation for now - could be more sophisticated
        attr_vectors = [
            attributes[attr.value] for attr in StyleAttribute if attr.value in attributes
        ]
        if not attr_vectors:
            return torch.zeros(1, self.style_dim, device=next(self.parameters()).device)

        combined = torch.cat(attr_vectors, dim=-1)

        # Project to style dimension
        if combined.shape[-1] != self.style_dim:
            projection = nn.Linear(combined.shape[-1], self.style_dim).to(combined.device)
            combined = projection(combined)

        return combined


class StyleInterpolator:
    """Advanced style interpolation with multiple modes."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def interpolate(
        self,
        style_a: torch.Tensor,
        style_b: torch.Tensor,
        alpha: float,
        mode: InterpolationMode = InterpolationMode.LINEAR,
    ) -> torch.Tensor:
        """
        Interpolate between two style vectors.

        Args:
            style_a: First style vector
            style_b: Second style vector
            alpha: Interpolation factor (0.0 = style_a, 1.0 = style_b)
            mode: Interpolation mode

        Returns:
            Interpolated style vector
        """

        alpha = torch.clamp(torch.tensor(alpha), 0.0, 1.0)

        if mode == InterpolationMode.LINEAR:
            return self._linear_interpolation(style_a, style_b, alpha)
        elif mode == InterpolationMode.SPHERICAL:
            return self._spherical_interpolation(style_a, style_b, alpha)
        elif mode == InterpolationMode.CUBIC:
            return self._cubic_interpolation(style_a, style_b, alpha)
        elif mode == InterpolationMode.SMOOTH_STEP:
            return self._smooth_step_interpolation(style_a, style_b, alpha)
        elif mode == InterpolationMode.PERLIN:
            return self._perlin_interpolation(style_a, style_b, alpha)
        else:
            return self._linear_interpolation(style_a, style_b, alpha)

    def _linear_interpolation(self, a: torch.Tensor, b: torch.Tensor, alpha: float) -> torch.Tensor:
        """Linear interpolation between two vectors."""
        return (1 - alpha) * a + alpha * b

    def _spherical_interpolation(
        self, a: torch.Tensor, b: torch.Tensor, alpha: float
    ) -> torch.Tensor:
        """Spherical linear interpolation (SLERP)."""
        # Normalize vectors
        a_norm = F.normalize(a, dim=-1)
        b_norm = F.normalize(b, dim=-1)

        # Calculate angle between vectors
        dot_product = torch.sum(a_norm * b_norm, dim=-1, keepdim=True)
        dot_product = torch.clamp(dot_product, -1.0, 1.0)
        theta = torch.acos(dot_product)

        # Avoid division by zero
        sin_theta = torch.sin(theta)
        epsilon = 1e-6

        if torch.any(sin_theta < epsilon):
            # Fall back to linear interpolation
            return self._linear_interpolation(a, b, alpha)

        # SLERP formula
        factor_a = torch.sin((1 - alpha) * theta) / sin_theta
        factor_b = torch.sin(alpha * theta) / sin_theta

        return factor_a * a_norm + factor_b * b_norm

    def _cubic_interpolation(self, a: torch.Tensor, b: torch.Tensor, alpha: float) -> torch.Tensor:
        """Cubic interpolation with smooth acceleration."""
        # Cubic ease-in-out
        alpha_cubic = 3 * alpha**2 - 2 * alpha**3
        return self._linear_interpolation(a, b, alpha_cubic)

    def _smooth_step_interpolation(
        self, a: torch.Tensor, b: torch.Tensor, alpha: float
    ) -> torch.Tensor:
        """Smooth step interpolation."""
        # Smoothstep function
        alpha_smooth = alpha * alpha * (3 - 2 * alpha)
        return self._linear_interpolation(a, b, alpha_smooth)

    def _perlin_interpolation(self, a: torch.Tensor, b: torch.Tensor, alpha: float) -> torch.Tensor:
        """Perlin noise-based interpolation for organic transitions."""
        # Add controlled randomness
        noise = torch.randn_like(a) * 0.1 * (1 - abs(2 * alpha - 1))  # Max noise at alpha=0.5

        # Smooth interpolation with noise
        alpha_smooth = alpha * alpha * (3 - 2 * alpha)
        base_interp = self._linear_interpolation(a, b, alpha_smooth)

        return base_interp + noise

    def multi_style_interpolation(
        self,
        styles: List[torch.Tensor],
        weights: List[float],
        mode: InterpolationMode = InterpolationMode.LINEAR,
    ) -> torch.Tensor:
        """
        Interpolate between multiple styles with given weights.

        Args:
            styles: List of style vectors
            weights: List of weights (should sum to 1.0)
            mode: Interpolation mode

        Returns:
            Interpolated style vector
        """

        if len(styles) != len(weights):
            raise ValueError("Number of styles must match number of weights")

        if abs(sum(weights) - 1.0) > 1e-6:
            # Normalize weights
            weights = [w / sum(weights) for w in weights]

        if len(styles) == 1:
            return styles[0]

        if len(styles) == 2:
            return self.interpolate(styles[0], styles[1], weights[1], mode)

        # For multiple styles, use weighted average with pairwise interpolation
        result = torch.zeros_like(styles[0])

        for i, (style, weight) in enumerate(zip(styles, weights)):
            if weight > 0:
                result += weight * style

        return result

    def create_transition_sequence(
        self,
        style_a: torch.Tensor,
        style_b: torch.Tensor,
        num_steps: int,
        mode: InterpolationMode = InterpolationMode.SMOOTH_STEP,
    ) -> List[torch.Tensor]:
        """
        Create a sequence of interpolated styles for smooth transitions.

        Args:
            style_a: Starting style
            style_b: Ending style
            num_steps: Number of interpolation steps
            mode: Interpolation mode

        Returns:
            List of interpolated style vectors
        """

        sequence = []
        for i in range(num_steps):
            alpha = i / (num_steps - 1) if num_steps > 1 else 0.0
            interpolated = self.interpolate(style_a, style_b, alpha, mode)
            sequence.append(interpolated)

        return sequence


class StyleTransferSystem:
    """Complete style transfer system for music generation."""

    def __init__(
        self,
        conditioning_system: AdvancedConditioningSystem,
        audio_encoder: CLAPAudioEncoder,
        audio_processor: AudioProcessor,
        style_dim: int = 512,
        device: torch.device = None,
    ):
        self.conditioning_system = conditioning_system
        self.audio_encoder = audio_encoder
        self.audio_processor = audio_processor
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.logger = logging.getLogger(__name__)

        # Initialize style encoder
        self.style_encoder = StyleEncoder(
            input_dim=conditioning_system.hidden_dim, style_dim=style_dim
        ).to(self.device)

        # Initialize interpolator
        self.interpolator = StyleInterpolator()

        # Style library
        self.style_library: Dict[str, StyleProfile] = {}

        # Load pre-trained weights if available
        self._load_pretrained_weights()

    def _load_pretrained_weights(self):
        """Load pre-trained style encoder weights."""
        # Placeholder for loading pre-trained weights
        # In practice, this would load weights from a saved checkpoint

    def extract_style_from_audio(
        self,
        audio: torch.Tensor,
        style_name: str,
        artist: Optional[str] = None,
        genre: Optional[str] = None,
    ) -> StyleProfile:
        """
        Extract style profile from reference audio.

        Args:
            audio: Reference audio tensor
            style_name: Name for the extracted style
            artist: Artist name (optional)
            genre: Genre classification (optional)

        Returns:
            Extracted style profile
        """

        # Encode audio using CLAP encoder
        with torch.no_grad():
            self.audio_encoder.encode(audio)

            # Convert to conditioning format
            conditioning = self.conditioning_system.create_comprehensive_conditioning(
                text_prompt="", reference_audio=audio  # No text prompt for pure audio analysis
            )

            # Fuse conditioning to get unified representation
            unified_conditioning = self.conditioning_system.fuse_conditioning(conditioning)

            # Extract style vector
            style_vector = self.style_encoder.encode(unified_conditioning)

            # Extract individual attributes
            attributes = self.style_encoder.extract_attributes(style_vector)

        # Convert attributes to features dict
        features = {}
        for attr_name, attr_tensor in attributes.items():
            # Convert to scalar features
            features[attr_name] = float(attr_tensor.mean().item())

        # Create style profile
        profile = StyleProfile(
            name=style_name,
            genre=genre or "unknown",
            artist=artist,
            features=features,
            conditioning_vector=style_vector,
            audio_examples=[audio],
            description=f"Style extracted from {artist or 'unknown'} audio",
        )

        # Add to library
        self.style_library[style_name] = profile

        return profile

    def transfer_style(
        self,
        source_audio: torch.Tensor,
        target_style: Union[str, StyleProfile],
        config: Optional[StyleTransferConfig] = None,
    ) -> torch.Tensor:
        """
        Transfer style from target to source audio.

        Args:
            source_audio: Original audio to transform
            target_style: Target style (name or profile)
            config: Transfer configuration

        Returns:
            Style-transferred audio conditioning
        """

        if config is None:
            config = StyleTransferConfig()

        # Get target style profile
        if isinstance(target_style, str):
            if target_style not in self.style_library:
                raise ValueError(f"Style '{target_style}' not found in library")
            target_profile = self.style_library[target_style]
        else:
            target_profile = target_style

        # Extract source style
        source_conditioning = self.conditioning_system.create_comprehensive_conditioning(
            text_prompt="", reference_audio=source_audio
        )
        source_unified = self.conditioning_system.fuse_conditioning(source_conditioning)
        source_style = self.style_encoder.encode(source_unified)

        # Get target style vector
        target_style_vector = target_profile.conditioning_vector
        if target_style_vector is None:
            raise ValueError("Target style profile missing conditioning vector")

        # Extract attributes from both styles
        source_attributes = self.style_encoder.extract_attributes(source_style)
        target_attributes = self.style_encoder.extract_attributes(target_style_vector)

        # Perform selective attribute transfer
        transferred_attributes = {}

        for attr in StyleAttribute:
            attr_name = attr.value

            if attr in config.preserve_attributes:
                # Preserve source attribute
                transferred_attributes[attr_name] = source_attributes[attr_name]
            elif attr in config.transfer_attributes:
                # Transfer target attribute
                transferred_attributes[attr_name] = target_attributes[attr_name]
            else:
                # Blend attributes based on weights
                source_weight = config.source_weight
                target_weight = config.target_weight

                transferred_attributes[attr_name] = (
                    source_weight * source_attributes[attr_name]
                    + target_weight * target_attributes[attr_name]
                )

        # Reconstruct style vector from transferred attributes
        transferred_style = self.style_encoder.combine_attributes(transferred_attributes)

        # Decode back to conditioning space
        transferred_conditioning = self.style_encoder.decode(transferred_style)

        return transferred_conditioning

    def interpolate_styles(
        self,
        style_names: List[str],
        weights: List[float],
        mode: InterpolationMode = InterpolationMode.LINEAR,
    ) -> torch.Tensor:
        """
        Interpolate between multiple styles in the library.

        Args:
            style_names: Names of styles to interpolate
            weights: Interpolation weights
            mode: Interpolation mode

        Returns:
            Interpolated conditioning vector
        """

        # Get style vectors
        style_vectors = []
        for name in style_names:
            if name not in self.style_library:
                raise ValueError(f"Style '{name}' not found in library")

            profile = self.style_library[name]
            if profile.conditioning_vector is None:
                raise ValueError(f"Style '{name}' missing conditioning vector")

            style_vectors.append(profile.conditioning_vector)

        # Interpolate in style space
        interpolated_style = self.interpolator.multi_style_interpolation(
            style_vectors, weights, mode
        )

        # Decode to conditioning space
        interpolated_conditioning = self.style_encoder.decode(interpolated_style)

        return interpolated_conditioning

    def create_style_morph_sequence(
        self,
        start_style: str,
        end_style: str,
        num_steps: int,
        mode: InterpolationMode = InterpolationMode.SMOOTH_STEP,
    ) -> List[torch.Tensor]:
        """
        Create a sequence of style morphs between two styles.

        Args:
            start_style: Starting style name
            end_style: Ending style name
            num_steps: Number of morphing steps
            mode: Interpolation mode

        Returns:
            List of conditioning vectors for morphing sequence
        """

        # Get style vectors
        if start_style not in self.style_library:
            raise ValueError(f"Style '{start_style}' not found in library")
        if end_style not in self.style_library:
            raise ValueError(f"Style '{end_style}' not found in library")

        start_vector = self.style_library[start_style].conditioning_vector
        end_vector = self.style_library[end_style].conditioning_vector

        if start_vector is None or end_vector is None:
            raise ValueError("Style profiles missing conditioning vectors")

        # Create interpolation sequence in style space
        style_sequence = self.interpolator.create_transition_sequence(
            start_vector, end_vector, num_steps, mode
        )

        # Decode all styles to conditioning space
        conditioning_sequence = []
        for style_vector in style_sequence:
            conditioning = self.style_encoder.decode(style_vector)
            conditioning_sequence.append(conditioning)

        return conditioning_sequence

    def analyze_style_similarity(self, style_a: str, style_b: str) -> Dict[str, float]:
        """
        Analyze similarity between two styles.

        Args:
            style_a: First style name
            style_b: Second style name

        Returns:
            Dictionary of similarity metrics
        """

        if style_a not in self.style_library or style_b not in self.style_library:
            raise ValueError("One or both styles not found in library")

        profile_a = self.style_library[style_a]
        profile_b = self.style_library[style_b]

        similarity = {}

        # Feature-based similarity
        feature_similarities = {}
        for feature_name in profile_a.features:
            if feature_name in profile_b.features:
                diff = abs(profile_a.features[feature_name] - profile_b.features[feature_name])
                feature_similarities[feature_name] = 1.0 - diff

        similarity["feature_similarity"] = feature_similarities
        similarity["average_feature_similarity"] = np.mean(list(feature_similarities.values()))

        # Vector-based similarity
        if profile_a.conditioning_vector is not None and profile_b.conditioning_vector is not None:
            # Cosine similarity
            vec_a = profile_a.conditioning_vector.flatten()
            vec_b = profile_b.conditioning_vector.flatten()

            cosine_sim = F.cosine_similarity(vec_a.unsqueeze(0), vec_b.unsqueeze(0))
            similarity["cosine_similarity"] = float(cosine_sim.item())

            # Euclidean distance (normalized)
            euclidean_dist = torch.norm(vec_a - vec_b).item()
            max_possible_dist = torch.norm(torch.ones_like(vec_a)).item() * 2  # Rough estimate
            similarity["euclidean_similarity"] = 1.0 - (euclidean_dist / max_possible_dist)

        # Genre/artist similarity
        if profile_a.genre == profile_b.genre:
            similarity["genre_match"] = 1.0
        else:
            similarity["genre_match"] = 0.0

        if profile_a.artist and profile_b.artist and profile_a.artist == profile_b.artist:
            similarity["artist_match"] = 1.0
        else:
            similarity["artist_match"] = 0.0

        return similarity

    def save_style_library(self, path: Path):
        """Save style library to disk."""
        library_data = {}

        for name, profile in self.style_library.items():
            profile_data = {
                "name": profile.name,
                "genre": profile.genre,
                "artist": profile.artist,
                "era": profile.era,
                "features": profile.features,
                "description": profile.description,
            }

            # Save conditioning vector separately (binary format)
            if profile.conditioning_vector is not None:
                vector_path = path.parent / f"{name}_conditioning.pt"
                torch.save(profile.conditioning_vector, vector_path)
                profile_data["conditioning_vector_path"] = str(vector_path)

            library_data[name] = profile_data

        # Save metadata
        with open(path, "w") as f:
            json.dump(library_data, f, indent=2)

        self.logger.info(f"Style library saved to {path}")

    def load_style_library(self, path: Path):
        """Load style library from disk."""
        with open(path, "r") as f:
            library_data = json.load(f)

        self.style_library = {}

        for name, profile_data in library_data.items():
            # Load conditioning vector if available
            conditioning_vector = None
            if "conditioning_vector_path" in profile_data:
                vector_path = Path(profile_data["conditioning_vector_path"])
                if vector_path.exists():
                    conditioning_vector = torch.load(vector_path, map_location=self.device)

            # Create profile
            profile = StyleProfile(
                name=profile_data["name"],
                genre=profile_data["genre"],
                artist=profile_data.get("artist"),
                era=profile_data.get("era"),
                features=profile_data["features"],
                conditioning_vector=conditioning_vector,
                description=profile_data.get("description", ""),
            )

            self.style_library[name] = profile

        self.logger.info(f"Style library loaded from {path}")


class StyleTransferManager:
    """High-level manager for style transfer workflows."""

    def __init__(self, style_system: StyleTransferSystem):
        self.style_system = style_system
        self.logger = logging.getLogger(__name__)

    def setup_artist_styles(self, artist_audio_samples: Dict[str, List[torch.Tensor]]):
        """Set up style profiles for multiple artists."""

        for artist_name, audio_samples in artist_audio_samples.items():
            self.logger.info(f"Setting up style for {artist_name}")

            # Use first sample for style extraction (could be improved by averaging)
            if audio_samples:
                style_profile = self.style_system.extract_style_from_audio(
                    audio_samples[0], style_name=artist_name, artist=artist_name
                )

                # Add additional samples
                for sample in audio_samples[1:]:
                    style_profile.audio_examples.append(sample)

    def create_genre_fusion(self, genre_a: str, genre_b: str, fusion_ratio: float = 0.5) -> str:
        """Create a fusion style between two genres."""

        fusion_name = f"{genre_a}_{genre_b}_fusion"

        # Find styles from each genre
        genre_a_styles = [
            name
            for name, profile in self.style_system.style_library.items()
            if profile.genre == genre_a
        ]
        genre_b_styles = [
            name
            for name, profile in self.style_system.style_library.items()
            if profile.genre == genre_b
        ]

        if not genre_a_styles or not genre_b_styles:
            raise ValueError(f"Not enough styles found for genres {genre_a} or {genre_b}")

        # Use first style from each genre (could be improved)
        style_a = genre_a_styles[0]
        style_b = genre_b_styles[0]

        # Create interpolated style
        fusion_conditioning = self.style_system.interpolate_styles(
            [style_a, style_b], [1 - fusion_ratio, fusion_ratio], InterpolationMode.SPHERICAL
        )

        # Create fusion profile
        fusion_profile = StyleProfile(
            name=fusion_name,
            genre=f"{genre_a}_fusion",
            description=f"Fusion of {genre_a} and {genre_b}",
            conditioning_vector=fusion_conditioning,
        )

        # Add to library
        self.style_system.style_library[fusion_name] = fusion_profile

        return fusion_name
