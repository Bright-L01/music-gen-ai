"""
Advanced Composition System for Commercial-Grade Music Generation

Implements extended composition (2-5 minutes) with intelligent song structure,
arrangement templates, and multi-section generation capabilities.
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..inference.generators.base import GenerationConfig
from ..models.conditioning.text_encoder import T5TextEncoder
from ..models.musicgen import MusicGenForConditionalGeneration
from ..utils.audio.processing import AudioProcessor
from .music_theory import MusicTheoryEngine, TimeSignature


class SongSection(Enum):
    """Song structure sections."""

    INTRO = "intro"
    VERSE = "verse"
    CHORUS = "chorus"
    BRIDGE = "bridge"
    OUTRO = "outro"
    SOLO = "solo"
    BREAKDOWN = "breakdown"
    PRE_CHORUS = "pre_chorus"


@dataclass
class SectionConfig:
    """Configuration for a song section."""

    section_type: SongSection
    duration: float  # seconds
    intensity: float  # 0.0 to 1.0
    tempo_factor: float = 1.0  # relative to base tempo
    key_modulation: Optional[str] = None
    chord_progression: Optional[List[str]] = None
    arrangement_density: float = 0.5  # 0.0 to 1.0
    dynamic_range: Tuple[float, float] = (0.3, 0.8)

    def __post_init__(self):
        """Validate section configuration."""
        if not 0.0 <= self.intensity <= 1.0:
            raise ValueError("Intensity must be between 0.0 and 1.0")
        if not 0.0 <= self.arrangement_density <= 1.0:
            raise ValueError("Arrangement density must be between 0.0 and 1.0")


@dataclass
class SongStructure:
    """Complete song structure definition."""

    sections: List[SectionConfig]
    total_duration: float
    base_tempo: int = 120
    base_key: str = "C"
    time_signature: TimeSignature = field(default_factory=lambda: TimeSignature(4, 4))
    genre: str = "pop"
    style_tags: List[str] = field(default_factory=list)

    @property
    def section_count(self) -> int:
        """Number of sections in the structure."""
        return len(self.sections)

    def get_section_at_time(self, time: float) -> Optional[SectionConfig]:
        """Get the section that should be playing at given time."""
        current_time = 0.0
        for section in self.sections:
            if current_time <= time < current_time + section.duration:
                return section
            current_time += section.duration
        return None


class AdvancedComposer(nn.Module):
    """
    Advanced composition system for generating extended musical pieces
    with intelligent song structure and arrangement.
    """

    def __init__(
        self,
        model: MusicGenForConditionalGeneration,
        text_encoder: T5TextEncoder,
        music_theory_engine: MusicTheoryEngine,
        audio_processor: AudioProcessor,
        device: torch.device = None,
    ):
        super().__init__()

        self.model = model
        self.text_encoder = text_encoder
        self.music_theory = music_theory_engine
        self.audio_processor = audio_processor
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.logger = logging.getLogger(__name__)

        # Load genre-specific templates
        self.genre_templates = self._load_genre_templates()

        # Section transition weights
        self.transition_weights = self._initialize_transition_weights()

    def _load_genre_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load genre-specific song structure templates."""
        templates = {
            "pop": {
                "structure": [
                    ("intro", 8),
                    ("verse", 16),
                    ("chorus", 16),
                    ("verse", 16),
                    ("chorus", 16),
                    ("bridge", 8),
                    ("chorus", 16),
                    ("outro", 8),
                ],
                "tempo_range": (100, 140),
                "typical_keys": ["C", "G", "D", "A", "F", "Bb"],
                "chord_progressions": {
                    "verse": ["vi", "IV", "I", "V"],
                    "chorus": ["I", "V", "vi", "IV"],
                    "bridge": ["ii", "V", "I", "vi"],
                },
            },
            "rock": {
                "structure": [
                    ("intro", 8),
                    ("verse", 16),
                    ("chorus", 16),
                    ("verse", 16),
                    ("chorus", 16),
                    ("solo", 16),
                    ("chorus", 16),
                    ("outro", 8),
                ],
                "tempo_range": (120, 180),
                "typical_keys": ["E", "A", "D", "G", "B"],
                "chord_progressions": {
                    "verse": ["i", "VII", "VI", "VII"],
                    "chorus": ["i", "v", "VI", "VII"],
                    "solo": ["i", "VII", "i", "VII"],
                },
            },
            "jazz": {
                "structure": [
                    ("intro", 16),
                    ("verse", 32),
                    ("chorus", 32),
                    ("verse", 32),
                    ("chorus", 32),
                    ("solo", 64),
                    ("chorus", 32),
                    ("outro", 16),
                ],
                "tempo_range": (80, 200),
                "typical_keys": ["F", "Bb", "Eb", "Ab", "Db"],
                "chord_progressions": {
                    "verse": ["ii7", "V7", "Imaj7", "vi7"],
                    "chorus": ["Imaj7", "vi7", "ii7", "V7"],
                    "solo": ["ii7", "V7", "Imaj7", "Imaj7"],
                },
            },
            "electronic": {
                "structure": [
                    ("intro", 16),
                    ("breakdown", 16),
                    ("verse", 16),
                    ("chorus", 32),
                    ("breakdown", 16),
                    ("chorus", 32),
                    ("outro", 16),
                ],
                "tempo_range": (120, 150),
                "typical_keys": ["Am", "Dm", "Em", "Bm"],
                "chord_progressions": {
                    "verse": ["vi", "IV", "I", "V"],
                    "chorus": ["I", "V", "vi", "IV"],
                    "breakdown": ["vi", "vi", "IV", "IV"],
                },
            },
        }

        return templates

    def _initialize_transition_weights(self) -> Dict[Tuple[SongSection, SongSection], float]:
        """Initialize section transition weights for smooth generation."""
        weights = {}

        # Common transitions with higher weights
        high_weight_transitions = [
            (SongSection.INTRO, SongSection.VERSE),
            (SongSection.VERSE, SongSection.CHORUS),
            (SongSection.CHORUS, SongSection.VERSE),
            (SongSection.VERSE, SongSection.BRIDGE),
            (SongSection.BRIDGE, SongSection.CHORUS),
            (SongSection.CHORUS, SongSection.OUTRO),
            (SongSection.SOLO, SongSection.CHORUS),
        ]

        # Initialize all transitions with base weight
        for section1 in SongSection:
            for section2 in SongSection:
                if (section1, section2) in high_weight_transitions:
                    weights[(section1, section2)] = 1.0
                else:
                    weights[(section1, section2)] = 0.3

        return weights

    def create_song_structure(
        self,
        duration: float,
        genre: str = "pop",
        complexity: float = 0.5,
        custom_structure: Optional[List[Tuple[str, float]]] = None,
    ) -> SongStructure:
        """
        Create an intelligent song structure based on genre and duration.

        Args:
            duration: Target duration in seconds
            genre: Musical genre for template selection
            complexity: Structure complexity (0.0 = simple, 1.0 = complex)
            custom_structure: Optional custom structure override

        Returns:
            Generated song structure
        """
        if custom_structure:
            sections = []
            for section_name, section_duration in custom_structure:
                section_type = SongSection(section_name.lower())
                sections.append(
                    SectionConfig(
                        section_type=section_type,
                        duration=section_duration,
                        intensity=self._get_section_intensity(section_type),
                        arrangement_density=self._get_section_density(section_type, complexity),
                    )
                )

            return SongStructure(
                sections=sections, total_duration=sum(s.duration for s in sections), genre=genre
            )

        # Use genre template
        template = self.genre_templates.get(genre, self.genre_templates["pop"])
        base_structure = template["structure"]

        # Scale structure to target duration
        base_duration = sum(duration for _, duration in base_structure)
        scale_factor = duration / base_duration

        sections = []
        for section_name, section_duration in base_structure:
            section_type = SongSection(section_name)
            scaled_duration = section_duration * scale_factor

            # Add complexity-based variations
            if complexity > 0.7:
                # Add extra repetitions for complex arrangements
                if section_type in [SongSection.VERSE, SongSection.CHORUS]:
                    scaled_duration *= 1.2

            sections.append(
                SectionConfig(
                    section_type=section_type,
                    duration=scaled_duration,
                    intensity=self._get_section_intensity(section_type),
                    tempo_factor=self._get_tempo_factor(section_type),
                    chord_progression=template["chord_progressions"].get(section_name),
                    arrangement_density=self._get_section_density(section_type, complexity),
                )
            )

        # Select appropriate key and tempo
        tempo = np.random.randint(*template["tempo_range"])
        key = np.random.choice(template["typical_keys"])

        return SongStructure(
            sections=sections,
            total_duration=sum(s.duration for s in sections),
            base_tempo=tempo,
            base_key=key,
            genre=genre,
            style_tags=self._generate_style_tags(genre, complexity),
        )

    def _get_section_intensity(self, section_type: SongSection) -> float:
        """Get typical intensity for section type."""
        intensity_map = {
            SongSection.INTRO: 0.3,
            SongSection.VERSE: 0.5,
            SongSection.CHORUS: 0.8,
            SongSection.BRIDGE: 0.6,
            SongSection.OUTRO: 0.4,
            SongSection.SOLO: 0.9,
            SongSection.BREAKDOWN: 0.2,
            SongSection.PRE_CHORUS: 0.7,
        }
        return intensity_map.get(section_type, 0.5)

    def _get_tempo_factor(self, section_type: SongSection) -> float:
        """Get tempo factor for section type."""
        tempo_map = {
            SongSection.INTRO: 0.9,
            SongSection.VERSE: 1.0,
            SongSection.CHORUS: 1.0,
            SongSection.BRIDGE: 0.95,
            SongSection.OUTRO: 0.9,
            SongSection.SOLO: 1.1,
            SongSection.BREAKDOWN: 0.8,
            SongSection.PRE_CHORUS: 1.05,
        }
        return tempo_map.get(section_type, 1.0)

    def _get_section_density(self, section_type: SongSection, complexity: float) -> float:
        """Get arrangement density for section type."""
        base_density = {
            SongSection.INTRO: 0.3,
            SongSection.VERSE: 0.5,
            SongSection.CHORUS: 0.8,
            SongSection.BRIDGE: 0.6,
            SongSection.OUTRO: 0.4,
            SongSection.SOLO: 0.9,
            SongSection.BREAKDOWN: 0.2,
            SongSection.PRE_CHORUS: 0.7,
        }

        density = base_density.get(section_type, 0.5)
        # Scale by complexity
        return min(1.0, density * (0.7 + 0.6 * complexity))

    def _generate_style_tags(self, genre: str, complexity: float) -> List[str]:
        """Generate style tags based on genre and complexity."""
        base_tags = [genre]

        if complexity > 0.7:
            base_tags.extend(["complex", "layered"])
        elif complexity < 0.3:
            base_tags.extend(["minimal", "simple"])
        else:
            base_tags.append("balanced")

        genre_specific_tags = {
            "pop": ["catchy", "melodic", "commercial"],
            "rock": ["energetic", "powerful", "driving"],
            "jazz": ["sophisticated", "improvisational", "swing"],
            "electronic": ["synthetic", "rhythmic", "atmospheric"],
        }

        base_tags.extend(genre_specific_tags.get(genre, []))
        return base_tags

    async def generate_extended_composition(
        self,
        prompt: str,
        song_structure: SongStructure,
        generation_config: GenerationConfig,
        section_overlap: float = 2.0,
        use_transitions: bool = True,
    ) -> torch.Tensor:
        """
        Generate an extended musical composition with song structure.

        Args:
            prompt: Base text prompt for generation
            song_structure: Song structure to follow
            generation_config: Generation parameters
            section_overlap: Overlap between sections in seconds
            use_transitions: Whether to generate smooth transitions

        Returns:
            Generated audio tensor
        """
        self.logger.info(f"Generating extended composition: {song_structure.total_duration:.1f}s")

        generated_sections = []
        previous_audio = None

        for i, section in enumerate(song_structure.sections):
            self.logger.info(
                f"Generating section {i+1}/{len(song_structure.sections)}: {section.section_type.value}"
            )

            # Create section-specific prompt
            section_prompt = self._create_section_prompt(prompt, section, song_structure)

            # Generate section with context
            section_audio = await self._generate_section(
                section_prompt,
                section,
                generation_config,
                previous_audio if use_transitions else None,
                section_overlap if i > 0 else 0.0,
            )

            generated_sections.append(section_audio)
            previous_audio = section_audio

        # Combine sections with crossfades
        complete_composition = self._combine_sections(
            generated_sections,
            [s.duration for s in song_structure.sections],
            section_overlap if use_transitions else 0.0,
        )

        # Apply final mastering
        complete_composition = self._apply_final_mastering(complete_composition, song_structure)

        self.logger.info(
            f"Extended composition generated: {complete_composition.shape[-1] / 32000:.1f}s"
        )
        return complete_composition

    def _create_section_prompt(
        self, base_prompt: str, section: SectionConfig, song_structure: SongStructure
    ) -> str:
        """Create section-specific prompt with musical context."""
        section_descriptors = {
            SongSection.INTRO: "introduction, opening, building",
            SongSection.VERSE: "verse, storytelling, melodic",
            SongSection.CHORUS: "chorus, hook, memorable, energetic",
            SongSection.BRIDGE: "bridge, contrast, different, transitional",
            SongSection.OUTRO: "outro, ending, conclusion, fade",
            SongSection.SOLO: "solo, instrumental, featured, expressive",
            SongSection.BREAKDOWN: "breakdown, minimal, stripped down",
            SongSection.PRE_CHORUS: "pre-chorus, building, anticipation",
        }

        # Combine base prompt with section-specific elements
        section_prompt = f"{base_prompt}, {section_descriptors[section.section_type]}"

        # Add intensity descriptors
        if section.intensity > 0.8:
            section_prompt += ", intense, energetic, powerful"
        elif section.intensity < 0.3:
            section_prompt += ", subtle, gentle, quiet"

        # Add tempo information
        if section.tempo_factor > 1.1:
            section_prompt += ", fast, driving"
        elif section.tempo_factor < 0.9:
            section_prompt += ", slow, relaxed"

        # Add arrangement density
        if section.arrangement_density > 0.8:
            section_prompt += ", full arrangement, layered, rich"
        elif section.arrangement_density < 0.3:
            section_prompt += ", minimal arrangement, sparse"

        # Add genre and style tags
        section_prompt += f", {song_structure.genre}"
        section_prompt += ", " + ", ".join(song_structure.style_tags)

        return section_prompt

    async def _generate_section(
        self,
        prompt: str,
        section: SectionConfig,
        config: GenerationConfig,
        previous_audio: Optional[torch.Tensor] = None,
        overlap_duration: float = 0.0,
    ) -> torch.Tensor:
        """Generate audio for a single section."""
        # Encode prompt
        text_encoding = self.text_encoder.encode(prompt)

        # Calculate generation length
        sample_rate = self.audio_processor.sample_rate
        target_length = int(section.duration * sample_rate)

        # Prepare conditioning
        conditioning = {
            "text_encoding": text_encoding,
            "intensity": section.intensity,
            "tempo_factor": section.tempo_factor,
            "arrangement_density": section.arrangement_density,
        }

        # Add previous audio context for smooth transitions
        if previous_audio is not None and overlap_duration > 0:
            overlap_samples = int(overlap_duration * sample_rate)
            conditioning["previous_context"] = previous_audio[..., -overlap_samples:]

        # Generate section
        with torch.no_grad():
            section_audio = self.model.generate(
                conditioning=conditioning, target_length=target_length, **config.to_dict()
            )

        return section_audio

    def _combine_sections(
        self, sections: List[torch.Tensor], durations: List[float], overlap_duration: float = 0.0
    ) -> torch.Tensor:
        """Combine sections with smooth crossfades."""
        if not sections:
            raise ValueError("No sections to combine")

        if len(sections) == 1:
            return sections[0]

        sample_rate = self.audio_processor.sample_rate
        overlap_samples = int(overlap_duration * sample_rate)

        # Calculate total length
        total_samples = sum(s.shape[-1] for s in sections)
        if overlap_duration > 0:
            total_samples -= overlap_samples * (len(sections) - 1)

        # Initialize output tensor
        batch_size, channels = sections[0].shape[0], sections[0].shape[1]
        combined = torch.zeros(batch_size, channels, total_samples, device=sections[0].device)

        current_pos = 0

        for i, section in enumerate(sections):
            section_length = section.shape[-1]

            if i == 0:
                # First section - no overlap
                combined[..., :section_length] = section
                current_pos = section_length
            else:
                # Subsequent sections - apply crossfade
                if overlap_samples > 0:
                    # Crossfade region
                    fade_start = current_pos - overlap_samples
                    fade_end = current_pos

                    # Create fade curves
                    fade_out = torch.linspace(1, 0, overlap_samples, device=section.device)
                    fade_in = torch.linspace(0, 1, overlap_samples, device=section.device)

                    # Apply crossfade
                    combined[..., fade_start:fade_end] *= fade_out
                    combined[..., fade_start:fade_end] += section[..., :overlap_samples] * fade_in

                    # Add remaining section
                    remaining_start = current_pos
                    remaining_end = current_pos + section_length - overlap_samples
                    combined[..., remaining_start:remaining_end] = section[..., overlap_samples:]

                    current_pos = remaining_end
                else:
                    # No overlap - direct concatenation
                    end_pos = current_pos + section_length
                    combined[..., current_pos:end_pos] = section
                    current_pos = end_pos

        return combined

    def _apply_final_mastering(
        self, audio: torch.Tensor, song_structure: SongStructure
    ) -> torch.Tensor:
        """Apply final mastering effects to the complete composition."""
        # Apply dynamic range compression
        audio = self.audio_processor.apply_compression(
            audio, ratio=4.0, threshold=-12.0, attack=0.003, release=0.1
        )

        # Apply EQ based on genre
        if song_structure.genre == "rock":
            # Boost mids and highs for rock
            audio = self.audio_processor.apply_eq(audio, [(800, 2.0), (3000, 1.5), (8000, 1.0)])
        elif song_structure.genre == "electronic":
            # Enhance bass and highs for electronic
            audio = self.audio_processor.apply_eq(audio, [(60, 2.0), (200, -1.0), (10000, 2.0)])

        # Final limiter to prevent clipping
        audio = self.audio_processor.apply_limiter(audio, threshold=-1.0, release=0.05)

        # Normalize to target loudness
        audio = self.audio_processor.normalize_loudness(audio, target_lufs=-14.0)

        return audio

    def analyze_composition_structure(self, audio: torch.Tensor) -> Dict[str, Any]:
        """Analyze the structure of a generated composition."""
        # This would typically use onset detection, beat tracking,
        # and harmonic analysis to identify sections

        # Placeholder implementation
        duration = audio.shape[-1] / self.audio_processor.sample_rate

        return {
            "total_duration": duration,
            "estimated_sections": [],
            "tempo_analysis": {},
            "key_analysis": {},
            "dynamic_range": float(torch.std(audio).item()),
            "spectral_centroid": self.audio_processor.spectral_centroid(audio).mean().item(),
        }


class CompositionManager:
    """High-level manager for advanced composition workflows."""

    def __init__(self, composer: AdvancedComposer):
        self.composer = composer
        self.logger = logging.getLogger(__name__)

    async def create_song(
        self,
        prompt: str,
        duration: float = 180.0,
        genre: str = "pop",
        complexity: float = 0.5,
        custom_structure: Optional[List[Tuple[str, float]]] = None,
    ) -> Tuple[torch.Tensor, SongStructure]:
        """
        Create a complete song with intelligent structure.

        Args:
            prompt: Text description of desired music
            duration: Target duration in seconds
            genre: Musical genre
            complexity: Arrangement complexity
            custom_structure: Optional custom structure

        Returns:
            Tuple of (generated audio, song structure)
        """
        # Create song structure
        structure = self.composer.create_song_structure(
            duration=duration, genre=genre, complexity=complexity, custom_structure=custom_structure
        )

        # Generate configuration
        config = GenerationConfig(temperature=0.8, top_k=50, top_p=0.9, guidance_scale=7.5)

        # Generate composition
        audio = await self.composer.generate_extended_composition(
            prompt=prompt,
            song_structure=structure,
            generation_config=config,
            section_overlap=2.0,
            use_transitions=True,
        )

        return audio, structure

    def save_composition(
        self,
        audio: torch.Tensor,
        structure: SongStructure,
        output_path: Path,
        include_metadata: bool = True,
    ):
        """Save composition with metadata."""
        # Save audio
        self.composer.audio_processor.save_audio(audio, output_path)

        if include_metadata:
            # Save structure metadata
            metadata_path = output_path.with_suffix(".json")
            structure_data = {
                "total_duration": structure.total_duration,
                "base_tempo": structure.base_tempo,
                "base_key": structure.base_key,
                "genre": structure.genre,
                "style_tags": structure.style_tags,
                "sections": [
                    {
                        "type": s.section_type.value,
                        "duration": s.duration,
                        "intensity": s.intensity,
                        "tempo_factor": s.tempo_factor,
                        "arrangement_density": s.arrangement_density,
                    }
                    for s in structure.sections
                ],
            }

            with open(metadata_path, "w") as f:
                json.dump(structure_data, f, indent=2)
