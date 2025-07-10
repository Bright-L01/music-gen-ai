"""
Intelligent Music Theory Engine

Provides advanced music theory analysis, chord progression generation,
harmonic analysis, and music structure intelligence for commercial-grade
music generation.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class ScaleType(Enum):
    """Musical scale types."""

    MAJOR = "major"
    MINOR = "minor"
    DORIAN = "dorian"
    PHRYGIAN = "phrygian"
    LYDIAN = "lydian"
    MIXOLYDIAN = "mixolydian"
    AEOLIAN = "aeolian"
    LOCRIAN = "locrian"
    HARMONIC_MINOR = "harmonic_minor"
    MELODIC_MINOR = "melodic_minor"
    PENTATONIC_MAJOR = "pentatonic_major"
    PENTATONIC_MINOR = "pentatonic_minor"
    BLUES = "blues"


class ChordQuality(Enum):
    """Chord quality types."""

    MAJOR = "major"
    MINOR = "minor"
    DIMINISHED = "diminished"
    AUGMENTED = "augmented"
    MAJOR_SEVENTH = "major_seventh"
    MINOR_SEVENTH = "minor_seventh"
    DOMINANT_SEVENTH = "dominant_seventh"
    HALF_DIMINISHED = "half_diminished"
    FULLY_DIMINISHED = "fully_diminished"
    SUSPENDED_SECOND = "sus2"
    SUSPENDED_FOURTH = "sus4"
    MAJOR_SIXTH = "major_sixth"
    MINOR_SIXTH = "minor_sixth"
    MAJOR_NINTH = "major_ninth"
    MINOR_NINTH = "minor_ninth"


@dataclass
class TimeSignature:
    """Time signature representation."""

    numerator: int
    denominator: int

    def __str__(self) -> str:
        return f"{self.numerator}/{self.denominator}"

    @property
    def beats_per_measure(self) -> int:
        """Number of beats per measure."""
        return self.numerator

    @property
    def beat_unit(self) -> int:
        """Note value that gets one beat."""
        return self.denominator


@dataclass
class Note:
    """Musical note representation."""

    pitch_class: int  # 0-11 (C=0, C#=1, D=2, etc.)
    octave: int = 4
    duration: float = 1.0  # in beats
    velocity: float = 0.7  # 0.0 to 1.0

    @property
    def midi_number(self) -> int:
        """MIDI note number."""
        return self.octave * 12 + self.pitch_class + 12

    @property
    def frequency(self) -> float:
        """Fundamental frequency in Hz."""
        return 440.0 * (2 ** ((self.midi_number - 69) / 12))

    def __str__(self) -> str:
        note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        return f"{note_names[self.pitch_class]}{self.octave}"


@dataclass
class Chord:
    """Musical chord representation."""

    root: int  # 0-11
    quality: ChordQuality
    bass: Optional[int] = None  # Bass note for inversions
    extensions: List[int] = field(default_factory=list)  # Additional notes

    @property
    def notes(self) -> List[int]:
        """Get all notes in the chord."""
        intervals = self._get_chord_intervals()
        notes = [(self.root + interval) % 12 for interval in intervals]

        # Add extensions
        for ext in self.extensions:
            notes.append((self.root + ext) % 12)

        return sorted(list(set(notes)))

    def _get_chord_intervals(self) -> List[int]:
        """Get intervals for chord quality."""
        intervals_map = {
            ChordQuality.MAJOR: [0, 4, 7],
            ChordQuality.MINOR: [0, 3, 7],
            ChordQuality.DIMINISHED: [0, 3, 6],
            ChordQuality.AUGMENTED: [0, 4, 8],
            ChordQuality.MAJOR_SEVENTH: [0, 4, 7, 11],
            ChordQuality.MINOR_SEVENTH: [0, 3, 7, 10],
            ChordQuality.DOMINANT_SEVENTH: [0, 4, 7, 10],
            ChordQuality.HALF_DIMINISHED: [0, 3, 6, 10],
            ChordQuality.FULLY_DIMINISHED: [0, 3, 6, 9],
            ChordQuality.SUSPENDED_SECOND: [0, 2, 7],
            ChordQuality.SUSPENDED_FOURTH: [0, 5, 7],
            ChordQuality.MAJOR_SIXTH: [0, 4, 7, 9],
            ChordQuality.MINOR_SIXTH: [0, 3, 7, 9],
            ChordQuality.MAJOR_NINTH: [0, 4, 7, 11, 14],
            ChordQuality.MINOR_NINTH: [0, 3, 7, 10, 14],
        }
        return intervals_map.get(self.quality, [0, 4, 7])

    def __str__(self) -> str:
        note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        chord_symbols = {
            ChordQuality.MAJOR: "",
            ChordQuality.MINOR: "m",
            ChordQuality.DIMINISHED: "dim",
            ChordQuality.AUGMENTED: "aug",
            ChordQuality.MAJOR_SEVENTH: "maj7",
            ChordQuality.MINOR_SEVENTH: "m7",
            ChordQuality.DOMINANT_SEVENTH: "7",
            ChordQuality.HALF_DIMINISHED: "m7b5",
            ChordQuality.FULLY_DIMINISHED: "dim7",
            ChordQuality.SUSPENDED_SECOND: "sus2",
            ChordQuality.SUSPENDED_FOURTH: "sus4",
            ChordQuality.MAJOR_SIXTH: "6",
            ChordQuality.MINOR_SIXTH: "m6",
            ChordQuality.MAJOR_NINTH: "maj9",
            ChordQuality.MINOR_NINTH: "m9",
        }

        symbol = f"{note_names[self.root]}{chord_symbols[self.quality]}"

        if self.bass is not None and self.bass != self.root:
            symbol += f"/{note_names[self.bass]}"

        return symbol


@dataclass
class ChordProgression:
    """Musical chord progression."""

    chords: List[Chord]
    key: str
    time_signature: TimeSignature = field(default_factory=lambda: TimeSignature(4, 4))
    chord_durations: Optional[List[float]] = None  # Duration of each chord in beats

    def __post_init__(self):
        if self.chord_durations is None:
            # Default to equal duration for all chords
            beats_per_chord = self.time_signature.beats_per_measure / len(self.chords)
            self.chord_durations = [beats_per_chord] * len(self.chords)

    @property
    def total_duration(self) -> float:
        """Total duration in beats."""
        return sum(self.chord_durations or [])

    def roman_numeral_analysis(self) -> List[str]:
        """Convert to Roman numeral analysis."""
        key_root = self._parse_key(self.key)
        is_minor = "m" in self.key.lower() or "minor" in self.key.lower()

        roman_numerals = []
        for chord in self.chords:
            roman = self._chord_to_roman(chord, key_root, is_minor)
            roman_numerals.append(roman)

        return roman_numerals

    def _parse_key(self, key: str) -> int:
        """Parse key string to root note."""
        note_map = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

        key_clean = key.replace("m", "").replace("minor", "").replace("major", "").strip()
        if "#" in key_clean:
            return (note_map[key_clean[0]] + 1) % 12
        elif "b" in key_clean:
            return (note_map[key_clean[0]] - 1) % 12
        else:
            return note_map[key_clean[0]]

    def _chord_to_roman(self, chord: Chord, key_root: int, is_minor: bool) -> str:
        """Convert chord to Roman numeral notation."""
        # Calculate degree
        degree = (chord.root - key_root) % 12

        # Map to scale degree
        if is_minor:
            degree_map = {0: "i", 2: "ii", 3: "III", 5: "iv", 7: "v", 8: "VI", 10: "VII"}
        else:
            degree_map = {0: "I", 2: "ii", 4: "iii", 5: "IV", 7: "V", 9: "vi", 11: "vii"}

        roman = degree_map.get(degree, f"?{degree}")

        # Add quality modifiers
        if chord.quality == ChordQuality.DIMINISHED:
            roman += "°"
        elif chord.quality == ChordQuality.AUGMENTED:
            roman += "+"
        elif chord.quality in [
            ChordQuality.DOMINANT_SEVENTH,
            ChordQuality.MAJOR_SEVENTH,
            ChordQuality.MINOR_SEVENTH,
        ]:
            roman += "7"

        return roman


class Scale:
    """Musical scale representation and analysis."""

    def __init__(self, root: int, scale_type: ScaleType):
        self.root = root
        self.scale_type = scale_type
        self.notes = self._generate_scale_notes()

    def _generate_scale_notes(self) -> List[int]:
        """Generate notes for the scale."""
        intervals_map = {
            ScaleType.MAJOR: [0, 2, 4, 5, 7, 9, 11],
            ScaleType.MINOR: [0, 2, 3, 5, 7, 8, 10],
            ScaleType.DORIAN: [0, 2, 3, 5, 7, 9, 10],
            ScaleType.PHRYGIAN: [0, 1, 3, 5, 7, 8, 10],
            ScaleType.LYDIAN: [0, 2, 4, 6, 7, 9, 11],
            ScaleType.MIXOLYDIAN: [0, 2, 4, 5, 7, 9, 10],
            ScaleType.AEOLIAN: [0, 2, 3, 5, 7, 8, 10],
            ScaleType.LOCRIAN: [0, 1, 3, 5, 6, 8, 10],
            ScaleType.HARMONIC_MINOR: [0, 2, 3, 5, 7, 8, 11],
            ScaleType.MELODIC_MINOR: [0, 2, 3, 5, 7, 9, 11],
            ScaleType.PENTATONIC_MAJOR: [0, 2, 4, 7, 9],
            ScaleType.PENTATONIC_MINOR: [0, 3, 5, 7, 10],
            ScaleType.BLUES: [0, 3, 5, 6, 7, 10],
        }

        intervals = intervals_map.get(self.scale_type, [0, 2, 4, 5, 7, 9, 11])
        return [(self.root + interval) % 12 for interval in intervals]

    def get_diatonic_chords(self) -> List[Chord]:
        """Get diatonic chords for the scale."""
        chords = []

        # Build triads on each scale degree
        for i, note in enumerate(self.notes[:7]):  # Only use first 7 notes for traditional harmony
            # Get chord tones (1st, 3rd, 5th)
            chord_tones = []
            for j in range(3):
                degree_index = (i + j * 2) % len(self.notes)
                chord_tones.append(self.notes[degree_index])

            # Determine chord quality based on intervals
            third_interval = (chord_tones[1] - chord_tones[0]) % 12
            fifth_interval = (chord_tones[2] - chord_tones[0]) % 12

            if third_interval == 4 and fifth_interval == 7:
                quality = ChordQuality.MAJOR
            elif third_interval == 3 and fifth_interval == 7:
                quality = ChordQuality.MINOR
            elif third_interval == 3 and fifth_interval == 6:
                quality = ChordQuality.DIMINISHED
            elif third_interval == 4 and fifth_interval == 8:
                quality = ChordQuality.AUGMENTED
            else:
                quality = ChordQuality.MAJOR  # Default

            chords.append(Chord(root=note, quality=quality))

        return chords


class MusicTheoryEngine:
    """Advanced music theory analysis and generation engine."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Load chord progression templates
        self.progression_templates = self._load_progression_templates()

        # Initialize harmonic analysis weights
        self.chord_transition_weights = self._initialize_transition_weights()

        # Genre-specific theory rules
        self.genre_rules = self._load_genre_rules()

    def _load_progression_templates(self) -> Dict[str, List[List[str]]]:
        """Load common chord progression templates."""
        return {
            "pop": [
                ["I", "V", "vi", "IV"],  # Very common pop progression
                ["vi", "IV", "I", "V"],  # vi-IV-I-V
                ["I", "vi", "IV", "V"],  # 50s progression
                ["vi", "V", "IV", "V"],  # Alternative
                ["I", "V", "vi", "iii", "IV", "I", "IV", "V"],  # Extended pop
            ],
            "rock": [
                ["i", "VII", "VI", "VII"],  # Minor rock progression
                ["I", "V", "VI", "IV"],  # Major rock progression
                ["i", "v", "VI", "v"],  # Power chord progression
                ["i", "iv", "VII", "VI"],  # Alternative rock
                ["VI", "VII", "i", "i"],  # Aggressive rock
            ],
            "jazz": [
                ["Imaj7", "vi7", "ii7", "V7"],  # Jazz standard
                ["ii7", "V7", "Imaj7", "Imaj7"],  # ii-V-I
                ["Imaj7", "I7", "IVmaj7", "iv7"],  # Jazz with chromatic
                ["vi7", "ii7", "V7", "Imaj7"],  # Circle of fifths
                ["Imaj7", "vi7", "ii7", "V7", "iii7", "vi7", "ii7", "V7"],  # Extended jazz
            ],
            "blues": [
                [
                    "I7",
                    "I7",
                    "I7",
                    "I7",
                    "IV7",
                    "IV7",
                    "I7",
                    "I7",
                    "V7",
                    "IV7",
                    "I7",
                    "V7",
                ],  # 12-bar blues
                ["I7", "IV7", "I7", "V7"],  # Simple blues
                ["i7", "iv7", "i7", "v7"],  # Minor blues
                [
                    "I7",
                    "I7",
                    "I7",
                    "I7",
                    "IV7",
                    "IV7",
                    "I7",
                    "I7",
                    "V7",
                    "V7",
                    "I7",
                    "I7",
                ],  # Traditional 12-bar
            ],
            "folk": [
                ["I", "V", "vi", "IV"],  # Simple folk
                ["vi", "I", "V", "V"],  # Modal folk
                ["I", "IV", "V", "I"],  # Traditional folk
                ["vi", "V", "I", "IV"],  # Alternative folk
            ],
            "electronic": [
                ["vi", "IV", "I", "V"],  # EDM progression
                ["i", "VII", "VI", "VII"],  # Dark electronic
                ["vi", "vi", "IV", "IV"],  # Minimal electronic
                ["I", "V", "vi", "vi"],  # Simple electronic
            ],
        }

    def _initialize_transition_weights(self) -> Dict[Tuple[str, str], float]:
        """Initialize chord transition weights for harmonic analysis."""
        weights = defaultdict(lambda: 0.1)  # Default low weight

        # Strong transitions (high weights)
        strong_transitions = [
            ("I", "V"),
            ("V", "I"),
            ("I", "vi"),
            ("vi", "IV"),
            ("IV", "V"),
            ("V", "vi"),
            ("ii", "V"),
            ("V", "I"),
            ("I", "IV"),
            ("IV", "I"),
            ("vi", "V"),
            ("iii", "vi"),
            ("VII", "I"),
            ("VI", "VII"),
            ("iv", "I"),
        ]

        for transition in strong_transitions:
            weights[transition] = 1.0

        # Medium transitions
        medium_transitions = [
            ("I", "ii"),
            ("I", "iii"),
            ("ii", "iii"),
            ("iii", "IV"),
            ("IV", "ii"),
            ("V", "ii"),
            ("V", "iii"),
            ("vi", "ii"),
            ("vi", "iii"),
            ("VII", "VI"),
            ("VI", "V"),
        ]

        for transition in medium_transitions:
            weights[transition] = 0.6

        return weights

    def _load_genre_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load genre-specific music theory rules."""
        return {
            "pop": {
                "preferred_keys": ["C", "G", "D", "A", "F", "Bb"],
                "common_time_signatures": [(4, 4), (3, 4)],
                "typical_chord_qualities": [ChordQuality.MAJOR, ChordQuality.MINOR],
                "avoid_progressions": [["vii", "I"]],  # Avoid leading tone to tonic in pop
                "tempo_range": (80, 140),
            },
            "rock": {
                "preferred_keys": ["E", "A", "D", "G", "B", "Em", "Am"],
                "common_time_signatures": [(4, 4), (2, 4)],
                "typical_chord_qualities": [
                    ChordQuality.MAJOR,
                    ChordQuality.MINOR,
                    ChordQuality.DOMINANT_SEVENTH,
                ],
                "power_chords": True,
                "tempo_range": (100, 180),
            },
            "jazz": {
                "preferred_keys": ["F", "Bb", "Eb", "Ab", "Db", "C", "G"],
                "common_time_signatures": [(4, 4), (3, 4), (6, 8)],
                "typical_chord_qualities": [
                    ChordQuality.MAJOR_SEVENTH,
                    ChordQuality.MINOR_SEVENTH,
                    ChordQuality.DOMINANT_SEVENTH,
                    ChordQuality.HALF_DIMINISHED,
                ],
                "complex_harmony": True,
                "tempo_range": (60, 200),
            },
            "blues": {
                "preferred_keys": ["E", "A", "B", "G", "C", "F"],
                "common_time_signatures": [(4, 4), (12, 8)],
                "typical_chord_qualities": [ChordQuality.DOMINANT_SEVENTH],
                "use_blue_notes": True,
                "tempo_range": (60, 140),
            },
            "electronic": {
                "preferred_keys": ["Am", "Dm", "Em", "Bm", "C", "F"],
                "common_time_signatures": [(4, 4), (7, 8)],
                "typical_chord_qualities": [ChordQuality.MINOR, ChordQuality.MINOR_SEVENTH],
                "modal_harmony": True,
                "tempo_range": (120, 150),
            },
        }

    def generate_chord_progression(
        self, key: str, genre: str = "pop", length: int = 4, complexity: float = 0.5
    ) -> ChordProgression:
        """
        Generate an intelligent chord progression.

        Args:
            key: Key signature (e.g., "C", "Am", "F#")
            genre: Musical genre for style rules
            length: Number of chords in progression
            complexity: Harmonic complexity (0.0 = simple, 1.0 = complex)

        Returns:
            Generated chord progression
        """
        # Get genre-specific templates
        templates = self.progression_templates.get(genre, self.progression_templates["pop"])
        self.genre_rules.get(genre, self.genre_rules["pop"])

        # Select template based on length and complexity
        suitable_templates = [t for t in templates if len(t) >= length]
        if not suitable_templates:
            suitable_templates = templates

        # Choose template with some randomness
        if complexity > 0.7:
            # Prefer longer, more complex progressions
            template = max(suitable_templates, key=len)[:length]
        else:
            # Choose randomly from suitable templates
            template = np.random.choice(suitable_templates)[:length]

        # Convert Roman numerals to actual chords
        chords = []
        key_root = self._parse_key(key)
        is_minor = "m" in key.lower() or "minor" in key.lower()

        for roman in template:
            chord = self._roman_to_chord(roman, key_root, is_minor, complexity)
            chords.append(chord)

        return ChordProgression(chords=chords, key=key, time_signature=TimeSignature(4, 4))

    def _parse_key(self, key: str) -> int:
        """Parse key string to root note."""
        note_map = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

        key_clean = key.replace("m", "").replace("minor", "").replace("major", "").strip()
        if "#" in key_clean:
            return (note_map[key_clean[0]] + 1) % 12
        elif "b" in key_clean:
            return (note_map[key_clean[0]] - 1) % 12
        else:
            return note_map[key_clean[0]]

    def _roman_to_chord(
        self, roman: str, key_root: int, is_minor: bool, complexity: float
    ) -> Chord:
        """Convert Roman numeral to Chord object."""
        # Parse Roman numeral
        roman_clean = roman.replace("maj7", "").replace("7", "").replace("°", "").replace("+", "")

        # Map Roman numerals to scale degrees
        if is_minor:
            degree_map = {"i": 0, "ii": 2, "III": 3, "iv": 5, "v": 7, "VI": 8, "VII": 10}
        else:
            degree_map = {"I": 0, "ii": 2, "iii": 4, "IV": 5, "V": 7, "vi": 9, "vii": 11}

        # Handle both upper and lower case
        for roman_num, degree in degree_map.items():
            if roman_clean.upper() == roman_num.upper():
                root = (key_root + degree) % 12
                break
        else:
            # Default to tonic if not found
            root = key_root

        # Determine chord quality
        if "maj7" in roman:
            quality = ChordQuality.MAJOR_SEVENTH
        elif "7" in roman:
            if roman_clean.islower():
                quality = (
                    ChordQuality.MINOR_SEVENTH
                    if "ii" in roman or "iii" in roman or "vi" in roman
                    else ChordQuality.DOMINANT_SEVENTH
                )
            else:
                quality = ChordQuality.DOMINANT_SEVENTH
        elif "°" in roman:
            quality = ChordQuality.DIMINISHED
        elif "+" in roman:
            quality = ChordQuality.AUGMENTED
        elif roman_clean.islower():
            quality = ChordQuality.MINOR
        else:
            quality = ChordQuality.MAJOR

        # Add complexity-based extensions
        extensions = []
        if complexity > 0.7:
            # Add 9ths, 11ths for complex harmony
            if np.random.random() > 0.5:
                extensions.append(14)  # 9th
            if np.random.random() > 0.8:
                extensions.append(17)  # 11th

        return Chord(root=root, quality=quality, extensions=extensions)

    def analyze_harmonic_function(self, progression: ChordProgression) -> List[str]:
        """Analyze the harmonic function of each chord in a progression."""
        functions = []
        roman_analysis = progression.roman_numeral_analysis()

        for roman in roman_analysis:
            if roman.upper() in ["I", "III", "VI", "i", "iii", "vi"]:
                functions.append("tonic")
            elif roman.upper() in ["II", "IV", "ii", "iv"]:
                functions.append("subdominant")
            elif roman.upper() in ["V", "VII", "v", "vii"]:
                functions.append("dominant")
            else:
                functions.append("other")

        return functions

    def suggest_next_chord(
        self,
        current_chord: Chord,
        key: str,
        genre: str = "pop",
        context: Optional[List[Chord]] = None,
    ) -> List[Tuple[Chord, float]]:
        """
        Suggest next chords based on music theory and genre rules.

        Returns:
            List of (chord, weight) tuples sorted by preference
        """
        key_root = self._parse_key(key)
        is_minor = "m" in key.lower()

        # Generate possible next chords
        scale = Scale(key_root, ScaleType.MINOR if is_minor else ScaleType.MAJOR)
        diatonic_chords = scale.get_diatonic_chords()

        suggestions = []

        for chord in diatonic_chords:
            # Calculate transition weight
            weight = self._calculate_transition_weight(current_chord, chord, key, genre, context)
            suggestions.append((chord, weight))

        # Sort by weight
        suggestions.sort(key=lambda x: x[1], reverse=True)

        return suggestions[:5]  # Return top 5 suggestions

    def _calculate_transition_weight(
        self,
        from_chord: Chord,
        to_chord: Chord,
        key: str,
        genre: str,
        context: Optional[List[Chord]] = None,
    ) -> float:
        """Calculate the weight for a chord transition."""
        weight = 0.0

        # Voice leading weight
        common_tones = len(set(from_chord.notes) & set(to_chord.notes))
        weight += common_tones * 0.3

        # Root movement weight
        root_interval = (to_chord.root - from_chord.root) % 12
        if root_interval in [5, 7]:  # Fourth/fifth movement
            weight += 0.5
        elif root_interval in [2, 10]:  # Second movement
            weight += 0.3
        elif root_interval in [3, 9]:  # Third movement
            weight += 0.4

        # Genre-specific weights
        rules = self.genre_rules.get(genre, {})
        preferred_qualities = rules.get("typical_chord_qualities", [])
        if to_chord.quality in preferred_qualities:
            weight += 0.3

        # Context-based weight
        if context and len(context) >= 2:
            # Avoid immediate repetition
            if to_chord.root == context[-1].root:
                weight -= 0.4

            # Sequential pattern detection
            if len(context) >= 3:
                recent_roots = [c.root for c in context[-3:]]
                if len(set(recent_roots)) == 1:  # Too much repetition
                    weight -= 0.3

        return max(0.0, weight)

    def modulate_progression(
        self, progression: ChordProgression, target_key: str, modulation_type: str = "pivot"
    ) -> ChordProgression:
        """
        Create a modulated version of a chord progression.

        Args:
            progression: Original progression
            target_key: Target key for modulation
            modulation_type: Type of modulation ("pivot", "chromatic", "direct")

        Returns:
            Modulated chord progression
        """
        if modulation_type == "direct":
            # Direct modulation - simply transpose
            interval = self._parse_key(target_key) - self._parse_key(progression.key)

            new_chords = []
            for chord in progression.chords:
                new_root = (chord.root + interval) % 12
                new_chords.append(
                    Chord(
                        root=new_root,
                        quality=chord.quality,
                        bass=((chord.bass + interval) % 12) if chord.bass else None,
                        extensions=chord.extensions,
                    )
                )

            return ChordProgression(
                chords=new_chords,
                key=target_key,
                time_signature=progression.time_signature,
                chord_durations=progression.chord_durations,
            )

        elif modulation_type == "pivot":
            # Pivot chord modulation
            # Find a chord that exists in both keys
            original_scale = Scale(self._parse_key(progression.key), ScaleType.MAJOR)
            target_scale = Scale(self._parse_key(target_key), ScaleType.MAJOR)

            original_notes = set(original_scale.notes)
            target_notes = set(target_scale.notes)
            common_notes = original_notes & target_notes

            # Create pivot progression
            pivot_chords = []
            for chord in progression.chords:
                if set(chord.notes) & common_notes:
                    # This chord can serve as a pivot
                    pivot_chords.append(chord)
                    break

            # Continue in target key
            target_progression = self.generate_chord_progression(
                target_key, length=len(progression.chords) - len(pivot_chords)
            )

            new_chords = (
                progression.chords[: -len(target_progression.chords)]
                + pivot_chords
                + target_progression.chords
            )

            return ChordProgression(
                chords=new_chords, key=target_key, time_signature=progression.time_signature
            )

        else:
            # For now, default to direct modulation
            return self.modulate_progression(progression, target_key, "direct")

    def analyze_progression_quality(self, progression: ChordProgression) -> Dict[str, Any]:
        """
        Analyze the quality and characteristics of a chord progression.

        Returns:
            Analysis results including harmonic rhythm, voice leading, etc.
        """
        analysis = {
            "length": len(progression.chords),
            "harmonic_functions": self.analyze_harmonic_function(progression),
            "roman_numerals": progression.roman_numeral_analysis(),
            "voice_leading_quality": 0.0,
            "harmonic_rhythm": "even",  # Could be "fast", "slow", "syncopated"
            "complexity_score": 0.0,
            "genre_appropriateness": {},
        }

        # Calculate voice leading quality
        total_movement = 0
        for i in range(len(progression.chords) - 1):
            common_tones = len(
                set(progression.chords[i].notes) & set(progression.chords[i + 1].notes)
            )
            total_movement += 3 - common_tones  # Less movement = better voice leading

        analysis["voice_leading_quality"] = 1.0 - (total_movement / (len(progression.chords) * 3))

        # Calculate complexity score
        unique_qualities = len(set(chord.quality for chord in progression.chords))
        has_extensions = any(chord.extensions for chord in progression.chords)
        analysis["complexity_score"] = (unique_qualities / len(progression.chords)) + (
            0.5 if has_extensions else 0.0
        )

        # Genre appropriateness
        for genre, templates in self.progression_templates.items():
            roman_analysis = analysis["roman_numerals"]
            best_match = 0.0
            for template in templates:
                if len(template) == len(roman_analysis):
                    matches = sum(
                        1
                        for i, r in enumerate(roman_analysis)
                        if i < len(template) and r == template[i]
                    )
                    match_score = matches / len(template)
                    best_match = max(best_match, match_score)
            analysis["genre_appropriateness"][genre] = best_match

        return analysis
