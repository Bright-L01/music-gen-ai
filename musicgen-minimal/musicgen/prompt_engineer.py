"""
Prompt Engineering Assistant for MusicGen.
Helps users craft better prompts based on what actually works.
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class PromptTemplate:
    """Template for a specific genre or style."""

    name: str
    description: str
    template: str
    example: str
    tips: List[str]


class PromptEngineer:
    """
    Helps users create effective prompts for MusicGen.
    Based on real-world testing and user feedback.
    """

    def __init__(self):
        self.templates = self._load_templates()
        self.instruments = self._load_instruments()
        self.modifiers = self._load_modifiers()
        self.avoid_words = self._load_avoid_words()

    def _load_templates(self) -> Dict[str, PromptTemplate]:
        """Load genre-specific templates that actually work."""
        return {
            "jazz": PromptTemplate(
                name="Jazz",
                description="Smooth jazz, bebop, or jazz fusion",
                template="{mood} jazz {subgenre} with {instruments}, {tempo} tempo",
                example="smooth jazz fusion with saxophone and piano, medium tempo",
                tips=[
                    "Mention specific instruments (sax, piano, upright bass)",
                    "Include tempo (slow, medium, upbeat)",
                    "Add mood descriptors (smooth, energetic, melancholic)",
                ],
            ),
            "electronic": PromptTemplate(
                name="Electronic/EDM",
                description="Electronic dance music, techno, house",
                template="{energy} {subgenre} electronic track with {elements}, {bpm} bpm",
                example="energetic progressive house electronic track with heavy bass and synth leads, 128 bpm",
                tips=[
                    "Specify BPM for dance music (120-140 common)",
                    "Mention key elements (bass, synths, pads)",
                    "Use energy descriptors (pumping, ambient, aggressive)",
                ],
            ),
            "classical": PromptTemplate(
                name="Classical",
                description="Orchestral, chamber music, or solo instruments",
                template="{period} classical {ensemble} piece, {mood} and {tempo}",
                example="romantic classical string quartet piece, melancholic and slow",
                tips=[
                    "Specify ensemble size (solo, quartet, orchestra)",
                    "Mention period (baroque, classical, romantic)",
                    "Describe mood and dynamics",
                ],
            ),
            "rock": PromptTemplate(
                name="Rock",
                description="Rock, metal, punk, or alternative",
                template="{intensity} {subgenre} rock with {instruments}, {tempo} and {mood}",
                example="heavy metal rock with distorted guitars and pounding drums, fast and aggressive",
                tips=[
                    "Specify guitar tone (distorted, clean, acoustic)",
                    "Mention drum style (pounding, steady, complex)",
                    "Include energy level (mellow, energetic, aggressive)",
                ],
            ),
            "ambient": PromptTemplate(
                name="Ambient",
                description="Atmospheric, meditative, or background music",
                template="{texture} ambient soundscape with {elements}, {mood} atmosphere",
                example="ethereal ambient soundscape with pad synths and nature sounds, peaceful atmosphere",
                tips=[
                    "Focus on texture (ethereal, dark, warm)",
                    "Mention atmospheric elements",
                    "Avoid requesting specific melodies",
                ],
            ),
            "folk": PromptTemplate(
                name="Folk/Country",
                description="Folk, country, bluegrass, or acoustic",
                template="{style} {genre} song with {instruments}, {mood} and {tempo}",
                example="traditional folk song with acoustic guitar and harmonica, nostalgic and moderate tempo",
                tips=[
                    "Specify acoustic instruments",
                    "Mention regional style if relevant",
                    "Include emotional tone",
                ],
            ),
            "hiphop": PromptTemplate(
                name="Hip Hop/Trap",
                description="Hip hop beats, trap, or lo-fi hip hop",
                template="{style} hip hop beat with {elements}, {tempo} bpm, {mood} vibe",
                example="lo-fi hip hop beat with jazzy samples and vinyl crackle, 70 bpm, chill vibe",
                tips=[
                    "Specify beat style (boom bap, trap, lo-fi)",
                    "Mention BPM (60-80 for lo-fi, 140-170 for trap)",
                    "Include vibe/mood descriptors",
                ],
            ),
            "world": PromptTemplate(
                name="World Music",
                description="Music from specific cultures or regions",
                template="{region} {style} music with {instruments}, {mood} and {tempo}",
                example="middle eastern traditional music with oud and percussion, mystical and moderate tempo",
                tips=[
                    "Be respectful and specific about cultural styles",
                    "Mention traditional instruments if known",
                    "Avoid stereotypes",
                ],
            ),
        }

    def _load_instruments(self) -> Dict[str, List[str]]:
        """Common instruments by category that MusicGen handles well."""
        return {
            "strings": [
                "violin",
                "cello",
                "guitar",
                "bass",
                "harp",
                "acoustic guitar",
                "electric guitar",
            ],
            "wind": ["saxophone", "trumpet", "flute", "clarinet", "harmonica"],
            "keys": ["piano", "synthesizer", "organ", "electric piano", "harpsichord"],
            "percussion": [
                "drums",
                "percussion",
                "timpani",
                "tambourine",
                "shaker",
                "808",
            ],
            "electronic": [
                "synth lead",
                "synth bass",
                "pad",
                "arpeggiator",
                "808 bass",
                "samples",
                "vinyl",
            ],
        }

    def _load_modifiers(self) -> Dict[str, List[str]]:
        """Effective modifier words grouped by category."""
        return {
            "energy": [
                "energetic",
                "calm",
                "intense",
                "relaxed",
                "aggressive",
                "peaceful",
            ],
            "mood": [
                "happy",
                "sad",
                "melancholic",
                "uplifting",
                "dark",
                "bright",
                "mysterious",
            ],
            "tempo": ["slow", "moderate", "fast", "upbeat", "downtempo", "mid-tempo"],
            "texture": [
                "smooth",
                "rough",
                "ethereal",
                "dense",
                "sparse",
                "rich",
                "minimal",
            ],
        }

    def _load_avoid_words(self) -> List[str]:
        """Words that often lead to poor results with MusicGen."""
        return [
            "vocal",
            "vocals",
            "singing",
            "voice",
            "lyrics",  # MusicGen is instrumental only
            "epic",
            "cinematic",
            "hollywood",  # Too vague, often disappoints
            "masterpiece",
            "professional",
            "high quality",  # Quality descriptors don't help
            "complex",
            "intricate",
            "sophisticated",  # Often leads to messy output
            "copyright",
            "style of",
            "sounds like",  # Legal/ethical issues
            "minute",
            "minutes",
            "seconds",  # Duration in prompt doesn't work
        ]

    def validate_prompt(self, prompt: str) -> Tuple[bool, List[str]]:
        """
        Validate a prompt and return issues found.

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        prompt_lower = prompt.lower()

        # Check for words to avoid
        for word in self.avoid_words:
            if word in prompt_lower:
                if word in ["vocal", "vocals", "singing", "voice", "lyrics"]:
                    issues.append(
                        f"Remove '{word}' - MusicGen only generates instrumentals"
                    )
                elif word in ["epic", "cinematic", "hollywood"]:
                    issues.append(
                        f"'{word}' is too vague - be more specific about instruments and mood"
                    )
                elif word in ["masterpiece", "professional", "high quality"]:
                    issues.append(
                        f"Remove '{word}' - quality descriptors don't improve output"
                    )
                else:
                    issues.append(f"Avoid using '{word}'")

        # Check length
        if len(prompt) < 10:
            issues.append(
                "Prompt too short - add more detail about style, instruments, and mood"
            )
        elif len(prompt) > 200:
            issues.append(
                "Prompt too long - MusicGen works better with concise descriptions"
            )

        # Check for key elements
        has_genre = any(
            genre in prompt_lower
            for genre in [
                "jazz",
                "rock",
                "electronic",
                "classical",
                "folk",
                "ambient",
                "hip hop",
                "trap",
            ]
        )
        has_instrument = any(
            inst in prompt_lower
            for instruments in self.instruments.values()
            for inst in instruments
        )
        has_descriptor = any(
            mod in prompt_lower
            for modifiers in self.modifiers.values()
            for mod in modifiers
        )

        # Also check for tempo-related words that aren't in our modifiers list
        tempo_words = ["bpm", "tempo", "groove", "rhythm", "beat"]
        has_tempo_context = any(word in prompt_lower for word in tempo_words)

        if not has_genre:
            issues.append("Consider adding a genre (jazz, rock, electronic, etc.)")
        if not has_instrument:
            issues.append("Mention specific instruments for better results")
        if not has_descriptor and not has_tempo_context:
            issues.append("Add descriptors for mood, energy, or tempo")

        return len(issues) == 0, issues

    def improve_prompt(self, prompt: str) -> str:
        """
        Automatically improve a prompt based on best practices.

        Args:
            prompt: Original user prompt

        Returns:
            Improved prompt
        """
        improved = prompt

        # Remove problematic words
        for word in self.avoid_words:
            pattern = r"\b" + re.escape(word) + r"\b"
            improved = re.sub(pattern, "", improved, flags=re.IGNORECASE)

        # Clean up extra spaces
        improved = " ".join(improved.split())

        # Add common improvements
        prompt_lower = improved.lower()

        # If no tempo mentioned, add one
        if not any(
            tempo in prompt_lower
            for tempo in ["slow", "fast", "moderate", "tempo", "bpm"]
        ):
            improved += ", moderate tempo"

        # If very short, add generic but helpful additions
        if len(improved) < 30:
            if "beat" in prompt_lower or "track" in prompt_lower:
                improved += " with clear rhythm and bass"
            else:
                improved += " with rich instrumentation"

        return improved.strip()

    def suggest_variations(self, prompt: str, count: int = 3) -> List[str]:
        """
        Generate variations of a prompt for A/B testing.

        Args:
            prompt: Base prompt
            count: Number of variations to generate

        Returns:
            List of prompt variations
        """
        variations = []
        base_prompt = self.improve_prompt(prompt)

        # Variation 1: Add energy descriptor
        if not any(e in base_prompt.lower() for e in self.modifiers["energy"]):
            variations.append(f"energetic {base_prompt}")

        # Variation 2: Add mood descriptor
        if not any(m in base_prompt.lower() for m in self.modifiers["mood"]):
            variations.append(f"{base_prompt}, uplifting mood")

        # Variation 3: Add specific instrument
        if "guitar" not in base_prompt.lower() and "piano" not in base_prompt.lower():
            if "electronic" in base_prompt.lower() or "edm" in base_prompt.lower():
                variations.append(f"{base_prompt} with prominent synth lead")
            else:
                variations.append(f"{base_prompt} with piano accompaniment")

        # Ensure we have enough variations
        while len(variations) < count:
            variations.append(base_prompt)

        return variations[:count]

    def interactive_build(self) -> str:
        """
        Interactive prompt builder (to be called from CLI).
        Returns a crafted prompt.
        """
        # This will be implemented in the CLI
        # For now, return a placeholder
        return ""

    def get_random_example(self, genre: Optional[str] = None) -> str:
        """
        Get a random example prompt that works well.

        Args:
            genre: Specific genre, or None for any

        Returns:
            Example prompt
        """
        import random

        # Genre-specific examples
        genre_examples = {
            "jazz": [
                "smooth jazz with saxophone and piano, medium tempo, relaxed mood",
                "bebop jazz with fast trumpet and upright bass, energetic tempo",
                "jazz fusion with electric guitar and synthesizer, funky groove",
            ],
            "electronic": [
                "energetic electronic dance track with heavy bass and synth leads, 128 bpm",
                "ambient electronic with ethereal pads and glitchy percussion, 90 bpm",
                "progressive house with driving bassline and uplifting synths, 124 bpm",
            ],
            "classical": [
                "melancholic classical string quartet, slow tempo with rich harmonies",
                "baroque harpsichord concerto, ornate melodies and counterpoint",
                "romantic era piano sonata, expressive dynamics and rubato",
            ],
            "rock": [
                "aggressive metal with distorted guitars and fast double-bass drums",
                "classic rock with blues guitar solo and steady drum beat",
                "indie rock with jangly guitars and melodic bass, moderate tempo",
            ],
            "hiphop": [
                "lo-fi hip hop beat with jazzy piano samples, 70 bpm, chill vibe",
                "trap beat with 808 bass and hi-hat rolls, 140 bpm, dark mood",
                "boom bap hip hop with vinyl crackle and soul samples, 90 bpm",
            ],
        }

        # All examples combined
        all_examples = []
        for examples_list in genre_examples.values():
            all_examples.extend(examples_list)

        # Add some additional varied examples
        all_examples.extend(
            [
                "ethereal ambient soundscape with pad synths and subtle percussion",
                "upbeat folk song with acoustic guitar and harmonica, cheerful mood",
                "mysterious middle eastern music with oud and hand percussion",
                "funky disco with bass guitar and brass section, danceable groove",
                "dramatic orchestral piece with strings and timpani, building intensity",
            ]
        )

        if genre and genre.lower() in genre_examples:
            # Return genre-specific example
            return random.choice(genre_examples[genre.lower()])
        elif genre:
            # Try to find any example containing the genre word
            matching = [e for e in all_examples if genre.lower() in e.lower()]
            if matching:
                return random.choice(matching)

        # Return any random example
        return random.choice(all_examples)

    def analyze_prompt(self, prompt: str) -> Dict[str, any]:
        """
        Analyze a prompt and return detailed information.

        Returns:
            Dictionary with analysis results
        """
        analysis = {
            "original": prompt,
            "improved": self.improve_prompt(prompt),
            "length": len(prompt),
            "detected_genre": None,
            "detected_instruments": [],
            "detected_modifiers": [],
            "issues": [],
        }

        prompt_lower = prompt.lower()

        # Detect genre
        for genre in self.templates.keys():
            if genre in prompt_lower:
                analysis["detected_genre"] = genre
                break

        # Detect instruments
        for category, instruments in self.instruments.items():
            for instrument in instruments:
                if instrument in prompt_lower:
                    analysis["detected_instruments"].append(instrument)

        # Detect modifiers
        for category, modifiers in self.modifiers.items():
            for modifier in modifiers:
                if modifier in prompt_lower:
                    analysis["detected_modifiers"].append(f"{category}: {modifier}")

        # Validate
        is_valid, issues = self.validate_prompt(prompt)
        analysis["issues"] = issues
        analysis["is_valid"] = is_valid

        return analysis
