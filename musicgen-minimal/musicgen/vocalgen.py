"""
VocalGen: Hybrid music generation with vocals.
Combines MusicGen instrumental generation with vocal synthesis.
"""

import logging
import os
import time
from typing import Optional, Tuple, Dict, Any, Callable
import tempfile

import numpy as np

from .generator import MusicGenerator
from .vocal_synthesis import VocalSynthesizer, LyricsProcessor, SimpleMelodySynthesizer
from .audio_enhancement import AudioEnhancer, AudioMixer

logger = logging.getLogger(__name__)


class VocalGenPipeline:
    """
    Main pipeline for generating music with vocals.
    Combines instrumental generation, vocal synthesis, and enhancement.
    """
    
    def __init__(
        self,
        model_name: str = "facebook/musicgen-small",
        device: Optional[str] = None,
        enhance_output: bool = True
    ):
        """
        Initialize VocalGen pipeline.
        
        Args:
            model_name: MusicGen model to use
            device: Device for computation
            enhance_output: Whether to apply audio enhancement
        """
        self.device = device
        self.enhance_output = enhance_output
        
        # Initialize components
        logger.info("Initializing VocalGen components...")
        
        # Instrumental generation
        self.music_generator = MusicGenerator(model_name, device)
        
        # Vocal synthesis
        self.vocal_synthesizer = VocalSynthesizer(device)
        
        # Audio enhancement
        if enhance_output:
            self.audio_enhancer = AudioEnhancer(device)
        else:
            self.audio_enhancer = None
        
        # Processors
        self.lyrics_processor = LyricsProcessor()
        self.melody_synthesizer = SimpleMelodySynthesizer()
        
        logger.info("✓ VocalGen pipeline ready")
    
    def generate_with_vocals(
        self,
        prompt: str,
        lyrics: Optional[str] = None,
        duration: float = 30.0,
        vocal_style: str = "auto",
        mix_style: str = "balanced",
        enhance_level: str = "moderate",
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Generate complete song with vocals.
        
        Args:
            prompt: Music generation prompt
            lyrics: Song lyrics (optional)
            duration: Duration in seconds
            vocal_style: Vocal style (auto, pop, rock, jazz)
            mix_style: Mix style (balanced, vocal_forward, instrumental_forward)
            enhance_level: Enhancement level (light, moderate, heavy)
            progress_callback: Progress callback function
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        start_time = time.time()
        
        # Step 1: Generate instrumental
        if progress_callback:
            progress_callback(0, "Generating instrumental track...")
        
        logger.info(f"Generating instrumental: '{prompt}' for {duration}s")
        
        # Use extended generation for long tracks
        if duration > 30:
            instrumental_audio, inst_sr = self.music_generator.generate_extended(
                prompt=prompt,
                duration=duration,
                progress_callback=lambda current, total, msg: 
                    progress_callback(int(30 * current / total), msg) if progress_callback else None
            )
        else:
            instrumental_audio, inst_sr = self.music_generator.generate(
                prompt=prompt,
                duration=duration
            )
        
        # If no lyrics, just enhance and return instrumental
        if not lyrics:
            if self.enhance_output and self.audio_enhancer:
                if progress_callback:
                    progress_callback(80, "Enhancing audio quality...")
                
                enhanced_audio, sr = self.audio_enhancer.enhance_audio(
                    instrumental_audio, inst_sr, style=vocal_style, enhance_level=enhance_level
                )
                
                if progress_callback:
                    progress_callback(100, "✓ Generation complete!")
                
                return enhanced_audio, sr
            else:
                if progress_callback:
                    progress_callback(100, "✓ Generation complete!")
                return instrumental_audio, inst_sr
        
        # Step 2: Process lyrics
        if progress_callback:
            progress_callback(35, "Processing lyrics...")
        
        structured_lyrics = self.lyrics_processor.parse_lyrics(lyrics)
        
        # Detect style from prompt if auto
        if vocal_style == "auto":
            vocal_style = self._detect_style_from_prompt(prompt)
        
        # Step 3: Generate vocals
        if progress_callback:
            progress_callback(40, "Generating vocals...")
        
        try:
            # Combine all lyrics into one text for now
            # In future, we'll generate section by section
            all_lyrics = []
            for verse in structured_lyrics.get('verses', []):
                all_lyrics.extend(verse)
            all_lyrics.extend(structured_lyrics.get('chorus', []))
            all_lyrics.extend(structured_lyrics.get('bridge', []))
            
            lyrics_text = ' '.join(all_lyrics)
            
            # Generate vocals
            vocal_audio, vocal_sr = self.vocal_synthesizer.generate_singing(
                lyrics=lyrics_text,
                style=vocal_style
            )
            
            # Apply simple melody
            if hasattr(self.melody_synthesizer, 'apply_melody_contour'):
                vocal_audio = self.melody_synthesizer.apply_melody_contour(
                    vocal_audio, vocal_sr, melody_pattern=self._get_melody_pattern(vocal_style)
                )
            
        except Exception as e:
            logger.error(f"Vocal generation failed: {e}")
            logger.warning("Returning instrumental only")
            return instrumental_audio, inst_sr
        
        # Step 4: Mix vocals with instrumental
        if progress_callback:
            progress_callback(70, "Mixing vocals with instrumental...")
        
        # Determine mix parameters
        vocal_level, reverb = self._get_mix_parameters(mix_style, vocal_style)
        
        mixed_audio, mixed_sr = AudioMixer.mix_tracks(
            instrumental=(instrumental_audio, inst_sr),
            vocals=(vocal_audio, vocal_sr),
            vocal_level=vocal_level,
            reverb_amount=reverb,
            style=vocal_style
        )
        
        # Step 5: Enhance final output
        if self.enhance_output and self.audio_enhancer:
            if progress_callback:
                progress_callback(85, "Applying final enhancement...")
            
            # Use stem separation for better enhancement
            if enhance_level == "heavy":
                final_audio, final_sr = self.audio_enhancer.enhance_stems_separately(
                    mixed_audio, mixed_sr, style=vocal_style
                )
            else:
                final_audio, final_sr = self.audio_enhancer.enhance_audio(
                    mixed_audio, mixed_sr, style=vocal_style, enhance_level=enhance_level
                )
        else:
            final_audio, final_sr = mixed_audio, mixed_sr
        
        # Log generation stats
        gen_time = time.time() - start_time
        logger.info(f"✓ Generated {duration}s song with vocals in {gen_time:.1f}s")
        
        if progress_callback:
            progress_callback(100, "✓ Generation complete!")
        
        return final_audio, final_sr
    
    def generate_instrumental_only(
        self,
        prompt: str,
        duration: float = 30.0,
        enhance: bool = True,
        **kwargs
    ) -> Tuple[np.ndarray, int]:
        """
        Generate instrumental music only (no vocals).
        
        Args:
            prompt: Music generation prompt
            duration: Duration in seconds
            enhance: Whether to enhance output
            **kwargs: Additional arguments for generation
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        # Generate
        if duration > 30:
            audio, sr = self.music_generator.generate_extended(prompt, duration, **kwargs)
        else:
            audio, sr = self.music_generator.generate(prompt, duration, **kwargs)
        
        # Enhance if requested
        if enhance and self.enhance_output and self.audio_enhancer:
            audio, sr = self.audio_enhancer.enhance_audio(
                audio, sr, style=self._detect_style_from_prompt(prompt)
            )
        
        return audio, sr
    
    def remix_with_vocals(
        self,
        instrumental_path: str,
        lyrics: str,
        vocal_style: str = "auto",
        output_path: Optional[str] = None
    ) -> str:
        """
        Add vocals to an existing instrumental track.
        
        Args:
            instrumental_path: Path to instrumental audio file
            lyrics: Lyrics to sing
            vocal_style: Vocal style
            output_path: Output file path (optional)
            
        Returns:
            Path to output file
        """
        # Load instrumental
        instrumental_audio, inst_sr = self.music_generator.load_audio(instrumental_path)
        
        # Generate vocals
        vocal_audio, vocal_sr = self.vocal_synthesizer.generate_singing(
            lyrics=lyrics,
            style=vocal_style
        )
        
        # Mix
        mixed_audio, mixed_sr = AudioMixer.mix_tracks(
            instrumental=(instrumental_audio, inst_sr),
            vocals=(vocal_audio, vocal_sr),
            style=vocal_style
        )
        
        # Save
        if not output_path:
            output_path = instrumental_path.replace('.', '_with_vocals.')
        
        self.music_generator.save_audio(mixed_audio, mixed_sr, output_path)
        
        return output_path
    
    def _detect_style_from_prompt(self, prompt: str) -> str:
        """Detect music style from prompt."""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ['jazz', 'swing', 'bebop']):
            return 'jazz'
        elif any(word in prompt_lower for word in ['rock', 'metal', 'punk']):
            return 'rock'
        elif any(word in prompt_lower for word in ['electronic', 'techno', 'house', 'edm']):
            return 'electronic'
        elif any(word in prompt_lower for word in ['classical', 'orchestra', 'symphony']):
            return 'classical'
        else:
            return 'pop'  # Default
    
    def _get_melody_pattern(self, style: str) -> str:
        """Get appropriate melody pattern for style."""
        patterns = {
            'pop': 'pop',
            'rock': 'ascending',
            'jazz': 'arc',
            'electronic': 'ascending',
            'classical': 'arc'
        }
        return patterns.get(style, 'pop')
    
    def _get_mix_parameters(self, mix_style: str, vocal_style: str) -> Tuple[float, float]:
        """Get mixing parameters based on style."""
        # Base parameters
        if mix_style == "vocal_forward":
            vocal_level = 0.9
            reverb = 0.15
        elif mix_style == "instrumental_forward":
            vocal_level = 0.6
            reverb = 0.25
        else:  # balanced
            vocal_level = 0.75
            reverb = 0.2
        
        # Adjust for vocal style
        if vocal_style == "rock":
            vocal_level *= 1.1  # Rock vocals typically more prominent
        elif vocal_style == "electronic":
            reverb *= 1.3  # More spacious vocals in electronic
        
        return vocal_level, reverb


def quick_generate_with_vocals(
    prompt: str,
    output_file: str,
    lyrics: Optional[str] = None,
    duration: float = 30.0,
    model_name: str = "facebook/musicgen-small",
    enhance: bool = True
) -> str:
    """
    Quick function to generate music with vocals.
    
    Args:
        prompt: Music description
        output_file: Output filename
        lyrics: Song lyrics (optional)
        duration: Duration in seconds
        model_name: MusicGen model name
        enhance: Whether to enhance output
        
    Returns:
        Path to generated file
    """
    # Initialize pipeline
    pipeline = VocalGenPipeline(model_name, enhance_output=enhance)
    
    # Generate
    audio, sample_rate = pipeline.generate_with_vocals(
        prompt=prompt,
        lyrics=lyrics,
        duration=duration
    )
    
    # Save
    # Determine format from filename
    if output_file.lower().endswith('.mp3'):
        output_path = pipeline.music_generator.save_audio_as_format(
            audio, sample_rate, output_file, format="mp3"
        )
    else:
        pipeline.music_generator.save_audio(audio, sample_rate, output_file)
        output_path = output_file
    
    return output_path