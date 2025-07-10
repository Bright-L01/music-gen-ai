"""
Vocal synthesis module for adding singing capabilities to MusicGen.
Uses various TTS and singing voice synthesis models.
"""

import os
import logging
import tempfile
from typing import Optional, Tuple, Dict, Any
import numpy as np

# Core dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# TTS dependencies
try:
    from TTS.api import TTS
    COQUI_AVAILABLE = True
except ImportError:
    COQUI_AVAILABLE = False

# Audio processing
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

logger = logging.getLogger(__name__)


class VocalSynthesizer:
    """
    Handles vocal synthesis using various TTS models.
    Falls back gracefully when advanced models aren't available.
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize vocal synthesizer.
        
        Args:
            device: Device to use ('cuda' or 'cpu')
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu") if TORCH_AVAILABLE else "cpu"
        self.tts_model = None
        self._initialize_tts()
    
    def _initialize_tts(self):
        """Initialize the best available TTS model."""
        if COQUI_AVAILABLE:
            try:
                # Try to load a singing-capable model
                # Note: In production, we'd use GPT-SoVITS or similar
                # For now, using Coqui TTS as a starting point
                logger.info("Loading Coqui TTS model...")
                
                # List available models
                available_models = TTS.list_models()
                
                # Look for models with good prosody for singing
                # YourTTS supports voice cloning which helps with singing
                if "tts_models/multilingual/multi-dataset/your_tts" in available_models:
                    model_name = "tts_models/multilingual/multi-dataset/your_tts"
                else:
                    # Fallback to a basic model
                    model_name = "tts_models/en/ljspeech/tacotron2-DDC"
                
                self.tts_model = TTS(model_name=model_name, progress_bar=False, gpu=(self.device == "cuda"))
                logger.info(f"✓ Loaded TTS model: {model_name}")
                
            except Exception as e:
                logger.warning(f"Failed to load Coqui TTS: {e}")
                self.tts_model = None
        else:
            logger.warning("Coqui TTS not available. Install with: pip install TTS")
    
    def text_to_speech(
        self, 
        text: str, 
        speaker_wav: Optional[str] = None,
        language: str = "en",
        emotion: Optional[str] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Convert text to speech/singing.
        
        Args:
            text: Text to synthesize
            speaker_wav: Optional reference audio for voice cloning
            language: Language code
            emotion: Emotion hint (not all models support this)
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        if not self.tts_model:
            raise RuntimeError("No TTS model available. Install TTS: pip install TTS")
        
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            # Generate speech
            if speaker_wav and hasattr(self.tts_model, 'tts_with_vc'):
                # Voice cloning for more natural singing
                self.tts_model.tts_with_vc_to_file(
                    text=text,
                    speaker_wav=speaker_wav,
                    file_path=tmp_path,
                    language=language
                )
            else:
                # Standard TTS
                self.tts_model.tts_to_file(
                    text=text,
                    file_path=tmp_path,
                    language=language
                )
            
            # Load the generated audio
            if LIBROSA_AVAILABLE:
                audio, sr = librosa.load(tmp_path, sr=None)
            else:
                # Fallback to pydub
                audio_segment = AudioSegment.from_wav(tmp_path)
                audio = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
                audio = audio / (2**15)  # Convert to [-1, 1]
                sr = audio_segment.frame_rate
            
            # Clean up
            os.unlink(tmp_path)
            
            return audio, sr
            
        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            raise
    
    def apply_singing_effects(
        self, 
        audio: np.ndarray, 
        sample_rate: int,
        pitch_shift: float = 0.0,
        vibrato_rate: float = 5.0,
        vibrato_depth: float = 0.1
    ) -> np.ndarray:
        """
        Apply effects to make speech more singing-like.
        
        Args:
            audio: Input audio array
            sample_rate: Sample rate
            pitch_shift: Semitones to shift pitch
            vibrato_rate: Vibrato frequency in Hz
            vibrato_depth: Vibrato depth (0-1)
            
        Returns:
            Processed audio
        """
        if not LIBROSA_AVAILABLE:
            logger.warning("Librosa not available for singing effects")
            return audio
        
        # Apply pitch shifting
        if pitch_shift != 0:
            audio = librosa.effects.pitch_shift(
                audio, sr=sample_rate, n_steps=pitch_shift
            )
        
        # Simple vibrato effect
        if vibrato_rate > 0 and vibrato_depth > 0:
            # Generate vibrato LFO
            duration = len(audio) / sample_rate
            t = np.linspace(0, duration, len(audio))
            vibrato = vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t)
            
            # Apply as pitch modulation (simplified)
            # In practice, we'd use a proper pitch shifter
            phase_shift = np.cumsum(vibrato) / sample_rate
            audio = audio * np.cos(2 * np.pi * phase_shift)
        
        return audio
    
    def generate_singing(
        self,
        lyrics: str,
        melody_notes: Optional[Dict[str, Any]] = None,
        style: str = "pop",
        speaker_wav: Optional[str] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Generate singing voice from lyrics.
        
        Args:
            lyrics: Song lyrics
            melody_notes: Optional melody information (pitch, duration)
            style: Singing style hint
            speaker_wav: Reference voice for cloning
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        # For now, this is a simplified version
        # In production, we'd use GPT-SoVITS or similar
        
        # Generate base speech
        audio, sr = self.text_to_speech(lyrics, speaker_wav=speaker_wav)
        
        # Apply singing-like effects
        # These are placeholder values - real implementation would be more sophisticated
        if style == "pop":
            audio = self.apply_singing_effects(audio, sr, pitch_shift=0, vibrato_rate=4.5)
        elif style == "rock":
            audio = self.apply_singing_effects(audio, sr, pitch_shift=-2, vibrato_rate=3.0)
        elif style == "jazz":
            audio = self.apply_singing_effects(audio, sr, pitch_shift=0, vibrato_rate=5.5, vibrato_depth=0.15)
        
        return audio, sr


class LyricsProcessor:
    """
    Handles lyrics processing and alignment for vocal synthesis.
    """
    
    @staticmethod
    def parse_lyrics(lyrics: str) -> Dict[str, Any]:
        """
        Parse lyrics into structured format.
        
        Args:
            lyrics: Raw lyrics text
            
        Returns:
            Structured lyrics with timing hints
        """
        lines = lyrics.strip().split('\n')
        
        # Simple structure detection
        structured = {
            'verses': [],
            'chorus': [],
            'bridge': [],
            'raw_lines': lines
        }
        
        current_section = 'verses'
        current_verse = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Detect section markers
            lower_line = line.lower()
            if lower_line.startswith('[') and lower_line.endswith(']'):
                # Section marker like [Chorus], [Verse 1], etc.
                if 'chorus' in lower_line:
                    current_section = 'chorus'
                elif 'bridge' in lower_line:
                    current_section = 'bridge'
                elif 'verse' in lower_line:
                    if current_verse:
                        structured['verses'].append(current_verse)
                        current_verse = []
                    current_section = 'verses'
                continue
            
            # Add line to current section
            if current_section == 'verses':
                current_verse.append(line)
            else:
                structured[current_section].append(line)
        
        # Don't forget the last verse
        if current_verse:
            structured['verses'].append(current_verse)
        
        return structured
    
    @staticmethod
    def align_lyrics_to_beat(
        lyrics: Dict[str, Any], 
        tempo: int = 120,
        time_signature: str = "4/4"
    ) -> Dict[str, Any]:
        """
        Align lyrics to musical beat.
        
        Args:
            lyrics: Structured lyrics
            tempo: BPM
            time_signature: Time signature
            
        Returns:
            Lyrics with timing information
        """
        # Calculate beat duration
        beat_duration = 60.0 / tempo  # seconds per beat
        
        # Simple alignment - one line per measure
        measures_per_line = 2  # Typical for pop music
        line_duration = beat_duration * 4 * measures_per_line  # For 4/4 time
        
        timed_lyrics = lyrics.copy()
        current_time = 0.0
        
        # Add timing to each section
        for section in ['verses', 'chorus', 'bridge']:
            if section == 'verses':
                timed_verses = []
                for verse in lyrics.get(section, []):
                    timed_verse = []
                    for line in verse:
                        timed_verse.append({
                            'text': line,
                            'start_time': current_time,
                            'duration': line_duration
                        })
                        current_time += line_duration
                    timed_verses.append(timed_verse)
                timed_lyrics[section] = timed_verses
            else:
                timed_lines = []
                for line in lyrics.get(section, []):
                    timed_lines.append({
                        'text': line,
                        'start_time': current_time,
                        'duration': line_duration
                    })
                    current_time += line_duration
                timed_lyrics[section] = timed_lines
        
        timed_lyrics['total_duration'] = current_time
        
        return timed_lyrics


# Simplified melody-guided synthesis for immediate use
class SimpleMelodySynthesizer:
    """
    Simple approach to add melody to vocals using pitch shifting.
    This is a placeholder until we integrate advanced models.
    """
    
    @staticmethod
    def apply_melody_contour(
        audio: np.ndarray,
        sample_rate: int,
        melody_pattern: str = "ascending"
    ) -> np.ndarray:
        """
        Apply simple melody patterns to vocals.
        
        Args:
            audio: Vocal audio
            sample_rate: Sample rate
            melody_pattern: Pattern type
            
        Returns:
            Audio with melody applied
        """
        if not LIBROSA_AVAILABLE:
            return audio
        
        # Define simple melody patterns (in semitones)
        patterns = {
            "ascending": [0, 2, 4, 5, 7],
            "descending": [7, 5, 4, 2, 0],
            "arc": [0, 2, 4, 5, 4, 2, 0],
            "pop": [0, 0, 2, 2, 0, -2, 0]
        }
        
        pattern = patterns.get(melody_pattern, patterns["pop"])
        
        # Split audio into segments
        segment_length = int(0.5 * sample_rate)  # 0.5 second segments
        segments = []
        
        for i in range(0, len(audio), segment_length):
            segment = audio[i:i + segment_length]
            if len(segment) < segment_length:
                # Pad last segment
                segment = np.pad(segment, (0, segment_length - len(segment)))
            segments.append(segment)
        
        # Apply pitch pattern
        processed_segments = []
        for i, segment in enumerate(segments):
            pitch_shift = pattern[i % len(pattern)]
            if pitch_shift != 0:
                shifted = librosa.effects.pitch_shift(
                    segment, sr=sample_rate, n_steps=pitch_shift
                )
                processed_segments.append(shifted)
            else:
                processed_segments.append(segment)
        
        # Concatenate
        result = np.concatenate(processed_segments)
        
        # Trim to original length
        return result[:len(audio)]