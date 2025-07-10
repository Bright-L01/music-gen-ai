"""
Core music generation functionality.
Dead simple wrapper around Facebook's MusicGen model.
"""

import logging
import os
import time
from typing import Tuple, Optional, Callable, Union

import numpy as np
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration, MusicgenMelodyForConditionalGeneration
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    import scipy.io.wavfile as wavfile
    SOUNDFILE_AVAILABLE = False

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

logger = logging.getLogger(__name__)


class GenerationProgress:
    """Track generation progress with time estimates."""
    
    def __init__(self, total_duration: float, total_steps: Optional[int] = None):
        """
        Initialize progress tracker.
        
        Args:
            total_duration: Total duration in seconds to generate
            total_steps: Total number of steps (tokens) to generate
        """
        self.total_duration = total_duration
        self.total_steps = total_steps or int(256 * total_duration / 5)  # Estimate
        self.start_time = time.time()
        self.current_step = 0
        self.current_duration = 0.0
        
    def update(self, step: int = None, duration: float = None):
        """Update progress with current step or duration."""
        if step is not None:
            self.current_step = step
        if duration is not None:
            self.current_duration = duration
            
    def get_progress_percent(self) -> float:
        """Get progress as percentage (0-100)."""
        if self.total_steps > 0:
            return min(100, (self.current_step / self.total_steps) * 100)
        elif self.total_duration > 0:
            return min(100, (self.current_duration / self.total_duration) * 100)
        return 0
        
    def estimate_time_remaining(self) -> Optional[float]:
        """Estimate remaining time based on current progress."""
        elapsed = time.time() - self.start_time
        progress_percent = self.get_progress_percent()
        
        if progress_percent > 0:
            total_estimated = elapsed * (100 / progress_percent)
            remaining = total_estimated - elapsed
            return max(0, remaining)
        return None
        
    def get_eta_string(self) -> str:
        """Get formatted ETA string."""
        remaining = self.estimate_time_remaining()
        if remaining is None:
            return "Calculating..."
        
        if remaining < 60:
            return f"{int(remaining)}s"
        elif remaining < 3600:
            return f"{int(remaining / 60)}m {int(remaining % 60)}s"
        else:
            hours = int(remaining / 3600)
            minutes = int((remaining % 3600) / 60)
            return f"{hours}h {minutes}m"


class MusicGenerator:
    """
    Dead simple music generation that actually works.
    
    No complex abstractions, no microservices, no enterprise patterns.
    Just a thin wrapper around the working MusicGen model.
    """
    
    def __init__(self, model_name: str = "facebook/musicgen-small", device: Optional[str] = None):
        """
        Initialize the generator.
        
        Args:
            model_name: HuggingFace model name. Options:
                - facebook/musicgen-small (default, fastest)
                - facebook/musicgen-medium (better quality, slower)
                - facebook/musicgen-large (best quality, slowest)
                - facebook/musicgen-melody (melody-guided generation)
            device: Device to use. Auto-detects if None.
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.is_melody_model = "melody" in model_name.lower()
        
        logger.info(f"Loading {model_name} on {self.device}...")
        self._load_model()
        logger.info("✓ Model loaded successfully")
        
    def _load_model(self):
        """Load the HuggingFace models with robust error handling."""
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            if self.is_melody_model:
                self.model = MusicgenMelodyForConditionalGeneration.from_pretrained(self.model_name)
            else:
                self.model = MusicgenForConditionalGeneration.from_pretrained(self.model_name)
            self.model.to(self.device)
        except Exception as e:
            error_msg = f"Failed to load model '{self.model_name}': {e}"
            
            # Provide helpful suggestions based on common errors
            if "ConnectTimeout" in str(e) or "ConnectionError" in str(e):
                error_msg += "\n\nSuggestion: Check your internet connection. Model downloads require internet access."
            elif "OutOfMemoryError" in str(e) or "CUDA out of memory" in str(e):
                error_msg += f"\n\nSuggestion: Try using CPU instead: device='cpu', or use a smaller model like 'facebook/musicgen-small'."
            elif "does not appear to have a file named" in str(e):
                error_msg += f"\n\nSuggestion: Verify the model name. Available models: facebook/musicgen-small, facebook/musicgen-medium, facebook/musicgen-large"
            
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
    def generate(
        self, 
        prompt: str, 
        duration: float = 10.0, 
        temperature: float = 1.0, 
        guidance_scale: float = 3.0,
        max_new_tokens: Optional[int] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Generate music from text prompt.
        
        Args:
            prompt: Text description of the music to generate
            duration: Duration in seconds (0.1 to 120.0)
            temperature: Sampling temperature (0.1-2.0, higher = more random)
            guidance_scale: Classifier-free guidance (1.0-10.0, higher = follows prompt better)
            max_new_tokens: Override automatic token calculation
            progress_callback: Optional callback function(percent, message)
            
        Returns:
            Tuple of (audio_array, sample_rate)
            
        Raises:
            ValueError: If parameters are out of valid ranges
            RuntimeError: If generation fails
        """
        # Input validation
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        if not (0.1 <= duration <= 120.0):
            raise ValueError(f"Duration must be between 0.1 and 120.0 seconds, got {duration}")
        
        if not (0.1 <= temperature <= 2.0):
            raise ValueError(f"Temperature must be between 0.1 and 2.0, got {temperature}")
        
        if not (1.0 <= guidance_scale <= 10.0):
            raise ValueError(f"Guidance scale must be between 1.0 and 10.0, got {guidance_scale}")
        
        logger.info(f"Generating: '{prompt}' for {duration}s")
        start_time = time.time()
        
        # Progress tracking
        if progress_callback:
            progress_callback(0, "Processing text prompt...")
        
        # Process the text prompt
        inputs = self.processor(
            text=[prompt],
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Calculate tokens needed (roughly 256 tokens = 5 seconds)
        if max_new_tokens is None:
            max_new_tokens = int(256 * duration / 5)
        
        if progress_callback:
            progress_callback(10, f"Generating {duration}s of audio...")
        
        # Generate audio
        # Note: HuggingFace generate doesn't provide step callbacks, so we estimate
        with torch.no_grad():
            audio_values = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                guidance_scale=guidance_scale
            )
        
        if progress_callback:
            progress_callback(90, "Processing audio output...")
        
        # Extract audio array
        audio = audio_values[0, 0].cpu().numpy()
        sample_rate = self.model.config.audio_encoder.sampling_rate
        
        generation_time = time.time() - start_time
        actual_duration = len(audio) / sample_rate
        
        if progress_callback:
            progress_callback(100, "Complete!")
        
        logger.info(
            f"✓ Generated {actual_duration:.1f}s in {generation_time:.1f}s "
            f"({actual_duration/generation_time:.2f}x realtime)"
        )
        
        return audio, sample_rate
    
    def save_audio(self, audio: np.ndarray, sample_rate: int, filename: str):
        """
        Save audio array to WAV file.
        
        Args:
            audio: Audio array from generate()
            sample_rate: Sample rate from generate()
            filename: Output filename (must be .wav)
        """
        # Ensure we're saving as WAV (soundfile doesn't support MP3)
        if filename.lower().endswith('.mp3'):
            logger.warning(f"save_audio only supports WAV format. Use save_audio_as_format for MP3.")
            # Force WAV extension
            filename = filename.rsplit('.', 1)[0] + '.wav'
            logger.info(f"Saving as: {filename}")
        
        # Normalize audio to prevent clipping
        audio = np.clip(audio, -1, 1)
        
        try:
            if SOUNDFILE_AVAILABLE:
                # Use soundfile for better quality and error handling
                sf.write(filename, audio, sample_rate, subtype='PCM_16')
            else:
                # Fallback to scipy
                audio_16bit = (audio * 32767).astype(np.int16)
                wavfile.write(filename, sample_rate, audio_16bit)
        except Exception as e:
            logger.error(f"Failed to save audio to {filename}: {e}")
            raise RuntimeError(f"Audio save failed: {e}")
        
        # Calculate file info
        file_size = len(audio) * 2 / 1024  # Approximate KB for 16-bit
        duration = len(audio) / sample_rate
        
        logger.info(f"✓ Saved {duration:.1f}s audio to {filename} ({file_size:.1f} KB)")
    
    def convert_to_mp3(self, wav_path: str, bitrate: str = "192k", delete_wav: bool = False) -> Tuple[str, bool]:
        """
        Convert WAV file to MP3 using pydub/ffmpeg.
        
        Args:
            wav_path: Path to WAV file to convert
            bitrate: MP3 bitrate (e.g., "128k", "192k", "320k")
            delete_wav: Whether to delete the original WAV file after conversion
            
        Returns:
            Tuple of (output_path, success_bool)
            - output_path: Path to MP3 file if successful, otherwise WAV path
            - success: True if MP3 conversion succeeded, False otherwise
            
        Raises:
            Warning: If pydub is not available, returns original path
        """
        if not PYDUB_AVAILABLE:
            logger.warning(
                "pydub not installed. Install with: pip install 'musicgen-minimal[audio]' or pip install pydub\n"
                "Also requires ffmpeg: https://ffmpeg.org/download.html"
            )
            return wav_path, False
        
        if not os.path.exists(wav_path):
            logger.error(f"WAV file not found: {wav_path}")
            return wav_path, False
        
        try:
            # Create MP3 path
            mp3_path = wav_path.rsplit('.', 1)[0] + '.mp3'
            
            logger.info(f"Converting {wav_path} to MP3...")
            
            # Load WAV file
            audio = AudioSegment.from_wav(wav_path)
            
            # Export as MP3 with specified bitrate
            audio.export(mp3_path, format="mp3", bitrate=bitrate)
            
            # Calculate file sizes
            wav_size = os.path.getsize(wav_path) / (1024 * 1024)  # MB
            mp3_size = os.path.getsize(mp3_path) / (1024 * 1024)  # MB
            compression_ratio = (1 - mp3_size / wav_size) * 100
            
            logger.info(f"✓ MP3 conversion complete: {mp3_path}")
            logger.info(f"  WAV: {wav_size:.1f} MB → MP3: {mp3_size:.1f} MB ({compression_ratio:.1f}% smaller)")
            
            # Optionally remove WAV file
            if delete_wav or os.environ.get('MUSICGEN_DELETE_WAV', 'false').lower() == 'true':
                try:
                    os.remove(wav_path)
                    logger.info(f"✓ Removed original WAV file: {wav_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove WAV file: {e}")
            
            return mp3_path, True
            
        except ImportError as e:
            logger.error(f"Missing dependency for MP3 conversion: {e}")
            logger.error("⚠️  ffmpeg is required but not found!")
            logger.error("Install ffmpeg:")
            logger.error("  macOS: brew install ffmpeg")
            logger.error("  Ubuntu: sudo apt install ffmpeg")
            logger.error("  Windows: Download from https://ffmpeg.org/download.html")
            return wav_path, False
        except Exception as e:
            logger.error(f"MP3 conversion failed: {e}")
            if "ffmpeg" in str(e).lower() or "avconv" in str(e).lower():
                logger.error("⚠️  ffmpeg is required but not found!")
                logger.error("Install ffmpeg:")
                logger.error("  macOS: brew install ffmpeg")
                logger.error("  Ubuntu: sudo apt install ffmpeg")
                logger.error("  Windows: Download from https://ffmpeg.org/download.html")
            return wav_path, False
    
    def save_audio_as_format(
        self, 
        audio: np.ndarray, 
        sample_rate: int, 
        filename: str, 
        format: str = "wav",
        bitrate: str = "192k",
        delete_wav: bool = False
    ) -> str:
        """
        Save audio in the specified format (WAV or MP3).
        
        Args:
            audio: Audio array from generate()
            sample_rate: Sample rate from generate()
            filename: Output filename (extension will be adjusted for format)
            format: Output format ("wav" or "mp3")
            bitrate: MP3 bitrate if format is "mp3"
            delete_wav: Whether to delete WAV file after MP3 conversion
            
        Returns:
            Path to saved file
        """
        # Validate format
        if format not in ["wav", "mp3"]:
            logger.warning(f"Unsupported format '{format}', using WAV")
            format = "wav"
        
        # Adjust filename extension
        base_name = filename.rsplit('.', 1)[0]
        
        if format == "wav":
            wav_filename = f"{base_name}.wav"
            self.save_audio(audio, sample_rate, wav_filename)
            return wav_filename
        
        elif format == "mp3":
            # Always save as WAV first, then convert
            wav_filename = f"{base_name}.wav"
            self.save_audio(audio, sample_rate, wav_filename)
            
            # Convert to MP3
            output_path, success = self.convert_to_mp3(wav_filename, bitrate=bitrate, delete_wav=delete_wav)
            if not success and output_path.endswith('.wav'):
                # Make it clear that MP3 conversion failed
                logger.warning("⚠️  MP3 conversion failed - saved as WAV instead")
            return output_path
    
    def _crossfade_audio(self, audio1: np.ndarray, audio2: np.ndarray, overlap_samples: int) -> np.ndarray:
        """
        Apply crossfade between two audio segments.
        
        Args:
            audio1: First audio segment
            audio2: Second audio segment  
            overlap_samples: Number of samples to crossfade
            
        Returns:
            Blended audio with crossfade applied
        """
        if overlap_samples <= 0:
            return np.concatenate([audio1, audio2])
        
        # Ensure we don't try to crossfade more than available audio
        overlap_samples = min(overlap_samples, len(audio1), len(audio2))
        
        # Create fade curves (linear fade for simplicity)
        fade_out = np.linspace(1.0, 0.0, overlap_samples)
        fade_in = np.linspace(0.0, 1.0, overlap_samples)
        
        # Apply crossfade
        audio1_fade = audio1.copy()
        audio2_fade = audio2.copy()
        
        # Fade out the end of audio1
        audio1_fade[-overlap_samples:] *= fade_out
        
        # Fade in the beginning of audio2
        audio2_fade[:overlap_samples] *= fade_in
        
        # Blend the overlapping region
        blended_region = audio1_fade[-overlap_samples:] + audio2_fade[:overlap_samples]
        
        # Concatenate: audio1 (minus overlap) + blended_region + audio2 (minus overlap)
        result = np.concatenate([
            audio1_fade[:-overlap_samples],
            blended_region,
            audio2_fade[overlap_samples:]
        ])
        
        return result
    
    def load_audio(self, audio_path: str, duration: Optional[float] = None) -> Tuple[np.ndarray, int]:
        """
        Load audio file for melody conditioning.
        
        Args:
            audio_path: Path to audio file (WAV, MP3, etc.)
            duration: Maximum duration in seconds to load (None = load all)
            
        Returns:
            Tuple of (audio_array, sample_rate)
            
        Raises:
            RuntimeError: If audio loading fails
        """
        try:
            # Try librosa first for better format support
            if LIBROSA_AVAILABLE:
                audio, sr = librosa.load(audio_path, sr=None, mono=True, duration=duration)
                logger.info(f"✓ Loaded audio: {audio_path} ({sr}Hz, {len(audio)/sr:.1f}s)")
                return audio, sr
            
            # Fallback to soundfile
            elif SOUNDFILE_AVAILABLE:
                audio, sr = sf.read(audio_path)
                if len(audio.shape) > 1:  # Convert to mono if stereo
                    audio = audio.mean(axis=1)
                if duration is not None:
                    max_samples = int(duration * sr)
                    audio = audio[:max_samples]
                logger.info(f"✓ Loaded audio: {audio_path} ({sr}Hz, {len(audio)/sr:.1f}s)")
                return audio, sr
            
            # Last resort: pydub
            elif PYDUB_AVAILABLE:
                audio_segment = AudioSegment.from_file(audio_path)
                if duration is not None:
                    audio_segment = audio_segment[:int(duration * 1000)]  # ms
                
                # Convert to numpy array
                samples = np.array(audio_segment.get_array_of_samples())
                
                # Convert to float32 and normalize
                if audio_segment.sample_width == 2:
                    audio = samples.astype(np.float32) / 32768.0
                elif audio_segment.sample_width == 1:
                    audio = (samples.astype(np.float32) - 128) / 128.0
                else:
                    audio = samples.astype(np.float32)
                
                # Convert stereo to mono
                if audio_segment.channels == 2:
                    audio = audio.reshape((-1, 2)).mean(axis=1)
                
                sr = audio_segment.frame_rate
                logger.info(f"✓ Loaded audio: {audio_path} ({sr}Hz, {len(audio)/sr:.1f}s)")
                return audio, sr
            
            else:
                raise RuntimeError(
                    "No audio loading library available. Install librosa, soundfile, or pydub:\n"
                    "pip install librosa soundfile pydub"
                )
                
        except Exception as e:
            error_msg = f"Failed to load audio file '{audio_path}': {e}"
            if not os.path.exists(audio_path):
                error_msg += f"\nFile does not exist: {audio_path}"
            elif "librosa" not in str(e).lower() and not LIBROSA_AVAILABLE:
                error_msg += "\n\nTip: Install librosa for better audio format support: pip install librosa"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def generate_with_melody(
        self,
        prompt: str,
        melody_path: str,
        duration: Optional[float] = None,
        temperature: float = 1.0,
        guidance_scale: float = 3.0,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Generate music guided by a melody/audio file.
        
        Args:
            prompt: Text description for style/genre/mood
            melody_path: Path to audio file to use as melody guide
            duration: Duration in seconds (uses melody duration if None)
            temperature: Sampling temperature (0.1-2.0)
            guidance_scale: How closely to follow the prompt (1.0-10.0)
            progress_callback: Optional callback function(percent, message)
            
        Returns:
            Tuple of (audio_array, sample_rate)
            
        Raises:
            RuntimeError: If not using a melody model
            ValueError: If parameters are invalid
        """
        # Validate we're using a melody model
        if not self.is_melody_model:
            raise RuntimeError(
                f"Model '{self.model_name}' does not support melody-guided generation.\n"
                "Use 'facebook/musicgen-melody' instead."
            )
        
        # Validate inputs
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        if not (0.1 <= temperature <= 2.0):
            raise ValueError(f"Temperature must be between 0.1 and 2.0, got {temperature}")
        
        if not (1.0 <= guidance_scale <= 10.0):
            raise ValueError(f"Guidance scale must be between 1.0 and 10.0, got {guidance_scale}")
        
        logger.info(f"Generating with melody: '{prompt}' guided by {melody_path}")
        start_time = time.time()
        
        # Progress tracking
        if progress_callback:
            progress_callback(0, "Loading melody audio...")
        
        # Load the melody audio
        try:
            melody_audio, melody_sr = self.load_audio(melody_path, duration=duration)
        except Exception as e:
            logger.error(f"Failed to load melody audio: {e}")
            raise
        
        # Use melody duration if not specified
        if duration is None:
            duration = len(melody_audio) / melody_sr
            logger.info(f"Using melody duration: {duration:.1f}s")
        
        # Limit to 30 seconds (model constraint)
        if duration > 30:
            logger.warning(f"Duration {duration}s exceeds 30s limit. Truncating to 30s.")
            duration = 30
            melody_audio = melody_audio[:int(30 * melody_sr)]
        
        if progress_callback:
            progress_callback(20, f"Processing melody and text prompt...")
        
        # Process inputs for the model
        # The processor expects both text and audio inputs for melody model
        inputs = self.processor(
            text=[prompt],
            audio=melody_audio,
            sampling_rate=melody_sr,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Calculate tokens needed
        max_new_tokens = int(256 * duration / 5)
        
        if progress_callback:
            progress_callback(30, f"Generating {duration:.1f}s of music...")
        
        # Generate audio with melody conditioning
        with torch.no_grad():
            audio_values = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                guidance_scale=guidance_scale,
                max_new_tokens=max_new_tokens
            )
        
        if progress_callback:
            progress_callback(90, "Processing generated audio...")
        
        # Extract audio
        audio = audio_values[0].cpu().numpy().squeeze()
        sample_rate = self.model.config.audio_encoder.sampling_rate
        
        # Log generation stats
        gen_time = time.time() - start_time
        audio_duration = len(audio) / sample_rate
        speed = audio_duration / gen_time
        
        logger.info(
            f"✓ Generated {audio_duration:.1f}s in {gen_time:.1f}s "
            f"({speed:.2f}x realtime)"
        )
        
        if progress_callback:
            progress_callback(100, "✓ Generation complete!")
        
        return audio, sample_rate
    
    def generate_extended(
        self,
        prompt: str, 
        duration: float,
        temperature: float = 1.0,
        guidance_scale: float = 3.0,
        segment_length: float = 25.0,
        overlap_seconds: float = 2.0,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Generate audio longer than 30 seconds using segment generation with crossfading.
        
        ⚠️ REALITY CHECK: This is NOT true context-aware generation like Meta's research.
        Each segment is generated independently then blended. This is a limitation of 
        the current HuggingFace MusicGen API which has a hard 30-second limit.
        
        Args:
            prompt: Text description of the music to generate
            duration: Total duration in seconds
            temperature: Sampling temperature (0.1-2.0)
            guidance_scale: Classifier-free guidance (1.0-10.0)
            segment_length: Length of each segment in seconds (max ~25-28s)
            overlap_seconds: Seconds of overlap between segments for crossfading
            progress_callback: Optional callback function(current, total, message)
            
        Returns:
            Tuple of (audio_array, sample_rate)
            
        Raises:
            ValueError: If parameters are out of valid ranges
            RuntimeError: If generation fails
        """
        # Input validation
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        if duration <= 30.0:
            # For short audio, use regular generation
            if progress_callback:
                progress_callback(0, 1, f"Generating {duration:.1f}s audio...")
            
            result = self.generate(
                prompt=prompt,
                duration=duration,
                temperature=temperature,
                guidance_scale=guidance_scale
            )
            
            if progress_callback:
                progress_callback(1, 1, "Complete!")
                
            return result
        
        if not (0.1 <= segment_length <= 28.0):
            raise ValueError(f"Segment length must be between 0.1 and 28.0 seconds, got {segment_length}")
        
        if not (0.0 <= overlap_seconds <= segment_length / 2):
            raise ValueError(f"Overlap must be between 0.0 and {segment_length/2:.1f} seconds, got {overlap_seconds}")
        
        # Calculate number of segments needed
        effective_segment_length = segment_length - overlap_seconds
        num_segments = max(1, int(np.ceil((duration - overlap_seconds) / effective_segment_length)))
        
        logger.info(f"Generating {duration:.1f}s audio using {num_segments} segments of {segment_length:.1f}s each")
        
        segments = []
        total_start_time = time.time()
        
        for i in range(num_segments):
            if progress_callback:
                progress_callback(i, num_segments, f"Generating segment {i+1}/{num_segments}...")
            
            # Calculate duration for this segment
            remaining_duration = duration - i * effective_segment_length
            current_segment_length = min(segment_length, remaining_duration + overlap_seconds)
            
            # Generate this segment
            segment_audio, sample_rate = self.generate(
                prompt=prompt,
                duration=current_segment_length,
                temperature=temperature,
                guidance_scale=guidance_scale
            )
            
            segments.append(segment_audio)
            
            logger.info(f"✓ Generated segment {i+1}/{num_segments}: {len(segment_audio)/sample_rate:.1f}s")
        
        # Blend segments together with crossfading
        if progress_callback:
            progress_callback(num_segments, num_segments, "Blending segments...")
        
        overlap_samples = int(overlap_seconds * sample_rate)
        final_audio = segments[0]
        
        for i in range(1, len(segments)):
            final_audio = self._crossfade_audio(final_audio, segments[i], overlap_samples)
        
        # Trim to exact duration if needed
        target_samples = int(duration * sample_rate)
        if len(final_audio) > target_samples:
            final_audio = final_audio[:target_samples]
        
        generation_time = time.time() - total_start_time
        actual_duration = len(final_audio) / sample_rate
        
        logger.info(
            f"✓ Extended generation complete: {actual_duration:.1f}s in {generation_time:.1f}s "
            f"({actual_duration/generation_time:.2f}x realtime)"
        )
        
        if progress_callback:
            progress_callback(num_segments, num_segments, "Complete!")
        
        return final_audio, sample_rate
    
    def generate_with_progress(
        self,
        prompt: str,
        duration: float = 10.0,
        temperature: float = 1.0,
        guidance_scale: float = 3.0,
        output_file: Optional[str] = None,
        format: str = "wav",
        bitrate: str = "192k",
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Tuple[np.ndarray, int, Optional[str]]:
        """
        Generate music with automatic progress tracking.
        
        Args:
            prompt: Text description of the music to generate
            duration: Duration in seconds
            temperature: Sampling temperature
            guidance_scale: Classifier-free guidance
            output_file: Optional output filename
            format: Output format ("wav" or "mp3")
            bitrate: MP3 bitrate if format is "mp3"
            progress_callback: Optional callback function(percent, message)
            
        Returns:
            Tuple of (audio_array, sample_rate, output_path)
        """
        # Use extended generation for > 30 seconds
        if duration > 30:
            audio, sample_rate = self.generate_extended(
                prompt=prompt,
                duration=duration,
                temperature=temperature,
                guidance_scale=guidance_scale,
                progress_callback=lambda c, t, m: progress_callback(
                    (c / t) * 100 if t > 0 else 0, m
                ) if progress_callback else None
            )
        else:
            audio, sample_rate = self.generate(
                prompt=prompt,
                duration=duration,
                temperature=temperature,
                guidance_scale=guidance_scale,
                progress_callback=progress_callback
            )
        
        # Save if output file specified
        output_path = None
        if output_file:
            output_path = self.save_audio_as_format(
                audio, sample_rate, output_file, 
                format=format, bitrate=bitrate, 
                delete_wav=(format == "mp3")
            )
            
            if progress_callback:
                file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
                progress_callback(100, f"✓ Saved {output_path} ({file_size:.1f} MB)")
        
        return audio, sample_rate, output_path
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "sample_rate": self.model.config.audio_encoder.sampling_rate,
            "audio_channels": self.model.config.audio_encoder.audio_channels,
            "vocab_size": self.model.config.decoder.vocab_size,
        }