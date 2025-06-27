#!/usr/bin/env python3
"""
CORRECT MusicGen Architecture - Using Facebook's Pre-trained Models
This shows how to properly use MusicGen without reinventing the wheel.
"""

import time
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import torch
import scipy.io.wavfile
import numpy as np

class RealMusicGenerator:
    """
    The CORRECT way to use MusicGen - with pre-trained models!
    Not building transformers from scratch.
    """
    
    def __init__(self, model_size='small'):
        print(f"Loading REAL MusicGen {model_size} model...")
        
        # Map model sizes to HuggingFace model IDs
        model_map = {
            'small': 'facebook/musicgen-small',    # 300M params
            'medium': 'facebook/musicgen-medium',  # 1.5B params
            'large': 'facebook/musicgen-large'     # 3.3B params
        }
        
        model_id = model_map.get(model_size, 'facebook/musicgen-small')
        
        # Load the ACTUAL pre-trained model
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = MusicgenForConditionalGeneration.from_pretrained(model_id)
        
        # Use GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        
        # Get sampling rate from model config
        self.sample_rate = self.model.config.audio_encoder.sampling_rate
        
        print(f"✓ Model loaded on {self.device}")
        print(f"✓ Sampling rate: {self.sample_rate} Hz")
    
    def generate(self, prompt, duration=10.0):
        """
        Generate audio from text prompt.
        This is REAL neural network generation!
        """
        print(f"\nGenerating: '{prompt}'")
        print(f"Duration: {duration} seconds")
        
        # Tokenize the prompt
        inputs = self.processor(
            text=[prompt],
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Calculate tokens needed for duration
        # MusicGen generates at 50Hz with 4 codebooks
        tokens_needed = int(duration * 50)
        
        # Generate audio
        start_time = time.time()
        with torch.no_grad():
            audio_values = self.model.generate(
                **inputs,
                max_new_tokens=tokens_needed,
                do_sample=True,
                temperature=1.0,
                top_p=0.9
            )
        
        gen_time = time.time() - start_time
        print(f"✓ Generated in {gen_time:.1f}s ({duration/gen_time:.1f}x realtime)")
        
        # Convert to numpy
        audio = audio_values[0, 0].cpu().numpy()
        
        return audio
    
    def save_audio(self, audio, filename):
        """Save audio to WAV file."""
        # Normalize to prevent clipping
        audio = audio / np.abs(audio).max()
        audio_int16 = (audio * 32767).astype(np.int16)
        
        scipy.io.wavfile.write(filename, rate=self.sample_rate, data=audio_int16)
        print(f"✓ Saved: {filename}")


def demonstrate_correct_architecture():
    """Show the correct way to implement MusicGen features."""
    
    print("=" * 60)
    print("🎵 MusicGen - Correct Architecture Demo")
    print("=" * 60)
    
    # 1. Basic Generation
    print("\n1. BASIC GENERATION")
    generator = RealMusicGenerator('small')
    
    audio = generator.generate("upbeat jazz piano melody", duration=10)
    generator.save_audio(audio, "correct_musicgen_output.wav")
    
    # 2. Multi-track Generation (the RIGHT way)
    print("\n2. MULTI-TRACK GENERATION")
    print("Generate each instrument separately, then mix:")
    
    tracks = {}
    instruments = {
        'piano': 'jazz piano comping chords',
        'bass': 'walking jazz bass line',
        'drums': 'jazz drum kit with brushes'
    }
    
    for instrument, prompt in instruments.items():
        print(f"\nGenerating {instrument}...")
        audio = generator.generate(prompt, duration=10)
        tracks[instrument] = audio
        generator.save_audio(audio, f"track_{instrument}.wav")
    
    # Simple mixing
    print("\n3. MIXING TRACKS")
    mixed = sum(tracks.values()) / len(tracks)
    mixed = mixed / np.abs(mixed).max()  # Normalize
    generator.save_audio(mixed, "mixed_output.wav")
    
    # 3. Streaming (practical approach)
    print("\n4. STREAMING GENERATION")
    print("MusicGen doesn't support true streaming, but we can chunk:")
    
    def stream_generate(prompt, chunk_duration=2):
        """Generate in small chunks to simulate streaming."""
        chunks = []
        for i in range(3):  # Generate 3 chunks
            print(f"  Chunk {i+1}/3...")
            chunk_prompt = f"{prompt} part {i+1}"
            audio = generator.generate(chunk_prompt, duration=chunk_duration)
            chunks.append(audio)
            yield audio
        
        # Save full stream
        full_audio = np.concatenate(chunks)
        generator.save_audio(full_audio, "streamed_output.wav")
    
    # Simulate streaming
    for i, chunk in enumerate(stream_generate("electronic dance music")):
        print(f"  ✓ Received chunk {i+1} ({len(chunk)/generator.sample_rate:.1f}s)")
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ CORRECT ARCHITECTURE DEMONSTRATED")
    print("=" * 60)
    print("\nKey Points:")
    print("1. Use facebook/musicgen models via transformers")
    print("2. Don't build transformers from scratch")
    print("3. Multi-track: Generate separately, then mix")
    print("4. Streaming: Chunk generation (not true streaming)")
    print("5. This produces REAL playable audio files!")
    
    print("\n🎵 All WAV files are ready to play!")
    print("This is how production systems should use MusicGen.")


if __name__ == "__main__":
    # Check dependencies
    try:
        import scipy
    except ImportError:
        print("Installing scipy...")
        import subprocess
        subprocess.check_call(["pip", "install", "scipy"])
    
    # Run demo
    demonstrate_correct_architecture()