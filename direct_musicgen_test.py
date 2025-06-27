#!/usr/bin/env python3
"""
Direct MusicGen Test - No fancy features, just generate audio NOW.
"""

print("🎵 Direct MusicGen Test - Let's make music!")
print("="*50)

# 1. Quick dependency check
import subprocess
import sys

for pkg in ['transformers', 'scipy']:
    try:
        __import__(pkg)
    except:
        print(f"Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# 2. Import and generate - no error handling, just do it
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile
import numpy as np
import time

print("\nLoading model (first run downloads ~1.5GB)...")
start = time.time()

# Load without any fancy options
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

print(f"Model loaded in {time.time()-start:.1f}s")

# 3. Generate THREE different styles to prove it works
prompts = [
    ("techno beat", 5),
    ("piano jazz", 5), 
    ("rock guitar", 5)
]

print("\nGenerating audio...")
for i, (prompt, duration) in enumerate(prompts):
    print(f"\n[{i+1}/3] Generating: {prompt}")
    start = time.time()
    
    # Tokenize
    inputs = processor(text=[prompt], return_tensors="pt")
    
    # Generate (5 seconds = 250 tokens at 50Hz)
    audio_values = model.generate(**inputs, max_new_tokens=250)
    
    # Extract audio
    audio = audio_values[0, 0].numpy()
    
    # Normalize and save
    audio = audio / (np.abs(audio).max() + 1e-7)
    audio_int16 = (audio * 32767).astype(np.int16)
    
    filename = f"REAL_MUSIC_{i+1}_{prompt.replace(' ', '_')}.wav"
    scipy.io.wavfile.write(filename, 32000, audio_int16)
    
    gen_time = time.time() - start
    print(f"✓ Generated in {gen_time:.1f}s")
    print(f"✓ Saved: {filename} ({len(audio)/32000:.1f}s, {np.std(audio):.3f} RMS)")

print("\n" + "="*50)
print("✅ SUCCESS! Check these files:")
import os
for f in os.listdir('.'):
    if f.startswith('REAL_MUSIC_') and f.endswith('.wav'):
        size = os.path.getsize(f) / 1024
        print(f"  🎵 {f} ({size:.1f} KB)")

print("\nThese are REAL AI-generated music files!")
print("Play them in any audio player to hear the music!")
print("="*50)