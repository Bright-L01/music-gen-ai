#!/usr/bin/env python3
"""
The SIMPLEST way to test if MusicGen actually works.
This will generate a real playable audio file in ~60 seconds.
"""

print("🎵 Simple MusicGen Test - Generating Real Audio")
print("=" * 50)

# 1. Install scipy if needed (for saving audio)
try:
    import scipy
except ImportError:
    print("Installing scipy for audio saving...")
    import subprocess
    subprocess.check_call(["pip", "install", "scipy"])

# 2. Import and generate
print("\n1. Loading MusicGen (will download ~1.5GB on first run)...")
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile
import time

# Load model
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
print("✓ Model loaded!")

# 3. Generate music
prompt = "upbeat electronic dance music with heavy bass"
print(f"\n2. Generating music: '{prompt}'")
print("   This will take ~30-60 seconds on CPU...")

start = time.time()
inputs = processor(text=[prompt], padding=True, return_tensors="pt")

# Generate 10 seconds of audio
audio_values = model.generate(**inputs, max_new_tokens=500)
elapsed = time.time() - start

print(f"✓ Generated in {elapsed:.1f} seconds!")

# 4. Save audio
audio = audio_values[0, 0].cpu().numpy()
# Normalize and convert to 16-bit
audio = audio / abs(audio).max()
audio_int16 = (audio * 32767).astype('int16')

filename = "real_musicgen_output.wav"
scipy.io.wavfile.write(filename, rate=32000, data=audio_int16)

print(f"\n✅ SUCCESS! Audio saved to: {filename}")
print(f"   Duration: {len(audio)/32000:.1f} seconds")
print(f"   Size: {len(audio_int16)*2/1024:.1f} KB")
print("\n🎧 You can now play this WAV file in any audio player!")
print("   This is REAL AI-generated music, not mock data!")

# 5. Quick analysis
import numpy as np
if np.std(audio) > 0.01:
    print("\n✓ Audio verification: Contains actual music (not silence)")
    print(f"  RMS level: {np.sqrt(np.mean(audio**2)):.3f}")
    print(f"  Peak level: {abs(audio).max():.3f}")
else:
    print("\n⚠️  Audio might be too quiet")

print("\n" + "=" * 50)
print("This is how simple MusicGen should be!")
print("Total lines of actual generation code: ~10")
print("=" * 50)