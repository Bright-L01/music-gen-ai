#!/usr/bin/env python3
"""
Quick MusicGen Test - Minimal dependencies, maximum compatibility
"""

print("🎵 Quick MusicGen Test - Let's make some music!")
print("="*50)

# 1. Basic imports and setup
import sys
import os
import time

# Install dependencies if needed
deps = ['transformers', 'scipy']
for dep in deps:
    try:
        __import__(dep)
    except ImportError:
        print(f"Installing {dep}...")
        os.system(f"{sys.executable} -m pip install {dep}")

# 2. Import what we need
print("\n1. Loading libraries...")
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile
import numpy as np

# 3. Load the model (simplest way possible)
print("\n2. Loading MusicGen model...")
print("   (First run will download ~1.5GB, please wait...)")

try:
    start = time.time()
    
    # Load without any fancy options
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
    
    elapsed = time.time() - start
    print(f"✓ Model loaded in {elapsed:.1f} seconds!")
    
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    print("\nTroubleshooting:")
    print("1. Check internet connection")
    print("2. Try: pip install --upgrade transformers")
    print("3. Make sure you have 4GB+ free disk space")
    sys.exit(1)

# 4. Generate some music!
prompts = [
    ("electronic dance music", 10),
    ("peaceful piano", 10),
    ("rock guitar", 10)
]

print("\n3. Generating music...")
print("   (Takes ~1-2 minutes per 10 seconds on CPU)")

success_count = 0
output_dir = "quick_outputs"
os.makedirs(output_dir, exist_ok=True)

for i, (prompt, duration) in enumerate(prompts):
    print(f"\n[{i+1}/3] '{prompt}'...")
    
    try:
        # Tokenize
        inputs = processor(
            text=[prompt],
            padding=True,
            return_tensors="pt"
        )
        
        # Generate (50 tokens per second of audio)
        tokens = int(duration * 50)
        
        start = time.time()
        audio_values = model.generate(
            **inputs, 
            max_new_tokens=min(tokens, 500),  # Cap at 10 seconds
            do_sample=True,
            temperature=1.0
        )
        elapsed = time.time() - start
        
        # Extract audio
        audio = audio_values[0, 0].cpu().numpy()
        
        # Check if we got sound
        if np.std(audio) > 0.001:
            # Normalize and save
            audio = audio / (np.abs(audio).max() + 1e-7)
            audio_int16 = (audio * 32767).astype(np.int16)
            
            filename = f"{output_dir}/musicgen_{i+1}_{prompt.replace(' ', '_')}.wav"
            scipy.io.wavfile.write(filename, 32000, audio_int16)
            
            print(f"✓ Generated in {elapsed:.1f}s - Saved to {filename}")
            print(f"  Duration: {len(audio)/32000:.1f}s, RMS: {np.sqrt(np.mean(audio**2)):.3f}")
            success_count += 1
        else:
            print("⚠️  Generated silent audio")
            
    except Exception as e:
        print(f"❌ Failed: {str(e)[:100]}")

# 5. Report results
print("\n" + "="*50)
print("📊 RESULTS")
print("="*50)
print(f"Successfully generated: {success_count}/3 audio files")

if success_count > 0:
    print(f"\n✅ SUCCESS! Check the '{output_dir}' folder")
    print("🎧 You now have AI-generated music!")
    
    # List files
    print("\nGenerated files:")
    for f in os.listdir(output_dir):
        if f.endswith('.wav'):
            print(f"  - {f}")
else:
    print("\n❌ No audio generated. Check errors above.")

print("\n💡 Tips:")
print("- Each file is playable in any audio player")
print("- Generation is slow on CPU (normal)")
print("- Try different prompts for different styles")
print("="*50)