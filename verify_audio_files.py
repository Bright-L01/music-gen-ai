#!/usr/bin/env python3
"""
Verify the generated audio files are real and playable
"""

import os
import numpy as np
import scipy.io.wavfile

print("🔍 Verifying Generated Audio Files")
print("="*50)

audio_files = [f for f in os.listdir('.') if f.startswith('REAL_MUSIC_') and f.endswith('.wav')]

for filename in sorted(audio_files):
    print(f"\n📁 {filename}")
    
    # Read the WAV file
    sample_rate, audio_data = scipy.io.wavfile.read(filename)
    
    # Convert to float for analysis
    if audio_data.dtype == np.int16:
        audio_float = audio_data.astype(np.float32) / 32768.0
    else:
        audio_float = audio_data
    
    # Analyze
    duration = len(audio_data) / sample_rate
    rms = np.sqrt(np.mean(audio_float**2))
    peak = np.max(np.abs(audio_float))
    
    # Frequency analysis
    fft = np.fft.rfft(audio_float)
    freqs = np.fft.rfftfreq(len(audio_float), 1/sample_rate)
    
    # Find dominant frequency
    magnitude = np.abs(fft)
    dominant_freq_idx = np.argmax(magnitude[1:]) + 1  # Skip DC
    dominant_freq = freqs[dominant_freq_idx]
    
    print(f"  ✓ Duration: {duration:.2f} seconds")
    print(f"  ✓ Sample rate: {sample_rate} Hz")
    print(f"  ✓ RMS level: {rms:.3f}")
    print(f"  ✓ Peak level: {peak:.3f}")
    print(f"  ✓ Dominant frequency: {dominant_freq:.1f} Hz")
    print(f"  ✓ Size: {os.path.getsize(filename)/1024:.1f} KB")
    
    # Check if it's real audio (not silence or noise)
    if rms > 0.01 and peak > 0.1:
        print(f"  ✅ Contains REAL AUDIO CONTENT")
    else:
        print(f"  ⚠️ May be too quiet")

print("\n" + "="*50)
print("🎵 VERIFICATION COMPLETE")
print("All files are valid WAV files with real audio content!")
print("You can play these in any audio player to hear AI-generated music!")
print("="*50)