#!/usr/bin/env python3
"""
Test REAL MusicGen using Hugging Face Transformers.
This actually generates playable audio files!
"""

import time
import torch
import numpy as np
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def install_dependencies():
    """Ensure required dependencies are installed."""
    try:
        import scipy
        logger.info("✓ scipy already installed")
    except ImportError:
        logger.info("Installing scipy for audio saving...")
        import subprocess
        subprocess.check_call(["pip", "install", "scipy"])
        logger.info("✓ scipy installed")

def test_musicgen():
    """Test real MusicGen audio generation."""
    print("=" * 60)
    print("🎵 Testing REAL MusicGen Audio Generation")
    print("=" * 60)
    
    # Import after ensuring dependencies
    from transformers import AutoProcessor, MusicgenForConditionalGeneration
    import scipy.io.wavfile
    
    # 1. Load model (downloads ~1.5GB on first run)
    print("\n1. Loading MusicGen model...")
    print("   This will download ~1.5GB on first run...")
    
    start_load = time.time()
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
    load_time = time.time() - start_load
    
    print(f"   ✓ Model loaded in {load_time:.1f} seconds")
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        model = model.to(device)
        print(f"   ✓ Using GPU acceleration")
    else:
        print(f"   ⚠️  Using CPU (will be slower)")
    
    # Model info
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"   Model size: {param_count:.1f}M parameters")
    
    # 2. Generate audio from text prompt
    text_prompt = "upbeat jazz piano with walking bassline"
    print(f"\n2. Generating audio for: '{text_prompt}'")
    print("   Duration: 10 seconds")
    print("   Please wait, this may take 30-60 seconds on CPU...")
    
    # Prepare inputs
    inputs = processor(
        text=[text_prompt],
        padding=True,
        return_tensors="pt"
    )
    
    if device == "cuda":
        inputs = inputs.to(device)
    
    # Generate audio
    start_gen = time.time()
    with torch.no_grad():
        # Generate ~10 seconds of audio
        # At 50Hz with 4 codebooks, 10 seconds = 500 tokens
        audio_values = model.generate(
            **inputs, 
            max_new_tokens=500,
            do_sample=True,
            temperature=1.0,
            top_p=0.9
        )
    
    gen_time = time.time() - start_gen
    print(f"   ✓ Generated in {gen_time:.1f} seconds")
    print(f"   Speed: {10.0/gen_time:.2f}x realtime")
    
    # 3. Convert and analyze audio
    print("\n3. Processing generated audio...")
    
    # Get sampling rate and convert to numpy
    sampling_rate = model.config.audio_encoder.sampling_rate
    audio_numpy = audio_values[0, 0].cpu().numpy()
    
    # Audio characteristics
    duration = len(audio_numpy) / sampling_rate
    print(f"   Sampling rate: {sampling_rate} Hz")
    print(f"   Duration: {duration:.2f} seconds")
    print(f"   Samples: {len(audio_numpy):,}")
    print(f"   Range: [{audio_numpy.min():.3f}, {audio_numpy.max():.3f}]")
    print(f"   RMS level: {np.sqrt(np.mean(audio_numpy**2)):.3f}")
    
    # Check if it's not silence
    if np.abs(audio_numpy).max() > 0.01:
        print("   ✓ Audio contains sound (not silence)")
    else:
        print("   ⚠️  Audio might be too quiet")
    
    # 4. Save audio file
    print("\n4. Saving audio file...")
    
    # Normalize to prevent clipping
    audio_normalized = audio_numpy / np.abs(audio_numpy).max()
    audio_int16 = (audio_normalized * 32767).astype(np.int16)
    
    output_file = "musicgen_output.wav"
    scipy.io.wavfile.write(output_file, rate=sampling_rate, data=audio_int16)
    
    # Verify file
    file_size = Path(output_file).stat().st_size / 1024
    print(f"   ✓ Saved to: {output_file}")
    print(f"   File size: {file_size:.1f} KB")
    
    # 5. Test variations
    print("\n5. Testing prompt variations...")
    test_prompts = [
        "classical piano sonata in minor key",
        "heavy metal guitar riff with drums",
        "ambient electronic soundscape"
    ]
    
    outputs = []
    for i, prompt in enumerate(test_prompts):
        print(f"   Generating: {prompt}")
        inputs = processor(text=[prompt], padding=True, return_tensors="pt")
        if device == "cuda":
            inputs = inputs.to(device)
        
        with torch.no_grad():
            audio = model.generate(**inputs, max_new_tokens=250)  # 5 seconds each
        
        # Save each variation
        audio_np = audio[0, 0].cpu().numpy()
        audio_norm = audio_np / np.abs(audio_np).max()
        audio_int = (audio_norm * 32767).astype(np.int16)
        
        filename = f"musicgen_variation_{i+1}.wav"
        scipy.io.wavfile.write(filename, rate=sampling_rate, data=audio_int)
        outputs.append(filename)
        print(f"      ✓ Saved: {filename}")
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ SUCCESS! Real Audio Generation Complete!")
    print("=" * 60)
    print("\nGenerated files:")
    print(f"  - {output_file} (main output)")
    for f in outputs:
        print(f"  - {f}")
    
    print("\n🎵 You can now play these WAV files in any audio player!")
    print("   They contain real AI-generated music, not mock data!")
    
    print("\nWhat just happened:")
    print("  1. Downloaded real MusicGen model from Facebook/Meta")
    print("  2. Processed text through neural network")
    print("  3. Generated audio tokens with transformer")
    print("  4. Decoded tokens to audio waveform")
    print("  5. Saved playable WAV files")
    
    return True

def main():
    """Main test function."""
    try:
        # Ensure dependencies
        install_dependencies()
        
        # Run test
        success = test_musicgen()
        
        if success:
            print("\n🎉 MusicGen is working correctly!")
            print("This is the proper way to use MusicGen - through transformers library.")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n💡 Troubleshooting:")
        print("1. Make sure you have internet connection for model download")
        print("2. Ensure you have at least 2GB free disk space")
        print("3. If on Mac, generation will be CPU-only (slower)")
        print("4. Try: pip install --upgrade transformers scipy")

if __name__ == "__main__":
    main()