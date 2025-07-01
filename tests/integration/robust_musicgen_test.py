#!/usr/bin/env python3
"""
Robust MusicGen Test with Error Recovery
Simplified but reliable test that handles common issues.
"""

import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Simple logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def test_musicgen_with_fallbacks():
    """Test MusicGen with multiple fallback strategies."""

    print("🎵 Robust MusicGen Test")
    print("=" * 50)

    # 1. Check and install dependencies
    print("\n1. Checking dependencies...")
    try:
        import scipy

        print("✓ scipy installed")
    except ImportError:
        print("Installing scipy...")
        os.system(f"{sys.executable} -m pip install scipy")

    try:
        import transformers

        print("✓ transformers installed")
    except ImportError:
        print("Installing transformers...")
        os.system(f"{sys.executable} -m pip install transformers")

    # 2. Try different import methods
    print("\n2. Loading MusicGen...")
    model = None
    processor = None

    # Method 1: Standard import
    try:
        from transformers import AutoProcessor, MusicgenForConditionalGeneration

        print("✓ Imports successful")
    except Exception as e:
        print(f"Import error: {e}")
        print("Trying alternative import...")
        try:
            import transformers

            AutoProcessor = transformers.AutoProcessor
            MusicgenForConditionalGeneration = transformers.MusicgenForConditionalGeneration
            print("✓ Alternative import successful")
        except Exception as e2:
            print(f"Failed to import: {e2}")
            return

    # 3. Load model with timeout and fallbacks
    print("\n3. Loading model (this may take 2-5 minutes)...")

    # Create cache directory
    cache_dir = Path.home() / ".cache" / "huggingface"
    cache_dir.mkdir(parents=True, exist_ok=True)

    model_loaded = False
    attempts = 0
    max_attempts = 3

    while not model_loaded and attempts < max_attempts:
        attempts += 1
        print(f"\nAttempt {attempts}/{max_attempts}...")

        try:
            # Set timeout for downloads
            os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"  # 10 minutes

            # Try loading
            start = time.time()
            processor = AutoProcessor.from_pretrained(
                "facebook/musicgen-small",
                cache_dir=cache_dir,
                resume_download=True,  # Resume if interrupted
            )

            model = MusicgenForConditionalGeneration.from_pretrained(
                "facebook/musicgen-small",
                cache_dir=cache_dir,
                resume_download=True,
                low_cpu_mem_usage=True,  # Reduce memory usage
            )

            elapsed = time.time() - start
            print(f"✓ Model loaded in {elapsed:.1f} seconds")
            model_loaded = True

        except Exception as e:
            print(f"❌ Attempt {attempts} failed: {str(e)[:100]}...")

            if "connection" in str(e).lower() or "timeout" in str(e).lower():
                print("Network issue detected. Waiting 30 seconds before retry...")
                time.sleep(30)
            elif "memory" in str(e).lower():
                print("Memory issue detected. Trying CPU-only mode...")
                os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU
            else:
                print("Unknown error. Trying with minimal config...")

            # Clean up before retry
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not model_loaded:
        print("\n❌ Failed to load model after all attempts")
        return

    # 4. Test generation with multiple prompts
    print("\n4. Testing audio generation...")

    test_prompts = [
        ("simple piano melody", 5),  # Short test first
        ("upbeat jazz with saxophone", 10),
        ("electronic dance music", 15),
        ("classical orchestra", 10),
    ]

    successful_generations = 0
    output_dir = Path("musicgen_outputs")
    output_dir.mkdir(exist_ok=True)

    for i, (prompt, duration) in enumerate(test_prompts):
        print(f"\n--- Test {i+1}/{len(test_prompts)} ---")
        print(f"Prompt: '{prompt}'")
        print(f"Duration: {duration} seconds")

        try:
            # Prepare inputs
            inputs = processor(text=[prompt], padding=True, return_tensors="pt")

            # Move to CPU to avoid GPU issues
            if hasattr(model, "to"):
                model = model.to("cpu")
                inputs = {k: v.to("cpu") for k, v in inputs.items()}

            # Generate with progress
            print("Generating... (this may take 1-3 minutes)")
            start = time.time()

            # Calculate tokens
            tokens = int(duration * 50)  # 50Hz

            # Generate
            with torch.no_grad():
                audio_values = model.generate(
                    **inputs,
                    max_new_tokens=min(tokens, 1000),  # Limit for testing
                    do_sample=True,
                    temperature=1.0,
                )

            elapsed = time.time() - start
            print(f"✓ Generated in {elapsed:.1f} seconds")

            # Extract audio
            audio = audio_values[0, 0].cpu().numpy()

            # Validate
            if len(audio) > 0 and np.std(audio) > 0.001:
                print("✓ Audio contains sound (not silence)")

                # Save
                filename = f"test_{i+1}_{prompt.replace(' ', '_')}.wav"
                filepath = output_dir / filename

                # Normalize
                audio = audio / np.abs(audio).max()
                audio_int16 = (audio * 32767).astype(np.int16)

                # Save with scipy
                import scipy.io.wavfile

                scipy.io.wavfile.write(str(filepath), 32000, audio_int16)  # Sample rate

                file_size = filepath.stat().st_size / 1024
                print(f"✓ Saved: {filename} ({file_size:.1f} KB)")
                successful_generations += 1

            else:
                print("⚠️  Generated audio is silent or invalid")

        except KeyboardInterrupt:
            print("\n\nTest interrupted by user")
            break
        except Exception as e:
            print(f"❌ Generation failed: {str(e)[:200]}...")

            # Try to recover
            if "memory" in str(e).lower():
                print("Clearing memory and continuing...")
                import gc

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Small delay between tests
        time.sleep(2)

    # 5. Final report
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS")
    print("=" * 50)
    print(f"Tests attempted: {len(test_prompts)}")
    print(f"Successful: {successful_generations}")

    if successful_generations > 0:
        print(f"\n✅ SUCCESS! Generated {successful_generations} audio files")
        print(f"Check the '{output_dir}' folder for your AI music!")
        print("\nFiles generated:")
        for f in output_dir.glob("*.wav"):
            print(f"  - {f.name}")
    else:
        print("\n❌ No successful generations")
        print("Troubleshooting:")
        print("1. Check your internet connection")
        print("2. Ensure you have 4GB+ free RAM")
        print("3. Try running with: python -u robust_musicgen_test.py")
        print("4. Check musicgen_test.log for details")

    print("\n" + "=" * 50)


if __name__ == "__main__":
    try:
        test_musicgen_with_fallbacks()
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback

        traceback.print_exc()
        print("\nPlease check your Python environment and try again.")
