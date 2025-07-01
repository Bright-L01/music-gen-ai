#!/usr/bin/env python3
"""
Batch MusicGen Testing - Efficient multi-prompt generation
Optimized for generating multiple audio clips with proper resource management.
"""

import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


def setup_environment():
    """Setup optimal environment for generation."""
    # Force CPU if low on GPU memory
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_mem < 4:
            print(f"GPU memory ({gpu_mem:.1f}GB) too low, using CPU")
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Optimize memory usage
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"

    # Create directories
    Path("batch_outputs").mkdir(exist_ok=True)
    Path("cache").mkdir(exist_ok=True)


def generate_batch():
    """Generate multiple audio clips efficiently."""
    print("🎵 Batch MusicGen Test")
    print("=" * 60)

    setup_environment()

    # Install dependencies quietly
    required = ["transformers", "scipy", "soundfile"]
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            print(f"Installing {pkg}...")
            os.system(f"{sys.executable} -m pip install -q {pkg}")

    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    # Define test prompts with varying styles
    test_prompts = [
        # (prompt, duration, style_tag)
        ("upbeat electronic dance music with driving bass", 10, "electronic"),
        ("peaceful piano melody in a minor key", 15, "classical"),
        ("energetic rock guitar solo with distortion", 10, "rock"),
        ("smooth jazz saxophone over walking bassline", 20, "jazz"),
        ("ambient soundscape with nature sounds", 15, "ambient"),
        ("hip hop beat with heavy 808 drums", 10, "hiphop"),
        ("folk acoustic guitar fingerpicking", 15, "folk"),
        ("cinematic orchestral film score", 20, "orchestral"),
        ("reggae rhythm with offbeat guitar", 10, "reggae"),
        ("techno loop with acid bass", 10, "techno"),
    ]

    # Load model once
    print("\nLoading MusicGen model...")
    try:
        model = MusicgenForConditionalGeneration.from_pretrained(
            "facebook/musicgen-small",
            cache_dir="cache",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
        )
        processor = AutoProcessor.from_pretrained("facebook/musicgen-small", cache_dir="cache")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()  # Set to evaluation mode

        print(f"✓ Model loaded on {device}")

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Batch generation
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\nGenerating {len(test_prompts)} audio clips...")
    print("-" * 60)

    for i, (prompt, duration, style) in enumerate(test_prompts):
        result = {
            "index": i + 1,
            "prompt": prompt,
            "duration": duration,
            "style": style,
            "status": "pending",
            "file": None,
            "generation_time": None,
            "error": None,
        }

        print(f"\n[{i+1}/{len(test_prompts)}] {style.upper()}: {prompt[:50]}...")

        try:
            # Generate
            start_time = time.time()

            inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(device)

            # Calculate tokens (50Hz generation rate)
            max_tokens = min(int(duration * 50), 1500)  # Cap at 30s

            # Generate with memory optimization
            with torch.no_grad():
                if device == "cuda":
                    with torch.cuda.amp.autocast():
                        audio_values = model.generate(
                            **inputs,
                            max_new_tokens=max_tokens,
                            do_sample=True,
                            temperature=1.0,
                            top_p=0.9,
                            guidance_scale=3.0,  # Better quality
                        )
                else:
                    audio_values = model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=True,
                        temperature=1.0,
                        top_p=0.9,
                    )

            # Extract and process audio
            audio = audio_values[0, 0].cpu().numpy()
            generation_time = time.time() - start_time

            # Validate audio
            if len(audio) > 0 and np.std(audio) > 0.001:
                # Normalize
                audio = audio / (np.abs(audio).max() + 1e-7)

                # Save with descriptive filename
                filename = f"musicgen_{timestamp}_{i+1:02d}_{style}.wav"
                filepath = Path("batch_outputs") / filename

                # Try soundfile first, fallback to scipy
                try:
                    import soundfile as sf

                    sf.write(str(filepath), audio, 32000, subtype="PCM_16")
                except:
                    import scipy.io.wavfile

                    audio_int16 = (audio * 32767).astype(np.int16)
                    scipy.io.wavfile.write(str(filepath), 32000, audio_int16)

                result["status"] = "success"
                result["file"] = str(filepath)
                result["generation_time"] = generation_time

                print(f"✓ Generated in {generation_time:.1f}s - Saved as {filename}")

            else:
                result["status"] = "failed"
                result["error"] = "Silent or invalid audio"
                print("❌ Generated audio is silent")

        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)[:100]
            print(f"❌ Failed: {str(e)[:100]}")

        results.append(result)

        # Memory cleanup between generations
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

        # Save progress
        progress_file = Path("batch_outputs") / f"progress_{timestamp}.json"
        with open(progress_file, "w") as f:
            json.dump(results, f, indent=2)

    # Generate final report
    print("\n" + "=" * 60)
    print("📊 BATCH GENERATION REPORT")
    print("=" * 60)

    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]

    print(f"\nTotal attempts: {len(results)}")
    print(f"Successful: {len(successful)} ({len(successful)/len(results)*100:.0f}%)")
    print(f"Failed: {len(failed)}")

    if successful:
        # Performance stats
        gen_times = [r["generation_time"] for r in successful]
        avg_time = np.mean(gen_times)
        total_duration = sum(r["duration"] for r in successful)

        print("\nPerformance:")
        print(f"  Average generation time: {avg_time:.1f}s")
        print(f"  Total audio generated: {total_duration}s")
        print(f"  Average speed: {total_duration/sum(gen_times):.1f}x realtime")

        # List files by style
        print("\nGenerated files by style:")
        styles = {}
        for r in successful:
            style = r["style"]
            if style not in styles:
                styles[style] = []
            styles[style].append(Path(r["file"]).name)

        for style, files in sorted(styles.items()):
            print(f"\n  {style.upper()}:")
            for f in files:
                print(f"    - {f}")

    # Save final report
    report = {
        "timestamp": timestamp,
        "total_prompts": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "results": results,
    }

    report_file = Path("batch_outputs") / f"batch_report_{timestamp}.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n📄 Full report saved to: {report_file}")

    if successful:
        print(f"\n✅ SUCCESS! Generated {len(successful)} audio files")
        print("🎵 Check the 'batch_outputs' folder to listen to your AI music!")

    # Cleanup
    del model
    del processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    try:
        generate_batch()
    except KeyboardInterrupt:
        print("\n\nBatch generation interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback

        traceback.print_exc()
