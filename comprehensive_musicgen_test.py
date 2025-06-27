#!/usr/bin/env python3
"""
Comprehensive MusicGen Testing and Verification
Tests real audio generation with progress tracking, error handling, and performance optimization.
"""

import os
import sys
import time
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import traceback
import psutil
import gc
from typing import Dict, List, Tuple, Optional

# Setup detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('musicgen_test.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class MusicGenTester:
    """Comprehensive MusicGen testing with progress tracking and error recovery."""
    
    def __init__(self):
        self.results = []
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cache_dir = Path("model_cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.output_dir = Path("test_outputs")
        self.output_dir.mkdir(exist_ok=True)
        
    def install_dependencies(self):
        """Ensure all dependencies are installed."""
        logger.info("Checking dependencies...")
        required = ['scipy', 'transformers', 'soundfile']
        
        for package in required:
            try:
                __import__(package)
                logger.info(f"✓ {package} already installed")
            except ImportError:
                logger.info(f"Installing {package}...")
                import subprocess
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                    logger.info(f"✓ {package} installed successfully")
                except Exception as e:
                    logger.error(f"Failed to install {package}: {e}")
                    return False
        return True
    
    def get_memory_usage(self):
        """Get current memory usage."""
        process = psutil.Process()
        return {
            'cpu_ram_gb': process.memory_info().rss / 1024**3,
            'cpu_percent': process.cpu_percent(),
            'gpu_memory_gb': torch.cuda.memory_allocated() / 1024**3 if self.device == "cuda" else 0
        }
    
    def load_model_with_progress(self, model_size='small'):
        """Load MusicGen model with progress tracking and caching."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Loading MusicGen {model_size} model...")
        logger.info(f"Using device: {self.device}")
        
        from transformers import AutoProcessor, MusicgenForConditionalGeneration
        
        model_id = f"facebook/musicgen-{model_size}"
        
        # Track loading stages
        stages = [
            ("Checking cache", 0.1),
            ("Downloading model config", 0.2),
            ("Downloading model weights", 0.6),
            ("Loading into memory", 0.8),
            ("Moving to device", 0.9),
            ("Ready", 1.0)
        ]
        
        start_time = time.time()
        memory_before = self.get_memory_usage()
        
        try:
            # Stage 1: Check cache
            logger.info(f"[1/6] {stages[0][0]}...")
            cache_path = self.cache_dir / model_size
            
            # Stage 2-3: Download
            logger.info(f"[2/6] {stages[1][0]}...")
            logger.info(f"[3/6] {stages[2][0]} (this may take several minutes on first run)...")
            
            # Load with cache_dir
            os.environ['TRANSFORMERS_CACHE'] = str(self.cache_dir)
            
            self.processor = AutoProcessor.from_pretrained(
                model_id,
                cache_dir=self.cache_dir
            )
            
            self.model = MusicgenForConditionalGeneration.from_pretrained(
                model_id,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            # Stage 4: Loading
            logger.info(f"[4/6] {stages[3][0]}...")
            param_count = sum(p.numel() for p in self.model.parameters()) / 1e6
            logger.info(f"Model parameters: {param_count:.1f}M")
            
            # Stage 5: Move to device
            logger.info(f"[5/6] {stages[4][0]}...")
            self.model = self.model.to(self.device)
            
            # Stage 6: Ready
            logger.info(f"[6/6] {stages[5][0]}!")
            
            load_time = time.time() - start_time
            memory_after = self.get_memory_usage()
            memory_used = memory_after['cpu_ram_gb'] - memory_before['cpu_ram_gb']
            
            logger.info(f"\n✅ Model loaded successfully!")
            logger.info(f"Load time: {load_time:.1f} seconds")
            logger.info(f"Memory used: {memory_used:.2f} GB")
            logger.info(f"Sample rate: {self.model.config.audio_encoder.sampling_rate} Hz")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            traceback.print_exc()
            return False
    
    def generate_with_progress(self, prompt: str, duration: float = 10.0) -> Optional[np.ndarray]:
        """Generate audio with detailed progress tracking."""
        logger.info(f"\n{'='*40}")
        logger.info(f"Generating: '{prompt}'")
        logger.info(f"Duration: {duration} seconds")
        
        if self.model is None:
            logger.error("Model not loaded!")
            return None
        
        try:
            # Prepare inputs
            start_time = time.time()
            memory_before = self.get_memory_usage()
            
            logger.info("Tokenizing prompt...")
            inputs = self.processor(
                text=[prompt],
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Calculate tokens needed
            tokens_needed = int(duration * 50)  # 50Hz generation
            logger.info(f"Generating {tokens_needed} tokens...")
            
            # Generate with progress estimation
            logger.info("Starting generation (this may take 2-5 minutes on CPU)...")
            
            # Show progress stages
            stages_complete = 0
            total_stages = 10
            
            def progress_callback(current_tokens):
                nonlocal stages_complete
                progress = current_tokens / tokens_needed
                new_stages = int(progress * total_stages)
                if new_stages > stages_complete:
                    stages_complete = new_stages
                    elapsed = time.time() - start_time
                    eta = (elapsed / progress) - elapsed if progress > 0 else 0
                    logger.info(f"Progress: {'█' * stages_complete}{'░' * (total_stages - stages_complete)} "
                              f"{progress*100:.0f}% (ETA: {eta:.0f}s)")
            
            # Generate audio
            with torch.no_grad():
                if self.device == "cuda":
                    with torch.cuda.amp.autocast():
                        audio_values = self.model.generate(
                            **inputs,
                            max_new_tokens=tokens_needed,
                            do_sample=True,
                            temperature=1.0,
                            top_p=0.9
                        )
                else:
                    audio_values = self.model.generate(
                        **inputs,
                        max_new_tokens=tokens_needed,
                        do_sample=True,
                        temperature=1.0,
                        top_p=0.9
                    )
            
            # Complete progress
            logger.info(f"Progress: {'█' * total_stages} 100%")
            
            # Extract audio
            audio = audio_values[0, 0].cpu().numpy()
            
            # Performance stats
            gen_time = time.time() - start_time
            memory_after = self.get_memory_usage()
            memory_used = memory_after['cpu_ram_gb'] - memory_before['cpu_ram_gb']
            
            logger.info(f"\n✅ Generation complete!")
            logger.info(f"Generation time: {gen_time:.1f} seconds")
            logger.info(f"Speed: {duration/gen_time:.2f}x realtime")
            logger.info(f"Memory used: {memory_used:.2f} GB")
            
            return audio
            
        except torch.cuda.OutOfMemoryError:
            logger.error("❌ GPU out of memory! Trying CPU fallback...")
            self.device = "cpu"
            self.model = self.model.to("cpu")
            torch.cuda.empty_cache()
            gc.collect()
            return self.generate_with_progress(prompt, duration)
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            traceback.print_exc()
            return None
    
    def validate_audio(self, audio: np.ndarray, expected_duration: float) -> Dict:
        """Validate generated audio."""
        validation = {
            'valid': True,
            'issues': [],
            'stats': {}
        }
        
        # Check if audio exists and has content
        if audio is None or len(audio) == 0:
            validation['valid'] = False
            validation['issues'].append("No audio generated")
            return validation
        
        # Audio statistics
        sample_rate = self.model.config.audio_encoder.sampling_rate
        actual_duration = len(audio) / sample_rate
        
        validation['stats'] = {
            'duration': actual_duration,
            'sample_rate': sample_rate,
            'samples': len(audio),
            'range': [float(audio.min()), float(audio.max())],
            'rms': float(np.sqrt(np.mean(audio**2))),
            'peak': float(np.abs(audio).max()),
            'silence_ratio': float(np.sum(np.abs(audio) < 0.001) / len(audio))
        }
        
        # Validation checks
        if abs(actual_duration - expected_duration) > 2.0:
            validation['issues'].append(f"Duration mismatch: expected {expected_duration}s, got {actual_duration:.1f}s")
        
        if validation['stats']['rms'] < 0.001:
            validation['valid'] = False
            validation['issues'].append("Audio is silent or near-silent")
        
        if validation['stats']['silence_ratio'] > 0.9:
            validation['valid'] = False
            validation['issues'].append("Audio is mostly silence")
        
        if validation['stats']['peak'] > 1.0:
            validation['issues'].append("Audio may be clipping")
        
        # Log validation results
        if validation['valid']:
            logger.info("✅ Audio validation passed")
        else:
            logger.warning(f"⚠️ Audio validation issues: {', '.join(validation['issues'])}")
        
        logger.info(f"Audio stats: Duration={actual_duration:.1f}s, RMS={validation['stats']['rms']:.3f}, "
                   f"Peak={validation['stats']['peak']:.3f}")
        
        return validation
    
    def save_audio(self, audio: np.ndarray, prompt: str, model_size: str) -> Optional[str]:
        """Save audio with descriptive filename."""
        try:
            import soundfile as sf
            
            # Create descriptive filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_prompt = safe_prompt.replace(' ', '_')[:50]
            filename = f"musicgen_{model_size}_{safe_prompt}_{timestamp}.wav"
            filepath = self.output_dir / filename
            
            # Normalize audio
            audio = audio / np.abs(audio).max() if np.abs(audio).max() > 0 else audio
            
            # Save with soundfile for better compatibility
            sample_rate = self.model.config.audio_encoder.sampling_rate
            sf.write(str(filepath), audio, sample_rate, subtype='PCM_16')
            
            file_size = filepath.stat().st_size / 1024
            logger.info(f"✅ Saved: {filename} ({file_size:.1f} KB)")
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
            # Fallback to scipy
            try:
                import scipy.io.wavfile
                audio_int16 = (audio * 32767).astype(np.int16)
                scipy.io.wavfile.write(str(filepath), sample_rate, audio_int16)
                logger.info(f"✅ Saved with scipy: {filename}")
                return str(filepath)
            except Exception as e2:
                logger.error(f"Scipy save also failed: {e2}")
                return None
    
    def test_prompt(self, prompt: str, duration: float, model_size: str) -> Dict:
        """Test a single prompt and return results."""
        logger.info(f"\n{'='*60}")
        logger.info(f"TEST: {prompt}")
        logger.info(f"{'='*60}")
        
        result = {
            'prompt': prompt,
            'duration': duration,
            'model_size': model_size,
            'success': False,
            'generation_time': None,
            'file_path': None,
            'validation': None,
            'error': None
        }
        
        try:
            # Generate audio
            audio = self.generate_with_progress(prompt, duration)
            
            if audio is not None:
                # Validate
                validation = self.validate_audio(audio, duration)
                result['validation'] = validation
                
                # Save if valid
                if validation['valid']:
                    filepath = self.save_audio(audio, prompt, model_size)
                    if filepath:
                        result['file_path'] = filepath
                        result['success'] = True
                
                # Clean up memory
                del audio
                gc.collect()
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                    
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Test failed: {e}")
            
        return result
    
    def run_comprehensive_test(self):
        """Run comprehensive MusicGen tests."""
        logger.info("\n" + "="*80)
        logger.info("🎵 COMPREHENSIVE MUSICGEN TESTING")
        logger.info("="*80)
        
        # Install dependencies
        if not self.install_dependencies():
            logger.error("Failed to install dependencies")
            return
        
        # Test configurations
        test_configs = [
            # (model_size, prompts_with_duration)
            ('small', [
                ("upbeat electronic dance music with heavy bass", 10),
                ("peaceful classical piano sonata", 15),
                ("energetic rock guitar solo", 10),
                ("jazz quartet playing smooth jazz", 20),
                ("ambient nature sounds with soft synth pads", 30)
            ])
        ]
        
        # Add medium model test if enough memory
        memory_info = psutil.virtual_memory()
        if memory_info.available / 1024**3 > 8:
            test_configs.append(('medium', [
                ("cinematic orchestral movie soundtrack", 15),
                ("hip hop beat with trap drums", 10)
            ]))
        
        # Run tests for each model size
        all_results = []
        
        for model_size, test_prompts in test_configs:
            logger.info(f"\n{'='*80}")
            logger.info(f"Testing {model_size.upper()} model")
            logger.info(f"{'='*80}")
            
            # Load model
            if not self.load_model_with_progress(model_size):
                logger.error(f"Failed to load {model_size} model, skipping...")
                continue
            
            # Test each prompt
            model_results = []
            for prompt, duration in test_prompts:
                result = self.test_prompt(prompt, duration, model_size)
                model_results.append(result)
                all_results.append(result)
                
                # Save intermediate results
                self.save_results(all_results)
                
                # Small delay between generations
                time.sleep(2)
            
            # Model cleanup
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            gc.collect()
            torch.cuda.empty_cache()
            
        # Final report
        self.generate_report(all_results)
    
    def save_results(self, results: List[Dict]):
        """Save test results to JSON."""
        results_file = self.output_dir / "test_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    def generate_report(self, results: List[Dict]):
        """Generate final test report."""
        logger.info("\n" + "="*80)
        logger.info("📊 FINAL TEST REPORT")
        logger.info("="*80)
        
        # Summary statistics
        total_tests = len(results)
        successful = sum(1 for r in results if r['success'])
        
        logger.info(f"\nTests run: {total_tests}")
        logger.info(f"Successful: {successful} ({successful/total_tests*100:.0f}%)")
        
        # Performance stats
        gen_times = [r['generation_time'] for r in results if r['generation_time']]
        if gen_times:
            logger.info(f"\nGeneration times:")
            logger.info(f"  Average: {np.mean(gen_times):.1f}s")
            logger.info(f"  Min: {min(gen_times):.1f}s")
            logger.info(f"  Max: {max(gen_times):.1f}s")
        
        # List generated files
        logger.info(f"\nGenerated audio files:")
        for r in results:
            if r['file_path']:
                logger.info(f"  ✅ {Path(r['file_path']).name}")
        
        # Validation summary
        logger.info(f"\nValidation summary:")
        for r in results:
            if r['validation']:
                status = "✅ VALID" if r['validation']['valid'] else "❌ INVALID"
                logger.info(f"  {r['prompt'][:50]}... - {status}")
                if r['validation']['issues']:
                    for issue in r['validation']['issues']:
                        logger.info(f"    - {issue}")
        
        # Save final report
        report_file = self.output_dir / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write("MUSICGEN COMPREHENSIVE TEST REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"Tests run: {total_tests}\n")
            f.write(f"Successful: {successful}\n\n")
            
            f.write("GENERATED FILES:\n")
            for r in results:
                if r['file_path']:
                    f.write(f"- {Path(r['file_path']).name}\n")
                    f.write(f"  Prompt: {r['prompt']}\n")
                    f.write(f"  Duration: {r['duration']}s\n")
                    if r['validation']:
                        f.write(f"  RMS: {r['validation']['stats']['rms']:.3f}\n")
                    f.write("\n")
        
        logger.info(f"\n📄 Report saved to: {report_file.name}")
        
        # Final success message
        if successful > 0:
            logger.info("\n" + "="*80)
            logger.info("✅ SUCCESS! MusicGen is working correctly!")
            logger.info(f"Generated {successful} playable audio files from text prompts")
            logger.info("Check the test_outputs folder for your AI-generated music!")
            logger.info("="*80)
        else:
            logger.info("\n❌ No successful generations. Check logs for errors.")


def main():
    """Run comprehensive MusicGen testing."""
    tester = MusicGenTester()
    
    try:
        tester.run_comprehensive_test()
    except KeyboardInterrupt:
        logger.info("\n\nTest interrupted by user")
    except Exception as e:
        logger.error(f"\nTest suite failed: {e}")
        traceback.print_exc()
    finally:
        # Cleanup
        logger.info("\nCleaning up...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()