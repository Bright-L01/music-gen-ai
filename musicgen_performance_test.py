#!/usr/bin/env python3
"""
MusicGen Performance Benchmark
Tests different model sizes and measures performance metrics.
"""

import os
import sys
import time
import json
import psutil
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

class PerformanceBenchmark:
    def __init__(self):
        self.results = []
        self.output_dir = Path("performance_results")
        self.output_dir.mkdir(exist_ok=True)
        
    def get_system_info(self):
        """Get system information."""
        info = {
            'cpu': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq().current if psutil.cpu_freq() else 0,
            'ram_gb': psutil.virtual_memory().total / 1024**3,
            'gpu': 'Available' if torch.cuda.is_available() else 'Not Available'
        }
        
        if torch.cuda.is_available():
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        return info
    
    def measure_performance(self, model_name, prompt, duration):
        """Measure generation performance."""
        from transformers import AutoProcessor, MusicgenForConditionalGeneration
        
        metrics = {
            'model': model_name,
            'prompt': prompt,
            'duration': duration,
            'load_time': 0,
            'generation_time': 0,
            'peak_memory_gb': 0,
            'audio_quality': {},
            'success': False,
            'error': None
        }
        
        try:
            # Monitor memory
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024**3
            
            # Load model
            print(f"\nLoading {model_name}...")
            load_start = time.time()
            
            processor = AutoProcessor.from_pretrained(
                f"facebook/musicgen-{model_name}",
                cache_dir="cache"
            )
            
            # Choose dtype based on available memory
            dtype = torch.float32
            if torch.cuda.is_available():
                free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
                if free_memory > 8 * 1024**3:  # 8GB free
                    dtype = torch.float16
            
            model = MusicgenForConditionalGeneration.from_pretrained(
                f"facebook/musicgen-{model_name}",
                cache_dir="cache",
                torch_dtype=dtype,
                low_cpu_mem_usage=True
            )
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            model.eval()
            
            load_time = time.time() - load_start
            metrics['load_time'] = load_time
            
            # Measure model size
            param_count = sum(p.numel() for p in model.parameters())
            metrics['parameters_millions'] = param_count / 1e6
            
            print(f"Model loaded in {load_time:.1f}s ({param_count/1e6:.0f}M params)")
            
            # Generate audio
            print(f"Generating {duration}s audio...")
            gen_start = time.time()
            
            inputs = processor(
                text=[prompt],
                padding=True,
                return_tensors="pt"
            ).to(device)
            
            tokens = min(int(duration * 50), 1500)
            
            with torch.no_grad():
                if device == "cuda":
                    with torch.cuda.amp.autocast():
                        audio_values = model.generate(
                            **inputs,
                            max_new_tokens=tokens,
                            do_sample=True,
                            temperature=1.0,
                            top_p=0.9
                        )
                else:
                    audio_values = model.generate(
                        **inputs,
                        max_new_tokens=tokens,
                        do_sample=True,
                        temperature=1.0,
                        top_p=0.9
                    )
            
            generation_time = time.time() - gen_start
            metrics['generation_time'] = generation_time
            
            # Extract audio
            audio = audio_values[0, 0].cpu().numpy()
            
            # Audio quality metrics
            metrics['audio_quality'] = {
                'samples': len(audio),
                'actual_duration': len(audio) / 32000,
                'rms': float(np.sqrt(np.mean(audio**2))),
                'peak': float(np.abs(audio).max()),
                'silence_ratio': float(np.sum(np.abs(audio) < 0.001) / len(audio)),
                'dynamic_range': float(np.abs(audio).max() - np.abs(audio).min())
            }
            
            # Memory usage
            peak_memory = process.memory_info().rss / 1024**3
            metrics['peak_memory_gb'] = peak_memory - initial_memory
            
            if device == "cuda":
                metrics['gpu_memory_gb'] = torch.cuda.max_memory_allocated() / 1024**3
                torch.cuda.reset_peak_memory_stats()
            
            # Performance ratios
            metrics['realtime_factor'] = duration / generation_time
            metrics['tokens_per_second'] = tokens / generation_time
            
            metrics['success'] = True
            
            print(f"✓ Generated in {generation_time:.1f}s ({metrics['realtime_factor']:.1f}x realtime)")
            print(f"  Memory used: {metrics['peak_memory_gb']:.1f}GB")
            
            # Cleanup
            del model
            del processor
            torch.cuda.empty_cache() if device == "cuda" else None
            
        except Exception as e:
            metrics['error'] = str(e)[:200]
            print(f"❌ Failed: {str(e)[:100]}")
        
        return metrics
    
    def run_benchmark(self):
        """Run comprehensive benchmark."""
        print("🎯 MusicGen Performance Benchmark")
        print("="*60)
        
        # System info
        system_info = self.get_system_info()
        print("\nSystem Information:")
        for key, value in system_info.items():
            print(f"  {key}: {value}")
        
        # Test configurations
        test_configs = [
            # (model_size, test_name, prompt, duration)
            ("small", "short", "simple piano melody", 5),
            ("small", "medium", "jazz quartet playing", 15),
            ("small", "long", "orchestral symphony", 30),
        ]
        
        # Add medium model tests if enough memory
        if system_info['ram_gb'] > 12:
            test_configs.extend([
                ("medium", "short", "electronic dance music", 5),
                ("medium", "medium", "rock band playing", 15),
            ])
        
        # Run benchmarks
        print(f"\nRunning {len(test_configs)} benchmark tests...")
        print("-"*60)
        
        for model_size, test_name, prompt, duration in test_configs:
            print(f"\nTest: {model_size}-{test_name}")
            
            result = self.measure_performance(model_size, prompt, duration)
            result['test_name'] = test_name
            self.results.append(result)
            
            # Save intermediate results
            self._save_results()
            
            # Delay between tests
            time.sleep(5)
            
            # Memory cleanup
            import gc
            gc.collect()
        
        # Generate report
        self._generate_report(system_info)
    
    def _save_results(self):
        """Save results to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"benchmark_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def _generate_report(self, system_info):
        """Generate performance report with visualizations."""
        print("\n" + "="*60)
        print("📊 PERFORMANCE REPORT")
        print("="*60)
        
        successful = [r for r in self.results if r['success']]
        
        if not successful:
            print("No successful benchmarks to report")
            return
        
        # Summary statistics
        print("\nGeneration Performance:")
        for r in successful:
            speed = r['realtime_factor']
            print(f"  {r['model']}-{r['test_name']}: {speed:.1f}x realtime, "
                  f"{r['generation_time']:.1f}s for {r['duration']}s audio")
        
        print("\nMemory Usage:")
        for r in successful:
            print(f"  {r['model']}: {r['peak_memory_gb']:.1f}GB RAM")
            if 'gpu_memory_gb' in r:
                print(f"    GPU: {r['gpu_memory_gb']:.1f}GB")
        
        # Create visualizations
        self._create_plots(successful)
        
        # Save detailed report
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_info': system_info,
            'results': self.results,
            'summary': {
                'total_tests': len(self.results),
                'successful': len(successful),
                'average_speed': np.mean([r['realtime_factor'] for r in successful])
            }
        }
        
        report_file = self.output_dir / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Report saved to: {report_file}")
        print(f"📊 Plots saved to: {self.output_dir}")
    
    def _create_plots(self, results):
        """Create performance visualization plots."""
        # Speed comparison
        plt.figure(figsize=(10, 6))
        
        models = list(set(r['model'] for r in results))
        durations = sorted(set(r['duration'] for r in results))
        
        for model in models:
            model_results = [r for r in results if r['model'] == model]
            x = [r['duration'] for r in model_results]
            y = [r['realtime_factor'] for r in model_results]
            plt.plot(x, y, marker='o', label=f"musicgen-{model}", linewidth=2)
        
        plt.xlabel('Audio Duration (seconds)')
        plt.ylabel('Generation Speed (x realtime)')
        plt.title('MusicGen Performance by Duration')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_dir / 'performance_speed.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Memory usage
        plt.figure(figsize=(10, 6))
        
        model_names = [f"{r['model']}-{r['test_name']}" for r in results]
        memory_usage = [r['peak_memory_gb'] for r in results]
        
        bars = plt.bar(range(len(model_names)), memory_usage)
        
        # Color by model
        colors = {'small': 'blue', 'medium': 'orange', 'large': 'red'}
        for i, r in enumerate(results):
            bars[i].set_color(colors.get(r['model'], 'gray'))
        
        plt.xlabel('Test Configuration')
        plt.ylabel('Memory Usage (GB)')
        plt.title('Memory Usage by Model and Test')
        plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'memory_usage.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("\n✓ Performance plots created")


def main():
    """Run performance benchmark."""
    # Check dependencies
    required = ['transformers', 'scipy', 'matplotlib', 'psutil']
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            print(f"Installing {pkg}...")
            os.system(f"{sys.executable} -m pip install {pkg}")
    
    # Run benchmark
    benchmark = PerformanceBenchmark()
    
    try:
        benchmark.run_benchmark()
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()