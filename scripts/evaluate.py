"""
Evaluation script for MusicGen models.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

from music_gen.data.datasets import create_dataloader, create_dataset
from music_gen.evaluation.metrics import AudioQualityMetrics
from music_gen.models.musicgen import MusicGenModel
from music_gen.utils.audio import save_audio_file

logger = logging.getLogger(__name__)


def evaluate_model(
    model_path: str,
    dataset_name: str = "synthetic",
    data_dir: str = "data/",
    num_samples: int = 100,
    output_dir: str = "evaluation_outputs/",
    device: str = "auto",
) -> Dict:
    """Evaluate model on dataset."""

    # Setup device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Load model
    logger.info(f"Loading model from {model_path}")
    model = MusicGenModel.from_pretrained(model_path)
    model = model.to(device)
    model.eval()

    # Create dataset
    logger.info(f"Creating {dataset_name} dataset")
    dataset = create_dataset(
        dataset_name=dataset_name,
        data_dir=data_dir,
        split="test",
        max_audio_length=10.0,  # Short clips for evaluation
    )

    # Limit samples if specified
    if num_samples > 0:
        dataset.metadata = dataset.metadata[:num_samples]

    # Create dataloader
    dataloader = create_dataloader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    # Initialize metrics
    metrics_evaluator = AudioQualityMetrics()

    # Evaluation loop
    logger.info("Starting evaluation...")
    generated_audio_list = []
    reference_audio_list = []
    prompts = []

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for i, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
        try:
            # Extract prompt
            prompt = batch["texts"][0] if "texts" in batch else f"Sample {i}"
            prompts.append(prompt)

            # Generate audio
            with torch.no_grad():
                generated_audio = model.generate_audio(
                    texts=[prompt],
                    duration=5.0,
                    temperature=0.9,
                    device=device,
                )

            # Convert to numpy
            if generated_audio.dim() > 2:
                generated_audio = generated_audio[0]  # Remove batch dim
            if generated_audio.dim() > 1:
                generated_audio = generated_audio.mean(dim=0)  # Convert to mono

            generated_np = generated_audio.cpu().numpy()
            generated_audio_list.append(generated_np)

            # Save generated audio
            save_path = output_path / f"generated_{i:03d}.wav"
            save_audio_file(
                generated_audio.unsqueeze(0),
                str(save_path),
                sample_rate=model.audio_tokenizer.sample_rate,
            )

            # Get reference audio if available
            if "audio_tokens" in batch:
                # This would require decoding the reference tokens
                # For now, we'll skip reference audio comparison
                pass

        except Exception as e:
            logger.warning(f"Failed to process sample {i}: {e}")
            continue

    # Compute overall metrics
    logger.info("Computing metrics...")
    overall_metrics = metrics_evaluator.evaluate_audio_quality(
        generated_audio=generated_audio_list,
        reference_audio=reference_audio_list if reference_audio_list else None,
    )

    # Additional metrics
    evaluation_results = {
        "model_path": model_path,
        "dataset": dataset_name,
        "num_samples_evaluated": len(generated_audio_list),
        "metrics": overall_metrics,
        "sample_prompts": prompts[:10],  # Save first 10 prompts as examples
    }

    # Save results
    results_path = output_path / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(evaluation_results, f, indent=2)

    logger.info(f"Evaluation complete. Results saved to {results_path}")

    return evaluation_results


def compare_models(
    model_paths: List[str],
    dataset_name: str = "synthetic",
    num_samples: int = 50,
    output_dir: str = "model_comparison/",
) -> Dict:
    """Compare multiple models on the same dataset."""

    comparison_results = {
        "models": {},
        "dataset": dataset_name,
        "num_samples": num_samples,
    }

    for model_path in model_paths:
        model_name = Path(model_path).stem
        logger.info(f"Evaluating model: {model_name}")

        try:
            results = evaluate_model(
                model_path=model_path,
                dataset_name=dataset_name,
                num_samples=num_samples,
                output_dir=f"{output_dir}/{model_name}/",
            )

            comparison_results["models"][model_name] = results["metrics"]

        except Exception as e:
            logger.error(f"Failed to evaluate {model_name}: {e}")
            comparison_results["models"][model_name] = {"error": str(e)}

    # Save comparison results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    comparison_path = output_path / "model_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(comparison_results, f, indent=2)

    logger.info(f"Model comparison complete. Results saved to {comparison_path}")

    return comparison_results


def benchmark_generation_speed(
    model_path: str,
    num_trials: int = 10,
    duration: float = 10.0,
    device: str = "auto",
) -> Dict:
    """Benchmark model generation speed."""

    # Setup device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Load model
    logger.info(f"Loading model from {model_path}")
    model = MusicGenModel.from_pretrained(model_path)
    model = model.to(device)
    model.eval()

    # Warmup
    logger.info("Warming up...")
    with torch.no_grad():
        for _ in range(3):
            model.generate_audio(
                texts=["Warmup generation"],
                duration=duration,
                device=device,
            )

    # Benchmark
    logger.info(f"Benchmarking generation speed ({num_trials} trials)")
    generation_times = []

    for i in tqdm(range(num_trials), desc="Benchmarking"):
        start_time = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        end_time = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None

        if device.type == "cuda":
            start_time.record()

        import time

        cpu_start = time.time()

        with torch.no_grad():
            audio = model.generate_audio(
                texts=[f"Benchmark generation {i}"],
                duration=duration,
                device=device,
            )

        cpu_end = time.time()

        if device.type == "cuda":
            end_time.record()
            torch.cuda.synchronize()
            gpu_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
            generation_times.append(gpu_time)
        else:
            generation_times.append(cpu_end - cpu_start)

    # Compute statistics
    mean_time = np.mean(generation_times)
    std_time = np.std(generation_times)
    min_time = np.min(generation_times)
    max_time = np.max(generation_times)

    rtf = duration / mean_time  # Real-time factor

    benchmark_results = {
        "model_path": model_path,
        "device": str(device),
        "duration": duration,
        "num_trials": num_trials,
        "generation_times": generation_times,
        "statistics": {
            "mean_time": mean_time,
            "std_time": std_time,
            "min_time": min_time,
            "max_time": max_time,
            "real_time_factor": rtf,
        },
        "throughput": {
            "audio_per_second": duration / mean_time,
            "generations_per_minute": 60.0 / mean_time,
        },
    }

    logger.info("Benchmark results:")
    logger.info(f"  Mean generation time: {mean_time:.3f}s ± {std_time:.3f}s")
    logger.info(f"  Real-time factor: {rtf:.2f}x")
    logger.info(f"  Throughput: {duration / mean_time:.2f} seconds of audio per second")

    return benchmark_results


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate MusicGen models")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to model checkpoint or saved model"
    )
    parser.add_argument("--dataset", type=str, default="synthetic", help="Dataset to evaluate on")
    parser.add_argument("--data_dir", type=str, default="data/", help="Data directory")
    parser.add_argument(
        "--num_samples", type=int, default=100, help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--output_dir", type=str, default="evaluation_outputs/", help="Output directory for results"
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to use (auto, cpu, cuda)"
    )
    parser.add_argument("--compare", nargs="+", type=str, help="Compare multiple models")
    parser.add_argument("--benchmark", action="store_true", help="Run generation speed benchmark")
    parser.add_argument(
        "--duration", type=float, default=10.0, help="Duration for benchmark generation"
    )
    parser.add_argument("--trials", type=int, default=10, help="Number of trials for benchmark")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    try:
        if args.compare:
            # Compare multiple models
            results = compare_models(
                model_paths=args.compare,
                dataset_name=args.dataset,
                num_samples=args.num_samples,
                output_dir=args.output_dir,
            )
        elif args.benchmark:
            # Benchmark generation speed
            results = benchmark_generation_speed(
                model_path=args.model_path,
                num_trials=args.trials,
                duration=args.duration,
                device=args.device,
            )
        else:
            # Evaluate single model
            results = evaluate_model(
                model_path=args.model_path,
                dataset_name=args.dataset,
                data_dir=args.data_dir,
                num_samples=args.num_samples,
                output_dir=args.output_dir,
                device=args.device,
            )

        # Print summary
        if "metrics" in results:
            print("\n=== Evaluation Summary ===")
            for metric, value in results["metrics"].items():
                print(f"{metric}: {value:.4f}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
