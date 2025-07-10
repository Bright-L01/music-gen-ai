"""
Batch processing functionality for musicgen-minimal.
Handles CSV-based batch generation with multiprocessing support.
"""

import os
import csv
import time
import logging
from typing import List, Dict, Any, Optional, Callable
from multiprocessing import Pool, cpu_count
from pathlib import Path
import json

import pandas as pd
from rich.console import Console
from rich.progress import (
    Progress,
    TaskID,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from .generator import MusicGenerator

logger = logging.getLogger(__name__)
console = Console()


class BatchProcessor:
    """
    Handles batch music generation from CSV files with multiprocessing support.

    CSV Format Expected:
    - prompt: Text description of music to generate
    - duration: Duration in seconds (optional, default: 10.0)
    - output_file: Output filename (optional, auto-generated if missing)
    - temperature: Sampling temperature (optional, default: 1.0)
    - guidance_scale: Guidance scale (optional, default: 3.0)
    - model: Model size (optional, default: "small")
    - format: Output format "wav" or "mp3" (optional, default: "wav")
    - bitrate: MP3 bitrate "128k", "192k", "320k" (optional, default: "192k")
    """

    def __init__(
        self,
        model_name: str = "facebook/musicgen-small",
        output_dir: str = "batch_output",
        max_workers: Optional[int] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize batch processor.

        Args:
            model_name: HuggingFace model name
            output_dir: Directory to save generated files
            max_workers: Number of parallel workers (default: CPU count)
            device: Device to use (auto-detect if None)
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.max_workers = max_workers or min(
            cpu_count(), 4
        )  # Cap at 4 to avoid memory issues
        self.device = device

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Batch processor initialized: {self.max_workers} workers, output: {self.output_dir}"
        )

    def load_csv(self, csv_file: str) -> List[Dict[str, Any]]:
        """
        Load and validate CSV file.

        Args:
            csv_file: Path to CSV file

        Returns:
            List of dictionaries with generation parameters

        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If CSV format is invalid
        """
        csv_path = Path(csv_file)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_file}")

        # Load CSV with pandas for better error handling
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            raise ValueError(f"Failed to read CSV file: {e}")

        # Validate required columns
        required_cols = {"prompt"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise ValueError(f"CSV missing required columns: {missing}")

        # Convert to list of dictionaries and add defaults
        jobs = []
        for idx, row in df.iterrows():
            # Auto-detect format from output_file if not specified
            output_file = str(row.get("output_file", f"generated_{idx:03d}.wav"))
            if "format" not in row or pd.isna(row.get("format")):
                if output_file.lower().endswith(".mp3"):
                    default_format = "mp3"
                else:
                    default_format = "wav"
            else:
                default_format = "wav"

            job = {
                "id": idx,
                "prompt": str(row["prompt"]).strip(),
                "duration": float(row.get("duration", 10.0)),
                "output_file": output_file,
                "temperature": float(row.get("temperature", 1.0)),
                "guidance_scale": float(row.get("guidance_scale", 3.0)),
                "model": str(row.get("model", "small")),
                "format": str(row.get("format", default_format)).lower(),
                "bitrate": str(row.get("bitrate", "192k")),
            }

            # Validate job parameters
            if not job["prompt"]:
                logger.warning(f"Row {idx}: Empty prompt, skipping")
                continue

            if not (0.1 <= job["duration"] <= 120.0):
                logger.warning(
                    f"Row {idx}: Invalid duration {job['duration']}, using 10.0"
                )
                job["duration"] = 10.0

            if not (0.1 <= job["temperature"] <= 2.0):
                logger.warning(
                    f"Row {idx}: Invalid temperature {job['temperature']}, using 1.0"
                )
                job["temperature"] = 1.0

            if not (1.0 <= job["guidance_scale"] <= 10.0):
                logger.warning(
                    f"Row {idx}: Invalid guidance_scale {job['guidance_scale']}, using 3.0"
                )
                job["guidance_scale"] = 3.0

            # Validate format
            if job["format"] not in ["wav", "mp3"]:
                logger.warning(
                    f"Row {idx}: Invalid format '{job['format']}', using 'wav'"
                )
                job["format"] = "wav"

            # Validate bitrate for MP3
            valid_bitrates = ["128k", "192k", "256k", "320k"]
            if job["format"] == "mp3" and job["bitrate"] not in valid_bitrates:
                logger.warning(
                    f"Row {idx}: Invalid bitrate '{job['bitrate']}', using '192k'"
                )
                job["bitrate"] = "192k"

            # Ensure output file has correct extension
            base_name = (
                job["output_file"].rsplit(".", 1)[0]
                if "." in job["output_file"]
                else job["output_file"]
            )
            if job["format"] == "wav" and not job["output_file"].lower().endswith(
                ".wav"
            ):
                job["output_file"] = base_name + ".wav"
            elif job["format"] == "mp3" and not job["output_file"].lower().endswith(
                ".mp3"
            ):
                job["output_file"] = base_name + ".mp3"

            # Make output file path absolute
            job["output_file"] = str(self.output_dir / job["output_file"])

            jobs.append(job)

        logger.info(f"Loaded {len(jobs)} valid jobs from {csv_file}")
        return jobs

    def generate_single(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a single music file (worker function).

        Args:
            job: Dictionary with generation parameters

        Returns:
            Dictionary with job results
        """
        start_time = time.time()
        result = {
            "id": job["id"],
            "prompt": job["prompt"],
            "output_file": job["output_file"],
            "success": False,
            "error": None,
            "duration": job["duration"],
            "generation_time": 0.0,
            "file_size": 0,
        }

        try:
            # Map model name
            model_names = {
                "small": "facebook/musicgen-small",
                "medium": "facebook/musicgen-medium",
                "large": "facebook/musicgen-large",
            }
            model_name = model_names.get(job["model"], "facebook/musicgen-small")

            # Create generator (each worker gets its own)
            generator = MusicGenerator(model_name=model_name, device=self.device)

            # Generate audio
            if job["duration"] > 30:
                audio, sample_rate = generator.generate_extended(
                    prompt=job["prompt"],
                    duration=job["duration"],
                    temperature=job["temperature"],
                    guidance_scale=job["guidance_scale"],
                )
            else:
                audio, sample_rate = generator.generate(
                    prompt=job["prompt"],
                    duration=job["duration"],
                    temperature=job["temperature"],
                    guidance_scale=job["guidance_scale"],
                )

            # Save audio in specified format
            final_output = generator.save_audio_as_format(
                audio,
                sample_rate,
                job["output_file"],
                format=job["format"],
                bitrate=job["bitrate"],
                delete_wav=(job["format"] == "mp3"),
            )

            # Update output file path in case it changed during conversion
            job["output_file"] = final_output

            # Update result
            result["success"] = True
            result["generation_time"] = time.time() - start_time
            result["file_size"] = os.path.getsize(job["output_file"])

        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Job {job['id']} failed: {e}")

        return result

    def process_batch(
        self,
        jobs: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of jobs using multiprocessing.

        Args:
            jobs: List of job dictionaries
            progress_callback: Optional progress callback function

        Returns:
            List of result dictionaries
        """
        if not jobs:
            return []

        logger.info(
            f"Starting batch processing: {len(jobs)} jobs with {self.max_workers} workers"
        )

        # For small batches, use single process to avoid overhead
        if len(jobs) <= 2 or self.max_workers == 1:
            results = []
            for i, job in enumerate(jobs):
                if progress_callback:
                    progress_callback(i, len(jobs), f"Processing job {i+1}/{len(jobs)}")
                result = self.generate_single(job)
                results.append(result)
            return results

        # Use multiprocessing for larger batches
        with Pool(processes=self.max_workers) as pool:
            # Submit all jobs
            async_results = []
            for job in jobs:
                async_result = pool.apply_async(self.generate_single, (job,))
                async_results.append(async_result)

            # Collect results with progress tracking
            results = []
            for i, async_result in enumerate(async_results):
                if progress_callback:
                    progress_callback(i, len(jobs), f"Processing job {i+1}/{len(jobs)}")

                try:
                    result = async_result.get(timeout=300)  # 5 minute timeout per job
                    results.append(result)
                except Exception as e:
                    # Create error result for timeout/failed jobs
                    error_result = {
                        "id": jobs[i]["id"],
                        "prompt": jobs[i]["prompt"],
                        "output_file": jobs[i]["output_file"],
                        "success": False,
                        "error": f"Worker timeout or error: {e}",
                        "duration": jobs[i]["duration"],
                        "generation_time": 0.0,
                        "file_size": 0,
                    }
                    results.append(error_result)
                    logger.error(f"Job {jobs[i]['id']} failed with timeout/error: {e}")

        return results

    def save_results(self, results: List[Dict[str, Any]], output_file: str):
        """
        Save batch results to JSON file.

        Args:
            results: List of result dictionaries
            output_file: Path to save results
        """
        results_path = self.output_dir / output_file

        # Calculate summary statistics
        total_jobs = len(results)
        successful_jobs = sum(1 for r in results if r["success"])
        failed_jobs = total_jobs - successful_jobs
        total_duration = sum(r["duration"] for r in results if r["success"])
        total_generation_time = sum(
            r["generation_time"] for r in results if r["success"]
        )
        total_file_size = sum(r["file_size"] for r in results if r["success"])

        summary = {
            "total_jobs": total_jobs,
            "successful_jobs": successful_jobs,
            "failed_jobs": failed_jobs,
            "success_rate": successful_jobs / total_jobs if total_jobs > 0 else 0,
            "total_audio_duration": total_duration,
            "total_generation_time": total_generation_time,
            "average_speed": total_duration / total_generation_time
            if total_generation_time > 0
            else 0,
            "total_file_size": total_file_size,
            "timestamp": time.time(),
        }

        output_data = {"summary": summary, "results": results}

        with open(results_path, "w") as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"Results saved to {results_path}")

    def print_summary(self, results: List[Dict[str, Any]]):
        """
        Print a summary table of batch results.

        Args:
            results: List of result dictionaries
        """
        # Create summary table
        table = Table(title="Batch Processing Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        total_jobs = len(results)
        successful_jobs = sum(1 for r in results if r["success"])
        failed_jobs = total_jobs - successful_jobs

        table.add_row("Total Jobs", str(total_jobs))
        table.add_row("Successful", str(successful_jobs))
        table.add_row("Failed", str(failed_jobs))
        table.add_row(
            "Success Rate",
            f"{successful_jobs/total_jobs*100:.1f}%" if total_jobs > 0 else "0%",
        )

        if successful_jobs > 0:
            total_duration = sum(r["duration"] for r in results if r["success"])
            total_generation_time = sum(
                r["generation_time"] for r in results if r["success"]
            )
            total_file_size = sum(r["file_size"] for r in results if r["success"])

            table.add_row("Total Audio Duration", f"{total_duration:.1f}s")
            table.add_row("Total Generation Time", f"{total_generation_time:.1f}s")
            table.add_row(
                "Average Speed", f"{total_duration/total_generation_time:.2f}x realtime"
            )
            table.add_row("Total File Size", f"{total_file_size/1024/1024:.1f} MB")

        console.print(table)

        # Show failed jobs if any
        if failed_jobs > 0:
            console.print("\n[red]Failed Jobs:[/red]")
            for result in results:
                if not result["success"]:
                    console.print(f"  • Job {result['id']}: {result['error']}")


def create_sample_csv(filename: str = "sample_batch.csv"):
    """
    Create a sample CSV file for batch processing.

    Args:
        filename: Name of the CSV file to create
    """
    sample_data = [
        {
            "prompt": "upbeat jazz piano with walking bass",
            "duration": 30,
            "output_file": "jazz_piano.wav",
            "temperature": 1.0,
            "guidance_scale": 3.0,
            "model": "small",
            "format": "wav",
            "bitrate": "192k",
        },
        {
            "prompt": "peaceful ambient soundscape with nature sounds",
            "duration": 45,
            "output_file": "ambient_nature.mp3",
            "temperature": 0.8,
            "guidance_scale": 4.0,
            "model": "medium",
            "format": "mp3",
            "bitrate": "192k",
        },
        {
            "prompt": "energetic electronic dance music",
            "duration": 60,
            "output_file": "edm_track.mp3",
            "temperature": 1.2,
            "guidance_scale": 3.5,
            "model": "small",
            "format": "mp3",
            "bitrate": "320k",
        },
        {
            "prompt": "classical orchestral piece with strings",
            "duration": 90,
            "output_file": "orchestral.wav",
            "temperature": 0.9,
            "guidance_scale": 4.5,
            "model": "large",
            "format": "wav",
            "bitrate": "192k",
        },
    ]

    with open(filename, "w", newline="") as csvfile:
        fieldnames = [
            "prompt",
            "duration",
            "output_file",
            "temperature",
            "guidance_scale",
            "model",
            "format",
            "bitrate",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sample_data)

    console.print(f"[green]✓ Sample CSV created: {filename}[/green]")
    console.print("Edit this file with your music generation tasks, then run:")
    console.print(f"[dim]musicgen batch {filename}[/dim]")
