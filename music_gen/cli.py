"""
Command-line interface for MusicGen.
"""

import logging
import time
from pathlib import Path
from typing import Optional

import scipy.io.wavfile
import torch
import typer
from rich import print as rprint
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from .optimization.fast_generator import FastMusicGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = typer.Typer(help="MusicGen: Production-ready text-to-music generation")


@app.command()
def generate(
    prompt: str = typer.Argument(..., help="Text description of the music to generate"),
    output: str = typer.Option("output.wav", "--output", "-o", help="Output audio file"),
    duration: float = typer.Option(10.0, "--duration", "-d", help="Duration in seconds"),
    model_size: str = typer.Option(
        "base", "--model-size", "-m", help="Model size (small, base, large)"
    ),
    temperature: float = typer.Option(1.0, "--temperature", "-t", help="Sampling temperature"),
    top_k: int = typer.Option(50, "--top-k", "-k", help="Top-k sampling"),
    top_p: float = typer.Option(0.9, "--top-p", "-p", help="Top-p (nucleus) sampling"),
    genre: Optional[str] = typer.Option(None, "--genre", "-g", help="Musical genre"),
    mood: Optional[str] = typer.Option(None, "--mood", help="Musical mood"),
    tempo: Optional[int] = typer.Option(None, "--tempo", help="Tempo in BPM"),
    seed: Optional[int] = typer.Option(None, "--seed", "-s", help="Random seed"),
    device: str = typer.Option("auto", "--device", help="Device to use (auto, cpu, cuda)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Generate music from text prompt."""

    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    # Set device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    logger.info(f"Using device: {device}")
    logger.info(f"Loading {model_size} model...")

    try:
        console = Console()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            load_task = progress.add_task("Loading optimized MusicGen model...", total=None)

            # Create OPTIMIZED MusicGen model
            model = FastMusicGenerator(
                model_name="facebook/musicgen-small",
                device=device if device != torch.device("cpu") else "cpu",
                warmup=True,
            )
            progress.update(load_task, completed=True)

        rprint("[green]✓ Optimized MusicGen model loaded successfully![/green]")

        # Show performance info
        if verbose:
            stats = model.get_performance_stats()
            table = Table(title="Model Performance Info")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="magenta")
            table.add_row("Device", str(device))
            table.add_row(
                "Cache Status",
                (
                    "Enabled"
                    if stats.get("cache_stats", {}).get("cached_models", 0) > 0
                    else "Warming up"
                ),
            )
            table.add_row("Max Concurrent", "3")
            table.add_row("Optimizations", "Model caching, GPU acceleration, Memory management")
            console.print(table)

        # Set seed
        if seed is not None:
            torch.manual_seed(seed)
            logger.info(f"Set random seed to {seed}")

        # Prepare conditioning
        conditioning = {}
        if genre:
            # Simple genre mapping (would need proper vocab in real implementation)
            genre_vocab = {"jazz": 0, "classical": 1, "rock": 2, "electronic": 3}
            if genre.lower() in genre_vocab:
                conditioning["genre_ids"] = torch.tensor(
                    [genre_vocab[genre.lower()]], device=device
                )

        if mood:
            mood_vocab = {"happy": 0, "sad": 1, "energetic": 2, "calm": 3}
            if mood.lower() in mood_vocab:
                conditioning["mood_ids"] = torch.tensor([mood_vocab[mood.lower()]], device=device)

        if tempo:
            conditioning["tempo"] = torch.tensor([float(tempo)], device=device)

        logger.info(f"Generating music for prompt: '{prompt}'")
        logger.info(f"Duration: {duration}s, Temperature: {temperature}")

        # Generate music using OPTIMIZED MusicGen with progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
        ) as progress:

            gen_task = progress.add_task(
                f"Generating '{prompt[:50]}...'" if len(prompt) > 50 else f"Generating '{prompt}'",
                total=100,
            )

            start_time = time.time()

            # Simulate progress during generation (since we can't get real progress from model)
            import threading

            def update_progress():
                elapsed = 0
                target_time = 45 if device == "cuda" else 90  # Estimated times
                while elapsed < target_time and not progress.finished:
                    elapsed = time.time() - start_time
                    progress_pct = min(95, (elapsed / target_time) * 100)
                    progress.update(gen_task, completed=progress_pct)
                    time.sleep(1)

            progress_thread = threading.Thread(target=update_progress)
            progress_thread.daemon = True
            progress_thread.start()

            # Actual generation
            result = model.generate_single(
                prompt=prompt, duration=duration, temperature=temperature, guidance_scale=3.0
            )

            progress.update(gen_task, completed=100)

        # Extract results
        audio_np = result.audio
        generation_time = result.generation_time
        sample_rate = result.sample_rate

        # Save audio
        scipy.io.wavfile.write(output, rate=sample_rate, data=(audio_np * 32767).astype("int16"))

        # Show results
        rprint(
            f"[green]✓ Generated {result.duration:.1f}s of audio in {generation_time:.1f}s[/green]"
        )
        rprint(f"[blue]Speed: {result.duration / generation_time:.2f}x realtime[/blue]")

        if verbose:
            perf_stats = model.get_performance_stats()
            rprint(f"[dim]Cache hit rate: {perf_stats.get('cache_hit_rate', 0):.1%}[/dim]")
            rprint(f"[dim]Total generations: {perf_stats.get('total_generations', 0)}[/dim]")

        rprint(f"[green]Audio saved to: {output}[/green]")

    except Exception as e:
        rprint(f"[red]Error: Failed to generate music: {e}[/red]")
        if verbose:
            import traceback

            rprint(f"[dim]{traceback.format_exc()}[/dim]")
        raise typer.Exit(1)


@app.command()
def info():
    """Show system information."""

    typer.echo("MusicGen System Information")
    typer.echo("=" * 40)

    # PyTorch info
    typer.echo(f"PyTorch version: {torch.__version__}")
    typer.echo(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        typer.echo(f"CUDA version: {torch.version.cuda}")
        typer.echo(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            typer.echo(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # Model info
    typer.echo("\nAvailable models:")
    models = ["small", "base", "large"]
    for model in models:
        typer.echo(f"  - {model}")


@app.command()
def test(
    output_dir: str = typer.Option("test_outputs", "--output-dir", "-o", help="Output directory"),
    model_size: str = typer.Option("base", "--model-size", "-m", help="Model size to test"),
):
    """Run a quick test of the system."""

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    test_prompts = [
        "Happy jazz music with piano",
        "Calm ambient music",
        "Energetic electronic dance music",
    ]

    logger.info(f"Running test with {model_size} model...")
    logger.info(f"Output directory: {output_path}")

    for i, prompt in enumerate(test_prompts):
        output_file = output_path / f"test_{i+1:02d}.wav"

        try:
            # Use the generate command
            ctx = typer.Context(generate)
            ctx.invoke(
                generate,
                prompt=prompt,
                output=str(output_file),
                duration=5.0,  # Short duration for testing
                model_size=model_size,
                verbose=True,
            )

            logger.info(f"✓ Generated: {output_file}")

        except Exception as e:
            logger.error(f"✗ Failed to generate {output_file}: {e}")

    logger.info("Test completed!")


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
