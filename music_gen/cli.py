"""
Command-line interface for MusicGen.
"""
import typer
import torch
from pathlib import Path
from typing import Optional, List
import logging

from .models.musicgen import create_musicgen_model
from .utils.audio import save_audio_file

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = typer.Typer(help="MusicGen: Production-ready text-to-music generation")


@app.command()
def generate(
    prompt: str = typer.Argument(..., help="Text description of the music to generate"),
    output: str = typer.Option("output.wav", "--output", "-o", help="Output audio file"),
    duration: float = typer.Option(10.0, "--duration", "-d", help="Duration in seconds"),
    model_size: str = typer.Option("base", "--model-size", "-m", help="Model size (small, base, large)"),
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
        # Create model
        model = create_musicgen_model(model_size)
        model = model.to(device)
        model.eval()
        
        logger.info("Model loaded successfully")
        
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
                conditioning["genre_ids"] = torch.tensor([genre_vocab[genre.lower()]], device=device)
        
        if mood:
            mood_vocab = {"happy": 0, "sad": 1, "energetic": 2, "calm": 3}
            if mood.lower() in mood_vocab:
                conditioning["mood_ids"] = torch.tensor([mood_vocab[mood.lower()]], device=device)
        
        if tempo:
            conditioning["tempo"] = torch.tensor([float(tempo)], device=device)
        
        logger.info(f"Generating music for prompt: '{prompt}'")
        logger.info(f"Duration: {duration}s, Temperature: {temperature}")
        
        # Generate music
        with torch.no_grad():
            audio = model.generate_audio(
                texts=[prompt],
                duration=duration,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                **conditioning,
            )
        
        # Save audio
        if audio.dim() > 2:
            audio = audio[0]  # Take first batch item
        
        save_audio_file(
            audio,
            output,
            sample_rate=model.audio_tokenizer.sample_rate,
        )
        
        logger.info(f"Generated audio saved to: {output}")
        
    except Exception as e:
        logger.error(f"Failed to generate music: {e}")
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