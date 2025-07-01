#!/usr/bin/env python3
"""
Download pre-trained model weights for MusicGen.
"""

import logging
from pathlib import Path

import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configurations
MODEL_CONFIGS = {
    "musicgen-small": {
        "url": "https://huggingface.co/facebook/musicgen-small",
        "size": "1.2GB",
        "description": "Small model (300M params) - Fast, lower quality",
        "files": ["pytorch_model.bin", "config.json", "tokenizer_config.json"],
    },
    "musicgen-base": {
        "url": "https://huggingface.co/facebook/musicgen-medium",
        "size": "3.8GB",
        "description": "Base model (1.5B params) - Balanced speed/quality",
        "files": ["pytorch_model.bin", "config.json", "tokenizer_config.json"],
    },
    "encodec": {
        "url": "https://huggingface.co/facebook/encodec_24khz",
        "size": "200MB",
        "description": "EnCodec audio tokenizer",
        "files": ["pytorch_model.bin", "config.json"],
    },
    "t5-base": {
        "url": "https://huggingface.co/t5-base",
        "size": "850MB",
        "description": "T5 text encoder",
        "files": ["pytorch_model.bin", "config.json", "tokenizer_config.json", "spiece.model"],
    },
}


def check_disk_space(required_gb):
    """Check if enough disk space is available."""
    import shutil

    stat = shutil.disk_usage(".")
    available_gb = stat.free / (1024**3)

    if available_gb < required_gb:
        logger.warning(
            f"Low disk space: {available_gb:.1f}GB available, {required_gb:.1f}GB required"
        )
        return False
    return True


def download_file(url, dest_path, expected_size=None):
    """Download a file with progress bar."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with open(dest_path, "wb") as f:
            with tqdm(total=total_size, unit="iB", unit_scale=True, desc=dest_path.name) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    pbar.update(size)

        return True

    except Exception as e:
        logger.error(f"Download failed: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False


def download_from_huggingface(model_name, output_dir):
    """Download model from HuggingFace (mock implementation)."""
    logger.info(f"\nDownloading {model_name}...")

    config = MODEL_CONFIGS.get(model_name)
    if not config:
        logger.error(f"Unknown model: {model_name}")
        return False

    # Create model directory
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Model: {config['description']}")
    logger.info(f"Size: {config['size']}")
    logger.info(f"URL: {config['url']}")

    # In real implementation, this would download from HuggingFace
    # For now, create placeholder files
    logger.warning("⚠️  Real download not implemented - creating placeholders")

    for file_name in config["files"]:
        file_path = model_dir / file_name
        if file_path.exists():
            logger.info(f"✓ {file_name} already exists")
        else:
            # Create placeholder
            file_path.write_text(f"Placeholder for {file_name}\nModel: {model_name}")
            logger.info(f"✓ Created placeholder: {file_name}")

    # Create model info file
    info_path = model_dir / "MODEL_INFO.txt"
    info_path.write_text(
        f"""Model: {model_name}
Description: {config['description']}
Size: {config['size']}
Source: {config['url']}

Note: These are placeholder files. To download real models:
1. Install huggingface-cli: pip install huggingface-hub
2. Run: huggingface-cli download {config['url'].split('/')[-1]}
3. Or download manually from {config['url']}
"""
    )

    return True


def setup_model_directory():
    """Setup the models directory structure."""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # Create subdirectories
    subdirs = ["checkpoints", "configs", "cache"]
    for subdir in subdirs:
        (models_dir / subdir).mkdir(exist_ok=True)

    return models_dir


def verify_models(models_dir):
    """Verify downloaded models."""
    logger.info("\nVerifying models...")

    for model_name in MODEL_CONFIGS:
        model_path = models_dir / model_name
        if model_path.exists():
            files = list(model_path.glob("*"))
            logger.info(f"✓ {model_name}: {len(files)} files")
        else:
            logger.warning(f"✗ {model_name}: Not found")


def main():
    """Main download function."""
    print("=" * 60)
    print("MusicGen Model Downloader")
    print("=" * 60)

    # Check disk space
    required_space = 10.0  # GB
    if not check_disk_space(required_space):
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != "y":
            return

    # Setup directory
    models_dir = setup_model_directory()
    logger.info(f"Models directory: {models_dir.absolute()}")

    # Show available models
    print("\nAvailable models:")
    for i, (name, config) in enumerate(MODEL_CONFIGS.items(), 1):
        print(f"{i}. {name} ({config['size']}) - {config['description']}")

    print("\nOptions:")
    print("a. Download all models")
    print("s. Download small model only (recommended for testing)")
    print("c. Cancel")

    choice = input("\nYour choice: ").lower()

    if choice == "c":
        print("Cancelled")
        return
    elif choice == "a":
        models_to_download = list(MODEL_CONFIGS.keys())
    elif choice == "s":
        models_to_download = ["musicgen-small", "encodec", "t5-base"]
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(MODEL_CONFIGS):
                models_to_download = [list(MODEL_CONFIGS.keys())[idx]]
            else:
                print("Invalid choice")
                return
        except ValueError:
            print("Invalid choice")
            return

    # Download selected models
    logger.info(f"\nWill download: {', '.join(models_to_download)}")

    for model_name in models_to_download:
        success = download_from_huggingface(model_name, models_dir)
        if not success:
            logger.error(f"Failed to download {model_name}")

    # Verify
    verify_models(models_dir)

    print("\n" + "=" * 60)
    print("Download complete!")
    print("\nNext steps:")
    print("1. Replace placeholder files with real model weights")
    print("2. Run: python scripts/verify_core_functionality.py")
    print("3. Test generation: python scripts/test_generation.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
