# Music Gen AI 🎵

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Code Style](https://img.shields.io/badge/Code%20Style-Black-black)](https://github.com/psf/black)

A production-ready text-to-music generation system that creates high-quality music from text prompts using state-of-the-art transformer architecture with EnCodec audio tokenization.

## 🚀 Features

### Core Capabilities
- **Text-to-Music Generation**: Generate music from natural language descriptions
- **Multiple Conditioning**: Support for text, genre, mood, tempo, and other musical attributes
- **High-Quality Audio**: Using EnCodec for efficient audio tokenization and reconstruction
- **Flexible Generation**: Multiple sampling strategies (nucleus, top-k, temperature control)
- **Progressive Training**: Scalable training from short to long sequences
- **Real-time Inference**: Fast generation suitable for interactive applications

### Technical Features
- **Transformer Architecture**: Modern attention-based generation model
- **T5 Text Encoding**: Powerful text understanding with cross-attention
- **PyTorch Lightning**: Professional training infrastructure
- **Hydra Configuration**: Flexible experiment management
- **WandB Integration**: Comprehensive experiment tracking
- **Distributed Training**: Multi-GPU and multi-node support
- **Production API**: FastAPI-based REST API for inference
- **Docker Support**: Containerized deployment

### Audio Capabilities
- **Multiple Formats**: WAV, MP3, FLAC export support
- **MIDI Export**: Musical notation export
- **Quality Metrics**: Automated audio quality evaluation
- **Iterative Refinement**: Progressive improvement of generated audio

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Text Input    │───▶│   T5 Encoder     │───▶│  Cross-Attention│
│ "Upbeat jazz"   │    │  (Text→Tokens)   │    │   Transformer   │
└─────────────────┘    └──────────────────┘    │                 │
                                               │                 │
┌─────────────────┐    ┌──────────────────┐    │                 │
│ Conditioning    │───▶│   Embeddings     │───▶│                 │
│ (Genre, Tempo)  │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    │                 │
                                               │                 │
┌─────────────────┐    ┌──────────────────┐    │                 │    ┌─────────────────┐
│  Audio Tokens   │───▶│ Position Encoding│───▶│                 │───▶│  EnCodec Decode │
│  (Previous)     │    │                  │    │                 │    │  (Tokens→Audio) │
└─────────────────┘    └──────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- 16GB+ RAM
- 10GB+ free disk space

### Quick Start
```bash
# Clone the repository
git clone https://github.com/Bright-L01/music-gen-ai.git
cd music-gen-ai

# Install dependencies
pip install -e .

# For development
pip install -e ".[dev]"

# For deployment
pip install -e ".[deployment]"
```

### Conda Environment
```bash
# Create environment
conda env create -f environment.yml
conda activate music-gen-ai

# Install package
pip install -e .
```

### Docker Setup
```bash
# Build image
docker build -t music-gen-ai .

# Run container
docker run -p 8000:8000 music-gen-ai
```

## 🎯 Quick Usage

### Command Line Interface
```bash
# Generate music from text
music-gen generate "Upbeat jazz with saxophone solo"

# Train a model
music-gen-train --config configs/training/default.yaml

# Start API server
music-gen-api --host 0.0.0.0 --port 8000
```

### Python API
```python
from music_gen import MusicGenerator

# Initialize generator
generator = MusicGenerator.from_pretrained("models/musicgen-base")

# Generate music
audio = generator.generate(
    prompt="Relaxing ambient music with nature sounds",
    duration=30.0,
    genre="ambient",
    tempo=60
)

# Save audio
generator.save_audio(audio, "output.wav")
```

### REST API
```bash
# Start server
uvicorn music_gen.api.main:app --reload

# Generate music
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Epic orchestral theme",
    "duration": 30,
    "genre": "orchestral",
    "tempo": 120
  }'
```

## 🔧 Configuration

The system uses Hydra for configuration management. Configurations are organized by component:

```
configs/
├── model/           # Model architectures
├── training/        # Training configurations
├── data/           # Dataset configurations
└── inference/      # Generation settings
```

### Example Configuration
```yaml
# configs/training/default.yaml
model:
  name: "musicgen_transformer"
  hidden_size: 768
  num_layers: 12
  num_heads: 12
  vocab_size: 2048

training:
  batch_size: 16
  learning_rate: 5e-4
  max_epochs: 100
  gradient_clip_val: 1.0

data:
  dataset: "musiccaps"
  sample_rate: 32000
  duration: 10.0
```

## 📊 Training

### Dataset Preparation
```bash
# Download and prepare datasets
python scripts/prepare_data.py --dataset musiccaps --output_dir data/

# Preprocess audio
python scripts/preprocess_audio.py --input_dir data/raw --output_dir data/processed
```

### Training Pipeline
```bash
# Single GPU training
music-gen-train --config configs/training/base.yaml

# Multi-GPU training
music-gen-train --config configs/training/base.yaml trainer.devices=4

# Resume from checkpoint
music-gen-train --config configs/training/base.yaml --resume_from checkpoints/last.ckpt
```

### Monitoring
Training metrics are automatically logged to:
- **WandB**: Real-time experiment tracking
- **TensorBoard**: Local visualization
- **Lightning Logs**: Detailed training logs

## 🎼 Model Details

### Architecture Components
- **Text Encoder**: T5-Base (220M parameters)
- **Audio Tokenizer**: EnCodec (24 kHz, 8 quantizers)
- **Generator**: Transformer decoder with cross-attention
- **Conditioning**: Multi-modal embedding layers

### Model Variants
- **MusicGen-Small**: 300M parameters, fast inference
- **MusicGen-Base**: 1.5B parameters, balanced quality/speed
- **MusicGen-Large**: 3.3B parameters, highest quality

### Performance Metrics
- **FAD**: Fréchet Audio Distance
- **CLAP Score**: Text-audio alignment
- **IS**: Inception Score for audio quality
- **Human Evaluation**: Musicality and coherence ratings

## 🔬 Evaluation

### Automated Metrics
```bash
# Evaluate model on test set
python scripts/evaluate.py --model_path checkpoints/best.ckpt --test_set data/test

# Generate evaluation report
python scripts/generate_report.py --eval_results outputs/evaluation.json
```

### Custom Evaluation
```python
from music_gen.evaluation import AudioEvaluator

evaluator = AudioEvaluator()
metrics = evaluator.evaluate_generation(
    generated_audio=audio,
    reference_text=prompt,
    reference_audio=ground_truth  # optional
)
```

## 🚀 Deployment

### Production API
```bash
# Production server with Gunicorn
gunicorn music_gen.api.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

### Docker Deployment
```bash
# Production build
docker build -t music-gen-ai:prod -f Dockerfile.prod .

# Deploy with Docker Compose
docker-compose up -d
```

### Kubernetes
```bash
# Deploy to Kubernetes
kubectl apply -f deployment/kubernetes/
```

## 📈 Monitoring

### Health Checks
- API health endpoint: `/health`
- Model readiness: `/ready`
- Metrics endpoint: `/metrics` (Prometheus format)

### Observability
- **Prometheus**: Metrics collection
- **Grafana**: Visualization dashboards
- **OpenTelemetry**: Distributed tracing
- **ELK Stack**: Log aggregation

## 🧪 Testing

```bash
# Run all tests
pytest

# Unit tests only
pytest tests/unit/

# Integration tests
pytest tests/integration/

# With coverage
pytest --cov=music_gen tests/
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Run pre-commit hooks: `pre-commit run --all-files`
5. Submit a pull request

### Development Setup
```bash
# Install pre-commit hooks
pre-commit install

# Install development dependencies
pip install -e ".[dev]"

# Run code formatting
black music_gen tests
isort music_gen tests

# Type checking
mypy music_gen
```

## 📖 Documentation

- [API Reference](docs/api.md)
- [Model Architecture](docs/architecture.md)
- [Training Guide](docs/training.md)
- [Deployment Guide](docs/deployment.md)
- [Contributing Guide](docs/contributing.md)

## 🎵 Examples

### Generated Samples
- [Jazz Examples](examples/jazz/)
- [Classical Examples](examples/classical/)
- [Electronic Examples](examples/electronic/)
- [Ambient Examples](examples/ambient/)

### Notebooks
- [Quick Start Tutorial](notebooks/01_quickstart.ipynb)
- [Training Custom Models](notebooks/02_training.ipynb)
- [Advanced Generation](notebooks/03_advanced_generation.ipynb)
- [Model Analysis](notebooks/04_model_analysis.ipynb)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Meta's MusicGen**: Inspiration for architecture design
- **Google's T5**: Text encoding capabilities
- **Meta's EnCodec**: Audio tokenization technology
- **Hugging Face**: Transformer implementations
- **PyTorch Lightning**: Training infrastructure

## 📬 Contact

- **Author**: Bright Liu
- **Email**: your.email@example.com
- **GitHub**: [@Bright-L01](https://github.com/Bright-L01)
- **Issues**: [GitHub Issues](https://github.com/Bright-L01/music-gen-ai/issues)

## 🏆 Citation

If you use this work in your research, please cite:

```bibtex
@software{music_gen_ai,
  title={Music Gen AI: Production-Ready Text-to-Music Generation},
  author={Liu, Bright},
  year={2024},
  url={https://github.com/Bright-L01/music-gen-ai}
}
```

---

**Status**: 🚧 Active Development | **Version**: 0.1.0 | **Last Updated**: 2024-06-26
