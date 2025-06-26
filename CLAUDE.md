# Music Gen AI - Claude Development Context

This file contains important context and commands for Claude Code when working on the Music Gen AI project.

## 🎯 Project Overview

Music Gen AI is a production-ready text-to-music generation system that creates high-quality music from text prompts using transformer architecture with EnCodec audio tokenization.

**Key Technologies:**
- PyTorch & PyTorch Lightning for ML infrastructure
- Transformer architecture with T5 text encoding
- EnCodec for audio tokenization
- Hydra for configuration management
- FastAPI for REST API
- WandB for experiment tracking

**Architecture:**
Text Input → T5 Encoder → Cross-Attention Transformer → Audio Tokens → EnCodec Decoder → Audio Output

## 📁 Project Structure

```
music_gen/
├── models/                 # Core model implementations
│   ├── transformer/       # Transformer architecture
│   ├── encodec/           # Audio tokenization
│   └── conditioning/      # Conditioning modules
├── data/                  # Data processing and loading
│   ├── datasets/          # Dataset implementations
│   ├── preprocessing/     # Audio/text preprocessing
│   └── loaders/          # DataLoader implementations
├── training/              # Training infrastructure
│   ├── trainers/         # Lightning trainers
│   ├── callbacks/        # Training callbacks
│   └── schedulers/       # Learning rate schedulers
├── inference/             # Generation and sampling
│   ├── generators/       # Music generation logic
│   └── strategies/       # Sampling strategies
├── evaluation/            # Metrics and evaluation
│   ├── metrics/          # Audio quality metrics
│   └── validators/       # Model validation
├── utils/                 # Utility functions
│   ├── audio/            # Audio processing utilities
│   ├── text/             # Text processing utilities
│   └── io/               # Input/output utilities
├── configs/               # Configuration files
│   ├── model/            # Model configurations
│   ├── training/         # Training configurations
│   └── data/             # Data configurations
├── api/                   # REST API implementation
│   ├── routes/           # API endpoints
│   ├── schemas/          # Pydantic schemas
│   └── middleware/       # API middleware
└── deployment/            # Deployment configurations
    ├── docker/           # Docker files
    └── scripts/          # Deployment scripts
```

## 🔧 Development Commands

### Setup and Installation
```bash
# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Create conda environment
conda env create -f environment.yml
conda activate music-gen-ai
```

### Code Quality
```bash
# Format code
black music_gen tests scripts
isort music_gen tests scripts

# Lint code
flake8 music_gen tests scripts

# Type checking
mypy music_gen

# Run all quality checks
pre-commit run --all-files
```

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=music_gen

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m slow          # Slow tests only
pytest -m gpu           # GPU-required tests only

# Run specific test file
pytest tests/test_models.py

# Run with verbose output
pytest -v -s
```

### Training
```bash
# Basic training
music-gen-train --config configs/training/default.yaml

# Multi-GPU training
music-gen-train --config configs/training/default.yaml trainer.devices=4

# Resume from checkpoint
music-gen-train --config configs/training/default.yaml --resume_from checkpoints/last.ckpt

# Override config values
music-gen-train --config configs/training/default.yaml training.batch_size=32 model.hidden_size=1024

# Training with specific experiment name
music-gen-train --config configs/training/default.yaml experiment_name=my_experiment
```

### Inference
```bash
# Generate music from text
music-gen generate "Upbeat jazz with saxophone solo" --duration 30 --output output.wav

# Generate with specific model
music-gen generate "Classical piano piece" --model_path checkpoints/best.ckpt

# Batch generation
music-gen generate-batch --prompts prompts.txt --output_dir outputs/

# Interactive generation
music-gen interactive
```

### API Development
```bash
# Start development server
uvicorn music_gen.api.main:app --reload --host 0.0.0.0 --port 8000

# Start with specific workers
music-gen-api --workers 4 --host 0.0.0.0 --port 8000

# Test API endpoints
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Upbeat jazz", "duration": 30}'
```

### Data Processing
```bash
# Prepare datasets
python scripts/prepare_data.py --dataset musiccaps --output_dir data/

# Preprocess audio files
python scripts/preprocess_audio.py --input_dir data/raw --output_dir data/processed

# Validate dataset
python scripts/validate_dataset.py --dataset_path data/processed
```

### Model Operations
```bash
# Convert model to different formats
python scripts/convert_model.py --input_path checkpoints/best.ckpt --output_format onnx

# Model inference benchmark
python scripts/benchmark_model.py --model_path checkpoints/best.ckpt

# Export model for deployment
python scripts/export_model.py --model_path checkpoints/best.ckpt --output_dir models/
```

### Evaluation
```bash
# Evaluate model on test set
python scripts/evaluate.py --model_path checkpoints/best.ckpt --test_set data/test

# Generate evaluation report
python scripts/generate_report.py --eval_results outputs/evaluation.json

# Compare models
python scripts/compare_models.py --models checkpoints/model1.ckpt checkpoints/model2.ckpt
```

### Docker Operations
```bash
# Build development image
docker build -t music-gen-ai:dev .

# Build production image
docker build -t music-gen-ai:prod -f Dockerfile.prod .

# Run container with GPU support
docker run --gpus all -p 8000:8000 music-gen-ai:prod

# Docker Compose for full stack
docker-compose up -d
```

## 🧠 Technical Implementation Notes

### Model Architecture Details
- **Text Encoder**: T5-Base (220M parameters) for robust text understanding
- **Audio Tokenizer**: EnCodec with 24kHz sampling rate, 8 quantization levels
- **Generator**: Transformer decoder with cross-attention to text embeddings
- **Conditioning**: Multi-modal embeddings for genre, mood, tempo, etc.

### Key Design Decisions
1. **Progressive Training**: Start with short sequences (10s) and gradually increase to long sequences (60s+)
2. **Memory Optimization**: Gradient checkpointing, mixed precision training, sequence packing
3. **Distributed Training**: Support for multi-GPU and multi-node training via PyTorch Lightning
4. **Configuration Management**: Hydra for flexible experiment configuration
5. **Monitoring**: WandB integration for comprehensive experiment tracking

### Performance Considerations
- **Inference Speed**: Optimized for real-time generation with proper batching
- **Memory Usage**: Efficient attention patterns and gradient checkpointing
- **Audio Quality**: High-quality reconstruction with EnCodec
- **Scalability**: Modular design for easy scaling and modification

### Error Handling Patterns
- Graceful degradation for missing conditioning inputs
- Robust audio file handling with format validation
- Comprehensive logging with structured error messages
- Recovery mechanisms for interrupted training

## 🚨 Common Issues and Solutions

### Training Issues
```bash
# Out of memory error
# Solution: Reduce batch_size, enable gradient_checkpointing, use mixed precision
music-gen-train --config configs/training/default.yaml training.batch_size=8 model.gradient_checkpointing=true trainer.precision=16

# Diverging loss
# Solution: Lower learning rate, increase warmup steps, check data quality
music-gen-train --config configs/training/default.yaml training.learning_rate=1e-4 training.warmup_steps=10000
```

### Inference Issues
```bash
# Slow generation
# Solution: Use smaller model, enable compilation, batch processing
music-gen generate "prompt" --model_size small --compile true

# Poor quality output
# Solution: Check model checkpoint, verify conditioning inputs, adjust sampling parameters
music-gen generate "prompt" --temperature 0.8 --top_k 50 --top_p 0.9
```

### Data Issues
```bash
# Corrupted audio files
# Solution: Validate and clean dataset
python scripts/clean_dataset.py --input_dir data/raw --output_dir data/clean

# Inconsistent audio format
# Solution: Standardize audio preprocessing
python scripts/standardize_audio.py --input_dir data/raw --target_sr 32000 --target_format wav
```

## 📊 Monitoring and Debugging

### Key Metrics to Track
- **Training**: Loss curves, learning rate, gradient norms
- **Audio Quality**: FAD, CLAP score, Inception Score
- **Performance**: Inference latency, memory usage, throughput
- **System**: GPU utilization, disk I/O, network bandwidth

### Debug Commands
```bash
# Profile model performance
python -m torch.profiler scripts/profile_model.py --model_path checkpoints/best.ckpt

# Debug data loading
python scripts/debug_dataloader.py --config configs/data/default.yaml

# Validate model outputs
python scripts/validate_outputs.py --model_path checkpoints/best.ckpt --test_prompts test_prompts.txt
```

### Log Analysis
```bash
# View training logs
tail -f logs/training.log

# Search for specific errors
grep "ERROR" logs/*.log

# Monitor GPU usage
nvidia-smi -l 1

# Monitor system resources
htop
```

## 🔄 Development Workflow

### Feature Development
1. Create feature branch: `git checkout -b feature/new-feature`
2. Implement changes with tests
3. Run quality checks: `pre-commit run --all-files`
4. Test locally: `pytest tests/`
5. Update documentation if needed
6. Submit PR with detailed description

### Model Experimentation
1. Create new config: `configs/training/experiment_name.yaml`
2. Run training: `music-gen-train --config configs/training/experiment_name.yaml`
3. Monitor with WandB: Check training metrics and losses
4. Evaluate results: `python scripts/evaluate.py --model_path checkpoints/best.ckpt`
5. Compare with baselines: `python scripts/compare_models.py`

### Production Deployment
1. Test thoroughly: `pytest tests/`
2. Build production image: `docker build -t music-gen-ai:prod -f Dockerfile.prod .`
3. Deploy to staging: Test API endpoints and performance
4. Monitor metrics: Check system health and model performance
5. Deploy to production: Gradual rollout with monitoring

## 📚 Important Resources

### Documentation
- [PyTorch Lightning Docs](https://pytorch-lightning.readthedocs.io/)
- [Hydra Configuration](https://hydra.cc/docs/intro/)
- [WandB Integration](https://docs.wandb.ai/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

### Research Papers
- "MusicGen: Simple and Controllable Music Generation" (Meta AI)
- "AudioLM: a Language Modeling Approach to Audio Generation" (Google)
- "EnCodec: High Fidelity Neural Audio Compression" (Meta AI)
- "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (T5)

### Model Checkpoints
- Production models: `models/production/`
- Experimental models: `models/experimental/`
- Baseline models: `models/baselines/`

## 🎵 Audio Processing Notes

### Supported Audio Formats
- Input: WAV, MP3, FLAC, OGG (auto-conversion to 32kHz WAV)
- Output: WAV (primary), MP3, FLAC (configurable)
- MIDI: Export support for musical notation

### Audio Quality Settings
- Sample Rate: 32kHz (configurable: 16kHz, 24kHz, 48kHz)
- Bit Depth: 16-bit (configurable: 24-bit, 32-bit float)
- Channels: Mono (primary), Stereo (experimental)

### EnCodec Configuration
- Quantizers: 8 levels (configurable: 4, 8, 16)
- Bandwidth: 6 kbps (configurable: 1.5, 3, 6, 12, 24)
- Model: encodec_24khz (alternatives: encodec_32khz, musicgen_decoder)

---

This context file should help Claude understand the project structure, commands, and development patterns for efficient collaboration on the Music Gen AI project.