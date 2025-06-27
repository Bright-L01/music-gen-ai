"""
Hydra configuration structures for MusicGen.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    
    # Model size preset
    size: str = "base"  # small, base, large
    
    # Transformer architecture
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    intermediate_size: int = 3072
    vocab_size: int = 2048
    max_sequence_length: int = 8192
    
    # Attention configuration
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    layer_norm_eps: float = 1e-12
    use_cache: bool = True
    use_rotary_positional_encoding: bool = False
    use_scaled_dot_product_attention: bool = True
    
    # Cross-attention
    cross_attention_layers: Optional[List[int]] = None
    text_hidden_size: int = 768
    
    # Generation settings
    max_generation_length: int = 4096
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    
    # Memory optimization
    gradient_checkpointing: bool = False
    
    # Conditioning
    use_conditioning: bool = True
    conditioning_dim: int = 512


@dataclass
class AudioConfig:
    """Audio processing configuration."""
    
    # EnCodec settings
    model_name: str = "facebook/encodec_24khz"
    sample_rate: int = 24000
    num_quantizers: int = 8
    bandwidth: float = 6.0
    normalize: bool = True
    chunk_length: Optional[float] = None
    overlap: float = 0.01


@dataclass
class TextConfig:
    """Text processing configuration."""
    
    # T5 settings
    model_name: str = "t5-base"
    hidden_size: int = 768
    max_text_length: int = 512
    tokenizer_name: str = "t5-base"
    freeze_encoder: bool = True
    dropout: float = 0.1
    use_cache: bool = True
    cache_dir: Optional[str] = None


@dataclass
class ConditioningConfig:
    """Conditioning configuration."""
    
    # Genre conditioning
    use_genre: bool = True
    genre_vocab_size: int = 50
    genre_embedding_dim: int = 128
    
    # Mood conditioning
    use_mood: bool = True
    mood_vocab_size: int = 20
    mood_embedding_dim: int = 128
    
    # Tempo conditioning
    use_tempo: bool = True
    tempo_bins: int = 100
    tempo_min: int = 60
    tempo_max: int = 200
    tempo_embedding_dim: int = 128
    
    # Duration conditioning
    use_duration: bool = True
    duration_max: float = 120.0
    duration_embedding_dim: int = 128
    
    # Instrumentation
    use_instruments: bool = False
    instrument_vocab_size: int = 128
    instrument_embedding_dim: int = 128
    
    # Fusion settings
    conditioning_dropout: float = 0.1
    conditioning_fusion: str = "concat"


@dataclass
class DataConfig:
    """Data loading and processing configuration."""
    
    # Dataset settings
    dataset_name: str = "synthetic"
    data_dir: str = "data/"
    split: str = "train"
    
    # Data processing
    max_audio_length: float = 30.0
    sample_rate: int = 24000
    max_text_length: int = 512
    
    # Augmentation
    augment_audio: bool = True
    augment_text: bool = False
    
    # Caching
    cache_audio_tokens: bool = True
    cache_dir: Optional[str] = None
    
    # DataLoader settings
    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True
    shuffle: bool = True
    drop_last: bool = True
    
    # Conditioning vocabulary
    conditioning_vocab: Dict[str, Dict[str, int]] = field(default_factory=lambda: {
        "genre": {
            "jazz": 0, "classical": 1, "rock": 2, "electronic": 3,
            "ambient": 4, "folk": 5, "blues": 6, "country": 7,
            "reggae": 8, "hip-hop": 9, "pop": 10, "metal": 11,
            "orchestral": 12, "piano": 13, "acoustic": 14, "instrumental": 15,
        },
        "mood": {
            "happy": 0, "sad": 1, "energetic": 2, "calm": 3,
            "dramatic": 4, "peaceful": 5, "melancholic": 6, "uplifting": 7,
            "mysterious": 8, "romantic": 9, "epic": 10, "nostalgic": 11,
            "playful": 12, "intense": 13, "serene": 14, "triumphant": 15,
        }
    })


@dataclass
class TrainingConfig:
    """Training configuration."""
    
    # Optimization
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    
    # Learning rate scheduling
    warmup_steps: int = 5000
    max_steps: int = 100000
    lr_scheduler: str = "cosine"  # linear, cosine, constant
    min_lr_ratio: float = 0.1
    
    # Training dynamics
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    max_epochs: int = 100
    
    # Validation
    val_check_interval: float = 0.25
    check_val_every_n_epoch: int = 1
    
    # Logging
    log_every_n_steps: int = 100
    save_top_k: int = 3
    
    # Progressive training
    use_progressive_training: bool = True
    sequence_length_schedule: List[tuple] = field(default_factory=lambda: [
        (0, 256),      # Start with short sequences
        (5000, 512),   # Increase at 5k steps
        (15000, 1024), # Increase at 15k steps
        (30000, 2048), # Increase at 30k steps
        (50000, 4096), # Full length at 50k steps
    ])
    
    # Mixed precision
    use_mixed_precision: bool = True
    precision: str = "16-mixed"  # 16, 16-mixed, bf16, bf16-mixed, 32
    
    # Compilation
    compile_model: bool = False


@dataclass 
class InferenceConfig:
    """Inference configuration."""
    
    # Generation parameters
    max_length: int = 1024
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    do_sample: bool = True
    num_beams: int = 1
    repetition_penalty: float = 1.1
    
    # Audio generation
    default_duration: float = 10.0
    max_duration: float = 60.0
    
    # Sampling strategies
    use_nucleus_sampling: bool = True
    use_typical_sampling: bool = False
    typical_p: float = 0.95
    
    # Batch processing
    batch_size: int = 1
    
    # Quality settings
    sample_rate: int = 24000
    normalize_output: bool = True
    apply_fade: bool = True
    fade_duration: float = 0.01


@dataclass
class ExperimentConfig:
    """Experiment tracking configuration."""
    
    # Experiment metadata
    name: str = "musicgen_experiment"
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    
    # Logging
    log_model: bool = True
    log_gradients: bool = False
    log_parameters: bool = True
    log_audio_samples: bool = True
    
    # WandB settings
    project: str = "musicgen"
    entity: Optional[str] = None
    group: Optional[str] = None
    job_type: str = "training"
    
    # Audio sampling during training
    sample_generation_steps: List[int] = field(default_factory=lambda: [
        1000, 5000, 10000, 25000, 50000
    ])
    num_audio_samples: int = 4
    sample_prompts: List[str] = field(default_factory=lambda: [
        "Upbeat jazz with saxophone solo",
        "Relaxing ambient music with nature sounds", 
        "Epic orchestral theme with dramatic crescendo",
        "Electronic dance music with heavy bass",
    ])


@dataclass
class CheckpointConfig:
    """Checkpointing configuration."""
    
    # Checkpoint saving
    save_every_n_steps: int = 1000
    save_every_n_epochs: int = 1
    save_on_train_epoch_end: bool = True
    save_weights_only: bool = False
    
    # Checkpoint management
    dirpath: str = "checkpoints/"
    filename: str = "{epoch:02d}-{step:06d}-{val_loss:.3f}"
    monitor: str = "val_loss"
    mode: str = "min"
    save_top_k: int = 3
    save_last: bool = True
    
    # Resuming
    resume_from_checkpoint: Optional[str] = None
    auto_resume: bool = True


@dataclass
class MusicGenTrainingConfig:
    """Complete training configuration."""
    
    # Sub-configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    text: TextConfig = field(default_factory=TextConfig)
    conditioning: ConditioningConfig = field(default_factory=ConditioningConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    
    # Environment
    seed: int = 42
    device: str = "auto"  # auto, cpu, cuda, cuda:0, etc.
    
    # Paths
    output_dir: str = "outputs/"
    log_dir: str = "logs/"
    
    # Debug settings
    debug: bool = False
    fast_dev_run: bool = False
    limit_train_batches: Optional[float] = None
    limit_val_batches: Optional[float] = None


def register_configs():
    """Register all configurations with Hydra ConfigStore."""
    cs = ConfigStore.instance()
    
    # Register main config
    cs.store(name="config", node=MusicGenTrainingConfig)
    
    # Register component configs
    cs.store(group="model", name="base", node=ModelConfig)
    cs.store(group="data", name="base", node=DataConfig)
    cs.store(group="training", name="base", node=TrainingConfig)
    cs.store(group="inference", name="base", node=InferenceConfig)
    cs.store(group="experiment", name="base", node=ExperimentConfig)
    cs.store(group="checkpoint", name="base", node=CheckpointConfig)
    
    # Model size presets
    cs.store(group="model", name="small", node=ModelConfig(
        size="small",
        hidden_size=512,
        num_layers=8,
        num_heads=8,
        intermediate_size=2048,
    ))
    
    cs.store(group="model", name="large", node=ModelConfig(
        size="large", 
        hidden_size=1024,
        num_layers=24,
        num_heads=16,
        intermediate_size=4096,
    ))
    
    # Dataset presets
    cs.store(group="data", name="musiccaps", node=DataConfig(
        dataset_name="musiccaps",
        data_dir="data/musiccaps/",
        batch_size=16,
        max_audio_length=30.0,
    ))
    
    cs.store(group="data", name="fma", node=DataConfig(
        dataset_name="fma",
        data_dir="data/fma/",
        batch_size=8,
        max_audio_length=15.0,
    ))
    
    cs.store(group="data", name="synthetic", node=DataConfig(
        dataset_name="synthetic",
        batch_size=32,
        max_audio_length=10.0,
    ))
    
    # Training presets
    cs.store(group="training", name="debug", node=TrainingConfig(
        max_steps=100,
        val_check_interval=10,
        log_every_n_steps=5,
        use_progressive_training=False,
        compile_model=False,
    ))
    
    cs.store(group="training", name="fast", node=TrainingConfig(
        learning_rate=1e-3,
        max_steps=10000,
        warmup_steps=500,
        use_progressive_training=False,
    ))
    
    cs.store(group="training", name="production", node=TrainingConfig(
        learning_rate=5e-4,
        max_steps=200000,
        warmup_steps=10000,
        use_progressive_training=True,
        compile_model=True,
    ))


# Model size configurations
SMALL_MODEL = ModelConfig(
    size="small",
    hidden_size=512,
    num_layers=8,
    num_heads=8,
    intermediate_size=2048,
    max_sequence_length=4096,
)

BASE_MODEL = ModelConfig(
    size="base",
    hidden_size=768,
    num_layers=12,
    num_heads=12,
    intermediate_size=3072,
    max_sequence_length=8192,
)

LARGE_MODEL = ModelConfig(
    size="large",
    hidden_size=1024,
    num_layers=24,
    num_heads=16,
    intermediate_size=4096,
    max_sequence_length=8192,
)