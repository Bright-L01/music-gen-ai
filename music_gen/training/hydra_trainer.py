"""
Hydra-based training script for MusicGen.
"""
import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar,
)
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
import torch
import wandb

from ..configs.config import MusicGenTrainingConfig, register_configs
from ..models.transformer.config import MusicGenConfig
from ..models.musicgen import MusicGenModel
from .lightning_module import MusicGenLightningModule, ProgressiveTrainingModule
from ..data.datasets import create_dataset, create_dataloader

logger = logging.getLogger(__name__)


def convert_hydra_to_model_config(hydra_cfg: DictConfig) -> MusicGenConfig:
    """Convert Hydra configuration to MusicGen model configuration."""
    
    # Create MusicGen config from Hydra config
    model_config = MusicGenConfig()
    
    # Update transformer config
    model_config.transformer.hidden_size = hydra_cfg.model.hidden_size
    model_config.transformer.num_layers = hydra_cfg.model.num_layers
    model_config.transformer.num_heads = hydra_cfg.model.num_heads
    model_config.transformer.intermediate_size = hydra_cfg.model.intermediate_size
    model_config.transformer.vocab_size = hydra_cfg.model.vocab_size
    model_config.transformer.max_sequence_length = hydra_cfg.model.max_sequence_length
    model_config.transformer.attention_dropout = hydra_cfg.model.attention_dropout
    model_config.transformer.hidden_dropout = hydra_cfg.model.hidden_dropout
    model_config.transformer.gradient_checkpointing = hydra_cfg.model.gradient_checkpointing
    model_config.transformer.use_conditioning = hydra_cfg.model.use_conditioning
    model_config.transformer.conditioning_dim = hydra_cfg.model.conditioning_dim
    
    # Update audio config
    model_config.encodec.model_name = hydra_cfg.audio.model_name
    model_config.encodec.sample_rate = hydra_cfg.audio.sample_rate
    model_config.encodec.num_quantizers = hydra_cfg.audio.num_quantizers
    model_config.encodec.bandwidth = hydra_cfg.audio.bandwidth
    
    # Update text config
    model_config.t5.model_name = hydra_cfg.text.model_name
    model_config.t5.max_text_length = hydra_cfg.text.max_text_length
    model_config.t5.freeze_encoder = hydra_cfg.text.freeze_encoder
    
    # Update conditioning config
    model_config.conditioning.use_genre = hydra_cfg.conditioning.use_genre
    model_config.conditioning.use_mood = hydra_cfg.conditioning.use_mood
    model_config.conditioning.use_tempo = hydra_cfg.conditioning.use_tempo
    model_config.conditioning.use_duration = hydra_cfg.conditioning.use_duration
    
    return model_config


def setup_logging(cfg: DictConfig) -> Optional[pl.loggers.Logger]:
    """Set up experiment logging."""
    
    loggers = []
    
    # WandB logger
    if cfg.experiment.project:
        try:
            wandb_logger = WandbLogger(
                project=cfg.experiment.project,
                entity=cfg.experiment.entity,
                name=cfg.experiment.name,
                tags=cfg.experiment.tags,
                notes=cfg.experiment.notes,
                group=cfg.experiment.group,
                job_type=cfg.experiment.job_type,
                log_model=cfg.experiment.log_model,
            )
            loggers.append(wandb_logger)
            logger.info("WandB logging enabled")
        except Exception as e:
            logger.warning(f"Failed to initialize WandB: {e}")
    
    # TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir=cfg.log_dir,
        name=cfg.experiment.name,
    )
    loggers.append(tb_logger)
    
    return loggers if loggers else None


def setup_callbacks(cfg: DictConfig) -> list:
    """Set up training callbacks."""
    
    callbacks = []
    
    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.checkpoint.dirpath,
        filename=cfg.checkpoint.filename,
        monitor=cfg.checkpoint.monitor,
        mode=cfg.checkpoint.mode,
        save_top_k=cfg.checkpoint.save_top_k,
        save_last=cfg.checkpoint.save_last,
        save_weights_only=cfg.checkpoint.save_weights_only,
        every_n_train_steps=cfg.checkpoint.save_every_n_steps,
        every_n_epochs=cfg.checkpoint.save_every_n_epochs,
        save_on_train_epoch_end=cfg.checkpoint.save_on_train_epoch_end,
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor=cfg.checkpoint.monitor,
        mode=cfg.checkpoint.mode,
        patience=10000,  # Very patient for music generation
        min_delta=0.001,
    )
    callbacks.append(early_stopping)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)
    
    # Rich progress bar
    if not cfg.debug:
        progress_bar = RichProgressBar()
        callbacks.append(progress_bar)
    
    return callbacks


def create_data_loaders(cfg: DictConfig) -> tuple:
    """Create training and validation data loaders."""
    
    # Training dataset
    train_dataset = create_dataset(
        dataset_name=cfg.data.dataset_name,
        data_dir=cfg.data.data_dir,
        split="train",
        max_audio_length=cfg.data.max_audio_length,
        sample_rate=cfg.data.sample_rate,
        max_text_length=cfg.data.max_text_length,
        conditioning_vocab=OmegaConf.to_container(cfg.data.conditioning_vocab),
        augment_audio=cfg.data.augment_audio,
        augment_text=cfg.data.augment_text,
        cache_audio_tokens=cfg.data.cache_audio_tokens,
    )
    
    # Validation dataset
    val_dataset = create_dataset(
        dataset_name=cfg.data.dataset_name,
        data_dir=cfg.data.data_dir,
        split="val",
        max_audio_length=cfg.data.max_audio_length,
        sample_rate=cfg.data.sample_rate,
        max_text_length=cfg.data.max_text_length,
        conditioning_vocab=OmegaConf.to_container(cfg.data.conditioning_vocab),
        augment_audio=False,  # No augmentation for validation
        augment_text=False,
        cache_audio_tokens=cfg.data.cache_audio_tokens,
    )
    
    # Create data loaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=cfg.data.shuffle,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=cfg.data.drop_last,
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=False,
    )
    
    return train_loader, val_loader


def create_lightning_module(cfg: DictConfig) -> pl.LightningModule:
    """Create PyTorch Lightning module."""
    
    # Convert config
    model_config = convert_hydra_to_model_config(cfg)
    
    # Choose module type
    if cfg.training.use_progressive_training:
        module = ProgressiveTrainingModule(
            config=model_config,
            learning_rate=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
            warmup_steps=cfg.training.warmup_steps,
            max_steps=cfg.training.max_steps,
            gradient_clip_val=cfg.training.gradient_clip_val,
            accumulate_grad_batches=cfg.training.accumulate_grad_batches,
            compile_model=cfg.training.compile_model,
            log_every_n_steps=cfg.training.log_every_n_steps,
            save_audio_samples=cfg.experiment.log_audio_samples,
            num_audio_samples=cfg.experiment.num_audio_samples,
            sample_generation_steps=cfg.experiment.sample_generation_steps,
            sequence_length_schedule=cfg.training.sequence_length_schedule,
        )
    else:
        module = MusicGenLightningModule(
            config=model_config,
            learning_rate=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
            warmup_steps=cfg.training.warmup_steps,
            max_steps=cfg.training.max_steps,
            gradient_clip_val=cfg.training.gradient_clip_val,
            accumulate_grad_batches=cfg.training.accumulate_grad_batches,
            compile_model=cfg.training.compile_model,
            log_every_n_steps=cfg.training.log_every_n_steps,
            save_audio_samples=cfg.experiment.log_audio_samples,
            num_audio_samples=cfg.experiment.num_audio_samples,
            sample_generation_steps=cfg.experiment.sample_generation_steps,
        )
    
    return module


def create_trainer(cfg: DictConfig) -> pl.Trainer:
    """Create PyTorch Lightning trainer."""
    
    # Device setup
    if cfg.device == "auto":
        if torch.cuda.is_available():
            accelerator = "gpu"
            devices = "auto"
        else:
            accelerator = "cpu"
            devices = 1
    elif cfg.device == "cpu":
        accelerator = "cpu"
        devices = 1
    elif cfg.device.startswith("cuda"):
        accelerator = "gpu"
        if ":" in cfg.device:
            devices = [int(cfg.device.split(":")[1])]
        else:
            devices = "auto"
    else:
        accelerator = "auto"
        devices = "auto"
    
    # Set up trainer
    trainer = pl.Trainer(
        # Hardware
        accelerator=accelerator,
        devices=devices,
        
        # Training control
        max_steps=cfg.training.max_steps,
        max_epochs=cfg.training.max_epochs,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        gradient_clip_val=cfg.training.gradient_clip_val,
        
        # Validation
        val_check_interval=cfg.training.val_check_interval,
        check_val_every_n_epoch=cfg.training.check_val_every_n_epoch,
        
        # Precision
        precision=cfg.training.precision if cfg.training.use_mixed_precision else 32,
        
        # Logging and callbacks
        logger=setup_logging(cfg),
        callbacks=setup_callbacks(cfg),
        log_every_n_steps=cfg.training.log_every_n_steps,
        
        # Debugging
        fast_dev_run=cfg.fast_dev_run,
        limit_train_batches=cfg.limit_train_batches,
        limit_val_batches=cfg.limit_val_batches,
        
        # Output
        default_root_dir=cfg.output_dir,
        
        # Reproducibility
        deterministic=True if cfg.seed else False,
        
        # Optimization
        enable_progress_bar=not cfg.debug,
        enable_model_summary=not cfg.debug,
    )
    
    return trainer


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def train(cfg: DictConfig) -> None:
    """Main training function."""
    
    # Register configurations
    register_configs()
    
    # Set random seed
    if cfg.seed:
        pl.seed_everything(cfg.seed, workers=True)
    
    # Print configuration
    logger.info("Training configuration:")
    logger.info(OmegaConf.to_yaml(cfg))
    
    # Create output directories
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.checkpoint.dirpath).mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_path = Path(cfg.output_dir) / "config.yaml"
    with open(config_path, "w") as f:
        OmegaConf.save(cfg, f)
    
    try:
        # Create data loaders
        logger.info("Creating data loaders...")
        train_loader, val_loader = create_data_loaders(cfg)
        logger.info(f"Train dataset size: {len(train_loader.dataset)}")
        logger.info(f"Validation dataset size: {len(val_loader.dataset)}")
        
        # Create Lightning module
        logger.info("Creating Lightning module...")
        module = create_lightning_module(cfg)
        
        # Create trainer
        logger.info("Creating trainer...")
        trainer = create_trainer(cfg)
        
        # Resume from checkpoint if specified
        ckpt_path = None
        if cfg.checkpoint.resume_from_checkpoint:
            ckpt_path = cfg.checkpoint.resume_from_checkpoint
            logger.info(f"Resuming from checkpoint: {ckpt_path}")
        elif cfg.checkpoint.auto_resume:
            # Auto-find last checkpoint
            checkpoint_dir = Path(cfg.checkpoint.dirpath)
            if checkpoint_dir.exists():
                checkpoints = list(checkpoint_dir.glob("*.ckpt"))
                if checkpoints:
                    ckpt_path = str(max(checkpoints, key=lambda x: x.stat().st_mtime))
                    logger.info(f"Auto-resuming from: {ckpt_path}")
        
        # Start training
        logger.info("Starting training...")
        trainer.fit(module, train_loader, val_loader, ckpt_path=ckpt_path)
        
        # Save final model
        final_model_path = Path(cfg.output_dir) / "final_model.ckpt"
        trainer.save_checkpoint(final_model_path)
        logger.info(f"Final model saved to: {final_model_path}")
        
        # Test if specified
        if hasattr(cfg, 'test') and cfg.test:
            logger.info("Running test...")
            trainer.test(module, val_loader)
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        # Cleanup
        if wandb.run:
            wandb.finish()


if __name__ == "__main__":
    train()