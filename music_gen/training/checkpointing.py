"""
Advanced checkpointing and model saving utilities.
"""
import os
import json
import torch
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime
import pytorch_lightning as pl

from ..models.musicgen import MusicGenModel
from ..models.transformer.config import MusicGenConfig

logger = logging.getLogger(__name__)


class ModelCheckpointManager:
    """Advanced model checkpoint management."""
    
    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        save_top_k: int = 3,
        monitor: str = "val_loss",
        mode: str = "min",
        save_last: bool = True,
        save_weights_only: bool = False,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_top_k = save_top_k
        self.monitor = monitor
        self.mode = mode
        self.save_last = save_last
        self.save_weights_only = save_weights_only
        
        # Track best checkpoints
        self.best_checkpoints = []
        self.best_scores = []
        
        # Metadata file
        self.metadata_file = self.checkpoint_dir / "checkpoint_metadata.json"
        self._load_metadata()
    
    def _load_metadata(self):
        """Load checkpoint metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                    self.best_checkpoints = metadata.get("best_checkpoints", [])
                    self.best_scores = metadata.get("best_scores", [])
            except Exception as e:
                logger.warning(f"Failed to load checkpoint metadata: {e}")
                self.best_checkpoints = []
                self.best_scores = []
    
    def _save_metadata(self):
        """Save checkpoint metadata."""
        metadata = {
            "best_checkpoints": self.best_checkpoints,
            "best_scores": self.best_scores,
            "last_updated": datetime.now().isoformat(),
        }
        
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save checkpoint metadata: {e}")
    
    def _is_better_score(self, score: float, best_score: float) -> bool:
        """Check if current score is better than best score."""
        if self.mode == "min":
            return score < best_score
        else:
            return score > best_score
    
    def save_checkpoint(
        self,
        model: Union[pl.LightningModule, torch.nn.Module],
        metrics: Dict[str, float],
        epoch: int,
        step: int,
        optimizer_states: Optional[Dict] = None,
        scheduler_states: Optional[Dict] = None,
        extra_data: Optional[Dict] = None,
    ) -> str:
        """Save model checkpoint with metadata."""
        
        # Create checkpoint filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        monitor_value = metrics.get(self.monitor, 0.0)
        filename = f"epoch_{epoch:03d}_step_{step:06d}_{self.monitor}_{monitor_value:.4f}_{timestamp}.ckpt"
        checkpoint_path = self.checkpoint_dir / filename
        
        # Prepare checkpoint data
        checkpoint_data = {
            "epoch": epoch,
            "step": step,
            "metrics": metrics,
            "timestamp": timestamp,
        }
        
        # Add model state
        if isinstance(model, pl.LightningModule):
            checkpoint_data["state_dict"] = model.state_dict()
            if hasattr(model, 'config'):
                checkpoint_data["model_config"] = model.config.__dict__
        else:
            checkpoint_data["state_dict"] = model.state_dict()
        
        # Add optimizer and scheduler states
        if optimizer_states:
            checkpoint_data["optimizer_states"] = optimizer_states
        if scheduler_states:
            checkpoint_data["scheduler_states"] = scheduler_states
        
        # Add extra data
        if extra_data:
            checkpoint_data.update(extra_data)
        
        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Update best checkpoints tracking
        self._update_best_checkpoints(str(checkpoint_path), monitor_value)
        
        # Save as last checkpoint
        if self.save_last:
            last_path = self.checkpoint_dir / "last.ckpt"
            shutil.copy2(checkpoint_path, last_path)
        
        # Clean up old checkpoints
        self._cleanup_old_checkpoints()
        
        return str(checkpoint_path)
    
    def _update_best_checkpoints(self, checkpoint_path: str, score: float):
        """Update list of best checkpoints."""
        
        # Add current checkpoint
        self.best_checkpoints.append(checkpoint_path)
        self.best_scores.append(score)
        
        # Sort by score
        paired = list(zip(self.best_scores, self.best_checkpoints))
        if self.mode == "min":
            paired.sort(key=lambda x: x[0])
        else:
            paired.sort(key=lambda x: x[0], reverse=True)
        
        # Keep only top-k
        if len(paired) > self.save_top_k:
            paired = paired[:self.save_top_k]
        
        # Update lists
        self.best_scores, self.best_checkpoints = zip(*paired) if paired else ([], [])
        self.best_scores = list(self.best_scores)
        self.best_checkpoints = list(self.best_checkpoints)
        
        # Save metadata
        self._save_metadata()
    
    def _cleanup_old_checkpoints(self):
        """Clean up old checkpoints that are not in top-k."""
        
        # Get all checkpoint files
        all_checkpoints = list(self.checkpoint_dir.glob("*.ckpt"))
        
        # Keep best checkpoints, last checkpoint, and any manual saves
        keep_files = set(self.best_checkpoints)
        keep_files.add(str(self.checkpoint_dir / "last.ckpt"))
        keep_files.add(str(self.checkpoint_dir / "best.ckpt"))
        
        # Remove old checkpoints
        for checkpoint_path in all_checkpoints:
            if str(checkpoint_path) not in keep_files:
                try:
                    checkpoint_path.unlink()
                    logger.debug(f"Removed old checkpoint: {checkpoint_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove checkpoint {checkpoint_path}: {e}")
    
    def get_best_checkpoint(self) -> Optional[str]:
        """Get path to best checkpoint."""
        if self.best_checkpoints:
            return self.best_checkpoints[0]
        return None
    
    def get_last_checkpoint(self) -> Optional[str]:
        """Get path to last checkpoint."""
        last_path = self.checkpoint_dir / "last.ckpt"
        if last_path.exists():
            return str(last_path)
        return None
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: Optional[torch.nn.Module] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """Load checkpoint and optionally restore model state."""
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
        
        # Restore model state if provided
        if model is not None:
            model.load_state_dict(checkpoint_data["state_dict"], strict=strict)
            logger.info(f"Restored model state from: {checkpoint_path}")
        
        return checkpoint_data
    
    def list_checkpoints(self) -> Dict[str, Any]:
        """List all available checkpoints with metadata."""
        
        checkpoints = []
        
        # Get all checkpoint files
        for ckpt_path in self.checkpoint_dir.glob("*.ckpt"):
            try:
                ckpt_data = torch.load(ckpt_path, map_location="cpu")
                info = {
                    "path": str(ckpt_path),
                    "filename": ckpt_path.name,
                    "epoch": ckpt_data.get("epoch", -1),
                    "step": ckpt_data.get("step", -1),
                    "metrics": ckpt_data.get("metrics", {}),
                    "timestamp": ckpt_data.get("timestamp", "unknown"),
                    "size_mb": ckpt_path.stat().st_size / (1024 * 1024),
                }
                checkpoints.append(info)
            except Exception as e:
                logger.warning(f"Failed to read checkpoint {ckpt_path}: {e}")
        
        # Sort by step
        checkpoints.sort(key=lambda x: x["step"], reverse=True)
        
        return {
            "checkpoints": checkpoints,
            "best_checkpoints": self.best_checkpoints,
            "best_scores": self.best_scores,
            "total_count": len(checkpoints),
        }


class ModelSaver:
    """Utility for saving complete models for deployment."""
    
    def __init__(self, save_dir: Union[str, Path]):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def save_model_for_deployment(
        self,
        model: MusicGenModel,
        model_name: str = "musicgen_model",
        include_config: bool = True,
        include_tokenizer: bool = True,
        optimize_for_inference: bool = True,
    ) -> str:
        """Save model for deployment/inference."""
        
        model_dir = self.save_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Set model to eval mode
        model.eval()
        
        # Optimize for inference if requested
        if optimize_for_inference:
            model = self._optimize_for_inference(model)
        
        # Save model weights
        model_path = model_dir / "pytorch_model.bin"
        torch.save(model.state_dict(), model_path)
        
        # Save configuration
        if include_config:
            config_path = model_dir / "config.json"
            with open(config_path, 'w') as f:
                json.dump(model.config.__dict__, f, indent=2)
        
        # Save tokenizer info (placeholder for now)
        if include_tokenizer:
            tokenizer_path = model_dir / "tokenizer_config.json"
            tokenizer_config = {
                "model_name": model.audio_tokenizer.model_name if hasattr(model.audio_tokenizer, 'model_name') else "facebook/encodec_24khz",
                "sample_rate": model.audio_tokenizer.sample_rate if hasattr(model.audio_tokenizer, 'sample_rate') else 24000,
                "num_quantizers": model.audio_tokenizer.num_quantizers if hasattr(model.audio_tokenizer, 'num_quantizers') else 8,
            }
            with open(tokenizer_path, 'w') as f:
                json.dump(tokenizer_config, f, indent=2)
        
        # Save model info
        model_info = {
            "model_type": "MusicGenModel",
            "version": "1.0.0",
            "created_at": datetime.now().isoformat(),
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        }
        
        info_path = model_dir / "model_info.json"
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        logger.info(f"Model saved for deployment: {model_dir}")
        return str(model_dir)
    
    def _optimize_for_inference(self, model: torch.nn.Module) -> torch.nn.Module:
        """Optimize model for inference."""
        
        # Disable dropout
        model.eval()
        
        # Fuse operations if possible
        try:
            # This would include specific optimizations like:
            # - Fusing batch norm and conv layers
            # - Quantization
            # - TorchScript compilation
            pass
        except Exception as e:
            logger.warning(f"Model optimization failed: {e}")
        
        return model
    
    def export_to_onnx(
        self,
        model: torch.nn.Module,
        output_path: str,
        input_shape: tuple = (1, 100),
        opset_version: int = 11,
    ):
        """Export model to ONNX format."""
        
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randint(0, 256, input_shape)
        
        try:
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=["input_ids"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch_size", 1: "sequence_length"},
                    "logits": {0: "batch_size", 1: "sequence_length"},
                },
            )
            logger.info(f"Model exported to ONNX: {output_path}")
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            raise


class TrainingResumer:
    """Utility for resuming training from checkpoints."""
    
    def __init__(self, checkpoint_manager: ModelCheckpointManager):
        self.checkpoint_manager = checkpoint_manager
    
    def find_latest_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint for resuming."""
        
        # Try last checkpoint first
        last_ckpt = self.checkpoint_manager.get_last_checkpoint()
        if last_ckpt:
            return last_ckpt
        
        # Try best checkpoint
        best_ckpt = self.checkpoint_manager.get_best_checkpoint()
        if best_ckpt:
            return best_ckpt
        
        return None
    
    def resume_training(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        checkpoint_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Resume training from checkpoint."""
        
        if checkpoint_path is None:
            checkpoint_path = self.find_latest_checkpoint()
            
        if checkpoint_path is None:
            logger.info("No checkpoint found, starting from scratch")
            return {
                "epoch": 0,
                "step": 0,
                "metrics": {},
            }
        
        # Load checkpoint
        checkpoint_data = self.checkpoint_manager.load_checkpoint(
            checkpoint_path, model=model
        )
        
        # Restore optimizer state
        if optimizer and "optimizer_states" in checkpoint_data:
            try:
                optimizer.load_state_dict(checkpoint_data["optimizer_states"])
                logger.info("Restored optimizer state")
            except Exception as e:
                logger.warning(f"Failed to restore optimizer state: {e}")
        
        # Restore scheduler state
        if scheduler and "scheduler_states" in checkpoint_data:
            try:
                scheduler.load_state_dict(checkpoint_data["scheduler_states"])
                logger.info("Restored scheduler state")
            except Exception as e:
                logger.warning(f"Failed to restore scheduler state: {e}")
        
        logger.info(f"Resumed training from epoch {checkpoint_data.get('epoch', 0)}, "
                   f"step {checkpoint_data.get('step', 0)}")
        
        return {
            "epoch": checkpoint_data.get("epoch", 0),
            "step": checkpoint_data.get("step", 0),
            "metrics": checkpoint_data.get("metrics", {}),
        }


def create_checkpoint_manager(
    checkpoint_dir: str,
    **kwargs
) -> ModelCheckpointManager:
    """Factory function to create checkpoint manager."""
    
    return ModelCheckpointManager(checkpoint_dir, **kwargs)


def save_model_for_deployment(
    model: MusicGenModel,
    save_dir: str,
    model_name: str = "musicgen_model",
    **kwargs
) -> str:
    """Convenience function to save model for deployment."""
    
    saver = ModelSaver(save_dir)
    return saver.save_model_for_deployment(model, model_name, **kwargs)