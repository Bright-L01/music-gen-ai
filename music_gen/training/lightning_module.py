"""
PyTorch Lightning module for MusicGen training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import pytorch_lightning as pl
from typing import Dict, Any, Optional, List, Tuple
import wandb
import logging

from ..models.musicgen import MusicGenModel, MusicGenConfig
from ..evaluation.metrics import AudioQualityMetrics
from ..utils.audio import save_audio_sample

logger = logging.getLogger(__name__)


class MusicGenLightningModule(pl.LightningModule):
    """PyTorch Lightning module for training MusicGen."""
    
    def __init__(
        self,
        config: MusicGenConfig,
        learning_rate: float = 5e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 5000,
        max_steps: int = 100000,
        gradient_clip_val: float = 1.0,
        accumulate_grad_batches: int = 1,
        compile_model: bool = False,
        log_every_n_steps: int = 100,
        val_check_interval: float = 0.25,
        save_audio_samples: bool = True,
        num_audio_samples: int = 4,
        sample_generation_steps: List[int] = None,
    ):
        super().__init__()
        
        self.save_hyperparameters()
        
        # Model configuration
        self.config = config
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.gradient_clip_val = gradient_clip_val
        self.accumulate_grad_batches = accumulate_grad_batches
        self.log_every_n_steps = log_every_n_steps
        self.val_check_interval = val_check_interval
        
        # Audio sampling configuration
        self.save_audio_samples = save_audio_samples
        self.num_audio_samples = num_audio_samples
        if sample_generation_steps is None:
            self.sample_generation_steps = [1000, 5000, 10000, 25000, 50000]
        else:
            self.sample_generation_steps = sample_generation_steps
        
        # Initialize model
        self.model = MusicGenModel(config)
        
        # Compile model for optimization (PyTorch 2.0+)
        if compile_model and hasattr(torch, 'compile'):
            self.model = torch.compile(self.model)
        
        # Initialize metrics
        self.train_metrics = AudioQualityMetrics()
        self.val_metrics = AudioQualityMetrics()
        
        # Loss tracking
        self.automatic_optimization = True
        
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        
        # Filter parameters that require gradients
        params = [p for p in self.parameters() if p.requires_grad]
        
        # Separate parameters for different learning rates if needed
        text_encoder_params = []
        other_params = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            
            if 'text_encoder' in name or 't5' in name:
                text_encoder_params.append(param)
            else:
                other_params.append(param)
        
        # Create parameter groups
        param_groups = []
        
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': self.learning_rate,
                'weight_decay': self.weight_decay,
            })
        
        if text_encoder_params:
            # Lower learning rate for pre-trained text encoder
            param_groups.append({
                'params': text_encoder_params,
                'lr': self.learning_rate * 0.1,
                'weight_decay': self.weight_decay * 0.1,
            })
        
        # Optimizer
        optimizer = AdamW(
            param_groups,
            betas=(0.9, 0.95),
            eps=1e-8,
        )
        
        # Learning rate scheduler
        if self.warmup_steps > 0:
            # Warmup scheduler
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=self.warmup_steps,
            )
            
            # Cosine annealing scheduler
            cosine_steps = max(1, self.max_steps - self.warmup_steps)
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=cosine_steps,
                eta_min=self.learning_rate * 0.1,
            )
            
            # Combined scheduler
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[self.warmup_steps],
            )
        else:
            # Just cosine annealing
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.max_steps,
                eta_min=self.learning_rate * 0.1,
            )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
    
    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Forward pass through the model."""
        return self.model(**batch)
    
    def compute_loss(
        self,
        batch: Dict[str, Any],
        outputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute training loss."""
        
        # Primary language modeling loss
        loss = outputs["loss"]
        
        # Additional losses can be added here
        # e.g., perplexity, token accuracy, etc.
        
        losses = {
            "loss": loss,
            "lm_loss": loss,
        }
        
        # Calculate perplexity
        if loss is not None:
            losses["perplexity"] = torch.exp(loss)
        
        return losses
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step."""
        
        # Forward pass
        outputs = self.forward(batch)
        
        # Compute losses
        losses = self.compute_loss(batch, outputs)
        
        # Log metrics
        self.log("train/loss", losses["loss"], prog_bar=True, sync_dist=True)
        self.log("train/perplexity", losses["perplexity"], sync_dist=True)
        self.log("train/lr", self.optimizers().param_groups[0]["lr"], sync_dist=True)
        
        # Log additional metrics periodically
        if self.global_step % self.log_every_n_steps == 0:
            self._log_training_metrics(batch, outputs, losses)
        
        return losses["loss"]
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step."""
        
        # Forward pass
        outputs = self.forward(batch)
        
        # Compute losses
        losses = self.compute_loss(batch, outputs)
        
        # Log metrics
        self.log("val/loss", losses["loss"], prog_bar=True, sync_dist=True)
        self.log("val/perplexity", losses["perplexity"], sync_dist=True)
        
        return {
            "val_loss": losses["loss"],
            "val_perplexity": losses["perplexity"],
        }
    
    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch."""
        
        # Generate audio samples for evaluation
        if self.save_audio_samples and self.global_step in self.sample_generation_steps:
            self._generate_audio_samples()
    
    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Test step."""
        return self.validation_step(batch, batch_idx)
    
    def _log_training_metrics(
        self,
        batch: Dict[str, Any],
        outputs: Dict[str, torch.Tensor],
        losses: Dict[str, torch.Tensor],
    ) -> None:
        """Log detailed training metrics."""
        
        # Gradient norms
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1.0 / 2)
        
        self.log("train/grad_norm", total_norm, sync_dist=True)
        
        # Model statistics
        logits = outputs.get("logits")
        if logits is not None:
            # Token prediction accuracy
            labels = batch.get("labels")
            if labels is not None:
                predictions = torch.argmax(logits, dim=-1)
                
                # Shift for causal LM
                shift_predictions = predictions[..., :-1].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                # Calculate accuracy (excluding padding)
                mask = shift_labels != self.model.pad_token_id
                correct = (shift_predictions == shift_labels) & mask
                accuracy = correct.sum().float() / mask.sum().float()
                
                self.log("train/token_accuracy", accuracy, sync_dist=True)
            
            # Logit statistics
            self.log("train/logit_mean", logits.mean(), sync_dist=True)
            self.log("train/logit_std", logits.std(), sync_dist=True)
            self.log("train/logit_max", logits.max(), sync_dist=True)
            self.log("train/logit_min", logits.min(), sync_dist=True)
    
    @torch.no_grad()
    def _generate_audio_samples(self) -> None:
        """Generate audio samples for qualitative evaluation."""
        
        # Sample prompts for generation
        sample_prompts = [
            "Upbeat jazz with saxophone solo",
            "Relaxing ambient music with nature sounds",
            "Epic orchestral theme with dramatic crescendo",
            "Electronic dance music with heavy bass",
        ]
        
        # Select subset of prompts
        prompts = sample_prompts[:self.num_audio_samples]
        
        try:
            # Generate audio
            audio_tensors = self.model.generate_audio(
                texts=prompts,
                duration=10.0,  # 10 seconds
                temperature=0.9,
                top_p=0.95,
                do_sample=True,
            )
            
            # Log to wandb if available
            if wandb.run is not None:
                audio_samples = []
                for i, (prompt, audio) in enumerate(zip(prompts, audio_tensors)):
                    # Convert to numpy and ensure proper format
                    audio_np = audio.cpu().numpy()
                    if audio_np.ndim > 1:
                        audio_np = audio_np[0]  # Take first channel if stereo
                    
                    # Create wandb audio object
                    audio_sample = wandb.Audio(
                        audio_np,
                        sample_rate=self.model.audio_tokenizer.sample_rate,
                        caption=f"Step {self.global_step}: {prompt}",
                    )
                    audio_samples.append(audio_sample)
                
                # Log all samples
                wandb.log({
                    f"generated_audio/step_{self.global_step}": audio_samples,
                    "step": self.global_step,
                })
            
            logger.info(f"Generated {len(prompts)} audio samples at step {self.global_step}")
            
        except Exception as e:
            logger.warning(f"Failed to generate audio samples: {e}")
    
    def configure_gradient_clipping(
        self,
        optimizer,
        gradient_clip_val: Optional[float] = None,
        gradient_clip_algorithm: Optional[str] = None,
    ) -> None:
        """Configure gradient clipping."""
        
        if gradient_clip_val is None:
            gradient_clip_val = self.gradient_clip_val
        
        if gradient_clip_val > 0:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=gradient_clip_val,
                gradient_clip_algorithm="norm",
            )
    
    def on_before_optimizer_step(self, optimizer) -> None:
        """Called before optimizer step."""
        
        # Log learning rate
        if self.global_step % self.log_every_n_steps == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                self.log(f"train/lr_group_{i}", param_group["lr"], sync_dist=True)
    
    def on_train_epoch_start(self) -> None:
        """Called at the start of training epoch."""
        logger.info(f"Starting epoch {self.current_epoch}")
    
    def on_train_epoch_end(self) -> None:
        """Called at the end of training epoch."""
        logger.info(f"Completed epoch {self.current_epoch}")
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Called when saving checkpoint."""
        
        # Add custom metadata to checkpoint
        checkpoint["model_config"] = self.config.__dict__
        checkpoint["step"] = self.global_step
        checkpoint["epoch"] = self.current_epoch
        
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Called when loading checkpoint."""
        
        # Log checkpoint info
        step = checkpoint.get("step", "unknown")
        epoch = checkpoint.get("epoch", "unknown")
        logger.info(f"Loading checkpoint from step {step}, epoch {epoch}")


class ProgressiveTrainingModule(MusicGenLightningModule):
    """Extended Lightning module with progressive training capabilities."""
    
    def __init__(
        self,
        config: MusicGenConfig,
        sequence_length_schedule: List[Tuple[int, int]] = None,
        **kwargs
    ):
        super().__init__(config, **kwargs)
        
        # Progressive training schedule: [(step, max_seq_len), ...]
        if sequence_length_schedule is None:
            self.sequence_length_schedule = [
                (0, 256),      # Start with short sequences
                (5000, 512),   # Increase at 5k steps
                (15000, 1024), # Increase at 15k steps
                (30000, 2048), # Increase at 30k steps
                (50000, 4096), # Full length at 50k steps
            ]
        else:
            self.sequence_length_schedule = sequence_length_schedule
        
        self.current_max_seq_len = self.sequence_length_schedule[0][1]
    
    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        """Update sequence length based on training progress."""
        
        # Check if we need to update sequence length
        for step_threshold, seq_len in self.sequence_length_schedule:
            if self.global_step >= step_threshold:
                new_seq_len = seq_len
            else:
                break
        
        if new_seq_len != self.current_max_seq_len:
            self.current_max_seq_len = new_seq_len
            logger.info(f"Updated max sequence length to {new_seq_len} at step {self.global_step}")
            
            # Log the change
            self.log("train/max_seq_len", float(self.current_max_seq_len), sync_dist=True)
    
    def get_current_max_sequence_length(self) -> int:
        """Get current maximum sequence length for data loading."""
        return self.current_max_seq_len