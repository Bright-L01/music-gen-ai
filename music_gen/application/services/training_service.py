"""
Training service implementation.

Handles model training orchestration with proper job management,
monitoring, and error handling.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
import torch

from music_gen.core.config import AppConfig
from music_gen.core.exceptions import TrainingError
from music_gen.core.interfaces.repositories import MetadataRepository
from music_gen.core.interfaces.services import (
    ModelService,
    TrainingConfig,
    TrainingService,
)
from music_gen.data.lightning_datamodule import MusicDataModule
from music_gen.training.lightning_module import MusicGenLightningModule

logger = logging.getLogger(__name__)


class TrainingServiceImpl(TrainingService):
    """Implementation of training service with job management."""

    def __init__(
        self,
        model_service: ModelService,
        metadata_repository: MetadataRepository,
        config: AppConfig,
    ):
        """Initialize training service.

        Args:
            model_service: Service for model management
            metadata_repository: Repository for dataset metadata
            config: Application configuration
        """
        self._model_service = model_service
        self._metadata_repository = metadata_repository
        self._config = config
        self._training_jobs: Dict[str, Dict[str, Any]] = {}
        self._job_lock = asyncio.Lock()

    async def train_model(
        self, model_id: str, dataset_id: str, config: TrainingConfig
    ) -> Dict[str, Any]:
        """Train a model on a dataset."""
        job_id = str(uuid.uuid4())

        # Create job record
        job_data = {
            "id": job_id,
            "model_id": model_id,
            "dataset_id": dataset_id,
            "config": config.__dict__,
            "status": "initializing",
            "started_at": datetime.utcnow().isoformat(),
            "progress": 0.0,
        }

        async with self._job_lock:
            self._training_jobs[job_id] = job_data

        try:
            # Load model
            await self._update_job_status(job_id, "loading_model", 0.1)
            model = await self._model_service.load_model(model_id)

            # Load dataset metadata
            await self._update_job_status(job_id, "loading_dataset", 0.2)
            dataset_metadata = await self._metadata_repository.load_metadata(dataset_id)

            # Create data module
            data_module = await self._create_data_module(dataset_metadata, config)

            # Create Lightning module
            lightning_module = MusicGenLightningModule(
                model=model,
                learning_rate=config.learning_rate,
                warmup_steps=config.warmup_steps,
            )

            # Create trainer
            await self._update_job_status(job_id, "training", 0.3)
            trainer = self._create_trainer(config, job_id)

            # Run training
            trainer.fit(lightning_module, data_module)

            # Save trained model
            await self._update_job_status(job_id, "saving_model", 0.9)
            trained_model_id = f"{model_id}_trained_{job_id[:8]}"
            await self._model_service.save_model(
                lightning_module.model,
                trained_model_id,
                metadata={
                    "base_model": model_id,
                    "dataset": dataset_id,
                    "training_config": config.__dict__,
                    "training_job": job_id,
                },
            )

            # Update job as completed
            results = {
                "trained_model_id": trained_model_id,
                "final_loss": trainer.callback_metrics.get("train_loss", 0.0),
                "epochs_trained": trainer.current_epoch,
            }

            await self._update_job_status(job_id, "completed", 1.0, results)

            logger.info(f"Training completed: {job_id}")
            return results

        except Exception as e:
            logger.error(f"Training failed for job {job_id}: {e}")
            await self._update_job_status(job_id, "failed", error=str(e))
            raise TrainingError(f"Training failed: {e}")

    async def fine_tune_model(
        self, base_model_id: str, dataset_id: str, config: TrainingConfig
    ) -> str:
        """Fine-tune a pre-trained model."""
        # Fine-tuning is similar to training but with lower learning rate
        fine_tune_config = TrainingConfig(
            **{**config.__dict__, "learning_rate": config.learning_rate * 0.1}
        )

        results = await self.train_model(base_model_id, dataset_id, fine_tune_config)
        return results["trained_model_id"]

    async def get_training_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a training job."""
        async with self._job_lock:
            if job_id not in self._training_jobs:
                raise ValueError(f"Training job not found: {job_id}")

            return self._training_jobs[job_id].copy()

    async def stop_training(self, job_id: str) -> None:
        """Stop an ongoing training job."""
        async with self._job_lock:
            if job_id not in self._training_jobs:
                raise ValueError(f"Training job not found: {job_id}")

            job = self._training_jobs[job_id]

            if job["status"] in ["completed", "failed", "stopped"]:
                return

            # Mark as stopping
            job["status"] = "stopping"

            # In a real implementation, we would interrupt the trainer
            # For now, just mark as stopped
            job["status"] = "stopped"
            job["stopped_at"] = datetime.utcnow().isoformat()

    async def list_training_jobs(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List training jobs with optional status filter."""
        async with self._job_lock:
            jobs = list(self._training_jobs.values())

            if status:
                jobs = [job for job in jobs if job["status"] == status]

            # Sort by start time (newest first)
            jobs.sort(key=lambda x: x["started_at"], reverse=True)

            return jobs

    async def _create_data_module(
        self, dataset_metadata: Dict[str, Any], config: TrainingConfig
    ) -> MusicDataModule:
        """Create data module for training."""
        # Extract dataset information
        dataset_path = dataset_metadata.get("path", "data/musiccaps")

        # Create data module
        data_module = MusicDataModule(
            dataset_name="musiccaps",  # Simplified for now
            data_dir=dataset_path,
            batch_size=config.batch_size,
            num_workers=4,
        )

        return data_module

    def _create_trainer(self, config: TrainingConfig, job_id: str) -> pl.Trainer:
        """Create PyTorch Lightning trainer."""
        # Create callbacks
        callbacks = [
            pl.callbacks.ModelCheckpoint(
                dirpath=f"checkpoints/{job_id}",
                filename="{epoch}-{train_loss:.4f}",
                save_top_k=3,
                monitor="train_loss",
                mode="min",
            ),
            pl.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                mode="min",
            ),
            TrainingProgressCallback(job_id, self),
        ]

        # Create trainer
        trainer = pl.Trainer(
            max_epochs=config.num_epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            precision=16 if config.mixed_precision else 32,
            gradient_clip_val=1.0,
            accumulate_grad_batches=config.gradient_accumulation_steps,
            callbacks=callbacks,
            enable_checkpointing=True,
            logger=pl.loggers.TensorBoardLogger(
                save_dir="logs",
                name=f"training_{job_id}",
            ),
        )

        return trainer

    async def _update_job_status(
        self,
        job_id: str,
        status: str,
        progress: float = None,
        results: Dict[str, Any] = None,
        error: str = None,
    ) -> None:
        """Update training job status."""
        async with self._job_lock:
            if job_id in self._training_jobs:
                job = self._training_jobs[job_id]
                job["status"] = status

                if progress is not None:
                    job["progress"] = progress

                if results:
                    job["results"] = results

                if error:
                    job["error"] = error

                job["updated_at"] = datetime.utcnow().isoformat()


class TrainingProgressCallback(pl.Callback):
    """Callback to update training progress."""

    def __init__(self, job_id: str, training_service: TrainingServiceImpl):
        self.job_id = job_id
        self.training_service = training_service

    def on_train_epoch_end(self, trainer, pl_module):
        """Update progress at end of each epoch."""
        progress = (trainer.current_epoch + 1) / trainer.max_epochs

        # Run async update in sync context
        asyncio.create_task(
            self.training_service._update_job_status(
                self.job_id,
                "training",
                progress=0.3 + progress * 0.6,  # Training is 30-90% of total progress
            )
        )
