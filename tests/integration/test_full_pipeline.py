"""
Integration tests for the full MusicGen pipeline.
"""
import pytest
import torch
import asyncio
import time
import json
import os
from pathlib import Path
import numpy as np

# Import all major components
from music_gen.models import create_musicgen_model
from music_gen.data import create_dataset, create_dataloader
from music_gen.training import MusicGenLightningModule
from music_gen.evaluation import AudioQualityMetrics, evaluate_model
from music_gen.streaming import StreamingGenerator, SessionManager, StreamingRequest
from music_gen.generation import BeamSearchConfig, beam_search_generate
from music_gen.api.main import app
from music_gen.utils.audio import save_audio_file, load_audio_file

from fastapi.testclient import TestClient
import pytorch_lightning as pl


class TestFullPipeline:
    """Test the complete MusicGen pipeline end-to-end."""
    
    @pytest.fixture(scope="class")
    def model(self):
        """Create a small model for testing."""
        # Use smallest possible model for fast tests
        model = create_musicgen_model("small")
        model.eval()
        return model
    
    @pytest.fixture(scope="class")
    def test_client(self):
        """Create test client for API."""
        # Mock the model loading for faster tests
        with TestClient(app) as client:
            yield client
    
    @pytest.fixture
    def sample_prompts(self):
        """Sample prompts for testing."""
        return [
            "Peaceful piano music",
            "Upbeat jazz with saxophone",
            "Electronic dance music with heavy bass",
            "Classical orchestra with strings",
            "Acoustic guitar folk melody"
        ]
    
    def test_model_creation_and_config(self):
        """Test model creation with different configurations."""
        # Test different model sizes
        for size in ["small", "base"]:
            model = create_musicgen_model(size)
            assert model is not None
            assert hasattr(model, "generate")
            assert hasattr(model, "generate_audio")
    
    def test_basic_generation(self, model, sample_prompts):
        """Test basic music generation."""
        # Test single generation
        prompt = sample_prompts[0]
        
        with torch.no_grad():
            audio = model.generate_audio(
                texts=[prompt],
                duration=5.0,  # Short duration for testing
                temperature=1.0,
            )
        
        assert audio is not None
        assert audio.shape[0] == 1  # Batch size
        assert audio.shape[1] in [1, 2]  # Mono or stereo
        assert audio.shape[2] > 0  # Has samples
        
        # Check audio is in valid range
        assert audio.abs().max() <= 1.0
    
    def test_generation_with_conditioning(self, model):
        """Test generation with various conditioning options."""
        conditioning_tests = [
            {"genre": "jazz", "mood": "happy", "tempo": 120},
            {"genre": "classical", "mood": "sad", "tempo": 60},
            {"genre": "electronic", "mood": "energetic", "tempo": 140},
        ]
        
        for conditions in conditioning_tests:
            with torch.no_grad():
                audio = model.generate_audio(
                    texts=["Test music"],
                    duration=3.0,
                    **conditions
                )
            
            assert audio is not None
            assert audio.shape[2] > 0
    
    def test_beam_search_generation(self, model):
        """Test beam search generation."""
        config = BeamSearchConfig(
            num_beams=2,  # Small for testing
            max_length=100,
            temperature=0.8,
        )
        
        # Prepare inputs
        input_ids = torch.tensor([[model.bos_token_id]])
        encoder_outputs = model.prepare_inputs(
            texts=["Jazz piano"],
            device=next(model.parameters()).device,
        )
        
        with torch.no_grad():
            generated, scores = beam_search_generate(
                model=model,
                input_ids=input_ids,
                config=config,
                encoder_hidden_states=encoder_outputs["text_hidden_states"],
                encoder_attention_mask=encoder_outputs["text_attention_mask"],
            )
        
        assert generated.shape[0] == 1  # Batch size
        assert generated.shape[1] <= config.max_length
        assert scores.shape[0] == 1
    
    def test_audio_save_load(self, model, tmp_path):
        """Test audio saving and loading."""
        # Generate audio
        with torch.no_grad():
            audio = model.generate_audio(
                texts=["Test audio"],
                duration=2.0,
            )
        
        # Save audio
        audio_path = tmp_path / "test_audio.wav"
        save_audio_file(audio[0], str(audio_path), sample_rate=24000)
        
        assert audio_path.exists()
        
        # Load audio
        loaded_audio, sr = load_audio_file(str(audio_path))
        
        assert loaded_audio is not None
        assert sr == 24000
        assert loaded_audio.shape[0] in [1, 2]  # Channels
    
    def test_data_pipeline(self, tmp_path):
        """Test data loading and augmentation pipeline."""
        # Create synthetic dataset
        dataset = create_dataset(
            dataset_name="synthetic",
            data_dir=str(tmp_path),
            split="train",
            num_samples=10,
            max_audio_length=5.0,
            augment_audio=True,
            augmentation_strength="moderate",
        )
        
        assert len(dataset) == 10
        
        # Test data loading
        sample = dataset[0]
        assert "audio_tokens" in sample
        assert "text" in sample
        
        # Create dataloader
        dataloader = create_dataloader(
            dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0,
        )
        
        # Test batch
        batch = next(iter(dataloader))
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert batch["input_ids"].shape[0] == 2
    
    @pytest.mark.asyncio
    async def test_streaming_generation(self, model):
        """Test real-time streaming generation."""
        # Create streaming generator
        from music_gen.streaming import StreamingConfig, create_streaming_generator
        
        config = StreamingConfig(
            chunk_duration=0.5,  # Short chunks for testing
            quality_mode="fast",
        )
        
        generator = create_streaming_generator(model, config=config)
        
        # Prepare streaming
        result = generator.prepare_streaming(
            texts=["Streaming test music"],
        )
        
        assert result["status"] == "prepared"
        
        # Collect a few chunks
        chunks = []
        chunk_count = 0
        max_chunks = 3
        
        for chunk_data in generator.start_streaming():
            if chunk_data.get("type") == "chunk":
                chunks.append(chunk_data)
                chunk_count += 1
                
                if chunk_count >= max_chunks:
                    generator.stop_streaming()
                    break
            elif chunk_data.get("type") == "error":
                pytest.fail(f"Streaming error: {chunk_data}")
        
        assert len(chunks) >= 1
        assert all(chunk.get("audio") is not None for chunk in chunks)
    
    def test_evaluation_metrics(self, model):
        """Test evaluation metrics calculation."""
        # Generate test audio
        with torch.no_grad():
            audio = model.generate_audio(
                texts=["Test music for evaluation"],
                duration=5.0,
            )
        
        # Convert to numpy
        audio_np = audio[0, 0].cpu().numpy()  # First channel
        
        # Calculate metrics
        metrics_calculator = AudioQualityMetrics()
        metrics = metrics_calculator.evaluate_audio_quality([audio_np])
        
        # Check metrics are calculated
        assert "snr" in metrics
        assert "spectral_contrast" in metrics
        assert metrics["snr"] > 0  # Should have some signal
    
    def test_training_setup(self, model, tmp_path):
        """Test training setup and basic training loop."""
        # Create Lightning module
        lightning_module = MusicGenLightningModule(model.config)
        
        # Create minimal dataset
        dataset = create_dataset(
            dataset_name="synthetic",
            data_dir=str(tmp_path),
            split="train",
            num_samples=4,
            max_audio_length=2.0,
        )
        
        dataloader = create_dataloader(dataset, batch_size=2, num_workers=0)
        
        # Setup trainer with minimal epochs
        trainer = pl.Trainer(
            max_epochs=1,
            max_steps=2,
            accelerator="cpu",
            logger=False,
            enable_checkpointing=False,
        )
        
        # Test training step
        batch = next(iter(dataloader))
        loss = lightning_module.training_step(batch, 0)
        
        assert loss is not None
        assert loss.item() > 0
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
    def test_gpu_inference(self, model):
        """Test GPU inference if available."""
        device = torch.device("cuda")
        model = model.to(device)
        
        with torch.no_grad():
            audio = model.generate_audio(
                texts=["GPU test"],
                duration=2.0,
                device=device,
            )
        
        assert audio.device.type == "cuda"
        assert audio.shape[2] > 0
    
    def test_model_checkpoint_save_load(self, model, tmp_path):
        """Test model checkpointing."""
        # Save model
        save_path = tmp_path / "test_model"
        model.save_pretrained(str(save_path))
        
        assert (save_path / "pytorch_model.bin").exists()
        assert (save_path / "config.json").exists()
        
        # Load model
        from music_gen.models import MusicGenModel
        loaded_model = MusicGenModel.from_pretrained(str(save_path))
        
        assert loaded_model is not None
        
        # Test loaded model works
        with torch.no_grad():
            audio = loaded_model.generate_audio(
                texts=["Loaded model test"],
                duration=2.0,
            )
        
        assert audio is not None
    
    @pytest.mark.asyncio
    async def test_api_generation_flow(self, test_client):
        """Test complete API generation flow."""
        # Skip if model not loaded in test client
        health = test_client.get("/health")
        if health.json().get("model_loaded") is False:
            pytest.skip("Model not loaded in test environment")
        
        # Start generation
        response = test_client.post("/generate", json={
            "prompt": "API test music",
            "duration": 5.0,
            "temperature": 1.0,
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        
        task_id = data["task_id"]
        
        # Poll for completion (with timeout)
        max_polls = 30
        for _ in range(max_polls):
            status_response = test_client.get(f"/generate/{task_id}")
            status = status_response.json()
            
            if status["status"] == "completed":
                assert "audio_url" in status
                break
            elif status["status"] == "failed":
                pytest.fail(f"Generation failed: {status.get('error')}")
            
            await asyncio.sleep(1)
        else:
            pytest.fail("Generation timed out")
    
    def test_config_overrides(self):
        """Test Hydra configuration overrides."""
        from omegaconf import OmegaConf
        from music_gen.models.transformer.config import MusicGenConfig
        
        # Load base config
        base_config = MusicGenConfig()
        
        # Test override
        overrides = {
            "transformer.hidden_size": 512,
            "transformer.num_layers": 6,
            "training.batch_size": 32,
        }
        
        config_dict = OmegaConf.to_container(OmegaConf.create(base_config.__dict__))
        override_config = OmegaConf.merge(config_dict, overrides)
        
        assert override_config.transformer.hidden_size == 512
        assert override_config.transformer.num_layers == 6
    
    def test_audio_augmentation_pipeline(self):
        """Test audio augmentation pipeline."""
        from music_gen.data.augmentation import create_training_augmentation_pipeline
        
        # Create pipeline
        pipeline = create_training_augmentation_pipeline(strong=True)
        
        # Test with synthetic audio
        sample_rate = 24000
        duration = 2.0
        samples = int(sample_rate * duration)
        audio = torch.randn(1, samples)
        
        # Apply augmentation
        augmented = pipeline(audio, sample_rate)
        
        assert augmented.shape == audio.shape
        assert not torch.allclose(augmented, audio)  # Should be different
    
    def test_end_to_end_quality(self, model, sample_prompts):
        """Test end-to-end generation quality."""
        results = []
        
        for prompt in sample_prompts[:3]:  # Test a few prompts
            start_time = time.time()
            
            with torch.no_grad():
                audio = model.generate_audio(
                    texts=[prompt],
                    duration=5.0,
                    temperature=0.8,
                    top_k=40,
                )
            
            generation_time = time.time() - start_time
            
            # Basic quality checks
            assert audio.abs().max() <= 1.0  # Valid range
            assert audio.abs().mean() > 0.001  # Not silence
            
            # Check generation speed (should be faster than real-time for small model)
            real_time_factor = 5.0 / generation_time
            
            results.append({
                "prompt": prompt,
                "generation_time": generation_time,
                "real_time_factor": real_time_factor,
                "audio_shape": audio.shape,
            })
        
        # Log results
        print("\nEnd-to-end test results:")
        for result in results:
            print(f"  Prompt: {result['prompt']}")
            print(f"  Generation time: {result['generation_time']:.2f}s")
            print(f"  Real-time factor: {result['real_time_factor']:.2f}x")
            print(f"  Audio shape: {result['audio_shape']}")
            print()


class TestSystemIntegration:
    """Test system-level integration."""
    
    def test_memory_usage(self, model):
        """Test memory usage during generation."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated()
        
        # Generate multiple sequences
        for i in range(3):
            with torch.no_grad():
                audio = model.generate_audio(
                    texts=[f"Test sequence {i}"],
                    duration=10.0,
                )
            
            # Force cleanup
            del audio
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated()
            memory_used = (peak_memory - initial_memory) / 1024**3  # GB
            
            print(f"\nPeak memory usage: {memory_used:.2f} GB")
            
            # Should not use excessive memory
            assert memory_used < 8.0  # Less than 8GB for small model
    
    def test_concurrent_generation(self, model):
        """Test concurrent generation requests."""
        import concurrent.futures
        
        def generate_audio(prompt, duration):
            with torch.no_grad():
                return model.generate_audio(
                    texts=[prompt],
                    duration=duration,
                )
        
        prompts = [
            "Concurrent test 1",
            "Concurrent test 2",
            "Concurrent test 3",
        ]
        
        # Test thread safety
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(generate_audio, prompt, 2.0)
                for prompt in prompts
            ]
            
            results = [f.result() for f in futures]
        
        assert len(results) == 3
        assert all(r is not None for r in results)
    
    def test_error_handling(self, model):
        """Test error handling and recovery."""
        
        # Test with invalid inputs
        with pytest.raises(Exception):
            model.generate_audio(
                texts=[],  # Empty list
                duration=10.0,
            )
        
        # Test with very long prompt
        long_prompt = " ".join(["word"] * 1000)
        try:
            audio = model.generate_audio(
                texts=[long_prompt],
                duration=2.0,
            )
            # Should either succeed or raise meaningful error
            assert audio is not None
        except Exception as e:
            assert "length" in str(e).lower() or "token" in str(e).lower()
        
        # Model should still work after errors
        audio = model.generate_audio(
            texts=["Recovery test"],
            duration=2.0,
        )
        assert audio is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])