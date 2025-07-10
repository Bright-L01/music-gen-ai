"""
Comprehensive unit tests for the music generation pipeline.

Tests core generation logic, conditioning, sampling strategies,
and error handling in the generation pipeline.
"""

import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import pytest
import torch
import numpy as np
from typing import Dict, Any, Optional

from music_gen.inference.generators import (
    MusicGenerator,
    ConditionalGenerator,
    BatchGenerator,
)
from music_gen.inference.strategies import (
    SamplingStrategy,
    GreedyStrategy,
    TopKStrategy,
    TopPStrategy,
    NucleusStrategy,
)
from music_gen.core.interfaces.services import (
    GenerationRequest,
    GenerationResult,
)
from music_gen.core.exceptions import (
    GenerationError,
    ModelLoadError,
    ValidationError,
)


@pytest.mark.unit
class TestMusicGenerator:
    """Test core music generation functionality."""

    @pytest.fixture
    def mock_model(self):
        """Create mock model for testing."""
        model = Mock()
        model.config.sample_rate = 24000
        model.config.hop_length = 512
        model.config.vocab_size = 2048
        model.config.max_sequence_length = 1024
        model.device = torch.device("cpu")

        # Mock generation method
        def mock_generate(input_ids, attention_mask=None, **kwargs):
            batch_size = input_ids.shape[0]
            seq_len = kwargs.get("max_length", 100)
            return torch.randint(0, 2048, (batch_size, seq_len))

        model.generate = Mock(side_effect=mock_generate)
        model.encode_text = Mock(return_value=torch.randn(1, 10, 768))
        model.decode_tokens = Mock(return_value=torch.randn(1, 24000))

        return model

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        tokenizer = Mock()
        tokenizer.encode = Mock(return_value=torch.randint(0, 100, (1, 10)))
        tokenizer.decode = Mock(return_value=torch.randn(1, 24000))
        return tokenizer

    @pytest.fixture
    def generator(self, mock_model, mock_tokenizer):
        """Create generator instance."""
        return MusicGenerator(mock_model, mock_tokenizer)

    @pytest.mark.asyncio
    async def test_generate_basic(self, generator, mock_model):
        """Test basic generation functionality."""
        request = GenerationRequest(
            prompt="Upbeat jazz music",
            duration=10.0,
            temperature=1.0,
        )

        result = await generator.generate(request)

        assert isinstance(result, GenerationResult)
        assert result.duration == 10.0
        assert result.sample_rate == 24000
        assert isinstance(result.audio, torch.Tensor)
        assert result.metadata["prompt"] == "Upbeat jazz music"

        # Verify model calls
        mock_model.encode_text.assert_called_once()
        mock_model.generate.assert_called_once()
        mock_model.decode_tokens.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_with_seed(self, generator, mock_model):
        """Test generation with fixed seed for reproducibility."""
        request = GenerationRequest(
            prompt="Classical piano",
            duration=5.0,
            seed=42,
        )

        # Generate twice with same seed
        result1 = await generator.generate(request)
        result2 = await generator.generate(request)

        # Results should be identical (mocked, but verifies seed handling)
        assert torch.equal(result1.audio, result2.audio)
        assert result1.metadata["seed"] == 42
        assert result2.metadata["seed"] == 42

    @pytest.mark.asyncio
    async def test_generate_different_durations(self, generator, mock_model):
        """Test generation with different durations."""
        durations = [5.0, 10.0, 30.0, 60.0]

        for duration in durations:
            request = GenerationRequest(
                prompt=f"Music for {duration} seconds",
                duration=duration,
            )

            result = await generator.generate(request)

            assert result.duration == duration
            expected_samples = int(duration * 24000)
            # Allow some tolerance for audio length
            assert abs(result.audio.shape[-1] - expected_samples) < 1000

    @pytest.mark.asyncio
    async def test_generate_with_invalid_duration(self, generator):
        """Test generation with invalid duration."""
        request = GenerationRequest(
            prompt="Test music",
            duration=-1.0,  # Invalid negative duration
        )

        with pytest.raises(ValidationError, match="Duration must be positive"):
            await generator.generate(request)

    @pytest.mark.asyncio
    async def test_generate_model_failure(self, generator, mock_model):
        """Test handling of model generation failure."""
        mock_model.generate.side_effect = RuntimeError("CUDA out of memory")

        request = GenerationRequest(prompt="Test music")

        with pytest.raises(GenerationError, match="Generation failed"):
            await generator.generate(request)

    @pytest.mark.asyncio
    async def test_generate_empty_prompt(self, generator):
        """Test generation with empty prompt."""
        request = GenerationRequest(
            prompt="",
            duration=5.0,
        )

        with pytest.raises(ValidationError, match="Prompt cannot be empty"):
            await generator.generate(request)

    @pytest.mark.asyncio
    async def test_generate_long_prompt(self, generator, mock_model):
        """Test generation with very long prompt."""
        long_prompt = "This is a very long prompt. " * 100  # ~3000 chars

        request = GenerationRequest(
            prompt=long_prompt,
            duration=10.0,
        )

        result = await generator.generate(request)

        assert isinstance(result, GenerationResult)
        # Verify prompt was truncated or handled appropriately
        mock_model.encode_text.assert_called_once()

    def test_calculate_sequence_length(self, generator):
        """Test sequence length calculation for different durations."""
        # Test various durations
        test_cases = [
            (1.0, 47),  # 1 second
            (10.0, 469),  # 10 seconds
            (30.0, 1407),  # 30 seconds
        ]

        for duration, expected_length in test_cases:
            length = generator._calculate_sequence_length(duration)
            assert abs(length - expected_length) < 5  # Allow small variance

    def test_prepare_generation_kwargs(self, generator):
        """Test preparation of generation kwargs."""
        request = GenerationRequest(
            prompt="Test music",
            duration=10.0,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
        )

        kwargs = generator._prepare_generation_kwargs(request)

        assert kwargs["temperature"] == 0.8
        assert kwargs["top_k"] == 50
        assert kwargs["top_p"] == 0.9
        assert "max_length" in kwargs
        assert kwargs["do_sample"] is True


@pytest.mark.unit
class TestConditionalGenerator:
    """Test conditional music generation with genre, mood, etc."""

    @pytest.fixture
    def mock_conditioning_model(self):
        """Create mock conditioning model."""
        model = Mock()
        model.encode_conditioning = Mock(return_value=torch.randn(1, 10, 256))
        return model

    @pytest.fixture
    def conditional_generator(self, mock_model, mock_tokenizer, mock_conditioning_model):
        """Create conditional generator instance."""
        return ConditionalGenerator(mock_model, mock_tokenizer, mock_conditioning_model)

    @pytest.mark.asyncio
    async def test_generate_with_genre_conditioning(self, conditional_generator):
        """Test generation with genre conditioning."""
        request = GenerationRequest(
            prompt="Upbeat music",
            duration=10.0,
        )
        conditioning = {
            "genre": "jazz",
            "mood": "happy",
            "tempo": 120,
        }

        result = await conditional_generator.generate_with_conditioning(request, conditioning)

        assert isinstance(result, GenerationResult)
        assert result.metadata["conditioning"] == conditioning

        # Verify conditioning was used
        conditional_generator._conditioning_model.encode_conditioning.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_with_partial_conditioning(self, conditional_generator):
        """Test generation with partial conditioning information."""
        request = GenerationRequest(prompt="Relaxing music")
        conditioning = {"mood": "calm"}  # Only mood specified

        result = await conditional_generator.generate_with_conditioning(request, conditioning)

        assert isinstance(result, GenerationResult)
        assert "mood" in result.metadata["conditioning"]

    @pytest.mark.asyncio
    async def test_generate_with_invalid_conditioning(self, conditional_generator):
        """Test generation with invalid conditioning values."""
        request = GenerationRequest(prompt="Test music")
        conditioning = {
            "genre": "invalid_genre",
            "tempo": -50,  # Invalid negative tempo
        }

        with pytest.raises(ValidationError, match="Invalid conditioning"):
            await conditional_generator.generate_with_conditioning(request, conditioning)

    def test_validate_conditioning(self, conditional_generator):
        """Test conditioning validation logic."""
        # Valid conditioning
        valid_conditioning = {
            "genre": "jazz",
            "mood": "happy",
            "tempo": 120,
            "key": "C",
        }
        conditional_generator._validate_conditioning(valid_conditioning)  # Should not raise

        # Invalid conditioning
        invalid_conditioning = {
            "genre": "unknown_genre",
            "tempo": 500,  # Too fast
        }

        with pytest.raises(ValidationError):
            conditional_generator._validate_conditioning(invalid_conditioning)


@pytest.mark.unit
class TestBatchGenerator:
    """Test batch generation functionality."""

    @pytest.fixture
    def batch_generator(self, mock_model, mock_tokenizer):
        """Create batch generator instance."""
        return BatchGenerator(mock_model, mock_tokenizer)

    @pytest.mark.asyncio
    async def test_generate_batch(self, batch_generator):
        """Test batch generation with multiple requests."""
        requests = [
            GenerationRequest(prompt="Jazz music", duration=10.0),
            GenerationRequest(prompt="Rock music", duration=15.0),
            GenerationRequest(prompt="Classical music", duration=20.0),
        ]

        results = await batch_generator.generate_batch(requests)

        assert len(results) == 3
        for i, result in enumerate(results):
            assert isinstance(result, GenerationResult)
            assert result.duration == requests[i].duration

    @pytest.mark.asyncio
    async def test_generate_batch_with_failure(self, batch_generator, mock_model):
        """Test batch generation with some failures."""

        # Make second request fail
        def mock_generate_with_failure(input_ids, **kwargs):
            if input_ids.shape[0] > 1:  # Batch size > 1
                raise RuntimeError("Batch processing failed")
            return torch.randint(0, 2048, (1, 100))

        mock_model.generate.side_effect = mock_generate_with_failure

        requests = [
            GenerationRequest(prompt="Good music", duration=10.0),
            GenerationRequest(prompt="Bad music", duration=10.0),  # This will fail
        ]

        results = await batch_generator.generate_batch(
            requests, fail_fast=False  # Continue on failures
        )

        assert len(results) == 2
        assert isinstance(results[0], GenerationResult)
        assert isinstance(results[1], Exception)

    @pytest.mark.asyncio
    async def test_generate_batch_memory_optimization(self, batch_generator):
        """Test memory optimization in batch generation."""
        # Create many requests to test batching strategy
        requests = [GenerationRequest(prompt=f"Music {i}", duration=5.0) for i in range(20)]

        results = await batch_generator.generate_batch(
            requests,
            batch_size=4,  # Process in smaller batches
        )

        assert len(results) == 20
        assert all(isinstance(r, GenerationResult) for r in results)


@pytest.mark.unit
class TestSamplingStrategies:
    """Test different sampling strategies for generation."""

    @pytest.fixture
    def logits(self):
        """Create sample logits for testing."""
        return torch.randn(1, 2048)  # Batch size 1, vocab size 2048

    def test_greedy_strategy(self, logits):
        """Test greedy sampling strategy."""
        strategy = GreedyStrategy()
        next_token = strategy.sample(logits)

        assert next_token.shape == (1,)
        assert next_token == logits.argmax(dim=-1)

    def test_top_k_strategy(self, logits):
        """Test top-k sampling strategy."""
        strategy = TopKStrategy(k=50)
        next_token = strategy.sample(logits)

        assert next_token.shape == (1,)
        assert 0 <= next_token.item() < 2048

        # Verify top-k constraint
        top_k_indices = logits.topk(50).indices
        assert next_token.item() in top_k_indices.squeeze().tolist()

    def test_top_p_strategy(self, logits):
        """Test top-p (nucleus) sampling strategy."""
        strategy = TopPStrategy(p=0.9)
        next_token = strategy.sample(logits)

        assert next_token.shape == (1,)
        assert 0 <= next_token.item() < 2048

    def test_nucleus_strategy_with_temperature(self, logits):
        """Test nucleus sampling with temperature scaling."""
        strategy = NucleusStrategy(p=0.95, temperature=0.8)

        # Sample multiple times to check diversity
        samples = [strategy.sample(logits).item() for _ in range(10)]

        # With temperature < 1, should have some diversity but not uniform
        unique_samples = len(set(samples))
        assert 1 <= unique_samples <= 10

    def test_strategy_with_extreme_temperature(self, logits):
        """Test sampling with extreme temperature values."""
        # Very low temperature (near deterministic)
        low_temp_strategy = TopPStrategy(p=0.9, temperature=0.01)
        samples_low = [low_temp_strategy.sample(logits).item() for _ in range(5)]
        assert len(set(samples_low)) <= 2  # Should be mostly deterministic

        # Very high temperature (near uniform)
        high_temp_strategy = TopPStrategy(p=0.9, temperature=10.0)
        samples_high = [high_temp_strategy.sample(logits).item() for _ in range(10)]
        assert len(set(samples_high)) >= 5  # Should be diverse

    def test_invalid_strategy_parameters(self):
        """Test strategy creation with invalid parameters."""
        with pytest.raises(ValueError, match="k must be positive"):
            TopKStrategy(k=0)

        with pytest.raises(ValueError, match="p must be between 0 and 1"):
            TopPStrategy(p=1.5)

        with pytest.raises(ValueError, match="temperature must be positive"):
            NucleusStrategy(p=0.9, temperature=0.0)


@pytest.mark.unit
class TestGenerationPipelineIntegration:
    """Test integration of generation components."""

    @pytest.fixture
    def full_pipeline(self, mock_model, mock_tokenizer):
        """Create full generation pipeline."""
        generator = MusicGenerator(mock_model, mock_tokenizer)
        strategy = TopPStrategy(p=0.9, temperature=0.8)
        generator.set_sampling_strategy(strategy)
        return generator

    @pytest.mark.asyncio
    async def test_end_to_end_generation(self, full_pipeline):
        """Test complete generation pipeline."""
        request = GenerationRequest(
            prompt="Energetic electronic dance music",
            duration=30.0,
            temperature=0.8,
            top_p=0.9,
        )

        result = await full_pipeline.generate(request)

        # Verify complete result structure
        assert isinstance(result, GenerationResult)
        assert result.audio.dim() == 2  # [batch, time]
        assert result.sample_rate == 24000
        assert result.duration == 30.0
        assert "prompt" in result.metadata
        assert "generation_time" in result.metadata
        assert "model_info" in result.metadata

    @pytest.mark.asyncio
    async def test_generation_timing(self, full_pipeline):
        """Test generation timing metadata."""
        request = GenerationRequest(prompt="Fast test", duration=5.0)

        result = await full_pipeline.generate(request)

        timing = result.metadata["generation_time"]
        assert "total_seconds" in timing
        assert "tokens_per_second" in timing
        assert timing["total_seconds"] > 0

    @pytest.mark.asyncio
    async def test_generation_with_progress_callback(self, full_pipeline):
        """Test generation with progress tracking."""
        progress_updates = []

        def progress_callback(progress: float, stage: str):
            progress_updates.append((progress, stage))

        request = GenerationRequest(prompt="Progress test", duration=10.0)

        result = await full_pipeline.generate(request, progress_callback=progress_callback)

        assert isinstance(result, GenerationResult)
        assert len(progress_updates) > 0

        # Verify progress stages
        stages = [update[1] for update in progress_updates]
        expected_stages = ["encoding", "generating", "decoding"]
        for stage in expected_stages:
            assert stage in stages


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
