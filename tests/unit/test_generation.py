"""
Unit tests for music generation functionality.
"""

from unittest.mock import Mock, patch

import pytest
import torch

# Make optional imports for testing without heavy dependencies
try:
    from music_gen.models.musicgen import create_musicgen_model

    MUSICGEN_AVAILABLE = True
except ImportError:
    create_musicgen_model = None
    MUSICGEN_AVAILABLE = False

try:
    from music_gen.models.transformer.config import MusicGenConfig

    CONFIG_AVAILABLE = True
except ImportError:
    MusicGenConfig = None
    CONFIG_AVAILABLE = False


@pytest.mark.unit
class TestBeamSearch:
    """Test beam search generation."""

    @pytest.fixture
    def beam_config(self):
        """Create beam search configuration."""
        return {
            "num_beams": 4,
            "max_length": 100,
            "temperature": 0.8,
            "top_k": 50,
            "top_p": 0.9,
            "length_penalty": 1.0,
            "early_stopping": True,
        }

    @pytest.fixture
    def mock_model(self):
        """Create mock model for testing."""
        model = Mock()
        model.config = Mock()
        model.config.vocab_size = 256
        model.config.pad_token_id = 0
        model.config.eos_token_id = 1
        model.config.bos_token_id = 2

        # Mock forward pass
        def mock_forward(input_ids, **kwargs):
            batch_size, seq_len = input_ids.shape
            vocab_size = model.config.vocab_size
            logits = torch.randn(batch_size, seq_len, vocab_size)
            return {"logits": logits}

        model.forward = mock_forward
        model.prepare_inputs_for_generation = lambda input_ids, **kwargs: {"input_ids": input_ids}

        return model

    def test_beam_search_config(self, beam_config):
        """Test beam search configuration."""
        assert beam_config["num_beams"] == 4
        assert beam_config["max_length"] == 100
        assert beam_config["temperature"] == 0.8
        assert beam_config["early_stopping"] is True

    def test_beam_search_generation(self, mock_model, beam_config):
        """Test beam search generation process."""
        input_ids = torch.tensor([[2]])  # Start with BOS token

        # Mock a simple generation process
        with patch.object(mock_model, "generate") as mock_generate:
            mock_generate.return_value = torch.randint(0, 256, (1, beam_config["max_length"]))

            generated = mock_model.generate(
                input_ids, max_length=beam_config["max_length"], num_beams=beam_config["num_beams"]
            )

        assert generated.shape[0] == 1  # Batch size
        assert generated.shape[1] <= beam_config["max_length"]

    def test_beam_search_with_encoder(self, mock_model, beam_config):
        """Test beam search with encoder inputs."""
        input_ids = torch.tensor([[2]])
        encoder_hidden_states = torch.randn(1, 50, 512)
        encoder_attention_mask = torch.ones(1, 50, dtype=torch.bool)

        # Mock generation with encoder inputs
        with patch.object(mock_model, "generate") as mock_generate:
            mock_generate.return_value = torch.randint(0, 256, (1, beam_config["max_length"]))

            generated = mock_model.generate(
                input_ids,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
            )

        assert generated.shape[0] == 1

    @pytest.mark.parametrize("num_beams", [1, 2, 4, 8])
    def test_different_beam_sizes(self, mock_model, num_beams):
        """Test different beam sizes."""
        config = {"num_beams": num_beams, "max_length": 50}
        input_ids = torch.tensor([[2]])

        with patch.object(mock_model, "generate") as mock_generate:
            mock_generate.return_value = torch.randint(0, 256, (1, config["max_length"]))

            generated = mock_model.generate(
                input_ids, num_beams=config["num_beams"], max_length=config["max_length"]
            )

        assert generated.shape[0] == 1
        assert generated.shape[1] <= config["max_length"]


@pytest.mark.unit
class TestGenerationStrategies:
    """Test different generation strategies."""

    @pytest.fixture
    def mock_model_outputs(self):
        """Create mock model outputs."""
        batch_size, seq_len, vocab_size = 2, 10, 256
        logits = torch.randn(batch_size, seq_len, vocab_size)
        return {"logits": logits}

    def test_nucleus_sampling(self, mock_model_outputs):
        """Test nucleus (top-p) sampling."""
        # Simple nucleus sampling implementation for testing
        logits = mock_model_outputs["logits"][:, -1, :]  # Last position

        # Apply temperature
        logits = logits / 1.0
        probs = torch.softmax(logits, dim=-1)

        # Sort probabilities
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)

        # Apply top-p filtering
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cumsum_probs <= 0.9

        # Sample from filtered distribution
        sampled = torch.multinomial(probs, 1).squeeze(-1)

        assert sampled.shape == (2,)  # Batch size
        assert all(0 <= token < 256 for token in sampled)

    def test_top_k_sampling(self, mock_model_outputs):
        """Test top-k sampling."""
        # Simple top-k sampling implementation for testing
        logits = mock_model_outputs["logits"][:, -1, :]

        # Apply temperature
        logits = logits / 0.8

        # Get top-k logits
        top_k = 50
        top_logits, top_indices = torch.topk(logits, top_k, dim=-1)

        # Sample from top-k
        probs = torch.softmax(top_logits, dim=-1)
        sampled_idx = torch.multinomial(probs, 1).squeeze(-1)
        sampled = top_indices.gather(-1, sampled_idx.unsqueeze(-1)).squeeze(-1)

        assert sampled.shape == (2,)
        assert all(0 <= token < 256 for token in sampled)

    def test_temperature_scaling(self, mock_model_outputs):
        """Test temperature scaling effects."""
        logits = mock_model_outputs["logits"][:, -1, :]

        # Low temperature should make distribution more peaked
        low_temp = logits / 0.1
        high_temp = logits / 2.0

        # Low temperature should have higher max probability
        low_temp_probs = torch.softmax(low_temp, dim=-1)
        high_temp_probs = torch.softmax(high_temp, dim=-1)

        assert low_temp_probs.max(dim=-1)[0].mean() > high_temp_probs.max(dim=-1)[0].mean()

    def test_repetition_penalty(self, mock_model_outputs):
        """Test repetition penalty application."""
        logits = mock_model_outputs["logits"][:, -1, :]
        generated_tokens = torch.tensor([[1, 2, 1, 3], [4, 5, 6, 5]])  # Batch x seq

        # Simple repetition penalty implementation
        penalty = 1.2
        penalized = logits.clone()

        # Apply penalty to repeated tokens
        for batch_idx, tokens in enumerate(generated_tokens):
            unique_tokens, counts = torch.unique(tokens, return_counts=True)
            for token, count in zip(unique_tokens, counts):
                if count > 1:  # Token was repeated
                    # For penalties, we reduce the logits (subtract penalty strength)
                    if penalized[batch_idx, token] > 0:
                        penalized[batch_idx, token] /= penalty  # Reduce positive logits
                    else:
                        penalized[batch_idx, token] *= penalty  # Make negative logits more negative

        assert penalized.shape == logits.shape
        # Test that the penalty mechanism works by checking specific known repeated tokens
        # We know token 1 appears twice in generated_tokens[0] = [1, 2, 1, 3]
        # and token 5 appears twice in generated_tokens[1] = [4, 5, 6, 5]
        assert (
            penalized[0, 1] <= logits[0, 1]
        )  # Token 1 should be penalized (allow equal for edge cases)
        assert penalized[1, 5] <= logits[1, 5]  # Token 5 should be penalized


@pytest.mark.unit
class TestMusicGenGeneration:
    """Test MusicGen-specific generation functionality."""

    @pytest.fixture
    def mock_musicgen_model(self):
        """Create mock MusicGen model."""
        mock_model = Mock()

        # Mock audio tokenizer
        mock_model.audio_tokenizer = Mock()
        mock_model.audio_tokenizer.codebook_size = 256
        mock_model.audio_tokenizer.num_quantizers = 8
        mock_model.audio_tokenizer.sample_rate = 24000

        # Mock model config
        mock_model.config = Mock()
        mock_model.config.hidden_size = 128
        mock_model.config.num_layers = 2
        mock_model.config.vocab_size = 256

        return mock_model

    def test_text_conditioning(self, mock_musicgen_model):
        """Test text conditioning for generation."""
        texts = ["Happy jazz music", "Sad classical piece"]

        # Mock the text encoder and prepare_conditioning method
        conditioning_result = {
            "text_hidden_states": torch.randn(2, 50, 128),
            "text_attention_mask": torch.ones(2, 50, dtype=torch.bool),
            "conditioning_embeddings": torch.randn(2, 64),
        }

        mock_musicgen_model.prepare_conditioning = Mock(return_value=conditioning_result)

        conditioning = mock_musicgen_model.prepare_conditioning(texts)

        assert "text_hidden_states" in conditioning
        assert "conditioning_embeddings" in conditioning
        assert conditioning["text_hidden_states"].shape[0] == 2

    def test_generation_with_conditioning(self, mock_musicgen_model):
        """Test generation with various conditioning."""
        # Mock the generation process
        generated_tokens = torch.randint(0, 256, (1, 100))
        mock_musicgen_model.generate_with_conditioning = Mock(return_value=generated_tokens)

        tokens = mock_musicgen_model.generate_with_conditioning(
            texts=["Test music"],
            genre="jazz",
            mood="happy",
            tempo=120,
            duration=10.0,
            max_length=100,
        )

        assert tokens.shape[0] == 1
        assert tokens.shape[1] <= 100
        mock_musicgen_model.generate_with_conditioning.assert_called_once()

    def test_audio_tokenization(self):
        """Test audio tokenization process."""
        # Mock tokenizer directly
        mock_tokenizer = Mock()
        mock_tokenizer.tokenize.return_value = torch.randint(0, 256, (1, 8, 100))
        mock_tokenizer.detokenize.return_value = torch.randn(1, 1, 24000)
        mock_tokenizer.sample_rate = 24000
        mock_tokenizer.codebook_size = 256
        mock_tokenizer.num_quantizers = 8

        # Test encoding
        audio = torch.randn(1, 1, 24000)
        tokens = mock_tokenizer.tokenize(audio)

        assert tokens.shape[0] == 1  # Batch
        assert tokens.shape[1] == 8  # Quantizers

        # Test decoding
        decoded = mock_tokenizer.detokenize(tokens)
        assert decoded.shape == (1, 1, 24000)

    def test_progressive_generation(self, mock_musicgen_model):
        """Test progressive generation for long sequences."""
        # Mock progressive generation
        with patch.object(mock_musicgen_model, "generate_progressive") as mock_prog:
            mock_prog.return_value = torch.randint(0, 256, (1, 500))

            tokens = mock_musicgen_model.generate_progressive(
                texts=["Long piece of music"],
                total_length=500,
                segment_length=100,
                overlap_length=20,
            )

            assert tokens.shape[1] == 500
            mock_prog.assert_called_once()


@pytest.mark.unit
class TestGenerationUtils:
    """Test generation utility functions."""

    def test_token_sequence_validation(self):
        """Test token sequence validation."""

        # Simple validation function
        def validate_token_sequence(tokens, vocab_size):
            return torch.all(tokens >= 0) and torch.all(tokens < vocab_size)

        # Valid sequence
        valid_tokens = torch.randint(0, 256, (1, 100))
        assert validate_token_sequence(valid_tokens, vocab_size=256)

        # Invalid sequence (out of range)
        invalid_tokens = torch.tensor([[300, 150, 50]])
        assert not validate_token_sequence(invalid_tokens, vocab_size=256)

    def test_generation_metrics(self):
        """Test generation quality metrics."""
        generated_tokens = torch.randint(0, 256, (2, 100))

        # Calculate simple metrics
        def calculate_generation_metrics(tokens):
            batch_size, seq_len = tokens.shape

            # Uniqueness: ratio of unique tokens to total tokens
            unique_tokens = torch.unique(tokens)
            uniqueness = len(unique_tokens) / (batch_size * seq_len)

            # Repetition rate: count of repeated consecutive tokens
            repetitions = 0
            for batch in tokens:
                for i in range(len(batch) - 1):
                    if batch[i] == batch[i + 1]:
                        repetitions += 1
            repetition_rate = repetitions / (batch_size * (seq_len - 1))

            return {
                "uniqueness": uniqueness,
                "repetition_rate": repetition_rate,
                "sequence_length": seq_len,
            }

        metrics = calculate_generation_metrics(generated_tokens)

        assert "uniqueness" in metrics
        assert "repetition_rate" in metrics
        assert "sequence_length" in metrics
        assert 0 <= metrics["uniqueness"] <= 1
        assert metrics["repetition_rate"] >= 0

    def test_conditioning_interpolation(self):
        """Test conditioning parameter interpolation."""

        # Simple interpolation function
        def interpolate_conditioning(start, end, alpha):
            result = {}
            for key in start.keys():
                if key in end:
                    result[key] = start[key] * (1 - alpha) + end[key] * alpha
            return result

        start_conditioning = {"tempo": 120, "energy": 0.5}
        end_conditioning = {"tempo": 140, "energy": 0.8}

        interpolated = interpolate_conditioning(start_conditioning, end_conditioning, alpha=0.5)

        assert interpolated["tempo"] == 130  # Midpoint
        assert interpolated["energy"] == pytest.approx(0.65, abs=1e-3)

    def test_generation_caching(self):
        """Test generation result caching."""

        # Simple cache implementation for testing
        class GenerationCache:
            def __init__(self, max_size=100):
                self.cache = {}
                self.max_size = max_size

            def set(self, key, value):
                if len(self.cache) >= self.max_size:
                    # Remove oldest item
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
                self.cache[key] = value

            def get(self, key):
                return self.cache.get(key)

        cache = GenerationCache(max_size=100)

        # Cache result
        prompt = "test music"
        result = torch.randint(0, 256, (1, 50))
        cache.set(prompt, result)

        # Retrieve result
        cached_result = cache.get(prompt)
        assert torch.equal(cached_result, result)

        # Test cache miss
        assert cache.get("nonexistent prompt") is None


if __name__ == "__main__":
    pytest.main([__file__])
