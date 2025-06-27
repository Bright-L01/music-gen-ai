"""
Tests for streaming generation functionality.
"""
import pytest
import torch
import asyncio
import json
import time
import numpy as np
from unittest.mock import Mock, AsyncMock, patch

from music_gen.streaming.generator import (
    StreamingGenerator,
    StreamingConfig,
    StreamingState,
    create_streaming_generator,
)
from music_gen.streaming.audio_streamer import (
    AudioChunk,
    AudioStreamer,
    CrossfadeProcessor,
    StreamingBuffer,
)
from music_gen.streaming.session import (
    StreamingSession,
    SessionManager,
    StreamingRequest,
    SessionState,
)
from music_gen.streaming.utils import (
    audio_to_base64,
    base64_to_audio,
    StreamingMetrics,
    LatencyTracker,
    validate_streaming_request,
)


class MockAudioTokenizer:
    """Mock audio tokenizer for testing."""
    
    def __init__(self):
        self.sample_rate = 24000
        self.frame_rate = 50  # 50 Hz
        self.num_quantizers = 4
    
    def get_sequence_length(self, duration: float) -> int:
        return int(duration * self.frame_rate * self.num_quantizers)
    
    def detokenize(self, tokens: torch.Tensor, time_frames: int) -> torch.Tensor:
        # Generate simple sine wave
        samples = int(time_frames * self.sample_rate / self.frame_rate)
        t = torch.linspace(0, samples / self.sample_rate, samples)
        audio = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0).unsqueeze(0)
        return audio


class MockModel:
    """Mock MusicGen model for testing."""
    
    def __init__(self):
        self.audio_tokenizer = MockAudioTokenizer()
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.pad_token_id = 0
        
        # Mock transformer
        self.transformer = Mock()
        self.transformer.return_value = {
            "logits": torch.randn(1, 1, 100),  # Random logits
            "past_key_values": None,
        }
    
    def parameters(self):
        return [torch.tensor([1.0])]  # Dummy parameter
    
    def prepare_inputs(self, texts, device, **kwargs):
        return {
            "text_hidden_states": torch.randn(1, 10, 64),
            "text_attention_mask": torch.ones(1, 10),
            "conditioning_embeddings": torch.randn(1, 32),
        }


class TestStreamingConfig:
    """Test streaming configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = StreamingConfig()
        
        assert config.chunk_duration == 1.0
        assert config.overlap_duration == 0.25
        assert config.lookahead_chunks == 2
        assert config.temperature == 0.9
        assert config.quality_mode == "balanced"
        assert config.enable_interruption == True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = StreamingConfig(
            chunk_duration=2.0,
            temperature=0.8,
            quality_mode="quality",
            enable_interruption=False,
        )
        
        assert config.chunk_duration == 2.0
        assert config.temperature == 0.8
        assert config.quality_mode == "quality"
        assert config.enable_interruption == False


class TestStreamingState:
    """Test streaming state management."""
    
    def test_initialization(self):
        """Test state initialization."""
        config = StreamingConfig()
        state = StreamingState(config)
        
        assert state.generated_tokens == []
        assert state.past_key_values is None
        assert state.current_chunk_idx == 0
        assert state.is_active == False
    
    def test_update_context(self):
        """Test context updating."""
        config = StreamingConfig()
        state = StreamingState(config)
        
        # Add tokens
        tokens = torch.tensor([1, 2, 3, 4])
        state.update_context(tokens, None)
        
        assert state.generated_tokens == [1, 2, 3, 4]
        
        # Add more tokens
        more_tokens = torch.tensor([5, 6])
        state.update_context(more_tokens, None)
        
        assert state.generated_tokens == [1, 2, 3, 4, 5, 6]
    
    def test_context_window_sliding(self):
        """Test context window sliding."""
        config = StreamingConfig(max_context_length=10, context_window_overlap=2)
        state = StreamingState(config)
        
        # Add tokens beyond max length
        long_tokens = torch.arange(15)  # 15 tokens
        state.update_context(long_tokens, None)
        
        # Should slide the window
        assert len(state.generated_tokens) <= config.max_context_length
        assert state.generated_tokens[-1] == 14  # Last token preserved


class TestAudioChunk:
    """Test audio chunk functionality."""
    
    def test_audio_chunk_creation(self):
        """Test audio chunk creation."""
        audio = torch.randn(1, 1, 1000)
        chunk = AudioChunk(
            chunk_id=1,
            audio=audio,
            sample_rate=24000,
            duration=1.0,
            timestamp=time.time(),
        )
        
        assert chunk.chunk_id == 1
        assert chunk.sample_rate == 24000
        assert chunk.duration == 1.0
        assert chunk.num_samples == 1000
        assert chunk.num_channels == 1
    
    def test_audio_shape_conversion(self):
        """Test automatic audio shape conversion."""
        # Test 1D audio
        audio_1d = torch.randn(1000)
        chunk = AudioChunk(
            chunk_id=1,
            audio=audio_1d,
            sample_rate=24000,
            duration=1.0,
            timestamp=time.time(),
        )
        assert chunk.audio.shape == (1, 1, 1000)
        
        # Test 2D audio
        audio_2d = torch.randn(2, 1000)  # Stereo
        chunk = AudioChunk(
            chunk_id=2,
            audio=audio_2d,
            sample_rate=24000,
            duration=1.0,
            timestamp=time.time(),
        )
        assert chunk.audio.shape == (1, 2, 1000)


class TestCrossfadeProcessor:
    """Test crossfade processing."""
    
    def test_crossfade_creation(self):
        """Test crossfade processor creation."""
        processor = CrossfadeProcessor(fade_duration=0.1, sample_rate=24000)
        
        assert processor.fade_duration == 0.1
        assert processor.sample_rate == 24000
        assert processor.fade_samples == 2400  # 0.1 * 24000
    
    def test_crossfade_chunks(self):
        """Test crossfading two chunks."""
        processor = CrossfadeProcessor(fade_duration=0.01, sample_rate=1000)  # 10 samples fade
        
        # Create test chunks
        audio1 = torch.ones(1, 1, 100)  # 100 samples of ones
        audio2 = torch.zeros(1, 1, 100)  # 100 samples of zeros
        
        chunk1 = AudioChunk(1, audio1, 1000, 0.1, time.time())
        chunk2 = AudioChunk(2, audio2, 1000, 0.1, time.time())
        
        # Crossfade
        result = processor.crossfade_chunks(chunk1, chunk2, overlap_samples=10)
        
        # Result should be longer than individual chunks due to overlap
        assert result.num_samples == 190  # 100 + 100 - 10 overlap
        
        # Check that crossfade region has intermediate values
        crossfade_region = result.audio[0, 0, 90:100]  # The overlap region
        assert torch.all(crossfade_region > 0)  # Should be between 0 and 1
        assert torch.all(crossfade_region < 1)
    
    def test_no_overlap_concatenation(self):
        """Test concatenation without overlap."""
        processor = CrossfadeProcessor(fade_duration=0.01, sample_rate=1000)
        
        audio1 = torch.ones(1, 1, 50)
        audio2 = torch.zeros(1, 1, 50)
        
        chunk1 = AudioChunk(1, audio1, 1000, 0.05, time.time())
        chunk2 = AudioChunk(2, audio2, 1000, 0.05, time.time())
        
        # No overlap
        result = processor.crossfade_chunks(chunk1, chunk2, overlap_samples=0)
        
        assert result.num_samples == 100  # Simple concatenation
        assert torch.all(result.audio[0, 0, :50] == 1)  # First half ones
        assert torch.all(result.audio[0, 0, 50:] == 0)  # Second half zeros


class TestStreamingBuffer:
    """Test streaming buffer functionality."""
    
    def test_buffer_initialization(self):
        """Test buffer initialization."""
        buffer = StreamingBuffer(buffer_size=5, min_buffer_size=2)
        
        assert buffer.buffer_size == 5
        assert buffer.min_buffer_size == 2
        assert buffer.is_buffering == True
        assert len(buffer.chunks) == 0
    
    def test_add_chunks(self):
        """Test adding chunks to buffer."""
        buffer = StreamingBuffer(buffer_size=3, min_buffer_size=2)
        
        # Add chunks
        for i in range(2):
            audio = torch.randn(1, 1, 100)
            chunk = AudioChunk(i, audio, 24000, 0.1, time.time())
            buffer.add_chunk(chunk)
        
        # Should stop buffering after min_buffer_size
        assert buffer.is_buffering == False
        assert len(buffer.chunks) == 2
    
    def test_buffer_overflow(self):
        """Test buffer overflow handling."""
        buffer = StreamingBuffer(buffer_size=2, min_buffer_size=1)
        
        # Add more chunks than buffer size
        for i in range(3):
            audio = torch.randn(1, 1, 100)
            chunk = AudioChunk(i, audio, 24000, 0.1, time.time())
            buffer.add_chunk(chunk)
        
        # Should maintain buffer size limit
        assert len(buffer.chunks) == 2
        # Should drop oldest chunk (chunk 0)
        assert buffer.chunks[0].chunk_id == 1
        assert buffer.chunks[1].chunk_id == 2
    
    def test_get_chunks(self):
        """Test getting chunks from buffer."""
        buffer = StreamingBuffer(buffer_size=3, min_buffer_size=2)
        
        # Add chunks
        chunks = []
        for i in range(3):
            audio = torch.randn(1, 1, 100)
            chunk = AudioChunk(i, audio, 24000, 0.1, time.time())
            chunks.append(chunk)
            buffer.add_chunk(chunk)
        
        # Get chunks
        first = buffer.get_next_chunk()
        assert first.chunk_id == 0
        
        second = buffer.get_next_chunk()
        assert second.chunk_id == 1
        
        # Should have one chunk left
        assert len(buffer.chunks) == 1


class TestStreamingUtilities:
    """Test streaming utility functions."""
    
    def test_audio_base64_conversion(self):
        """Test audio to/from base64 conversion."""
        # Create test audio
        original_audio = torch.randn(1, 1000)
        sample_rate = 24000
        
        # Convert to base64
        audio_b64 = audio_to_base64(original_audio, sample_rate)
        assert isinstance(audio_b64, str)
        assert len(audio_b64) > 0
        
        # Convert back
        recovered_audio = base64_to_audio(audio_b64, sample_rate)
        
        # Should have same length (may have slight differences due to int16 conversion)
        assert recovered_audio.shape[-1] == original_audio.shape[-1]
    
    def test_streaming_metrics(self):
        """Test streaming metrics tracking."""
        metrics = StreamingMetrics()
        
        assert metrics.total_chunks == 0
        assert metrics.real_time_factor == 0.0
        
        # Simulate some activity
        metrics.total_chunks = 10
        metrics.total_duration = 10.0
        metrics.total_generation_time = 5.0
        
        assert metrics.average_chunk_time == 0.5
        
        # Convert to dict
        metrics_dict = metrics.to_dict()
        assert "total_chunks" in metrics_dict
        assert "real_time_factor" in metrics_dict
    
    def test_latency_tracker(self):
        """Test latency tracking."""
        tracker = LatencyTracker(window_size=5)
        
        # Add measurements
        latencies = [0.1, 0.2, 0.15, 0.3, 0.25]
        for latency in latencies:
            tracker.add_measurement(latency)
        
        assert abs(tracker.average_latency - 0.2) < 0.01
        assert tracker.p95_latency >= 0.25  # Should be near the 95th percentile
    
    def test_request_validation(self):
        """Test streaming request validation."""
        # Valid request
        valid_request = {
            "prompt": "test music",
            "chunk_duration": 1.0,
            "temperature": 0.9,
            "genre": "jazz",
        }
        
        errors = validate_streaming_request(valid_request)
        assert len(errors) == 0
        
        # Invalid request
        invalid_request = {
            "prompt": "",  # Empty prompt
            "chunk_duration": 10.0,  # Too long
            "temperature": 3.0,  # Too high
            "genre": "invalid_genre",
        }
        
        errors = validate_streaming_request(invalid_request)
        assert len(errors) > 0
        assert any("prompt" in error.lower() for error in errors)
        assert any("chunk_duration" in error for error in errors)


class TestStreamingGenerator:
    """Test streaming generator functionality."""
    
    def test_generator_creation(self):
        """Test streaming generator creation."""
        model = MockModel()
        config = StreamingConfig(chunk_duration=1.0)
        
        generator = StreamingGenerator(model, config)
        
        assert generator.config == config
        assert generator.model == model
        assert generator.chunk_tokens == 200  # 1.0 * 50 * 4
    
    def test_factory_function(self):
        """Test factory function for creating generators."""
        model = MockModel()
        
        generator = create_streaming_generator(
            model=model,
            chunk_duration=2.0,
            quality_mode="fast",
        )
        
        assert generator.config.chunk_duration == 2.0
        assert generator.config.quality_mode == "fast"
    
    @pytest.mark.asyncio
    async def test_prepare_streaming(self):
        """Test streaming preparation."""
        model = MockModel()
        config = StreamingConfig()
        generator = StreamingGenerator(model, config)
        
        result = generator.prepare_streaming(
            texts=["test prompt"],
        )
        
        assert result["status"] == "prepared"
        assert "chunk_duration" in result
        assert generator.current_state.is_active == True


class TestStreamingSession:
    """Test streaming session management."""
    
    @pytest.mark.asyncio
    async def test_session_creation(self):
        """Test session creation."""
        model = MockModel()
        request = StreamingRequest(prompt="test music")
        
        session = StreamingSession("test-session", model, request)
        
        assert session.session_id == "test-session"
        assert session.request == request
        assert session.info.state == SessionState.CREATED
    
    @pytest.mark.asyncio
    async def test_session_preparation(self):
        """Test session preparation."""
        model = MockModel()
        request = StreamingRequest(prompt="test music")
        session = StreamingSession("test-session", model, request)
        
        # Mock the streaming generator creation
        with patch('music_gen.streaming.session.create_streaming_generator') as mock_create:
            mock_generator = Mock()
            mock_generator.prepare_streaming.return_value = {"status": "prepared"}
            mock_create.return_value = mock_generator
            
            result = await session.prepare()
            
            assert result["status"] == "prepared"
            assert session.info.state == SessionState.READY


class TestSessionManager:
    """Test session manager functionality."""
    
    @pytest.mark.asyncio
    async def test_session_manager_creation(self):
        """Test session manager creation."""
        model = MockModel()
        manager = SessionManager(model, max_concurrent_sessions=5)
        
        assert manager.model == model
        assert manager.max_concurrent_sessions == 5
        assert len(manager.sessions) == 0
    
    @pytest.mark.asyncio
    async def test_create_session(self):
        """Test creating a session."""
        model = MockModel()
        manager = SessionManager(model, max_concurrent_sessions=5)
        
        request = StreamingRequest(prompt="test music")
        session_id = await manager.create_session(request)
        
        assert session_id in manager.sessions
        assert len(manager.sessions) == 1
    
    @pytest.mark.asyncio
    async def test_session_limit(self):
        """Test session limit enforcement."""
        model = MockModel()
        manager = SessionManager(model, max_concurrent_sessions=2)
        
        # Create maximum sessions
        request = StreamingRequest(prompt="test music")
        session1 = await manager.create_session(request)
        session2 = await manager.create_session(request)
        
        # Third session should fail
        with pytest.raises(RuntimeError, match="Maximum concurrent sessions"):
            await manager.create_session(request)
    
    @pytest.mark.asyncio
    async def test_session_cleanup(self):
        """Test session cleanup."""
        model = MockModel()
        manager = SessionManager(model, max_concurrent_sessions=5)
        
        request = StreamingRequest(prompt="test music")
        session_id = await manager.create_session(request)
        
        # Remove session
        success = await manager.remove_session(session_id)
        
        assert success == True
        assert len(manager.sessions) == 0
    
    @pytest.mark.asyncio
    async def test_session_stats(self):
        """Test session statistics."""
        model = MockModel()
        manager = SessionManager(model, max_concurrent_sessions=5)
        
        stats = await manager.get_stats()
        
        assert "total_sessions" in stats
        assert "active_sessions" in stats
        assert stats["total_sessions"] == 0


if __name__ == "__main__":
    pytest.main([__file__])