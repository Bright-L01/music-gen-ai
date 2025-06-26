"""
EnCodec integration for audio tokenization and reconstruction.
"""
import torch
import torch.nn as nn
import torchaudio
from typing import Optional, Tuple, List, Union
import numpy as np
from encodec import EncodecModel
from encodec.utils import convert_audio
import logging

logger = logging.getLogger(__name__)


class EnCodecTokenizer(nn.Module):
    """EnCodec-based audio tokenizer for music generation."""
    
    def __init__(
        self,
        model_name: str = "facebook/encodec_24khz",
        sample_rate: int = 24000,
        bandwidth: float = 6.0,
        normalize: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.bandwidth = bandwidth
        self.normalize = normalize
        
        # Load EnCodec model
        try:
            self.encodec = EncodecModel.get_pretrained(model_name)
            self.encodec.set_target_bandwidth(bandwidth)
            
            if device is not None:
                self.encodec = self.encodec.to(device)
            
            # Set to evaluation mode by default
            self.encodec.eval()
            
        except Exception as e:
            logger.error(f"Failed to load EnCodec model {model_name}: {e}")
            raise
        
        # Model properties
        self.num_quantizers = self.encodec.quantizer.n_q
        self.codebook_size = self.encodec.quantizer.bins
        self.frame_rate = self.encodec.frame_rate
        self.hop_length = int(self.sample_rate / self.frame_rate)
        
        logger.info(f"Loaded EnCodec model with {self.num_quantizers} quantizers, "
                   f"codebook size {self.codebook_size}, frame rate {self.frame_rate}")
    
    def preprocess_audio(
        self,
        audio: torch.Tensor,
        original_sample_rate: Optional[int] = None,
    ) -> torch.Tensor:
        """Preprocess audio for EnCodec encoding."""
        
        # Ensure tensor is on correct device
        if hasattr(self.encodec, 'device'):
            audio = audio.to(self.encodec.device)
        
        # Convert sample rate if needed
        if original_sample_rate is not None and original_sample_rate != self.sample_rate:
            audio = convert_audio(
                audio,
                original_sample_rate,
                self.sample_rate,
                self.encodec.channels,
            )
        
        # Ensure correct number of channels
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # Add channel dimension
        if audio.dim() == 2 and audio.shape[0] != self.encodec.channels:
            if audio.shape[0] == 1 and self.encodec.channels == 1:
                pass  # Already mono
            elif audio.shape[0] == 2 and self.encodec.channels == 1:
                # Convert stereo to mono
                audio = audio.mean(dim=0, keepdim=True)
            elif audio.shape[0] == 1 and self.encodec.channels == 2:
                # Convert mono to stereo
                audio = audio.repeat(2, 1)
            else:
                raise ValueError(f"Cannot convert {audio.shape[0]} channels to {self.encodec.channels}")
        
        # Add batch dimension if needed
        if audio.dim() == 2:
            audio = audio.unsqueeze(0)
        
        # Normalize if requested
        if self.normalize:
            audio = audio / audio.abs().max().clamp(min=1e-8)
        
        return audio
    
    def encode(
        self,
        audio: torch.Tensor,
        sample_rate: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode audio to discrete tokens.
        
        Args:
            audio: Audio tensor of shape (batch, channels, time) or (channels, time) or (time,)
            sample_rate: Original sample rate of the audio
            
        Returns:
            codes: Discrete codes of shape (batch, num_quantizers, time_frames)
            scales: Scaling factors for reconstruction
        """
        
        # Preprocess audio
        audio = self.preprocess_audio(audio, sample_rate)
        
        # Encode with EnCodec
        with torch.no_grad():
            encoded_frames = self.encodec.encode(audio)
        
        # Extract codes from encoded frames
        codes_list = []
        scales_list = []
        
        for encoded_frame in encoded_frames:
            codes_list.append(encoded_frame[0])  # codes
            scales_list.append(encoded_frame[1])  # scales
        
        # Concatenate codes and scales
        codes = torch.cat(codes_list, dim=-1)  # (batch, num_quantizers, total_frames)
        scales = torch.cat(scales_list, dim=-1) if scales_list[0] is not None else None
        
        return codes, scales
    
    def decode(
        self,
        codes: torch.Tensor,
        scales: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decode discrete tokens back to audio.
        
        Args:
            codes: Discrete codes of shape (batch, num_quantizers, time_frames)
            scales: Optional scaling factors
            
        Returns:
            audio: Reconstructed audio of shape (batch, channels, time)
        """
        
        # Prepare encoded frames format expected by EnCodec
        encoded_frames = [(codes, scales)]
        
        # Decode with EnCodec
        with torch.no_grad():
            audio = self.encodec.decode(encoded_frames)
        
        return audio
    
    def tokenize(
        self,
        audio: torch.Tensor,
        sample_rate: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Tokenize audio into discrete tokens for language modeling.
        
        Args:
            audio: Input audio tensor
            sample_rate: Original sample rate
            
        Returns:
            tokens: Flattened tokens of shape (batch, sequence_length)
        """
        
        codes, _ = self.encode(audio, sample_rate)
        
        # Flatten codes for language modeling
        # Shape: (batch, num_quantizers, time_frames) -> (batch, num_quantizers * time_frames)
        batch_size, num_quantizers, time_frames = codes.shape
        tokens = codes.view(batch_size, -1)
        
        return tokens
    
    def detokenize(
        self,
        tokens: torch.Tensor,
        time_frames: int,
    ) -> torch.Tensor:
        """
        Convert flattened tokens back to audio.
        
        Args:
            tokens: Flattened tokens of shape (batch, sequence_length)
            time_frames: Number of time frames to reshape to
            
        Returns:
            audio: Reconstructed audio tensor
        """
        
        batch_size = tokens.shape[0]
        
        # Reshape tokens back to codes format
        codes = tokens.view(batch_size, self.num_quantizers, time_frames)
        
        # Decode to audio
        audio = self.decode(codes)
        
        return audio
    
    def get_sequence_length(self, audio_duration: float) -> int:
        """Calculate the sequence length for a given audio duration."""
        num_frames = int(audio_duration * self.frame_rate)
        return num_frames * self.num_quantizers
    
    def get_audio_duration(self, sequence_length: int) -> float:
        """Calculate audio duration for a given sequence length."""
        num_frames = sequence_length // self.num_quantizers
        return num_frames / self.frame_rate
    
    def forward(
        self,
        audio: Optional[torch.Tensor] = None,
        tokens: Optional[torch.Tensor] = None,
        sample_rate: Optional[int] = None,
        mode: str = "encode",
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass - can be used for encoding or decoding.
        
        Args:
            audio: Input audio (for encoding)
            tokens: Input tokens (for decoding)
            sample_rate: Sample rate of input audio
            mode: "encode" or "decode"
            
        Returns:
            For encoding: (codes, scales)
            For decoding: audio
        """
        
        if mode == "encode":
            if audio is None:
                raise ValueError("Audio must be provided for encoding")
            return self.encode(audio, sample_rate)
        
        elif mode == "decode":
            if tokens is None:
                raise ValueError("Tokens must be provided for decoding")
            return self.decode(tokens)
        
        else:
            raise ValueError(f"Unknown mode: {mode}")


class MultiResolutionTokenizer(nn.Module):
    """Multi-resolution audio tokenizer using multiple EnCodec models."""
    
    def __init__(
        self,
        model_configs: List[dict],
        fusion_method: str = "hierarchical",
    ):
        super().__init__()
        
        self.fusion_method = fusion_method
        self.tokenizers = nn.ModuleList()
        
        for config in model_configs:
            tokenizer = EnCodecTokenizer(**config)
            self.tokenizers.append(tokenizer)
        
        self.num_tokenizers = len(self.tokenizers)
        
    def encode_multi_resolution(
        self,
        audio: torch.Tensor,
        sample_rate: Optional[int] = None,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Encode audio at multiple resolutions."""
        
        results = []
        for tokenizer in self.tokenizers:
            codes, scales = tokenizer.encode(audio, sample_rate)
            results.append((codes, scales))
        
        return results
    
    def decode_multi_resolution(
        self,
        multi_codes: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> List[torch.Tensor]:
        """Decode codes at multiple resolutions."""
        
        results = []
        for i, (codes, scales) in enumerate(multi_codes):
            audio = self.tokenizers[i].decode(codes, scales)
            results.append(audio)
        
        return results
    
    def forward(
        self,
        audio: Optional[torch.Tensor] = None,
        multi_codes: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        sample_rate: Optional[int] = None,
        mode: str = "encode",
    ) -> Union[List[Tuple[torch.Tensor, torch.Tensor]], List[torch.Tensor]]:
        """Forward pass for multi-resolution processing."""
        
        if mode == "encode":
            return self.encode_multi_resolution(audio, sample_rate)
        elif mode == "decode":
            return self.decode_multi_resolution(multi_codes)
        else:
            raise ValueError(f"Unknown mode: {mode}")


def create_audio_tokenizer(
    model_name: str = "facebook/encodec_24khz",
    **kwargs
) -> EnCodecTokenizer:
    """Factory function to create an audio tokenizer."""
    
    return EnCodecTokenizer(model_name=model_name, **kwargs)


def load_audio_file(
    file_path: str,
    target_sample_rate: int = 24000,
    normalize: bool = True,
) -> Tuple[torch.Tensor, int]:
    """Load and preprocess an audio file."""
    
    try:
        # Load audio file
        audio, sample_rate = torchaudio.load(file_path)
        
        # Resample if needed
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=target_sample_rate,
            )
            audio = resampler(audio)
            sample_rate = target_sample_rate
        
        # Normalize if requested
        if normalize:
            audio = audio / audio.abs().max().clamp(min=1e-8)
        
        return audio, sample_rate
        
    except Exception as e:
        logger.error(f"Failed to load audio file {file_path}: {e}")
        raise


def save_audio_file(
    audio: torch.Tensor,
    file_path: str,
    sample_rate: int = 24000,
    normalize: bool = True,
) -> None:
    """Save audio tensor to file."""
    
    try:
        # Ensure audio is on CPU
        if audio.is_cuda:
            audio = audio.cpu()
        
        # Normalize if requested
        if normalize:
            audio = audio / audio.abs().max().clamp(min=1e-8)
        
        # Save audio file
        torchaudio.save(file_path, audio, sample_rate)
        
    except Exception as e:
        logger.error(f"Failed to save audio file {file_path}: {e}")
        raise