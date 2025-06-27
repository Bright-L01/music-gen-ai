"""Configuration for multi-instrument generation."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from music_gen.models.transformer.config import TransformerConfig


@dataclass
class InstrumentConfig:
    """Configuration for a single instrument."""
    
    name: str
    midi_program: int  # General MIDI program number
    frequency_range: tuple[float, float] = (20.0, 20000.0)  # Hz
    typical_octave_range: tuple[int, int] = (2, 7)
    polyphonic: bool = True
    percussion: bool = False
    sustained: bool = True
    default_volume: float = 0.7
    default_pan: float = 0.0  # -1.0 (left) to 1.0 (right)
    

@dataclass
class MultiInstrumentConfig(TransformerConfig):
    """Configuration for multi-instrument MusicGen model."""
    
    # Instrument settings
    num_instruments: int = 32
    instrument_embedding_dim: int = 256
    max_tracks: int = 8
    
    # Architecture extensions
    use_instrument_attention: bool = True
    instrument_cross_attention_layers: List[int] = field(default_factory=lambda: [4, 8, 12, 16])
    parallel_generation: bool = True
    
    # Track separation
    use_source_separation: bool = True
    separation_model: str = "demucs"  # "demucs" or "spleeter"
    separation_checkpoint: Optional[str] = None
    
    # Mixing settings
    use_automatic_mixing: bool = True
    mixing_latent_dim: int = 128
    
    # Generation settings
    instrument_dropout: float = 0.1
    track_dropout: float = 0.1
    
    # Supported instruments (expandable)
    instruments: Dict[str, InstrumentConfig] = field(default_factory=lambda: {
        # Keyboards
        "piano": InstrumentConfig("piano", 0, (27.5, 4186.0), (1, 8), True, False, True),
        "electric_piano": InstrumentConfig("electric_piano", 4, (27.5, 4186.0), (1, 8), True, False, True),
        "harpsichord": InstrumentConfig("harpsichord", 6, (27.5, 4186.0), (2, 7), True, False, False),
        "organ": InstrumentConfig("organ", 19, (16.35, 8372.0), (1, 9), True, False, True),
        "synthesizer": InstrumentConfig("synthesizer", 38, (20.0, 20000.0), (0, 10), True, False, True),
        
        # Strings
        "violin": InstrumentConfig("violin", 40, (196.0, 3520.0), (3, 7), True, False, True),
        "viola": InstrumentConfig("viola", 41, (130.8, 1760.0), (3, 6), True, False, True),
        "cello": InstrumentConfig("cello", 42, (65.4, 880.0), (2, 5), True, False, True),
        "double_bass": InstrumentConfig("double_bass", 43, (41.2, 440.0), (1, 4), True, False, True),
        "harp": InstrumentConfig("harp", 46, (32.7, 3322.4), (1, 7), True, False, False),
        
        # Guitars
        "acoustic_guitar": InstrumentConfig("acoustic_guitar", 24, (82.4, 880.0), (2, 5), True, False, False),
        "electric_guitar": InstrumentConfig("electric_guitar", 26, (82.4, 1318.5), (2, 6), True, False, False),
        "bass_guitar": InstrumentConfig("bass_guitar", 33, (41.2, 440.0), (1, 4), False, False, False),
        
        # Brass
        "trumpet": InstrumentConfig("trumpet", 56, (146.8, 1174.7), (3, 6), False, False, True),
        "trombone": InstrumentConfig("trombone", 57, (58.3, 698.5), (2, 5), False, False, True),
        "french_horn": InstrumentConfig("french_horn", 60, (58.3, 698.5), (2, 5), False, False, True),
        "tuba": InstrumentConfig("tuba", 58, (29.1, 349.2), (1, 4), False, False, True),
        
        # Woodwinds
        "flute": InstrumentConfig("flute", 73, (261.6, 2093.0), (4, 7), False, False, True),
        "clarinet": InstrumentConfig("clarinet", 71, (146.8, 1568.0), (3, 6), False, False, True),
        "saxophone": InstrumentConfig("saxophone", 66, (103.8, 1396.9), (2, 6), False, False, True),
        "oboe": InstrumentConfig("oboe", 68, (233.1, 1568.0), (3, 6), False, False, True),
        
        # Percussion
        "drums": InstrumentConfig("drums", 128, (20.0, 20000.0), (0, 10), True, True, False),
        "timpani": InstrumentConfig("timpani", 47, (65.4, 523.3), (2, 4), False, True, True),
        "xylophone": InstrumentConfig("xylophone", 13, (523.3, 4186.0), (4, 7), False, True, False),
        "vibraphone": InstrumentConfig("vibraphone", 11, (174.6, 1396.9), (3, 6), True, True, True),
        
        # Voice
        "soprano": InstrumentConfig("soprano", 52, (261.6, 1046.5), (4, 6), False, False, True),
        "alto": InstrumentConfig("alto", 53, (174.6, 698.5), (3, 5), False, False, True),
        "tenor": InstrumentConfig("tenor", 54, (130.8, 523.3), (3, 5), False, False, True),
        "bass_voice": InstrumentConfig("bass_voice", 55, (87.3, 349.2), (2, 4), False, False, True),
        
        # Additional
        "choir": InstrumentConfig("choir", 52, (87.3, 1046.5), (2, 6), True, False, True),
        "synth_pad": InstrumentConfig("synth_pad", 88, (20.0, 20000.0), (0, 10), True, False, True),
        "synth_lead": InstrumentConfig("synth_lead", 80, (65.4, 4186.0), (2, 7), False, False, True),
    })
    
    def get_instrument_config(self, instrument_name: str) -> Optional[InstrumentConfig]:
        """Get configuration for a specific instrument."""
        return self.instruments.get(instrument_name.lower())
    
    def get_instrument_names(self) -> List[str]:
        """Get list of all supported instrument names."""
        return list(self.instruments.keys())