"""
Advanced Music Generation Features

This module contains advanced features for commercial-grade music generation
including extended composition, intelligent music theory, sophisticated
conditioning systems, style transfer, automatic mixing/mastering,
quality assurance, and real-time control.
"""

from .composition import (
    AdvancedComposer,
    CompositionManager,
    SectionConfig,
    SongSection,
    SongStructure,
)
from .conditioning import (
    AdvancedConditioningSystem,
    ArrangementStyle,
    ArrangementTemplate,
    ConditioningManager,
    GenreArrangementLibrary,
    InstrumentConfig,
    InstrumentFamily,
)
from .mixing_mastering import (
    AutoMasteringManager,
    AutoMixingEngine,
    IntelligentEQ,
    LoudnessStandard,
    MasteringChain,
    MasteringEngine,
    MixingPreset,
    ProcessingQuality,
    SmartCompressor,
    StereoProcessor,
)
from .music_theory import (
    Chord,
    ChordProgression,
    ChordQuality,
    MusicTheoryEngine,
    Note,
    Scale,
    ScaleType,
    TimeSignature,
)
from .quality_assurance import (
    AudioQualityAnalyzer,
    MusicalCoherenceAnalyzer,
    QualityAssuranceManager,
    QualityAssuranceSystem,
    QualityIssue,
    QualityLevel,
    QualityMetric,
    QualityReport,
    TechnicalQualityAnalyzer,
)
from .realtime_control import (
    AutomationCurve,
    ControlEvent,
    ControlMode,
    GenerationState,
    GestureRecognizer,
    ParameterControl,
    ParameterType,
    RealTimeControlAPI,
    RealTimeController,
    RealTimeGenerationManager,
)
from .style_transfer import (
    InterpolationMode,
    StyleAttribute,
    StyleEncoder,
    StyleInterpolator,
    StyleProfile,
    StyleTransferManager,
    StyleTransferSystem,
)

__all__ = [
    # Composition
    "AdvancedComposer",
    "CompositionManager",
    "SongSection",
    "SectionConfig",
    "SongStructure",
    # Music Theory
    "MusicTheoryEngine",
    "ChordProgression",
    "Chord",
    "ChordQuality",
    "Scale",
    "ScaleType",
    "Note",
    "TimeSignature",
    # Conditioning
    "AdvancedConditioningSystem",
    "ConditioningManager",
    "ArrangementTemplate",
    "ArrangementStyle",
    "InstrumentFamily",
    "InstrumentConfig",
    "GenreArrangementLibrary",
    # Style Transfer
    "StyleTransferSystem",
    "StyleTransferManager",
    "StyleProfile",
    "StyleAttribute",
    "InterpolationMode",
    "StyleEncoder",
    "StyleInterpolator",
    # Mixing & Mastering
    "AutoMixingEngine",
    "MasteringEngine",
    "AutoMasteringManager",
    "IntelligentEQ",
    "SmartCompressor",
    "StereoProcessor",
    "ProcessingQuality",
    "LoudnessStandard",
    "MixingPreset",
    "MasteringChain",
    # Quality Assurance
    "QualityAssuranceSystem",
    "QualityAssuranceManager",
    "QualityReport",
    "QualityIssue",
    "QualityMetric",
    "QualityLevel",
    "AudioQualityAnalyzer",
    "MusicalCoherenceAnalyzer",
    "TechnicalQualityAnalyzer",
    # Real-Time Control
    "RealTimeController",
    "RealTimeGenerationManager",
    "RealTimeControlAPI",
    "ParameterControl",
    "ParameterType",
    "ControlMode",
    "ControlEvent",
    "GenerationState",
    "AutomationCurve",
    "GestureRecognizer",
]
