#!/usr/bin/env python3
"""System verification script to check all components are working."""

import importlib
import sys
from pathlib import Path

print("=== MusicGen AI System Check ===\n")

# Check Python version
print(f"Python version: {sys.version}")

# Test core imports
modules_to_test = [
    ("Core Model", "music_gen.models.musicgen"),
    ("Transformer", "music_gen.models.transformer.model"),
    ("Multi-Instrument", "music_gen.models.multi_instrument"),
    ("Audio Mixing", "music_gen.audio.mixing"),
    ("Track Separation", "music_gen.audio.separation"),
    ("MIDI Export", "music_gen.export.midi"),
    ("API Main", "music_gen.api.main"),
    ("Streaming", "music_gen.streaming"),
    ("Web App", "music_gen.web.app"),
    ("Utils", "music_gen.utils.audio"),
]

print("\n--- Module Import Test ---")
failed_imports = []
for name, module_path in modules_to_test:
    try:
        importlib.import_module(module_path)
        print(f"✓ {name}: {module_path}")
    except ImportError as e:
        print(f"✗ {name}: {module_path} - {e}")
        failed_imports.append((name, str(e)))

# Check configuration files
print("\n--- Configuration Files ---")
config_files = [
    "configs/config.yaml",
    "configs/model/base.yaml",
    "configs/training/base.yaml",
    "pyproject.toml",
    "environment.yml",
]

for config_file in config_files:
    path = Path(config_file)
    if path.exists():
        print(f"✓ {config_file}")
    else:
        print(f"✗ {config_file} - Not found")

# Check documentation
print("\n--- Documentation ---")
doc_files = [
    "README.md",
    "CLAUDE.md",
    "COMPREHENSIVE_DOCUMENTATION.md",
    "DEVELOPMENT_ROADMAP.md",
    "PROJECT_SUMMARY.md",
    "MULTI_INSTRUMENT_SUMMARY.md",
]

for doc_file in doc_files:
    path = Path(doc_file)
    if path.exists():
        size = path.stat().st_size
        print(f"✓ {doc_file} ({size:,} bytes)")
    else:
        print(f"✗ {doc_file} - Not found")

# Check test files
print("\n--- Test Coverage ---")
test_dirs = ["tests/unit", "tests/integration", "tests/e2e"]
total_tests = 0
for test_dir in test_dirs:
    path = Path(test_dir)
    if path.exists():
        test_files = list(path.glob("test_*.py"))
        print(f"✓ {test_dir}: {len(test_files)} test files")
        total_tests += len(test_files)
    else:
        print(f"✗ {test_dir} - Not found")

print(f"\nTotal test files: {total_tests}")

# Summary
print("\n=== Summary ===")
if failed_imports:
    print(f"⚠️  {len(failed_imports)} modules failed to import:")
    for name, error in failed_imports:
        print(f"   - {name}: {error}")
else:
    print("✅ All modules imported successfully!")

print("\n--- Quick Feature Test ---")

# Test multi-instrument configuration
try:
    from music_gen.models.multi_instrument import MultiInstrumentConfig

    config = MultiInstrumentConfig()
    instruments = config.get_instrument_names()
    print(f"✓ Multi-instrument system: {len(instruments)} instruments available")
    print(f"  Sample instruments: {', '.join(instruments[:5])}...")
except Exception as e:
    print(f"✗ Multi-instrument test failed: {e}")

# Test mixing engine
try:
    from music_gen.audio.mixing import MixingConfig

    mix_config = MixingConfig()
    print(f"✓ Mixing engine: {mix_config.sample_rate}Hz, {mix_config.channels} channels")
except Exception as e:
    print(f"✗ Mixing engine test failed: {e}")

# Test MIDI export
try:
    from music_gen.export.midi import MIDIExportConfig

    midi_config = MIDIExportConfig()
    print(
        f"✓ MIDI export: {midi_config.tempo} BPM, quantization={'enabled' if midi_config.quantize else 'disabled'}"
    )
except Exception as e:
    print(f"✗ MIDI export test failed: {e}")

print("\n✅ System check complete!")
