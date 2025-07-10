#!/usr/bin/env python3
"""
Check system requirements and available optional dependencies.

This script helps diagnose which dependencies are available and which
might need to be installed for full functionality.
"""

import sys
import json
from pathlib import Path

# Add music_gen to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from music_gen.utils import (
    check_system_requirements,
    get_available_backends,
    suggest_installations,
)


def main():
    """Check and report system requirements."""
    print("=" * 60)
    print("🎵 Music Gen AI - System Requirements Check")
    print("=" * 60)

    # Get system info
    system_info = check_system_requirements()

    print(f"\n📍 System Information:")
    print(f"   Python Version: {system_info['python_version']}")
    print(f"   Platform: {system_info['platform']}")

    # Check core dependencies
    core_deps = system_info["missing_core_deps"]
    if not core_deps:
        print(f"\n✅ Core Dependencies: All available")
    else:
        print(f"\n❌ Missing Core Dependencies:")
        for dep in core_deps:
            print(f"   - {dep}")
        print(f"\n   Install with:")
        print(f"   {suggest_installations(core_deps)}")

    # Check optional dependencies
    backends = system_info["available_backends"]
    available = [name for name, status in backends.items() if status]
    missing = [name for name, status in backends.items() if not status]

    print(f"\n📦 Available Optional Dependencies ({len(available)}/{len(backends)}):")
    for dep in available:
        print(f"   ✅ {dep}")

    if missing:
        print(f"\n📦 Missing Optional Dependencies:")
        for dep in missing:
            print(f"   ❌ {dep}")

        print(f"\n💡 To install missing dependencies:")
        print(suggest_installations(missing))

    # Feature availability summary
    print(f"\n🚀 Feature Availability:")
    print(f"   ✅ Core Music Generation: {'Yes' if not core_deps else 'No (missing core deps)'}")
    print(
        f"   ✅ Audio Processing: {'Yes' if backends.get('torchaudio', False) else 'Limited (fallback mode)'}"
    )
    print(f"   ✅ Audio Effects: {'Enhanced' if backends.get('pedalboard', False) else 'Basic'}")
    print(f"   ✅ Redis Task Storage: {'Yes' if backends.get('aioredis', False) else 'No'}")
    print(f"   ✅ PostgreSQL Storage: {'Yes' if backends.get('asyncpg', False) else 'No'}")
    print(f"   ✅ Advanced Metrics: {'Yes' if backends.get('librosa', False) else 'Limited'}")
    print(f"   ✅ Experiment Tracking: {'Yes' if backends.get('wandb', False) else 'No'}")

    # Recommendations
    print(f"\n🎯 Recommendations:")

    if core_deps:
        print(f"   🚨 CRITICAL: Install core dependencies first")
    elif not backends.get("torchaudio", False):
        print(f"   ⚠️  IMPORTANT: Install torchaudio for better audio processing")
    elif len(missing) > 5:
        print(f"   💡 OPTIONAL: Install additional dependencies for enhanced features")
    else:
        print(f"   🎉 EXCELLENT: System is well configured!")

    # Export to JSON for debugging
    output_file = Path(__file__).parent / "system_requirements.json"
    with open(output_file, "w") as f:
        json.dump(system_info, f, indent=2, default=str)

    print(f"\n💾 Detailed report saved to: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
