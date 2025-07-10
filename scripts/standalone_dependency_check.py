#!/usr/bin/env python3
"""
Standalone dependency validation script.

This script can run independently of the main Music Gen AI module
to validate dependencies before system initialization.
"""

import sys
import importlib
from typing import Dict, List, Tuple


class StandaloneDependencyChecker:
    """Standalone dependency checker that doesn't require main module."""

    CRITICAL_DEPENDENCIES = [
        ("torch", "PyTorch ML framework", "2.0.0"),
        ("torchaudio", "PyTorch audio processing", "2.0.0"),
        ("transformers", "HuggingFace transformers library", "4.30.0"),
        ("encodec", "Facebook EnCodec audio codec", "0.1.1"),
        ("librosa", "Audio analysis library", "0.10.0"),
        ("soundfile", "Audio file I/O", "0.12.0"),
        ("fastapi", "Web framework for API", "0.100.0"),
        ("numpy", "Numerical computing", "1.21.0"),
        ("scipy", "Scientific computing", "1.9.0"),
    ]

    HIGH_PRIORITY_DEPENDENCIES = [
        ("spleeter", "Source separation library", "2.3.0"),
        ("demucs", "State-of-the-art source separation", "4.0.0"),
        ("sentence_transformers", "Text embeddings for CLAP scoring", "2.2.0"),
        ("scikit-learn", "Machine learning utilities", "1.3.0"),
    ]

    OPTIONAL_DEPENDENCIES = [
        ("aioredis", "Redis async client", "2.0.0"),
        ("asyncpg", "PostgreSQL async client", "0.28.0"),
        ("pedalboard", "High-quality audio effects", "0.8.0"),
        ("wandb", "Experiment tracking", None),
        ("tensorboard", "TensorBoard logging", None),
        ("pretty_midi", "MIDI processing", None),
    ]

    def __init__(self):
        self.results = {
            "critical_missing": [],
            "critical_ok": [],
            "high_priority_missing": [],
            "high_priority_ok": [],
            "optional_missing": [],
            "optional_ok": [],
        }

    def check_dependency(
        self, package_name: str, description: str, min_version: str = None
    ) -> Dict:
        """Check if a single dependency is available."""

        result = {
            "name": package_name,
            "description": description,
            "available": False,
            "version": None,
            "error": None,
        }

        try:
            # Try to import the package
            module = importlib.import_module(package_name)
            result["available"] = True

            # Try to get version
            if hasattr(module, "__version__"):
                result["version"] = module.__version__
            elif hasattr(module, "version"):
                result["version"] = module.version

        except ImportError as e:
            result["error"] = str(e)
        except Exception as e:
            result["error"] = f"Unexpected error: {str(e)}"

        return result

    def validate_all_dependencies(self):
        """Validate all dependencies."""

        print("🔍 Checking Critical Dependencies...")
        print("-" * 50)

        for package, description, min_version in self.CRITICAL_DEPENDENCIES:
            result = self.check_dependency(package, description, min_version)

            if result["available"]:
                version_str = f" (v{result['version']})" if result["version"] else ""
                print(f"✅ {package:<20} {version_str}")
                self.results["critical_ok"].append(result)
            else:
                print(f"❌ {package:<20} - {result['error']}")
                self.results["critical_missing"].append(result)

        print(f"\n⚡ Checking High Priority Dependencies...")
        print("-" * 50)

        for package, description, min_version in self.HIGH_PRIORITY_DEPENDENCIES:
            result = self.check_dependency(package, description, min_version)

            if result["available"]:
                version_str = f" (v{result['version']})" if result["version"] else ""
                print(f"✅ {package:<20} {version_str}")
                self.results["high_priority_ok"].append(result)
            else:
                print(f"⚠️  {package:<20} - Missing")
                self.results["high_priority_missing"].append(result)

        print(f"\n💡 Checking Optional Dependencies...")
        print("-" * 50)

        for package, description, min_version in self.OPTIONAL_DEPENDENCIES:
            result = self.check_dependency(package, description, min_version)

            if result["available"]:
                version_str = f" (v{result['version']})" if result["version"] else ""
                print(f"✅ {package:<20} {version_str}")
                self.results["optional_ok"].append(result)
            else:
                print(f"➖ {package:<20} - Not installed")
                self.results["optional_missing"].append(result)

    def generate_report(self):
        """Generate a comprehensive report."""

        critical_missing_count = len(self.results["critical_missing"])
        high_priority_missing_count = len(self.results["high_priority_missing"])

        print("\n" + "=" * 80)
        print("🎵 MUSIC GEN AI - DEPENDENCY STATUS REPORT")
        print("=" * 80)

        # Overall status
        if critical_missing_count == 0:
            if high_priority_missing_count == 0:
                print("\n🎉 ALL DEPENDENCIES SATISFIED - PRODUCTION READY! ✅")
            else:
                print(
                    f"\n⚠️  {high_priority_missing_count} high-priority dependencies missing - Reduced functionality"
                )
        else:
            print(
                f"\n🔴 {critical_missing_count} CRITICAL dependencies missing - PRODUCTION DEPLOYMENT BLOCKED"
            )

        # Critical missing details
        if critical_missing_count > 0:
            print(f"\n🚨 CRITICAL ISSUES (Must fix before production):")
            print("-" * 60)

            for dep in self.results["critical_missing"]:
                print(f"❌ {dep['name']:<20} - {dep['description']}")
                print(f"   Error: {dep['error']}")
                print()

            print("📋 RESOLUTION:")
            print("1. Install missing critical dependencies:")
            print("   pip install -r requirements-prod.txt")
            print()
            print("2. Or install individually:")
            for dep in self.results["critical_missing"]:
                print(f"   pip install {dep['name']}")

        # High priority missing
        if high_priority_missing_count > 0:
            print(f"\n⚡ HIGH PRIORITY MISSING (Recommended for full functionality):")
            print("-" * 60)

            for dep in self.results["high_priority_missing"]:
                print(f"⚠️  {dep['name']:<20} - {dep['description']}")

            print("\n📋 INSTALL COMMAND:")
            missing_packages = [dep["name"] for dep in self.results["high_priority_missing"]]
            print(f"   pip install {' '.join(missing_packages)}")

        # Success summary
        if critical_missing_count == 0:
            print(f"\n🎯 READY FOR DEPLOYMENT!")
            print("Next steps:")
            print("1. Set production mode: export MUSICGEN_PRODUCTION=true")
            print("2. Start the system: python -m music_gen.api.main")

            if high_priority_missing_count > 0:
                print("3. Consider installing high-priority dependencies for full functionality")

        print("\n" + "=" * 80)

        return critical_missing_count == 0

    def check_torch_specifically(self):
        """Perform specific PyTorch validation."""

        print("\n🔥 Detailed PyTorch Validation...")
        print("-" * 50)

        try:
            import torch

            print(f"✅ PyTorch imported successfully")
            print(f"   Version: {torch.__version__}")
            print(f"   CUDA available: {torch.cuda.is_available()}")

            if torch.cuda.is_available():
                print(f"   CUDA devices: {torch.cuda.device_count()}")
                print(f"   Current device: {torch.cuda.current_device()}")

            # Test basic tensor operations
            test_tensor = torch.randn(2, 3)
            print(f"✅ Basic tensor operations working")

            return True

        except Exception as e:
            print(f"❌ PyTorch validation failed: {e}")
            print(f"\nThis appears to be a library loading issue.")
            print(f"Common causes:")
            print(f"  - Missing system libraries (libgfortran, MKL)")
            print(f"  - Conda environment conflicts")
            print(f"  - PyTorch installation corruption")
            print(f"\nSuggested fixes:")
            print(
                f"  1. Reinstall PyTorch: pip uninstall torch torchaudio && pip install torch torchaudio"
            )
            print(f"  2. Or use conda: conda install pytorch torchaudio -c pytorch")
            print(f"  3. Check system dependencies: brew install gfortran (macOS)")

            return False


def main():
    """Main validation function."""

    print("🎵 Music Gen AI - Standalone Dependency Validation")
    print("=" * 60)
    print("Checking dependencies without loading main system...\n")

    checker = StandaloneDependencyChecker()

    # Run validation
    checker.validate_all_dependencies()

    # Special PyTorch check (most common issue)
    checker.check_torch_specifically()

    # Generate report
    production_ready = checker.generate_report()

    # Exit with appropriate code
    if production_ready:
        print("\n✅ System validation passed - ready to load Music Gen AI!")
        sys.exit(0)
    else:
        print("\n❌ System validation failed - resolve dependency issues before proceeding")
        sys.exit(1)


if __name__ == "__main__":
    main()
