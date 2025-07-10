import torch

"""
Production dependency validation system.

This module ensures all critical dependencies are available for production deployment
and provides clear error messages for missing dependencies.
"""

import importlib
import logging
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DependencyInfo:
    """Information about a dependency."""

    name: str
    package: str
    min_version: Optional[str] = None
    description: str = ""
    critical: bool = True
    fallback_available: bool = False


class DependencyValidator:
    """Validates that all required dependencies are available for production."""

    CRITICAL_DEPENDENCIES = [
        DependencyInfo(
            name="torch",
            package="torch",
            min_version="2.0.0",
            description="PyTorch ML framework",
            critical=True,
        ),
        DependencyInfo(
            name="torchaudio",
            package="torchaudio",
            min_version="2.0.0",
            description="PyTorch audio processing",
            critical=True,
        ),
        DependencyInfo(
            name="transformers",
            package="transformers",
            min_version="4.30.0",
            description="HuggingFace transformers library",
            critical=True,
        ),
        DependencyInfo(
            name="encodec",
            package="encodec",
            min_version="0.1.1",
            description="Facebook EnCodec audio codec",
            critical=True,
        ),
        DependencyInfo(
            name="librosa",
            package="librosa",
            min_version="0.10.0",
            description="Audio analysis library",
            critical=True,
        ),
        DependencyInfo(
            name="soundfile",
            package="soundfile",
            min_version="0.12.0",
            description="Audio file I/O",
            critical=True,
        ),
        DependencyInfo(
            name="fastapi",
            package="fastapi",
            min_version="0.100.0",
            description="Web framework for API",
            critical=True,
        ),
    ]

    HIGH_PRIORITY_DEPENDENCIES = [
        DependencyInfo(
            name="spleeter",
            package="spleeter",
            min_version="2.3.0",
            description="Source separation library",
            critical=False,
            fallback_available=True,
        ),
        DependencyInfo(
            name="demucs",
            package="demucs",
            min_version="4.0.0",
            description="State-of-the-art source separation",
            critical=False,
            fallback_available=True,
        ),
        DependencyInfo(
            name="sentence_transformers",
            package="sentence_transformers",
            min_version="2.2.0",
            description="Text embeddings for CLAP scoring",
            critical=False,
            fallback_available=True,
        ),
    ]

    OPTIONAL_DEPENDENCIES = [
        DependencyInfo(
            name="aioredis",
            package="aioredis",
            min_version="2.0.0",
            description="Redis async client",
            critical=False,
            fallback_available=True,
        ),
        DependencyInfo(
            name="asyncpg",
            package="asyncpg",
            min_version="0.28.0",
            description="PostgreSQL async client",
            critical=False,
            fallback_available=True,
        ),
        DependencyInfo(
            name="pedalboard",
            package="pedalboard",
            min_version="0.8.0",
            description="High-quality audio effects",
            critical=False,
            fallback_available=True,
        ),
        DependencyInfo(
            name="wandb",
            package="wandb",
            description="Experiment tracking",
            critical=False,
            fallback_available=True,
        ),
    ]

    def __init__(self, production_mode: bool = False):
        """Initialize validator.

        Args:
            production_mode: If True, critical dependencies are required
        """
        self.production_mode = production_mode
        self._validation_results = None

    def validate_all_dependencies(self) -> Dict[str, any]:
        """Validate all dependencies and return detailed results."""

        results = {
            "critical_missing": [],
            "critical_ok": [],
            "high_priority_missing": [],
            "high_priority_ok": [],
            "optional_missing": [],
            "optional_ok": [],
            "version_warnings": [],
            "production_ready": True,
            "summary": "",
        }

        # Check critical dependencies
        for dep in self.CRITICAL_DEPENDENCIES:
            status = self._check_dependency(dep)
            if status["available"]:
                results["critical_ok"].append(status)
            else:
                results["critical_missing"].append(status)
                results["production_ready"] = False

        # Check high priority dependencies
        for dep in self.HIGH_PRIORITY_DEPENDENCIES:
            status = self._check_dependency(dep)
            if status["available"]:
                results["high_priority_ok"].append(status)
            else:
                results["high_priority_missing"].append(status)

        # Check optional dependencies
        for dep in self.OPTIONAL_DEPENDENCIES:
            status = self._check_dependency(dep)
            if status["available"]:
                results["optional_ok"].append(status)
            else:
                results["optional_missing"].append(status)

        # Generate summary
        critical_missing_count = len(results["critical_missing"])
        high_priority_missing_count = len(results["high_priority_missing"])

        if critical_missing_count == 0:
            if high_priority_missing_count == 0:
                results["summary"] = "✅ All dependencies satisfied - Production ready!"
            else:
                results[
                    "summary"
                ] = f"⚠️  {high_priority_missing_count} high-priority dependencies missing - Reduced functionality"
        else:
            results[
                "summary"
            ] = f"🔴 {critical_missing_count} critical dependencies missing - Production deployment blocked"

        self._validation_results = results
        return results

    def _check_dependency(self, dep: DependencyInfo) -> Dict[str, any]:
        """Check if a single dependency is available."""

        status = {
            "name": dep.name,
            "package": dep.package,
            "description": dep.description,
            "critical": dep.critical,
            "available": False,
            "version": None,
            "version_ok": True,
            "fallback_available": dep.fallback_available,
            "error": None,
        }

        try:
            # Try to import the package
            module = importlib.import_module(dep.package)
            status["available"] = True

            # Try to get version
            if hasattr(module, "__version__"):
                status["version"] = module.__version__

                # Check version if minimum specified
                if dep.min_version:
                    status["version_ok"] = self._check_version(status["version"], dep.min_version)

        except ImportError as e:
            status["error"] = str(e)
            logger.debug(f"Dependency {dep.name} not available: {e}")

        except Exception as e:
            status["error"] = f"Unexpected error: {str(e)}"
            logger.warning(f"Error checking dependency {dep.name}: {e}")

        return status

    def _check_version(self, current: str, minimum: str) -> bool:
        """Check if current version meets minimum requirement."""
        try:
            from packaging import version

            return version.parse(current) >= version.parse(minimum)
        except ImportError:
            # packaging not available, do basic string comparison
            return current >= minimum
        except Exception:
            # If version parsing fails, assume OK
            return True

    def ensure_production_ready(self) -> None:
        """Ensure system is ready for production deployment.

        Raises:
            RuntimeError: If critical dependencies are missing
        """
        if self._validation_results is None:
            self.validate_all_dependencies()

        critical_missing = self._validation_results["critical_missing"]

        if critical_missing:
            missing_names = [dep["name"] for dep in critical_missing]
            error_msg = self._format_production_error(critical_missing)
            raise RuntimeError(
                f"Production deployment blocked. Missing critical dependencies: {missing_names}\n\n{error_msg}"
            )

        logger.info("✅ All critical dependencies satisfied - System ready for production")

    def _format_production_error(self, missing_deps: List[Dict]) -> str:
        """Format detailed error message for production deployment issues."""

        error_lines = [
            "🔴 PRODUCTION DEPLOYMENT BLOCKED 🔴",
            "",
            "The following critical dependencies are missing:",
            "",
        ]

        for dep in missing_deps:
            error_lines.extend(
                [
                    f"❌ {dep['name']} ({dep['package']})",
                    f"   Description: {dep['description']}",
                    f"   Error: {dep.get('error', 'Not installed')}",
                    "",
                ]
            )

        error_lines.extend(
            [
                "📋 RESOLUTION STEPS:",
                "",
                "1. Install missing dependencies:",
                "   pip install -r requirements-prod.txt",
                "",
                "2. Or install individually:",
            ]
        )

        for dep in missing_deps:
            min_ver = ""
            # Try to find min version from our dependency lists
            for dep_info in self.CRITICAL_DEPENDENCIES:
                if dep_info.name == dep["name"] and dep_info.min_version:
                    min_ver = f">={dep_info.min_version}"
                    break

            error_lines.append(f"   pip install {dep['package']}{min_ver}")

        error_lines.extend(
            [
                "",
                "3. Verify installation:",
                '   python -c "from music_gen.utils.dependency_validator import DependencyValidator; DependencyValidator().ensure_production_ready()"',
                "",
                "⚠️  DO NOT deploy to production with missing critical dependencies!",
                "   This will result in degraded functionality and poor user experience.",
            ]
        )

        return "\n".join(error_lines)

    def print_status_report(self) -> None:
        """Print a comprehensive dependency status report."""

        if self._validation_results is None:
            self.validate_all_dependencies()

        results = self._validation_results

        print("\n" + "=" * 80)
        print("🎵 MUSIC GEN AI - DEPENDENCY STATUS REPORT")
        print("=" * 80)
        print(f"\n{results['summary']}\n")

        # Critical dependencies
        if results["critical_ok"] or results["critical_missing"]:
            print("🔥 CRITICAL DEPENDENCIES")
            print("-" * 40)

            for dep in results["critical_ok"]:
                version_str = f" (v{dep['version']})" if dep["version"] else ""
                print(f"✅ {dep['name']:<20} {version_str}")

            for dep in results["critical_missing"]:
                print(f"❌ {dep['name']:<20} - {dep['error']}")

            print()

        # High priority dependencies
        if results["high_priority_ok"] or results["high_priority_missing"]:
            print("⚡ HIGH PRIORITY DEPENDENCIES")
            print("-" * 40)

            for dep in results["high_priority_ok"]:
                version_str = f" (v{dep['version']})" if dep["version"] else ""
                print(f"✅ {dep['name']:<20} {version_str}")

            for dep in results["high_priority_missing"]:
                fallback_str = " (fallback available)" if dep["fallback_available"] else ""
                print(f"⚠️  {dep['name']:<20} - Missing{fallback_str}")

            print()

        # Optional dependencies
        if results["optional_ok"] or results["optional_missing"]:
            print("💡 OPTIONAL DEPENDENCIES")
            print("-" * 40)

            for dep in results["optional_ok"]:
                version_str = f" (v{dep['version']})" if dep["version"] else ""
                print(f"✅ {dep['name']:<20} {version_str}")

            for dep in results["optional_missing"]:
                print(f"➖ {dep['name']:<20} - Not installed")

        print("\n" + "=" * 80)

        # Action items
        if results["critical_missing"] or results["high_priority_missing"]:
            print("\n📋 RECOMMENDED ACTIONS:")

            if results["critical_missing"]:
                print("\n🔴 CRITICAL - Install immediately:")
                print("   pip install -r requirements-prod.txt")

            if results["high_priority_missing"]:
                print("\n⚠️  HIGH PRIORITY - Install for full functionality:")
                for dep in results["high_priority_missing"]:
                    print(f"   pip install {dep['package']}")

        print()


def validate_startup_dependencies(production_mode: bool = False) -> bool:
    """Validate dependencies at application startup.

    Args:
        production_mode: If True, raise error for missing critical deps

    Returns:
        True if all critical dependencies are available
    """
    validator = DependencyValidator(production_mode=production_mode)

    try:
        results = validator.validate_all_dependencies()

        if production_mode:
            validator.ensure_production_ready()
        else:
            # In development mode, just log warnings
            if results["critical_missing"]:
                logger.warning(
                    f"Missing critical dependencies: {[d['name'] for d in results['critical_missing']]}"
                )

            if results["high_priority_missing"]:
                logger.info(
                    f"Missing optional dependencies: {[d['name'] for d in results['high_priority_missing']]}"
                )

        return results["production_ready"]

    except Exception as e:
        if production_mode:
            raise
        else:
            logger.error(f"Dependency validation failed: {e}")
            return False


def main():
    """Command-line interface for dependency validation."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate Music Gen AI dependencies")
    parser.add_argument(
        "--production", action="store_true", help="Use production mode (strict validation)"
    )
    parser.add_argument("--quiet", action="store_true", help="Only show errors and warnings")

    args = parser.parse_args()

    if not args.quiet:
        logging.basicConfig(level=logging.INFO)

    validator = DependencyValidator(production_mode=args.production)

    try:
        validator.validate_all_dependencies()

        if not args.quiet:
            validator.print_status_report()

        if args.production:
            validator.ensure_production_ready()
            print("\n✅ System ready for production deployment!")

    except RuntimeError as e:
        print(f"\n{e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
