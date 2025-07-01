#!/usr/bin/env python3
"""
Comprehensive repository reorganization script.
Converts from dual architecture to clean monolithic structure.
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Archive directory with timestamp
ARCHIVE_DIR = PROJECT_ROOT / f"archive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Files to move to different locations
RELOCATIONS = {
    # Test files from root to tests/
    "batch_musicgen_test.py": "tests/integration/batch_musicgen_test.py",
    "comprehensive_musicgen_test.py": "tests/integration/comprehensive_musicgen_test.py",
    "musicgen_performance_test.py": "tests/performance/musicgen_performance_test.py",
    "robust_musicgen_test.py": "tests/integration/robust_musicgen_test.py",
    "test_advanced_mixing.py": "tests/integration/test_advanced_mixing.py",
    "test_api_example.py": "tests/integration/test_api_example.py",
    "test_api_generation.py": "tests/integration/test_api_generation.py",
    "test_integrated_system.py": "tests/integration/test_integrated_system.py",
    "test_multi_instrument_generation.py": "tests/integration/test_multi_instrument_generation.py",
    "test_optimized_performance.py": "tests/performance/test_optimized_performance.py",
    "test_real_multi_instrument.py": "tests/integration/test_real_multi_instrument.py",
    "test_single_instrument_real.py": "tests/integration/test_single_instrument_real.py",
    "test_web_ui.py": "tests/integration/test_web_ui.py",
    "test_working_optimizations.py": "tests/performance/test_working_optimizations.py",
    "test_cli.py": "tests/integration/test_cli.py",
    "test_cli_optimization.py": "tests/performance/test_cli_optimization.py",
    # Utility scripts to scripts/
    "analyze_and_cleanup.py": "scripts/maintenance/analyze_and_cleanup.py",
    "check_system.py": "scripts/validation/check_system.py",
    "verify_audio_files.py": "scripts/validation/verify_audio_files.py",
    "validate_production_ready.py": "scripts/validation/validate_production_ready.py",
    "profile_generation_pipeline.py": "scripts/performance/profile_generation_pipeline.py",
    "generate_api_test.py": "scripts/examples/generate_api_test.py",
    "direct_api_test.py": "scripts/examples/direct_api_test.py",
    # Setup scripts
    "setup_git_and_cleanup.sh": "scripts/setup/setup_git_and_cleanup.sh",
    "cleanup_music_gen.sh": "scripts/maintenance/cleanup_music_gen.sh",
    "fix_docker_startup.sh": "scripts/docker/fix_docker_startup.sh",
    # Move demo scripts to examples
    "demo_multi_instrument_fast.py": "examples/demo_multi_instrument_fast.py",
    "proof_of_fix_demo.py": "examples/proof_of_fix_demo.py",
    "quick_multi_test.py": "examples/quick_multi_test.py",
    "simple_test.py": "examples/simple_test.py",
}

# Directories to remove completely
REMOVE_DIRS = [
    "services",  # Microservices implementation
    "core",  # Empty package
    "demo_venv",  # Virtual environment
    "music_gen_ai.egg-info",  # Build artifact
    "batch_outputs",  # Temporary outputs
    "quick_test_outputs",  # Temporary outputs
    "test_outputs",  # Temporary outputs
    "test_cache",  # Temporary cache
    "cache",  # Temporary cache
    ".archive",  # Previous archive attempts
]

# Files to remove completely
REMOVE_FILES = [
    "docker-compose.microservices.yml",  # Microservices config
    "comprehensive_sql_fix.py",  # One-time fix
    "correct_musicgen_architecture.py",  # One-time migration
    "fix_production_to_use_real_musicgen.py",  # One-time fix
    "fix_remaining_tests.py",  # One-time fix
    "optimization_status_report.json",  # Generated report
    "production_readiness_validation.json",  # Generated report
    "generation_profile.json",  # Generated report
    "demo-requirements.txt",  # Duplicate requirements
]

# Documentation to archive (too many status files)
ARCHIVE_DOCS = [
    "ADVANCED_MIXING_COMPLETE.md",
    "AUDIT_SUMMARY_AND_NEXT_STEPS.md",
    "DEMO_GUIDE.md",
    "DEPLOYMENT_GUIDE.md",
    "DOCKER_DEMO_STATUS.md",
    "ENTERPRISE_ARCHITECTURE_PLAN.md",
    "ENTERPRISE_MICROSERVICES_ARCHITECTURE.md",
    "FINAL_MUSICGEN_VERIFICATION_REPORT.md",
    "FINAL_PROJECT_SUMMARY.md",
    "GITHUB_ACTIONS_STATUS.md",
    "IMPLEMENTATION_COMPLETE.md",
    "MULTI_INSTRUMENT_REFACTOR_COMPLETE.md",
    "MULTI_INSTRUMENT_STATUS.md",
    "MUSICGEN_REFACTOR_SUMMARY.md",
    "MUSICGEN_VERIFICATION_COMPLETE.md",
    "PERFORMANCE_OPTIMIZATION_GUIDE.md",
    "PERFORMANCE_OPTIMIZATION_REPORT.md",
    "PERFORMANCE_TRANSFORMATION_COMPLETE.md",
    "PRODUCTION_READINESS_REPORT.md",
    "QUICK_FIX.md",
    "READY_TO_DEMO.md",
    "SYSTEM_READY_FOR_PRODUCTION.md",
    "SYSTEM_RELIABILITY_METRICS.md",
    "URGENT_MODEL_INTEGRATION_AUDIT.md",
]

# New .gitignore entries
GITIGNORE_ADDITIONS = [
    "# Virtual environments",
    "demo_venv/",
    "venv/",
    "env/",
    ".env/",
    "",
    "# Build artifacts",
    "*.egg-info/",
    "dist/",
    "build/",
    "__pycache__/",
    "*.pyc",
    "*.pyo",
    "",
    "# Cache directories",
    "cache/",
    "test_cache/",
    ".cache/",
    "",
    "# Output directories",
    "output/",
    "outputs/",
    "test_outputs/",
    "batch_outputs/",
    "quick_test_outputs/",
    "",
    "# Temporary files",
    "*.wav",
    "*.mp3",
    "*.log",
    "*.tmp",
    "",
    "# IDE files",
    ".idea/",
    ".vscode/",
    "*.swp",
    "*.swo",
    "",
    "# OS files",
    ".DS_Store",
    "Thumbs.db",
    "",
    "# Archive directories",
    "archive_*/",
    ".archive/",
]


class ProductionReorganizer:
    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.actions_log: List[str] = []
        self.errors: List[str] = []

    def log(self, message: str):
        """Log action with dry-run prefix if applicable."""
        prefix = "[DRY RUN] " if self.dry_run else ""
        print(f"{prefix}{message}")
        self.actions_log.append(message)

    def error(self, message: str):
        """Log error."""
        print(f"ERROR: {message}")
        self.errors.append(message)

    def ensure_dir(self, path: Path):
        """Create directory if it doesn't exist."""
        if not path.exists():
            self.log(f"Creating directory: {path}")
            if not self.dry_run:
                path.mkdir(parents=True, exist_ok=True)

    def move_file(self, src: Path, dst: Path):
        """Move file to destination."""
        if src.exists():
            self.ensure_dir(dst.parent)
            self.log(f"Moving: {src} -> {dst}")
            if not self.dry_run:
                shutil.move(str(src), str(dst))
        else:
            self.error(f"Source file not found: {src}")

    def archive_file(self, file_path: Path):
        """Archive file instead of deleting."""
        if file_path.exists():
            rel_path = file_path.relative_to(PROJECT_ROOT)
            archive_path = ARCHIVE_DIR / rel_path
            self.move_file(file_path, archive_path)

    def remove_file(self, file_path: Path):
        """Remove file permanently."""
        if file_path.exists():
            self.log(f"Removing file: {file_path}")
            if not self.dry_run:
                file_path.unlink()

    def remove_directory(self, dir_path: Path):
        """Remove directory and all contents."""
        if dir_path.exists():
            self.log(f"Removing directory: {dir_path}")
            if not self.dry_run:
                shutil.rmtree(dir_path)

    def archive_directory(self, dir_path: Path):
        """Archive entire directory."""
        if dir_path.exists():
            rel_path = dir_path.relative_to(PROJECT_ROOT)
            archive_path = ARCHIVE_DIR / rel_path
            self.log(f"Archiving directory: {dir_path} -> {archive_path}")
            if not self.dry_run:
                archive_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(dir_path), str(archive_path))

    def update_gitignore(self):
        """Update .gitignore with production entries."""
        gitignore_path = PROJECT_ROOT / ".gitignore"

        # Read existing entries
        existing_entries = set()
        if gitignore_path.exists():
            with open(gitignore_path, "r") as f:
                existing_entries = set(
                    line.strip() for line in f if line.strip() and not line.startswith("#")
                )

        # Check which entries need to be added
        new_entries = []
        for entry in GITIGNORE_ADDITIONS:
            if entry and not entry.startswith("#") and entry not in existing_entries:
                new_entries.append(entry)
            elif entry.startswith("#") or not entry:
                new_entries.append(entry)

        if new_entries:
            self.log(
                f"Adding {len([e for e in new_entries if e and not e.startswith('#')])} new entries to .gitignore"
            )
            if not self.dry_run:
                with open(gitignore_path, "a") as f:
                    f.write("\n")
                    for entry in GITIGNORE_ADDITIONS:
                        f.write(f"{entry}\n")

    def reorganize_files(self):
        """Move files to their new locations."""
        self.log("\n=== Reorganizing Files ===")

        # Create necessary directories
        directories_needed = set()
        for src, dst in RELOCATIONS.items():
            directories_needed.add(Path(dst).parent)

        for directory in sorted(directories_needed):
            self.ensure_dir(PROJECT_ROOT / directory)

        # Move files
        for src, dst in RELOCATIONS.items():
            src_path = PROJECT_ROOT / src
            dst_path = PROJECT_ROOT / dst
            if src_path.exists():
                self.move_file(src_path, dst_path)

    def archive_documentation(self):
        """Archive redundant documentation."""
        self.log("\n=== Archiving Documentation ===")

        docs_archive = PROJECT_ROOT / "docs" / "archive"
        self.ensure_dir(docs_archive)

        for doc in ARCHIVE_DOCS:
            doc_path = PROJECT_ROOT / doc
            if doc_path.exists():
                self.move_file(doc_path, docs_archive / doc)

    def remove_directories(self):
        """Remove unnecessary directories."""
        self.log("\n=== Removing Directories ===")

        for dir_name in REMOVE_DIRS:
            dir_path = PROJECT_ROOT / dir_name
            if dir_path.exists():
                # Archive important directories, remove others
                if dir_name in ["services"]:
                    self.archive_directory(dir_path)
                else:
                    self.remove_directory(dir_path)

    def remove_files(self):
        """Remove unnecessary files."""
        self.log("\n=== Removing Files ===")

        for file_name in REMOVE_FILES:
            file_path = PROJECT_ROOT / file_name
            if file_path.exists():
                self.archive_file(file_path)

    def clean_wav_files(self):
        """Move WAV files to appropriate locations."""
        self.log("\n=== Organizing Audio Files ===")

        examples_audio = PROJECT_ROOT / "examples" / "audio"
        self.ensure_dir(examples_audio)

        # Move example audio files
        for wav_file in PROJECT_ROOT.glob("REAL_MUSIC_*.wav"):
            self.move_file(wav_file, examples_audio / wav_file.name)

        # Archive other WAV files
        for wav_file in PROJECT_ROOT.glob("*.wav"):
            self.archive_file(wav_file)

    def create_docs_structure(self):
        """Create proper documentation structure."""
        self.log("\n=== Creating Documentation Structure ===")

        docs_dirs = [
            "docs",
            "docs/api",
            "docs/architecture",
            "docs/deployment",
            "docs/development",
            "docs/guides",
            "docs/archive",
        ]

        for docs_dir in docs_dirs:
            self.ensure_dir(PROJECT_ROOT / docs_dir)

    def generate_report(self):
        """Generate reorganization report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "dry_run": self.dry_run,
            "actions": self.actions_log,
            "errors": self.errors,
            "summary": {"total_actions": len(self.actions_log), "errors": len(self.errors)},
        }

        report_path = PROJECT_ROOT / "reorganization_report.json"
        self.log(f"\nGenerating report: {report_path}")

        if not self.dry_run:
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)

        # Print summary
        print("\n=== Reorganization Summary ===")
        print(f"Total actions: {report['summary']['total_actions']}")
        print(f"Errors: {report['summary']['errors']}")

        if self.errors:
            print("\nErrors encountered:")
            for error in self.errors:
                print(f"  - {error}")

        if self.dry_run:
            print("\nThis was a DRY RUN. No files were actually modified.")
            print("Run without --dry-run to execute the reorganization.")

    def run(self):
        """Execute the complete reorganization."""
        print("=== Production Repository Reorganization ===")
        print(f"Project root: {PROJECT_ROOT}")

        if not self.dry_run:
            self.ensure_dir(ARCHIVE_DIR)
            print(f"Archive directory: {ARCHIVE_DIR}")

        # Execute reorganization steps
        self.reorganize_files()
        self.archive_documentation()
        self.remove_directories()
        self.remove_files()
        self.clean_wav_files()
        self.create_docs_structure()
        self.update_gitignore()

        # Generate report
        self.generate_report()


def main():
    parser = argparse.ArgumentParser(description="Reorganize repository to production standards")
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview changes without modifying files"
    )
    args = parser.parse_args()

    reorganizer = ProductionReorganizer(dry_run=args.dry_run)
    reorganizer.run()


if __name__ == "__main__":
    main()
