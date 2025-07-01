#!/usr/bin/env python3
"""
Project Cleanup Script for Music Gen

This script automates the cleanup of the music_gen project by:
1. Moving files to archive
2. Deleting temporary files
3. Reorganizing project structure
4. Generating cleanup report

Usage:
    python scripts/cleanup_project.py --dry-run  # Preview changes
    python scripts/cleanup_project.py            # Execute cleanup
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import List

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Archive directory
ARCHIVE_DIR = PROJECT_ROOT / f"archive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Files to remove
REMOVE_PATTERNS = {
    "root_test_scripts": [
        "test_*.py",
        "batch_musicgen_test.py",
        "comprehensive_musicgen_test.py",
        "direct_musicgen_test.py",
        "simple_musicgen_test.py",
        "quick_musicgen_test.py",
        "robust_musicgen_test.py",
    ],
    "fix_scripts": [
        "fix_*.py",
        "quick_fix_and_test.py",
        "proof_of_fix_demo.py",
        "correct_musicgen_architecture.py",
        "comprehensive_sql_fix.py",
    ],
    "demo_scripts": [
        "demo_multi_instrument_fast.py",
        "demo_outputs.py",
        "instant_demo.py",
        "simple_test.py",
        "quick_multi_test.py",
    ],
    "utility_scripts": [
        "analyze_and_cleanup.py",
        "check_system.py",
        "generate_api_test.py",
        "direct_api_test.py",
        "profile_generation_pipeline.py",
        "validate_audio_outputs.py",
        "validate_production_ready.py",
        "verify_audio_files.py",
    ],
    "temp_directories": [
        "demo_venv",
        "test_cache",
        "music_gen_ai.egg-info",
        "api_new",
        "audio_new",
        "cli_new",
        "models_new",
        "utils_new",
        "web_new",
    ],
    "output_files": [
        "*.wav",
        "*.log",
        "optimization_status_report.json",
        "production_readiness_validation.json",
        "generation_profile.json",
    ],
}

# Documentation to consolidate
DOCS_TO_CONSOLIDATE = [
    "ADVANCED_MIXING_COMPLETE.md",
    "ARCHITECTURE_ANALYSIS.md",
    "AUDIT_SUMMARY_AND_NEXT_STEPS.md",
    "COMPLETE_SYSTEM_REVIEW.md",
    "COMPREHENSIVE_DOCUMENTATION.md",
    "CRITICAL_ARCHITECTURE_FIX.md",
    "DOCKER_DEMO_STATUS.md",
    "ENTERPRISE_ARCHITECTURE_PLAN.md",
    "ENTERPRISE_MICROSERVICES_ARCHITECTURE.md",
    "FINAL_MUSICGEN_VERIFICATION_REPORT.md",
    "FINAL_PROJECT_SUMMARY.md",
    "FINAL_SUMMARY.md",
    "GITHUB_ACTIONS_STATUS.md",
    "IMPLEMENTATION_COMPLETE.md",
    "MULTI_INSTRUMENT_REFACTOR_COMPLETE.md",
    "MULTI_INSTRUMENT_STATUS.md",
    "MULTI_INSTRUMENT_SUMMARY.md",
    "MUSICGEN_REFACTOR_SUMMARY.md",
    "MUSICGEN_SUCCESS_PROOF.md",
    "MUSICGEN_TEST_RESULTS.md",
    "MUSICGEN_VERIFICATION_COMPLETE.md",
    "NEXT_STEPS_URGENT.md",
    "PERFORMANCE_OPTIMIZATION_REPORT.md",
    "PERFORMANCE_TRANSFORMATION_COMPLETE.md",
    "PRODUCTION_READINESS_REPORT.md",
    "PROJECT_SUMMARY.md",
    "QUICK_FIX.md",
    "README_COMPREHENSIVE.md",
    "README_DOCKER_FIX.md",
    "README_MICROSERVICES.md",
    "READY_TO_DEMO.md",
    "REAL_GENERATION_PROOF.md",
    "STATUS_REPORT.md",
    "SYSTEM_DEMONSTRATION.md",
    "SYSTEM_READY_FOR_PRODUCTION.md",
    "SYSTEM_RELIABILITY_METRICS.md",
    "TEST_SUITE_README.md",
    "URGENT_MODEL_INTEGRATION_AUDIT.md",
]

# Files to keep
KEEP_FILES = {
    "README.md",
    "CLAUDE.md",
    "LICENSE",
    "Makefile",
    "pyproject.toml",
    "requirements.txt",
    "environment.yml",
    "pytest.ini",
    "Dockerfile",
    "docker-compose.yml",
    ".gitignore",
    "demo.py",  # Keep one demo file
}


class ProjectCleaner:
    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.removed_files: List[Path] = []
        self.moved_files: List[tuple] = []
        self.created_dirs: List[Path] = []

    def log(self, message: str):
        """Log action with dry-run prefix if applicable."""
        prefix = "[DRY RUN] " if self.dry_run else ""
        print(f"{prefix}{message}")

    def ensure_dir(self, path: Path):
        """Create directory if it doesn't exist."""
        if not path.exists():
            self.log(f"Creating directory: {path}")
            if not self.dry_run:
                path.mkdir(parents=True, exist_ok=True)
            self.created_dirs.append(path)

    def remove_file(self, file_path: Path):
        """Remove a file."""
        if file_path.exists():
            self.log(f"Removing: {file_path}")
            if not self.dry_run:
                file_path.unlink()
            self.removed_files.append(file_path)

    def move_file(self, src: Path, dst: Path):
        """Move a file to destination."""
        if src.exists():
            self.log(f"Moving: {src} -> {dst}")
            if not self.dry_run:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
            self.moved_files.append((src, dst))

    def archive_file(self, file_path: Path):
        """Archive a file instead of deleting."""
        if file_path.exists():
            rel_path = file_path.relative_to(PROJECT_ROOT)
            archive_path = ARCHIVE_DIR / rel_path
            self.move_file(file_path, archive_path)

    def cleanup_root_files(self):
        """Clean up files in project root."""
        self.log("\n=== Cleaning Root Directory ===")

        # Remove test scripts
        for pattern in REMOVE_PATTERNS["root_test_scripts"]:
            for file in PROJECT_ROOT.glob(pattern):
                if file.name not in KEEP_FILES:
                    self.archive_file(file)

        # Remove fix scripts
        for pattern in REMOVE_PATTERNS["fix_scripts"]:
            for file in PROJECT_ROOT.glob(pattern):
                self.archive_file(file)

        # Remove demo scripts (except main demo.py)
        for pattern in REMOVE_PATTERNS["demo_scripts"]:
            for file in PROJECT_ROOT.glob(pattern):
                self.archive_file(file)

        # Remove utility scripts
        for script in REMOVE_PATTERNS["utility_scripts"]:
            file = PROJECT_ROOT / script
            if file.exists():
                self.archive_file(file)

    def cleanup_output_files(self):
        """Clean up output files."""
        self.log("\n=== Cleaning Output Files ===")

        # Create proper output directory
        output_dir = PROJECT_ROOT / "output"
        self.ensure_dir(output_dir)

        # Move or remove output files
        for pattern in REMOVE_PATTERNS["output_files"]:
            for file in PROJECT_ROOT.glob(pattern):
                if file.suffix == ".wav" and file.name.startswith("REAL_MUSIC"):
                    # Keep example audio files, move to examples
                    examples_dir = PROJECT_ROOT / "examples" / "audio"
                    self.ensure_dir(examples_dir)
                    self.move_file(file, examples_dir / file.name)
                else:
                    self.archive_file(file)

    def cleanup_directories(self):
        """Clean up temporary directories."""
        self.log("\n=== Cleaning Directories ===")

        for dir_name in REMOVE_PATTERNS["temp_directories"]:
            dir_path = PROJECT_ROOT / dir_name
            if dir_path.exists():
                self.log(f"Removing directory: {dir_path}")
                if not self.dry_run:
                    shutil.rmtree(dir_path)

    def consolidate_docs(self):
        """Consolidate documentation files."""
        self.log("\n=== Consolidating Documentation ===")

        # Create docs directory structure
        docs_dir = PROJECT_ROOT / "docs"
        for subdir in ["architecture", "deployment", "development", "api", "archive"]:
            self.ensure_dir(docs_dir / subdir)

        # Archive old documentation
        for doc in DOCS_TO_CONSOLIDATE:
            doc_path = PROJECT_ROOT / doc
            if doc_path.exists():
                archive_path = docs_dir / "archive" / doc
                self.move_file(doc_path, archive_path)

    def reorganize_examples(self):
        """Reorganize example files."""
        self.log("\n=== Reorganizing Examples ===")

        examples_dir = PROJECT_ROOT / "examples"
        self.ensure_dir(examples_dir)
        self.ensure_dir(examples_dir / "notebooks")

        # Move notebook
        notebook = PROJECT_ROOT / "CS_109B_Music_Project.ipynb"
        if notebook.exists():
            self.move_file(notebook, examples_dir / "notebooks" / notebook.name)

    def generate_report(self):
        """Generate cleanup report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "dry_run": self.dry_run,
            "statistics": {
                "files_removed": len(self.removed_files),
                "files_moved": len(self.moved_files),
                "directories_created": len(self.created_dirs),
            },
            "actions": {
                "removed": [str(f) for f in self.removed_files],
                "moved": [(str(s), str(d)) for s, d in self.moved_files],
                "created_dirs": [str(d) for d in self.created_dirs],
            },
        }

        report_path = PROJECT_ROOT / "cleanup_report.json"
        self.log(f"\nGenerating report: {report_path}")

        if not self.dry_run:
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)

        # Print summary
        print("\n=== Cleanup Summary ===")
        print(f"Files removed/archived: {report['statistics']['files_removed']}")
        print(f"Files moved: {report['statistics']['files_moved']}")
        print(f"Directories created: {report['statistics']['directories_created']}")

        if self.dry_run:
            print("\nThis was a DRY RUN. No files were actually modified.")
            print("Run without --dry-run to execute the cleanup.")

    def run(self):
        """Run the complete cleanup process."""
        print("=== Music Gen Project Cleanup ===")
        print(f"Project root: {PROJECT_ROOT}")

        if not self.dry_run:
            self.ensure_dir(ARCHIVE_DIR)
            print(f"Archive directory: {ARCHIVE_DIR}")

        # Execute cleanup steps
        self.cleanup_root_files()
        self.cleanup_output_files()
        self.cleanup_directories()
        self.consolidate_docs()
        self.reorganize_examples()

        # Generate report
        self.generate_report()


def main():
    parser = argparse.ArgumentParser(description="Clean up Music Gen project")
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview changes without modifying files"
    )
    args = parser.parse_args()

    cleaner = ProjectCleaner(dry_run=args.dry_run)
    cleaner.run()


if __name__ == "__main__":
    main()
