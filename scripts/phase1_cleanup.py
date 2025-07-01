#!/usr/bin/env python3
"""
Phase 1: Foundation Cleanup Script
Implements the immediate cleanup tasks from the strategic roadmap.
"""

import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).parent.parent


class Phase1Cleanup:
    """Execute Phase 1 cleanup tasks."""

    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.report = {
            "timestamp": datetime.now().isoformat(),
            "actions": [],
            "errors": [],
            "statistics": {},
        }

    def log_action(self, action: str, details: str = ""):
        """Log an action taken."""
        print(f"{'[DRY RUN] ' if self.dry_run else ''}✓ {action}")
        if details:
            print(f"  → {details}")
        self.report["actions"].append(
            {"action": action, "details": details, "dry_run": self.dry_run}
        )

    def log_error(self, error: str):
        """Log an error."""
        print(f"✗ ERROR: {error}")
        self.report["errors"].append(error)

    def remove_backup_files(self):
        """Remove all .bak files from the repository."""
        print("\n📋 Removing backup files (.bak)...")

        bak_files = list(PROJECT_ROOT.rglob("*.bak"))
        self.report["statistics"]["backup_files_removed"] = len(bak_files)

        for bak_file in bak_files:
            if self.dry_run:
                self.log_action(f"Would remove: {bak_file.relative_to(PROJECT_ROOT)}")
            else:
                try:
                    bak_file.unlink()
                    self.log_action(f"Removed: {bak_file.relative_to(PROJECT_ROOT)}")
                except Exception as e:
                    self.log_error(f"Failed to remove {bak_file}: {e}")

    def remove_archive_directories(self):
        """Remove archive directories."""
        print("\n📋 Removing archive directories...")

        archive_dirs = [
            d for d in PROJECT_ROOT.iterdir() if d.is_dir() and d.name.startswith("archive_")
        ]
        self.report["statistics"]["archive_dirs_removed"] = len(archive_dirs)

        for archive_dir in archive_dirs:
            if self.dry_run:
                self.log_action(f"Would remove directory: {archive_dir.name}")
            else:
                try:
                    shutil.rmtree(archive_dir)
                    self.log_action(f"Removed directory: {archive_dir.name}")
                except Exception as e:
                    self.log_error(f"Failed to remove {archive_dir}: {e}")

    def clean_pycache(self):
        """Remove all __pycache__ directories."""
        print("\n📋 Cleaning __pycache__ directories...")

        pycache_dirs = list(PROJECT_ROOT.rglob("__pycache__"))
        self.report["statistics"]["pycache_dirs_removed"] = len(pycache_dirs)

        for pycache in pycache_dirs:
            if self.dry_run:
                self.log_action(f"Would remove: {pycache.relative_to(PROJECT_ROOT)}")
            else:
                try:
                    shutil.rmtree(pycache)
                    self.log_action(f"Removed: {pycache.relative_to(PROJECT_ROOT)}")
                except Exception as e:
                    self.log_error(f"Failed to remove {pycache}: {e}")

    def remove_experimental_files(self):
        """Remove identified experimental/abandoned files."""
        print("\n📋 Removing experimental files...")

        experimental_files = [
            "music_gen/inference/integrated_music_system.py",
            "music_gen/inference/real_multi_instrument.py",
            "music_gen/audio/advanced_mixing.py",  # Has unused imports
        ]

        removed_count = 0
        for file_path in experimental_files:
            full_path = PROJECT_ROOT / file_path
            if full_path.exists():
                if self.dry_run:
                    self.log_action(f"Would remove: {file_path}")
                else:
                    try:
                        full_path.unlink()
                        self.log_action(f"Removed: {file_path}")
                        removed_count += 1
                    except Exception as e:
                        self.log_error(f"Failed to remove {file_path}: {e}")
            else:
                self.log_action(f"Already removed: {file_path}", "File not found")

        self.report["statistics"]["experimental_files_removed"] = removed_count

    def consolidate_api_structure(self):
        """Create new consolidated API structure."""
        print("\n📋 Creating consolidated API structure...")

        api_dir = PROJECT_ROOT / "music_gen" / "api"
        new_structure = {
            "endpoints": ["generation.py", "streaming.py", "health.py", "models.py"],
            "middleware": ["__init__.py", "auth.py", "cors.py"],
            "schemas": ["__init__.py", "requests.py", "responses.py"],
        }

        for subdir, files in new_structure.items():
            dir_path = api_dir / subdir
            if self.dry_run:
                self.log_action(f"Would create directory: {dir_path.relative_to(PROJECT_ROOT)}")
            else:
                dir_path.mkdir(exist_ok=True)
                self.log_action(f"Created directory: {dir_path.relative_to(PROJECT_ROOT)}")

                # Create placeholder files
                for file_name in files:
                    file_path = dir_path / file_name
                    if not file_path.exists():
                        file_path.write_text(f'"""Module: {file_name}"""\n')
                        self.log_action(f"Created: {file_path.relative_to(PROJECT_ROOT)}")

    def update_init_files(self):
        """Update __init__.py files with proper exports."""
        print("\n📋 Updating __init__.py files...")

        # Main package init
        main_init = PROJECT_ROOT / "music_gen" / "__init__.py"
        main_init_content = '''"""Music Generation AI Package"""

__version__ = "0.1.0"

# Core imports
from .utils.exceptions import MusicGenException
from .utils.logging import setup_logging, get_logger

# Optional imports with graceful fallback
try:
    from .models.musicgen import MusicGenModel
    from .generation.generator import MusicGenerator
except ImportError as e:
    import warnings
    warnings.warn(f"Optional dependencies not installed: {e}")
    MusicGenModel = None
    MusicGenerator = None

__all__ = [
    "__version__",
    "MusicGenException",
    "setup_logging",
    "get_logger",
    "MusicGenModel",
    "MusicGenerator",
]
'''

        if self.dry_run:
            self.log_action("Would update main __init__.py")
        else:
            main_init.write_text(main_init_content)
            self.log_action("Updated main __init__.py")

    def fix_imports(self):
        """Fix import issues across the codebase."""
        print("\n📋 Fixing imports...")

        if self.dry_run:
            self.log_action("Would run isort to fix imports")
            self.log_action("Would run autoflake to remove unused imports")
        else:
            try:
                # Run isort
                subprocess.run(
                    ["isort", "music_gen", "tests", "scripts", "--profile", "black"],
                    check=True,
                    capture_output=True,
                )
                self.log_action("Fixed import sorting with isort")

                # Run autoflake to remove unused imports
                subprocess.run(
                    [
                        "autoflake",
                        "--in-place",
                        "--remove-unused-variables",
                        "--remove-all-unused-imports",
                        "--recursive",
                        "music_gen",
                    ],
                    check=True,
                    capture_output=True,
                )
                self.log_action("Removed unused imports with autoflake")

            except subprocess.CalledProcessError as e:
                self.log_error(f"Import fixing failed: {e}")
            except FileNotFoundError:
                self.log_error("Tools not installed. Run: pip install isort autoflake")

    def create_pre_commit_config(self):
        """Create pre-commit configuration."""
        print("\n📋 Setting up pre-commit hooks...")

        pre_commit_config = """.pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict
      
  - repo: https://github.com/psf/black
    rev: 24.1.0
    hooks:
      - id: black
        language_version: python3.11
        args: [--line-length=100]
        
  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
        args: [--profile=black, --line-length=100]
        
  - repo: https://github.com/pycqa/flake8
    rev: 7.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=100, --extend-ignore=E203,W503]
        
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
        args: [--ignore-missing-imports]
"""

        config_path = PROJECT_ROOT / ".pre-commit-config.yaml"
        if self.dry_run:
            self.log_action("Would create .pre-commit-config.yaml")
        else:
            config_path.write_text(pre_commit_config)
            self.log_action("Created .pre-commit-config.yaml")

    def generate_report(self):
        """Generate cleanup report."""
        report_path = PROJECT_ROOT / "phase1_cleanup_report.json"

        # Add summary
        self.report["summary"] = {
            "total_actions": len(self.report["actions"]),
            "total_errors": len(self.report["errors"]),
            "dry_run": self.dry_run,
            **self.report["statistics"],
        }

        if not self.dry_run:
            with open(report_path, "w") as f:
                json.dump(self.report, f, indent=2)
            print(f"\n📄 Report saved to: {report_path}")

        # Print summary
        print("\n" + "=" * 60)
        print("🎯 PHASE 1 CLEANUP SUMMARY")
        print("=" * 60)
        for key, value in self.report["statistics"].items():
            print(f"  {key}: {value}")
        print(f"  Total actions: {len(self.report['actions'])}")
        print(f"  Total errors: {len(self.report['errors'])}")

        if self.dry_run:
            print("\n⚠️  This was a DRY RUN. No changes were made.")
            print("Run without --dry-run to execute cleanup.")

    def run(self):
        """Execute all Phase 1 cleanup tasks."""
        print("🚀 Starting Phase 1: Foundation Cleanup")
        print("=" * 60)

        # Execute cleanup tasks
        self.remove_backup_files()
        self.remove_archive_directories()
        self.clean_pycache()
        self.remove_experimental_files()
        self.consolidate_api_structure()
        self.update_init_files()
        self.fix_imports()
        self.create_pre_commit_config()

        # Generate report
        self.generate_report()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Phase 1 cleanup script")
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview changes without modifying files"
    )
    args = parser.parse_args()

    cleanup = Phase1Cleanup(dry_run=args.dry_run)
    cleanup.run()


if __name__ == "__main__":
    main()
