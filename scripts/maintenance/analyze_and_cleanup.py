#!/usr/bin/env python3
"""
Music Generation Platform - Comprehensive Analysis and Cleanup Tool

This tool will:
1. Analyze the current project state
2. Identify issues and redundancies
3. Propose cleanup actions
4. Execute cleanup with user confirmation
5. Set up proper Git version control
"""

import hashlib
import json
import os
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List


class ProjectAnalyzer:
    """Analyze and clean up the music generation project"""

    def __init__(self, project_root: Path = Path.cwd()):
        self.project_root = project_root
        self.analysis_report = {
            "timestamp": datetime.now().isoformat(),
            "issues": [],
            "stats": {},
            "recommendations": [],
        }

    def analyze(self) -> Dict:
        """Perform comprehensive project analysis"""
        print("🔍 Analyzing project structure...")

        # Analyze files
        self._analyze_file_structure()
        self._find_duplicate_files()
        self._analyze_test_files()
        self._analyze_documentation()
        self._analyze_git_status()
        self._analyze_dependencies()

        return self.analysis_report

    def _analyze_file_structure(self):
        """Analyze overall file structure"""
        stats = {
            "total_files": 0,
            "python_files": 0,
            "test_files": 0,
            "wav_files": 0,
            "md_files": 0,
            "directories": 0,
            "total_size_mb": 0,
        }

        file_types = defaultdict(int)
        large_files = []

        for root, dirs, files in os.walk(self.project_root):
            # Skip hidden and backup directories
            dirs[:] = [d for d in dirs if not d.startswith(".") and not d.startswith("backup_")]

            stats["directories"] += len(dirs)

            for file in files:
                if file.startswith("."):
                    continue

                filepath = Path(root) / file
                stats["total_files"] += 1

                # Get file extension
                ext = filepath.suffix.lower()
                file_types[ext] += 1

                # Count specific types
                if ext == ".py":
                    stats["python_files"] += 1
                    if "test" in file.lower():
                        stats["test_files"] += 1
                elif ext == ".wav":
                    stats["wav_files"] += 1
                elif ext == ".md":
                    stats["md_files"] += 1

                # Check file size
                try:
                    size_mb = filepath.stat().st_size / (1024 * 1024)
                    stats["total_size_mb"] += size_mb

                    if size_mb > 10:  # Files larger than 10MB
                        large_files.append(
                            {
                                "path": str(filepath.relative_to(self.project_root)),
                                "size_mb": round(size_mb, 2),
                            }
                        )
                except:
                    pass

        self.analysis_report["stats"] = stats
        self.analysis_report["file_types"] = dict(file_types)
        self.analysis_report["large_files"] = large_files

    def _find_duplicate_files(self):
        """Find duplicate files based on content hash"""
        print("🔍 Finding duplicate files...")

        file_hashes = defaultdict(list)
        duplicates = []

        for root, dirs, files in os.walk(self.project_root):
            # Skip hidden and backup directories
            dirs[:] = [d for d in dirs if not d.startswith(".") and not d.startswith("backup_")]

            for file in files:
                if file.startswith("."):
                    continue

                filepath = Path(root) / file

                # Skip very large files (>100MB) for performance
                try:
                    if filepath.stat().st_size > 100 * 1024 * 1024:
                        continue
                except:
                    continue

                # Calculate file hash
                try:
                    with open(filepath, "rb") as f:
                        file_hash = hashlib.md5(f.read()).hexdigest()

                    file_hashes[file_hash].append(str(filepath.relative_to(self.project_root)))
                except:
                    pass

        # Find duplicates
        for file_hash, files in file_hashes.items():
            if len(files) > 1:
                duplicates.append({"files": files, "count": len(files)})

        if duplicates:
            self.analysis_report["issues"].append(
                {
                    "type": "duplicate_files",
                    "severity": "medium",
                    "description": f"Found {len(duplicates)} sets of duplicate files",
                    "details": duplicates[:10],  # Show first 10
                }
            )

    def _analyze_test_files(self):
        """Analyze test file organization"""
        print("🔍 Analyzing test files...")

        root_tests = []
        organized_tests = []

        # Find test files in root
        for file in self.project_root.glob("test_*.py"):
            root_tests.append(file.name)

        # Find test files in tests directory
        tests_dir = self.project_root / "tests"
        if tests_dir.exists():
            for file in tests_dir.rglob("test_*.py"):
                organized_tests.append(str(file.relative_to(tests_dir)))

        if root_tests:
            self.analysis_report["issues"].append(
                {
                    "type": "unorganized_tests",
                    "severity": "medium",
                    "description": f"Found {len(root_tests)} test files in root directory",
                    "details": root_tests,
                    "recommendation": "Move test files to tests/ directory",
                }
            )

    def _analyze_documentation(self):
        """Analyze documentation files"""
        print("🔍 Analyzing documentation...")

        doc_files = list(self.project_root.glob("*.md"))

        # Categorize documentation
        status_docs = []
        guide_docs = []
        other_docs = []

        for doc in doc_files:
            name_lower = doc.name.lower()
            if any(word in name_lower for word in ["status", "complete", "report", "summary"]):
                status_docs.append(doc.name)
            elif any(word in name_lower for word in ["guide", "readme", "contributing"]):
                guide_docs.append(doc.name)
            else:
                other_docs.append(doc.name)

        if len(status_docs) > 5:
            self.analysis_report["issues"].append(
                {
                    "type": "excessive_status_docs",
                    "severity": "low",
                    "description": f"Found {len(status_docs)} status/report documents",
                    "details": status_docs,
                    "recommendation": "Archive old status reports to docs/archive/",
                }
            )

    def _analyze_git_status(self):
        """Analyze Git repository status"""
        print("🔍 Analyzing Git status...")

        try:
            # Check if it's a git repository
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"], capture_output=True, text=True
            )

            if result.returncode == 0:
                # Get untracked files
                result = subprocess.run(
                    ["git", "ls-files", "--others", "--exclude-standard"],
                    capture_output=True,
                    text=True,
                )
                untracked = result.stdout.strip().split("\n") if result.stdout.strip() else []

                # Get modified files
                result = subprocess.run(["git", "ls-files", "-m"], capture_output=True, text=True)
                modified = result.stdout.strip().split("\n") if result.stdout.strip() else []

                self.analysis_report["git_status"] = {
                    "untracked_files": len(untracked),
                    "modified_files": len(modified),
                    "sample_untracked": untracked[:10],
                }

                if len(untracked) > 50:
                    self.analysis_report["issues"].append(
                        {
                            "type": "many_untracked_files",
                            "severity": "medium",
                            "description": f"Found {len(untracked)} untracked files",
                            "recommendation": "Review and add to .gitignore or commit",
                        }
                    )
        except:
            self.analysis_report["git_status"] = {
                "error": "Not a git repository or git not available"
            }

    def _analyze_dependencies(self):
        """Analyze project dependencies"""
        print("🔍 Analyzing dependencies...")

        # Check for requirements files
        req_files = []
        for pattern in ["requirements*.txt", "Pipfile", "pyproject.toml", "setup.py"]:
            req_files.extend(self.project_root.glob(pattern))

        self.analysis_report["dependency_files"] = [f.name for f in req_files]

        if len(req_files) > 2:
            self.analysis_report["issues"].append(
                {
                    "type": "multiple_dependency_files",
                    "severity": "low",
                    "description": f"Found {len(req_files)} dependency files",
                    "details": [f.name for f in req_files],
                    "recommendation": "Consolidate to single requirements.txt or pyproject.toml",
                }
            )

    def generate_cleanup_plan(self) -> List[Dict]:
        """Generate a cleanup plan based on analysis"""
        cleanup_actions = []

        # Plan for each issue type
        for issue in self.analysis_report.get("issues", []):
            if issue["type"] == "duplicate_files":
                for dup_set in issue.get("details", [])[:5]:  # Handle first 5 sets
                    files = dup_set["files"]
                    # Keep the first file, remove others
                    for file_to_remove in files[1:]:
                        cleanup_actions.append(
                            {"action": "remove_duplicate", "file": file_to_remove, "keep": files[0]}
                        )

            elif issue["type"] == "unorganized_tests":
                for test_file in issue.get("details", []):
                    cleanup_actions.append(
                        {
                            "action": "move_test",
                            "source": test_file,
                            "destination": f"tests/{test_file}",
                        }
                    )

            elif issue["type"] == "excessive_status_docs":
                for doc in issue.get("details", []):
                    cleanup_actions.append(
                        {
                            "action": "archive_doc",
                            "source": doc,
                            "destination": f"docs/archive/{doc}",
                        }
                    )

        # Add WAV file cleanup
        if self.analysis_report["stats"].get("wav_files", 0) > 10:
            cleanup_actions.append(
                {
                    "action": "move_outputs",
                    "description": "Move all WAV files to outputs/ directory",
                }
            )

        return cleanup_actions


def main():
    """Main cleanup workflow"""
    print("🎵 Music Generation Platform - Comprehensive Cleanup Tool")
    print("=" * 60)

    # Analyze project
    analyzer = ProjectAnalyzer()
    report = analyzer.analyze()

    # Display analysis results
    print("\n📊 Analysis Results:")
    print(f"  Total files: {report['stats']['total_files']}")
    print(f"  Python files: {report['stats']['python_files']}")
    print(f"  Test files: {report['stats']['test_files']}")
    print(f"  Documentation files: {report['stats']['md_files']}")
    print(f"  Total size: {report['stats']['total_size_mb']:.2f} MB")

    print(f"\n⚠️  Issues found: {len(report['issues'])}")
    for issue in report["issues"]:
        print(f"  - {issue['description']} [{issue['severity']}]")

    # Generate cleanup plan
    cleanup_plan = analyzer.generate_cleanup_plan()

    if cleanup_plan:
        print(f"\n🛠️  Proposed cleanup actions: {len(cleanup_plan)}")
        for i, action in enumerate(cleanup_plan[:10]):  # Show first 10
            if action["action"] == "remove_duplicate":
                print(f"  {i+1}. Remove duplicate: {action['file']}")
            elif action["action"] == "move_test":
                print(f"  {i+1}. Move test file: {action['source']} → {action['destination']}")
            elif action["action"] == "archive_doc":
                print(f"  {i+1}. Archive doc: {action['source']}")

        if len(cleanup_plan) > 10:
            print(f"  ... and {len(cleanup_plan) - 10} more actions")

    # Save detailed report
    report_path = Path("cleanup_analysis_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n📄 Detailed report saved to: {report_path}")

    # Ask for confirmation
    print("\n" + "=" * 60)
    response = input("Would you like to proceed with cleanup? (yes/no): ").lower().strip()

    if response == "yes":
        print("\n🚀 Starting cleanup...")
        print("(Run ./cleanup_music_gen.sh to execute the cleanup)")
    else:
        print("\n✅ Analysis complete. No changes made.")
        print("Review cleanup_analysis_report.json for details.")


if __name__ == "__main__":
    main()
