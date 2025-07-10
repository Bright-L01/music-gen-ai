#!/usr/bin/env python3
"""
Clean up the codebase by removing unused imports, fixing linting issues, and consolidating code.
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import List, Set, Dict
import json
import ast
import re

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


class CodebaseCleanup:
    """Automated codebase cleanup tool."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.music_gen_path = project_root / "music_gen"
        self.issues = {
            "unused_imports": [],
            "unused_functions": [],
            "commented_code": [],
            "duplicate_functions": [],
            "linting_errors": [],
            "security_issues": [],
        }

    def run_autoflake(self, fix: bool = False):
        """Remove unused imports and variables."""
        print("\n🔍 Running autoflake to find unused imports...")

        cmd = [
            "autoflake",
            "--recursive",
            "--remove-all-unused-imports",
            "--remove-unused-variables",
            "--remove-duplicate-keys",
            str(self.music_gen_path),
        ]

        if fix:
            cmd.extend(["--in-place"])
        else:
            cmd.extend(["--check"])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0 and not fix:
                print("❌ Found unused imports/variables. Run with --fix to clean them.")
                self.issues["unused_imports"] = result.stdout.splitlines()[:10]
            elif fix:
                print("✅ Fixed unused imports and variables")
        except Exception as e:
            print(f"Error running autoflake: {e}")

    def run_isort(self, fix: bool = False):
        """Sort and organize imports."""
        print("\n🔍 Running isort to organize imports...")

        cmd = ["isort", str(self.music_gen_path)]

        if not fix:
            cmd.append("--check-only")
            cmd.append("--diff")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0 and not fix:
                print("❌ Found unsorted imports. Run with --fix to sort them.")
            elif fix:
                print("✅ Sorted all imports")
        except Exception as e:
            print(f"Error running isort: {e}")

    def run_black(self, fix: bool = False):
        """Format code with black."""
        print("\n🔍 Running black to check formatting...")

        cmd = ["black", str(self.music_gen_path)]

        if not fix:
            cmd.append("--check")
            cmd.append("--diff")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0 and not fix:
                print("❌ Found formatting issues. Run with --fix to format code.")
            elif fix:
                print("✅ Formatted all code with black")
        except Exception as e:
            print(f"Error running black: {e}")

    def find_commented_code(self):
        """Find and report commented-out code."""
        print("\n🔍 Finding commented-out code...")

        commented_patterns = [
            r"^\s*#\s*(import|from|def|class|if|for|while|return|raise)",
            r"^\s*#.*\(.*\).*:",  # Commented function calls
            r'^\s*"""[\s\S]*?"""',  # Large docstring blocks that might be commented code
        ]

        for py_file in self.music_gen_path.rglob("*.py"):
            try:
                content = py_file.read_text()
                lines = content.splitlines()

                for i, line in enumerate(lines, 1):
                    for pattern in commented_patterns:
                        if re.match(pattern, line):
                            self.issues["commented_code"].append(f"{py_file}:{i} - {line.strip()}")
            except Exception as e:
                print(f"Error reading {py_file}: {e}")

        if self.issues["commented_code"]:
            print(f"❌ Found {len(self.issues['commented_code'])} lines of commented code")
        else:
            print("✅ No significant commented code found")

    def find_duplicate_functions(self):
        """Find duplicate or very similar functions."""
        print("\n🔍 Finding duplicate functions...")

        function_signatures = {}

        for py_file in self.music_gen_path.rglob("*.py"):
            try:
                content = py_file.read_text()
                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Create a simplified signature
                        params = [arg.arg for arg in node.args.args]
                        signature = f"{node.name}({','.join(params)})"

                        if signature in function_signatures:
                            self.issues["duplicate_functions"].append(
                                f"Duplicate: {signature} in {py_file} and {function_signatures[signature]}"
                            )
                        else:
                            function_signatures[signature] = py_file

            except Exception as e:
                print(f"Error parsing {py_file}: {e}")

        if self.issues["duplicate_functions"]:
            print(f"❌ Found {len(self.issues['duplicate_functions'])} duplicate functions")
        else:
            print("✅ No duplicate functions found")

    def fix_common_issues(self):
        """Fix common issues automatically."""
        print("\n🔧 Fixing common issues...")

        fixes_applied = 0

        # Fix missing imports
        import_fixes = {
            "datetime": "from datetime import datetime",
            "timedelta": "from datetime import timedelta",
            "Any": "from typing import Any",
            "torch": "import torch",
        }

        for py_file in self.music_gen_path.rglob("*.py"):
            try:
                content = py_file.read_text()
                original_content = content

                # Fix missing imports
                for name, import_stmt in import_fixes.items():
                    if (
                        f"'{name}'" in content
                        or f'"{name}"' in content
                        or f" {name}." in content
                        or f" {name}(" in content
                    ):
                        if import_stmt not in content:
                            # Add import after other imports
                            import_section_end = 0
                            lines = content.splitlines()
                            for i, line in enumerate(lines):
                                if line.strip() and not line.startswith(("import", "from", "#")):
                                    import_section_end = i
                                    break

                            lines.insert(import_section_end, import_stmt)
                            content = "\n".join(lines)
                            fixes_applied += 1

                # Remove trailing whitespace
                content = "\n".join(line.rstrip() for line in content.splitlines())

                # Ensure file ends with newline
                if content and not content.endswith("\n"):
                    content += "\n"

                if content != original_content:
                    py_file.write_text(content)

            except Exception as e:
                print(f"Error fixing {py_file}: {e}")

        print(f"✅ Applied {fixes_applied} fixes")

    def remove_unused_files(self):
        """Identify potentially unused files."""
        print("\n🔍 Identifying potentially unused files...")

        # Files that are commonly unused
        unused_patterns = [
            "*_old.py",
            "*_backup.py",
            "*_test_*.py",  # Test files outside test directory
            "*.pyc",
            "__pycache__",
        ]

        unused_files = []
        for pattern in unused_patterns:
            unused_files.extend(self.music_gen_path.rglob(pattern))

        if unused_files:
            print(f"❌ Found {len(unused_files)} potentially unused files:")
            for f in unused_files[:10]:
                print(f"  - {f}")
        else:
            print("✅ No obviously unused files found")

    def update_requirements(self):
        """Check for outdated dependencies."""
        print("\n🔍 Checking for outdated dependencies...")

        requirements_files = [
            self.project_root / "requirements.txt",
            self.project_root / "requirements-dev.txt",
            self.project_root / "requirements-prod.txt",
        ]

        for req_file in requirements_files:
            if req_file.exists():
                print(f"\nChecking {req_file.name}:")
                try:
                    # Just report, don't auto-update
                    result = subprocess.run(
                        ["pip", "list", "--outdated", "--format=json"],
                        capture_output=True,
                        text=True,
                    )

                    if result.returncode == 0:
                        outdated = json.loads(result.stdout)
                        if outdated:
                            print(f"  ❌ Found {len(outdated)} outdated packages:")
                            for pkg in outdated[:5]:
                                print(
                                    f"    - {pkg['name']}: {pkg['version']} → {pkg['latest_version']}"
                                )
                        else:
                            print("  ✅ All packages up to date")
                except Exception as e:
                    print(f"  Error checking dependencies: {e}")

    def generate_report(self):
        """Generate cleanup report."""
        print("\n" + "=" * 60)
        print("CODEBASE CLEANUP REPORT")
        print("=" * 60)

        total_issues = sum(len(issues) for issues in self.issues.values())
        print(f"\nTotal issues found: {total_issues}")

        for issue_type, issues in self.issues.items():
            if issues:
                print(f"\n{issue_type.replace('_', ' ').title()}:")
                for issue in issues[:5]:
                    print(f"  - {issue}")
                if len(issues) > 5:
                    print(f"  ... and {len(issues) - 5} more")

        print("\n" + "=" * 60)
        print("RECOMMENDED ACTIONS:")
        print("=" * 60)
        print("1. Run with --fix to automatically fix formatting and imports")
        print("2. Manually review and remove commented-out code")
        print("3. Consolidate duplicate functions")
        print("4. Update outdated dependencies carefully")
        print("5. Remove unused files after verification")

    def run(self, fix: bool = False):
        """Run all cleanup tasks."""
        print(f"🧹 Starting codebase cleanup for {self.project_root}")

        # Run automated tools
        self.run_autoflake(fix)
        self.run_isort(fix)
        self.run_black(fix)

        # Run analysis
        self.find_commented_code()
        self.find_duplicate_functions()
        self.remove_unused_files()
        self.update_requirements()

        # Fix common issues if requested
        if fix:
            self.fix_common_issues()

        # Generate report
        self.generate_report()


def main():
    """Run codebase cleanup."""
    import argparse

    parser = argparse.ArgumentParser(description="Clean up the codebase")
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Apply automatic fixes (use with caution)",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path.cwd(),
        help="Project root path",
    )

    args = parser.parse_args()

    cleanup = CodebaseCleanup(args.path)
    cleanup.run(fix=args.fix)


if __name__ == "__main__":
    main()
