#!/usr/bin/env python3
"""
Check for basic issues that would cause CI failures.
"""

import ast
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
MUSIC_GEN_DIR = PROJECT_ROOT / "music_gen"


def check_syntax_errors():
    """Check all Python files for syntax errors."""
    print("Checking for syntax errors...")
    errors = []

    for py_file in MUSIC_GEN_DIR.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue

        try:
            with open(py_file, "r", encoding="utf-8") as f:
                ast.parse(f.read())
        except SyntaxError as e:
            rel_path = py_file.relative_to(PROJECT_ROOT)
            errors.append((rel_path, str(e)))

    if errors:
        print("\nSyntax errors found:")
        for file_path, error in errors:
            print(f"  {file_path}: {error}")
    else:
        print("✓ No syntax errors found")

    return errors


def check_basic_style():
    """Check for basic style issues."""
    print("\nChecking for basic style issues...")
    issues = []

    for py_file in MUSIC_GEN_DIR.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue

        with open(py_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        rel_path = py_file.relative_to(PROJECT_ROOT)
        file_issues = []

        for i, line in enumerate(lines, 1):
            # Check line length
            if len(line.rstrip()) > 127:
                file_issues.append(f"Line {i}: Too long ({len(line.rstrip())} chars)")

            # Check for tabs
            if "\t" in line:
                file_issues.append(f"Line {i}: Contains tabs")

            # Check for trailing whitespace
            if line.rstrip() != line.rstrip("\n").rstrip("\r"):
                file_issues.append(f"Line {i}: Trailing whitespace")

        if file_issues:
            issues.append((rel_path, file_issues))

    if issues:
        print("\nStyle issues found:")
        for file_path, file_issues in issues[:5]:  # Show first 5 files
            print(f"\n  {file_path}:")
            for issue in file_issues[:3]:  # Show first 3 issues per file
                print(f"    {issue}")
        if len(issues) > 5:
            print(f"\n  ... and {len(issues) - 5} more files with issues")
    else:
        print("✓ No major style issues found")

    return issues


def check_undefined_names():
    """Check for potentially undefined names."""
    print("\nChecking for undefined names...")

    # Run flake8 to check for undefined names
    try:
        result = subprocess.run(
            ["python", "-m", "flake8", "music_gen", "--select=F821,F822,F823", "--count"],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print("✓ No undefined names found")
        else:
            print("Issues found:")
            print(result.stdout)

    except Exception as e:
        print(f"Could not run flake8: {e}")


def check_test_files():
    """Check if test files exist and are valid."""
    print("\nChecking test files...")

    test_dirs = [
        PROJECT_ROOT / "tests" / "unit",
        PROJECT_ROOT / "tests" / "integration",
        PROJECT_ROOT / "tests" / "e2e",
    ]

    test_count = 0
    for test_dir in test_dirs:
        if test_dir.exists():
            test_files = list(test_dir.glob("test_*.py"))
            test_count += len(test_files)
            print(f"  {test_dir.relative_to(PROJECT_ROOT)}: {len(test_files)} test files")

    print(f"\nTotal test files: {test_count}")

    # Check for common test issues
    for test_dir in test_dirs:
        if not test_dir.exists():
            continue

        for test_file in test_dir.glob("test_*.py"):
            try:
                with open(test_file, "r") as f:
                    content = f.read()

                # Check if test has actual tests
                if "def test_" not in content and "class Test" not in content:
                    print(f"  ⚠️  {test_file.relative_to(PROJECT_ROOT)} has no test functions")

            except Exception as e:
                print(f"  ❌ Error reading {test_file}: {e}")


def main():
    """Run all checks."""
    print("=== Basic CI Issue Check ===\n")

    # Check syntax
    syntax_errors = check_syntax_errors()

    # Check style
    style_issues = check_basic_style()

    # Check undefined names
    check_undefined_names()

    # Check tests
    check_test_files()

    # Summary
    print("\n=== Summary ===")
    if syntax_errors:
        print(f"❌ {len(syntax_errors)} files with syntax errors")
    else:
        print("✅ No syntax errors")

    if style_issues:
        print(f"⚠️  {len(style_issues)} files with style issues")
    else:
        print("✅ No style issues")

    print("\nRecommendations:")
    print("1. Fix any syntax errors first")
    print("2. Run 'black music_gen tests scripts' to fix formatting")
    print("3. Run 'isort music_gen tests scripts' to fix imports")
    print("4. Ensure all test files have actual test functions")


if __name__ == "__main__":
    main()
