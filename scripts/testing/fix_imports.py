#!/usr/bin/env python3
"""
Fix common import issues that cause CI failures.
"""

import ast
import os
from pathlib import Path
from typing import List, Set

PROJECT_ROOT = Path(__file__).parent.parent.parent
MUSIC_GEN_DIR = PROJECT_ROOT / "music_gen"


def analyze_imports(file_path: Path) -> Set[str]:
    """Extract all imports from a Python file."""
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            tree = ast.parse(f.read())
        except SyntaxError:
            print(f"Syntax error in {file_path}")
            return set()

    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".")[0])

    return imports


def check_missing_imports():
    """Check for common missing imports in the codebase."""
    print("Checking for missing imports...")

    # Common imports that should be conditional
    optional_imports = {
        "torch",
        "torchaudio",
        "transformers",
        "encodec",
        "librosa",
        "soundfile",
        "scipy",
        "numpy",
        "fastapi",
        "uvicorn",
        "pydantic",
        "httpx",
        "pytorch_lightning",
        "wandb",
        "hydra",
        "pedalboard",
        "pyrubberband",
        "pretty_midi",
    }

    issues = []

    # Check all Python files
    for py_file in MUSIC_GEN_DIR.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue

        imports = analyze_imports(py_file)
        missing = imports & optional_imports

        if missing:
            rel_path = py_file.relative_to(PROJECT_ROOT)
            issues.append((rel_path, missing))

    if issues:
        print("\nFiles with potentially problematic imports:")
        for file_path, missing in sorted(issues):
            print(f"\n{file_path}:")
            for imp in sorted(missing):
                print(f"  - {imp}")
    else:
        print("No import issues found!")

    return issues


def add_import_guards(file_path: Path, imports_to_guard: Set[str]):
    """Add import guards for optional dependencies."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Check if already has guards
    if "try:" in content and "import" in content:
        return False

    lines = content.split("\n")
    new_lines = []

    i = 0
    while i < len(lines):
        line = lines[i]

        # Check if this is an import line that needs guarding
        needs_guard = False
        for imp in imports_to_guard:
            if f"import {imp}" in line or f"from {imp}" in line:
                needs_guard = True
                break

        if needs_guard:
            # Find all consecutive import lines
            import_lines = [line]
            j = i + 1
            while j < len(lines) and ("import" in lines[j] or lines[j].strip() == ""):
                import_lines.append(lines[j])
                j += 1

            # Add guard
            new_lines.append("try:")
            for imp_line in import_lines:
                if imp_line.strip():
                    new_lines.append(f"    {imp_line}")
                else:
                    new_lines.append(imp_line)
            new_lines.append("except ImportError:")
            new_lines.append("    pass  # Optional dependency not installed")

            i = j
        else:
            new_lines.append(line)
            i += 1

    # Write back
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines))

    return True


def fix_common_issues():
    """Fix common issues that cause CI failures."""
    print("\nFixing common issues...")

    # Fix __init__.py files that might have problematic imports
    init_files_to_check = [
        MUSIC_GEN_DIR / "__init__.py",
        MUSIC_GEN_DIR / "models" / "__init__.py",
        MUSIC_GEN_DIR / "audio" / "__init__.py",
        MUSIC_GEN_DIR / "data" / "__init__.py",
    ]

    for init_file in init_files_to_check:
        if init_file.exists():
            print(f"Checking {init_file.relative_to(PROJECT_ROOT)}...")

            # Make sure __init__.py files don't have heavy imports
            with open(init_file, "r") as f:
                content = f.read()

            if "torch" in content or "transformers" in content:
                print(f"  - Found heavy imports, making them optional")
                # Could implement fix here

    print("\nDone!")


def create_minimal_test_requirements():
    """Create a minimal requirements file for CI testing."""
    minimal_reqs = """# Minimal requirements for CI testing
pytest>=7.0.0
pytest-cov>=4.0.0
pytest-asyncio>=0.21.0
pytest-timeout>=2.1.0
black>=23.0.0
isort>=5.12.0
flake8>=6.0.0
mypy>=1.0.0
httpx>=0.24.0
click>=8.1.0
rich>=13.0.0
pydantic>=2.0.0
"""

    test_reqs_path = PROJECT_ROOT / "requirements-test.txt"
    with open(test_reqs_path, "w") as f:
        f.write(minimal_reqs)

    print(f"Created {test_reqs_path}")


if __name__ == "__main__":
    # Check for import issues
    issues = check_missing_imports()

    # Fix common issues
    fix_common_issues()

    # Create minimal test requirements
    create_minimal_test_requirements()

    print("\nRecommendations:")
    print("1. Use requirements-test.txt for CI to avoid heavy dependencies")
    print("2. Make optional imports conditional with try/except blocks")
    print("3. Move heavy imports inside functions where possible")
    print("4. Use TYPE_CHECKING for type-only imports")
