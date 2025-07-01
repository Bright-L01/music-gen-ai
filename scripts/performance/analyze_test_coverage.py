#!/usr/bin/env python3
"""
Analyze test coverage and identify modules that need more tests.
"""

import ast
import os
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).parent.parent.parent
MUSIC_GEN_DIR = PROJECT_ROOT / "music_gen"
TESTS_DIR = PROJECT_ROOT / "tests"


def get_python_files(directory: Path) -> List[Path]:
    """Get all Python files in a directory recursively."""
    return list(directory.rglob("*.py"))


def analyze_module(file_path: Path) -> Dict[str, int]:
    """Analyze a Python module to count functions, classes, etc."""
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            tree = ast.parse(f.read())
        except:
            return {"functions": 0, "classes": 0, "methods": 0}

    stats = {"functions": 0, "classes": 0, "methods": 0, "lines": len(open(file_path).readlines())}

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Check if it's a method (inside a class)
            for parent in ast.walk(tree):
                if isinstance(parent, ast.ClassDef) and node in parent.body:
                    stats["methods"] += 1
                    break
            else:
                stats["functions"] += 1
        elif isinstance(node, ast.ClassDef):
            stats["classes"] += 1

    return stats


def find_test_file(module_path: Path) -> Path:
    """Find the corresponding test file for a module."""
    relative_path = module_path.relative_to(MUSIC_GEN_DIR)
    test_name = f"test_{relative_path.stem}.py"

    # Check various test locations
    possible_locations = [
        TESTS_DIR / "unit" / test_name,
        TESTS_DIR / test_name,
        TESTS_DIR / relative_path.parent / test_name,
    ]

    for location in possible_locations:
        if location.exists():
            return location

    return None


def analyze_test_coverage():
    """Analyze test coverage for the entire project."""
    print("=== Test Coverage Analysis ===\n")

    # Get all Python modules
    modules = get_python_files(MUSIC_GEN_DIR)
    modules = [m for m in modules if "__pycache__" not in str(m)]

    coverage_report = {
        "covered": [],
        "partial": [],
        "uncovered": [],
        "total_stats": {"modules": 0, "functions": 0, "classes": 0, "methods": 0, "lines": 0},
    }

    for module in sorted(modules):
        if module.name == "__init__.py" and len(open(module).read().strip()) == 0:
            continue

        stats = analyze_module(module)
        test_file = find_test_file(module)

        coverage_report["total_stats"]["modules"] += 1
        coverage_report["total_stats"]["functions"] += stats["functions"]
        coverage_report["total_stats"]["classes"] += stats["classes"]
        coverage_report["total_stats"]["methods"] += stats["methods"]
        coverage_report["total_stats"]["lines"] += stats["lines"]

        if test_file:
            test_stats = analyze_module(test_file)
            # Simple heuristic: if test has at least half as many functions/methods
            if (test_stats["functions"] + test_stats["methods"]) >= (
                stats["functions"] + stats["methods"]
            ) * 0.5:
                coverage_report["covered"].append((module, test_file))
            else:
                coverage_report["partial"].append((module, test_file))
        else:
            coverage_report["uncovered"].append(module)

    # Print summary
    print(f"Total Modules: {coverage_report['total_stats']['modules']}")
    print(f"Total Functions: {coverage_report['total_stats']['functions']}")
    print(f"Total Classes: {coverage_report['total_stats']['classes']}")
    print(f"Total Methods: {coverage_report['total_stats']['methods']}")
    print(f"Total Lines: {coverage_report['total_stats']['lines']}")
    print()

    print(f"Modules with good test coverage: {len(coverage_report['covered'])}")
    print(f"Modules with partial test coverage: {len(coverage_report['partial'])}")
    print(f"Modules without tests: {len(coverage_report['uncovered'])}")
    print()

    # Calculate estimated coverage
    covered_pct = len(coverage_report["covered"]) / coverage_report["total_stats"]["modules"] * 100
    partial_pct = len(coverage_report["partial"]) / coverage_report["total_stats"]["modules"] * 100
    estimated_coverage = covered_pct + (partial_pct * 0.5)

    print(f"Estimated test coverage: {estimated_coverage:.1f}%")
    print()

    # List modules that need tests
    if coverage_report["uncovered"]:
        print("=== Modules Without Tests ===")
        for module in sorted(coverage_report["uncovered"])[:20]:
            rel_path = module.relative_to(MUSIC_GEN_DIR)
            print(f"  - {rel_path}")
        if len(coverage_report["uncovered"]) > 20:
            print(f"  ... and {len(coverage_report['uncovered']) - 20} more")
        print()

    # List modules with partial coverage
    if coverage_report["partial"]:
        print("=== Modules With Partial Test Coverage ===")
        for module, test in sorted(coverage_report["partial"])[:10]:
            rel_path = module.relative_to(MUSIC_GEN_DIR)
            print(f"  - {rel_path}")
        if len(coverage_report["partial"]) > 10:
            print(f"  ... and {len(coverage_report['partial']) - 10} more")
        print()

    # Suggest priority modules for testing
    print("=== Priority Modules for Testing ===")
    priority_modules = []

    for module in coverage_report["uncovered"]:
        stats = analyze_module(module)
        complexity = stats["functions"] + stats["classes"] * 2 + stats["methods"]
        if complexity > 5:  # Non-trivial modules
            priority_modules.append((module, complexity))

    priority_modules.sort(key=lambda x: x[1], reverse=True)

    for module, complexity in priority_modules[:15]:
        rel_path = module.relative_to(MUSIC_GEN_DIR)
        print(f"  - {rel_path} (complexity: {complexity})")

    return coverage_report


def generate_test_templates(coverage_report: Dict):
    """Generate test file templates for uncovered modules."""
    print("\n=== Generating Test Templates ===")

    template_count = 0
    for module in coverage_report["uncovered"][:10]:  # Generate for top 10
        rel_path = module.relative_to(MUSIC_GEN_DIR)

        # Determine test directory
        if "api" in str(rel_path):
            test_dir = TESTS_DIR / "unit"
        elif "models" in str(rel_path):
            test_dir = TESTS_DIR / "unit"
        elif "utils" in str(rel_path):
            test_dir = TESTS_DIR / "unit"
        else:
            test_dir = TESTS_DIR / "unit"

        test_file = test_dir / f"test_{rel_path.stem}.py"

        if not test_file.exists():
            test_content = f'''"""
Tests for {rel_path}
"""

import pytest
from music_gen.{str(rel_path).replace(".py", "").replace("/", ".")} import *


class Test{rel_path.stem.title().replace("_", "")}:
    """Test cases for {rel_path.stem} module."""

    def test_placeholder(self):
        """Placeholder test - implement actual tests."""
        # TODO: Implement actual tests
        assert True
'''

            test_dir.mkdir(parents=True, exist_ok=True)
            with open(test_file, "w") as f:
                f.write(test_content)

            print(f"  Created: {test_file.relative_to(PROJECT_ROOT)}")
            template_count += 1

    print(f"\nGenerated {template_count} test templates")


if __name__ == "__main__":
    coverage_report = analyze_test_coverage()

    # Optionally generate test templates
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--generate":
        generate_test_templates(coverage_report)
