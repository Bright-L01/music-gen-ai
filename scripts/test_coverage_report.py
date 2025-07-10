#!/usr/bin/env python3
"""
Test Coverage Analysis and Reporting Script

Analyzes current test coverage and generates detailed reports
for the MusicGen AI testing infrastructure.
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class TestFile:
    """Represents a test file and its metadata."""

    path: Path
    test_count: int
    test_classes: List[str]
    covered_modules: List[str]
    test_types: Set[str]


@dataclass
class SourceModule:
    """Represents a source module and its testing status."""

    path: Path
    loc: int
    functions: List[str]
    classes: List[str]
    covered_by: List[str]
    coverage_level: str


class TestCoverageAnalyzer:
    """Analyzes test coverage across the codebase."""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.source_dir = self.project_root / "music_gen"
        self.test_dir = self.project_root / "tests"

        self.test_files: List[TestFile] = []
        self.source_modules: List[SourceModule] = []

    def analyze_test_files(self) -> None:
        """Analyze all test files to understand coverage."""
        test_patterns = ["test_*.py", "*_test.py"]

        for pattern in test_patterns:
            for test_file in self.test_dir.rglob(pattern):
                if test_file.is_file():
                    self._analyze_test_file(test_file)

    def _analyze_test_file(self, test_file: Path) -> None:
        """Analyze a single test file."""
        try:
            content = test_file.read_text(encoding="utf-8")

            # Count test functions
            test_functions = re.findall(r"def (test_\w+)", content)

            # Find test classes
            test_classes = re.findall(r"class (Test\w+)", content)

            # Find imported modules from music_gen
            imports = re.findall(r"from music_gen\.([.\w]+) import", content)
            imports.extend(re.findall(r"import music_gen\.([.\w]+)", content))

            # Determine test types
            test_types = set()
            if "@pytest.mark.unit" in content:
                test_types.add("unit")
            if "@pytest.mark.integration" in content:
                test_types.add("integration")
            if "@pytest.mark.e2e" in content:
                test_types.add("e2e")
            if "@pytest.mark.performance" in content:
                test_types.add("performance")
            if "@pytest.mark.security" in content:
                test_types.add("security")

            # Default to unit if no markers found
            if not test_types:
                test_types.add("unit")

            test_file_obj = TestFile(
                path=test_file,
                test_count=len(test_functions),
                test_classes=test_classes,
                covered_modules=list(set(imports)),
                test_types=test_types,
            )

            self.test_files.append(test_file_obj)

        except Exception as e:
            print(f"Error analyzing {test_file}: {e}")

    def analyze_source_modules(self) -> None:
        """Analyze source modules to understand what needs testing."""
        for py_file in self.source_dir.rglob("*.py"):
            if py_file.is_file() and py_file.name != "__init__.py":
                self._analyze_source_module(py_file)

    def _analyze_source_module(self, source_file: Path) -> None:
        """Analyze a single source module."""
        try:
            content = source_file.read_text(encoding="utf-8")

            # Count lines of code (excluding comments and empty lines)
            lines = content.split("\n")
            loc = sum(1 for line in lines if line.strip() and not line.strip().startswith("#"))

            # Find functions and classes
            functions = re.findall(r"def (\w+)", content)
            classes = re.findall(r"class (\w+)", content)

            # Find which test files cover this module
            module_path = (
                str(source_file.relative_to(self.source_dir)).replace("/", ".").replace(".py", "")
            )
            covered_by = []

            for test_file in self.test_files:
                if any(
                    module_path.startswith(mod) or mod.startswith(module_path)
                    for mod in test_file.covered_modules
                ):
                    covered_by.append(str(test_file.path.relative_to(self.test_dir)))

            # Determine coverage level
            coverage_level = self._determine_coverage_level(source_file, covered_by)

            source_module = SourceModule(
                path=source_file,
                loc=loc,
                functions=functions,
                classes=classes,
                covered_by=covered_by,
                coverage_level=coverage_level,
            )

            self.source_modules.append(source_module)

        except Exception as e:
            print(f"Error analyzing {source_file}: {e}")

    def _determine_coverage_level(self, source_file: Path, covered_by: List[str]) -> str:
        """Determine the coverage level for a module."""
        if not covered_by:
            return "none"

        # Check for comprehensive testing
        has_unit = any("unit" in test_path for test_path in covered_by)
        has_integration = any("integration" in test_path for test_path in covered_by)

        if has_unit and has_integration:
            return "comprehensive"
        elif has_unit:
            return "basic"
        elif has_integration:
            return "integration_only"
        else:
            return "minimal"

    def generate_coverage_report(self) -> Dict:
        """Generate comprehensive coverage report."""
        # Overall statistics
        total_source_files = len(self.source_modules)
        total_test_files = len(self.test_files)
        total_tests = sum(tf.test_count for tf in self.test_files)

        # Coverage statistics
        covered_modules = sum(1 for sm in self.source_modules if sm.covered_by)
        coverage_percentage = (
            (covered_modules / total_source_files * 100) if total_source_files > 0 else 0
        )

        # Test type distribution
        test_type_counts = defaultdict(int)
        for test_file in self.test_files:
            for test_type in test_file.test_types:
                test_type_counts[test_type] += test_file.test_count

        # Coverage gaps
        uncovered_modules = [sm for sm in self.source_modules if not sm.covered_by]
        poorly_covered = [
            sm for sm in self.source_modules if sm.coverage_level in ["minimal", "none"]
        ]

        # High-priority gaps (core modules without good coverage)
        core_patterns = ["models/", "inference/", "training/", "api/", "core/"]
        critical_gaps = []

        for module in uncovered_modules:
            module_path = str(module.path.relative_to(self.source_dir))
            if any(pattern in module_path for pattern in core_patterns):
                critical_gaps.append(
                    {
                        "module": module_path,
                        "loc": module.loc,
                        "functions": len(module.functions),
                        "classes": len(module.classes),
                        "priority": "high",
                    }
                )

        return {
            "summary": {
                "total_source_files": total_source_files,
                "total_test_files": total_test_files,
                "total_tests": total_tests,
                "covered_modules": covered_modules,
                "coverage_percentage": round(coverage_percentage, 1),
                "uncovered_modules": len(uncovered_modules),
            },
            "test_distribution": dict(test_type_counts),
            "coverage_levels": {
                level: sum(1 for sm in self.source_modules if sm.coverage_level == level)
                for level in ["none", "minimal", "basic", "integration_only", "comprehensive"]
            },
            "critical_gaps": critical_gaps[:10],  # Top 10 critical gaps
            "recommendations": self._generate_recommendations(),
        }

    def _generate_recommendations(self) -> List[str]:
        """Generate testing recommendations based on analysis."""
        recommendations = []

        # Coverage recommendations
        coverage_percentage = (
            (sum(1 for sm in self.source_modules if sm.covered_by) / len(self.source_modules) * 100)
            if self.source_modules
            else 0
        )

        if coverage_percentage < 50:
            recommendations.append(
                "CRITICAL: Test coverage is below 50%. Implement basic unit tests for core modules."
            )
        elif coverage_percentage < 80:
            recommendations.append(
                "Increase test coverage to 80%+ by adding unit tests for uncovered modules."
            )

        # Test type recommendations
        test_type_counts = defaultdict(int)
        for test_file in self.test_files:
            for test_type in test_file.test_types:
                test_type_counts[test_type] += 1

        if test_type_counts["integration"] < 5:
            recommendations.append(
                "Add more integration tests to verify service-to-service communication."
            )

        if test_type_counts["e2e"] < 3:
            recommendations.append("Implement end-to-end tests for complete user workflows.")

        if test_type_counts["performance"] < 2:
            recommendations.append(
                "Add performance tests with benchmarks and regression detection."
            )

        # Module-specific recommendations
        core_modules_uncovered = 0
        for module in self.source_modules:
            module_path = str(module.path.relative_to(self.source_dir))
            if any(pattern in module_path for pattern in ["models/", "inference/", "core/"]):
                if not module.covered_by:
                    core_modules_uncovered += 1

        if core_modules_uncovered > 0:
            recommendations.append(
                f"URGENT: {core_modules_uncovered} core modules lack test coverage. "
                "Prioritize testing for models/, inference/, and core/ modules."
            )

        return recommendations

    def save_detailed_report(self, output_file: str) -> None:
        """Save detailed coverage report to JSON file."""
        report = {
            "test_files": [
                {
                    "path": str(tf.path.relative_to(self.test_dir)),
                    "test_count": tf.test_count,
                    "test_classes": tf.test_classes,
                    "covered_modules": tf.covered_modules,
                    "test_types": list(tf.test_types),
                }
                for tf in self.test_files
            ],
            "source_modules": [
                {
                    "path": str(sm.path.relative_to(self.source_dir)),
                    "loc": sm.loc,
                    "functions": len(sm.functions),
                    "classes": len(sm.classes),
                    "covered_by": sm.covered_by,
                    "coverage_level": sm.coverage_level,
                }
                for sm in self.source_modules
            ],
            "coverage_analysis": self.generate_coverage_report(),
        }

        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)

    def print_summary_report(self) -> None:
        """Print human-readable summary report."""
        report = self.generate_coverage_report()

        print("=" * 60)
        print("MUSICGEN AI - TEST COVERAGE ANALYSIS")
        print("=" * 60)

        # Summary statistics
        summary = report["summary"]
        print(f"\n📊 COVERAGE SUMMARY:")
        print(f"   Total Source Files: {summary['total_source_files']}")
        print(f"   Total Test Files: {summary['total_test_files']}")
        print(f"   Total Tests: {summary['total_tests']}")
        print(
            f"   Coverage: {summary['coverage_percentage']}% ({summary['covered_modules']}/{summary['total_source_files']} modules)"
        )

        # Test distribution
        print(f"\n🧪 TEST DISTRIBUTION:")
        for test_type, count in report["test_distribution"].items():
            print(f"   {test_type.capitalize()}: {count} tests")

        # Coverage levels
        print(f"\n📈 COVERAGE LEVELS:")
        for level, count in report["coverage_levels"].items():
            if count > 0:
                print(f"   {level.replace('_', ' ').title()}: {count} modules")

        # Critical gaps
        if report["critical_gaps"]:
            print(f"\n🚨 CRITICAL COVERAGE GAPS:")
            for gap in report["critical_gaps"][:5]:
                print(f"   {gap['module']} ({gap['loc']} LOC, {gap['functions']} functions)")

        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(report["recommendations"], 1):
            print(f"   {i}. {rec}")

        print("\n" + "=" * 60)


def main():
    """Main function to run coverage analysis."""
    analyzer = TestCoverageAnalyzer("/Users/brightliu/Coding_Projects/music_gen")

    print("Analyzing test files...")
    analyzer.analyze_test_files()

    print("Analyzing source modules...")
    analyzer.analyze_source_modules()

    print("Generating coverage report...")
    analyzer.print_summary_report()

    # Save detailed report
    output_file = "/Users/brightliu/Coding_Projects/music_gen/coverage_analysis.json"
    analyzer.save_detailed_report(output_file)
    print(f"\nDetailed report saved to: {output_file}")


if __name__ == "__main__":
    main()
