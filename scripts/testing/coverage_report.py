#!/usr/bin/env python3
"""
Generate comprehensive test coverage report and track progress.
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).parent.parent.parent


def run_coverage_analysis() -> Dict:
    """Run pytest with coverage and parse results."""
    print("Running test coverage analysis...")

    # Run pytest with coverage
    cmd = [
        "python",
        "-m",
        "pytest",
        f"{PROJECT_ROOT}/tests",
        "--cov=music_gen",
        "--cov-report=json",
        "--cov-report=term-missing:skip-covered",
        "-q",
        "--tb=short",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)

        # Parse coverage.json if it exists
        coverage_file = PROJECT_ROOT / "coverage.json"
        if coverage_file.exists():
            with open(coverage_file) as f:
                coverage_data = json.load(f)
                return parse_coverage_data(coverage_data)
    except Exception as e:
        print(f"Error running coverage: {e}")

    return {"total_coverage": 0, "files": {}}


def parse_coverage_data(coverage_data: Dict) -> Dict:
    """Parse coverage.json data."""
    files = coverage_data.get("files", {})

    total_lines = 0
    covered_lines = 0
    file_coverage = {}

    for file_path, file_data in files.items():
        if "music_gen" not in file_path:
            continue

        summary = file_data.get("summary", {})
        lines_total = summary.get("num_statements", 0)
        lines_covered = summary.get("covered_statements", 0)
        percent = summary.get("percent_covered", 0)

        total_lines += lines_total
        covered_lines += lines_covered

        # Get relative path
        try:
            rel_path = Path(file_path).relative_to(PROJECT_ROOT)
        except:
            rel_path = file_path

        file_coverage[str(rel_path)] = {
            "lines_total": lines_total,
            "lines_covered": lines_covered,
            "percent": percent,
            "missing_lines": file_data.get("missing_lines", []),
        }

    total_coverage = (covered_lines / total_lines * 100) if total_lines > 0 else 0

    return {
        "total_coverage": total_coverage,
        "total_lines": total_lines,
        "covered_lines": covered_lines,
        "files": file_coverage,
    }


def generate_coverage_report(coverage_data: Dict) -> str:
    """Generate detailed coverage report."""
    report = []

    report.append("=" * 80)
    report.append("TEST COVERAGE REPORT")
    report.append("=" * 80)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    # Overall statistics
    total_coverage = coverage_data.get("total_coverage", 0)
    total_lines = coverage_data.get("total_lines", 0)
    covered_lines = coverage_data.get("covered_lines", 0)

    report.append("OVERALL COVERAGE")
    report.append("-" * 40)
    report.append(f"Total Coverage: {total_coverage:.1f}%")
    report.append(f"Lines Covered: {covered_lines:,} / {total_lines:,}")
    report.append(f"Lines Missing: {total_lines - covered_lines:,}")
    report.append("")

    # Progress towards goal
    goal = 90.0
    progress = min(total_coverage / goal * 100, 100)
    report.append("PROGRESS TOWARDS 90% GOAL")
    report.append("-" * 40)
    report.append(f"Current: {total_coverage:.1f}% / Goal: {goal}%")
    report.append(
        f"Progress: {'█' * int(progress / 5)}{'░' * (20 - int(progress / 5))} {progress:.1f}%"
    )

    if total_coverage >= goal:
        report.append("✅ GOAL ACHIEVED!")
    else:
        lines_needed = int((goal / 100 * total_lines) - covered_lines)
        report.append(f"Lines needed for {goal}%: {lines_needed:,}")

    report.append("")

    # Files with lowest coverage
    files = coverage_data.get("files", {})
    sorted_files = sorted(files.items(), key=lambda x: x[1]["percent"])

    report.append("FILES WITH LOWEST COVERAGE")
    report.append("-" * 40)

    for file_path, data in sorted_files[:20]:
        if data["lines_total"] < 10:  # Skip very small files
            continue
        percent = data["percent"]
        missing = data["lines_total"] - data["lines_covered"]
        report.append(f"{percent:5.1f}% | {missing:4d} missing | {file_path}")

    report.append("")

    # Module coverage summary
    module_coverage = calculate_module_coverage(files)

    report.append("MODULE COVERAGE SUMMARY")
    report.append("-" * 40)

    for module, data in sorted(module_coverage.items(), key=lambda x: x[1]["percent"]):
        percent = data["percent"]
        report.append(f"{percent:5.1f}% | {module}/")

    report.append("")

    # Recommendations
    report.append("RECOMMENDATIONS")
    report.append("-" * 40)

    recommendations = generate_recommendations(coverage_data, module_coverage)
    for i, rec in enumerate(recommendations[:10], 1):
        report.append(f"{i}. {rec}")

    return "\n".join(report)


def calculate_module_coverage(files: Dict) -> Dict:
    """Calculate coverage by module."""
    modules = {}

    for file_path, data in files.items():
        # Extract module from path
        parts = Path(file_path).parts
        if "music_gen" in parts:
            idx = parts.index("music_gen")
            if idx + 1 < len(parts):
                module = parts[idx + 1]

                if module not in modules:
                    modules[module] = {"lines_total": 0, "lines_covered": 0, "files": 0}

                modules[module]["lines_total"] += data["lines_total"]
                modules[module]["lines_covered"] += data["lines_covered"]
                modules[module]["files"] += 1

    # Calculate percentages
    for module, data in modules.items():
        if data["lines_total"] > 0:
            data["percent"] = data["lines_covered"] / data["lines_total"] * 100
        else:
            data["percent"] = 0

    return modules


def generate_recommendations(coverage_data: Dict, module_coverage: Dict) -> List[str]:
    """Generate specific recommendations for improving coverage."""
    recommendations = []

    # Overall coverage recommendations
    total_coverage = coverage_data.get("total_coverage", 0)

    if total_coverage < 30:
        recommendations.append(
            "Focus on unit tests for core utility modules (exceptions, logging, config)"
        )
        recommendations.append("Add basic tests for all API endpoints")
        recommendations.append("Create fixtures and mocks for common test scenarios")
    elif total_coverage < 60:
        recommendations.append("Add integration tests for complete workflows")
        recommendations.append("Test error handling and edge cases")
        recommendations.append("Increase coverage for model and training modules")
    elif total_coverage < 90:
        recommendations.append("Focus on untested edge cases and error paths")
        recommendations.append("Add performance and stress tests")
        recommendations.append("Test configuration variations and parameter combinations")

    # Module-specific recommendations
    for module, data in module_coverage.items():
        if data["percent"] < 50 and data["lines_total"] > 100:
            recommendations.append(
                f"Priority: Improve {module} module coverage ({data['percent']:.1f}%)"
            )

    # File-specific recommendations
    files = coverage_data.get("files", {})
    critical_files = []

    for file_path, data in files.items():
        if data["percent"] < 20 and data["lines_total"] > 50:
            critical_files.append((file_path, data))

    if critical_files:
        critical_files.sort(key=lambda x: x[1]["lines_total"], reverse=True)
        for file_path, data in critical_files[:3]:
            recommendations.append(
                f"Critical: Add tests for {file_path} (0% coverage, {data['lines_total']} lines)"
            )

    # Testing strategy recommendations
    recommendations.append("Use pytest-mock for mocking external dependencies")
    recommendations.append("Add pytest fixtures for common test data")
    recommendations.append("Consider property-based testing for data processing functions")
    recommendations.append("Add end-to-end tests for critical user workflows")

    return recommendations


def save_coverage_history(coverage_data: Dict):
    """Save coverage history for tracking progress."""
    history_file = PROJECT_ROOT / "test_coverage_history.json"

    # Load existing history
    if history_file.exists():
        with open(history_file) as f:
            history = json.load(f)
    else:
        history = []

    # Add current data
    entry = {
        "timestamp": datetime.now().isoformat(),
        "total_coverage": coverage_data.get("total_coverage", 0),
        "total_lines": coverage_data.get("total_lines", 0),
        "covered_lines": coverage_data.get("covered_lines", 0),
    }

    history.append(entry)

    # Keep last 50 entries
    history = history[-50:]

    # Save
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)


def show_progress_graph(history: List[Dict]):
    """Show coverage progress over time."""
    if len(history) < 2:
        return

    print("\nCOVERAGE PROGRESS")
    print("-" * 40)

    # Simple ASCII graph
    max_coverage = 100
    graph_height = 10

    # Get last 20 entries
    recent = history[-20:]

    for i in range(graph_height, -1, -1):
        threshold = i * 10
        line = f"{threshold:3d}% |"

        for entry in recent:
            coverage = entry.get("total_coverage", 0)
            if coverage >= threshold:
                line += "█"
            else:
                line += " "

        print(line)

    print("     +" + "-" * len(recent))
    print("      " + "".join([str(i % 10) for i in range(len(recent))]))


def main():
    """Main function."""
    # Run coverage analysis
    coverage_data = run_coverage_analysis()

    if not coverage_data.get("files"):
        print("No coverage data found. Make sure pytest-cov is installed.")
        print("Run: pip install pytest-cov")
        return

    # Generate report
    report = generate_coverage_report(coverage_data)
    print(report)

    # Save report
    report_file = PROJECT_ROOT / "test_coverage_report.txt"
    with open(report_file, "w") as f:
        f.write(report)

    print(f"\n✅ Report saved to: {report_file}")

    # Save history
    save_coverage_history(coverage_data)

    # Show progress graph
    history_file = PROJECT_ROOT / "test_coverage_history.json"
    if history_file.exists():
        with open(history_file) as f:
            history = json.load(f)
            if history:
                show_progress_graph(history)


if __name__ == "__main__":
    main()
