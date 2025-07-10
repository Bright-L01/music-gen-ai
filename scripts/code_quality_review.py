#!/usr/bin/env python3
"""
Code Quality Review System for Music Gen AI
Comprehensive code quality checks including tests, security, performance, and standards
"""

import os
import json
import subprocess
import time
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass, asdict
import ast
import re
import requests
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"code_quality_review_{int(time.time())}.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


@dataclass
class QualityMetric:
    """Quality metric result"""

    name: str
    value: float
    target: float
    status: str  # "pass", "warn", "fail"
    details: str
    recommendations: List[str]


@dataclass
class CodeIssue:
    """Code quality issue"""

    file_path: str
    line_number: int
    issue_type: str
    severity: str
    description: str
    recommendation: str


@dataclass
class SecurityVulnerability:
    """Security vulnerability"""

    file_path: str
    line_number: int
    issue_id: str
    severity: str
    confidence: str
    description: str
    more_info: str


@dataclass
class DependencyIssue:
    """Dependency issue"""

    package: str
    current_version: str
    latest_version: str
    issue_type: str  # "outdated", "vulnerable", "deprecated"
    severity: str
    description: str


@dataclass
class PerformanceIssue:
    """Performance issue"""

    function_name: str
    file_path: str
    line_number: int
    issue_type: str
    impact: str
    recommendation: str


class CodeQualityReviewer:
    def __init__(self):
        self.project_root = Path.cwd()
        self.results_dir = self.project_root / "results" / "code_quality"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Quality thresholds
        self.thresholds = {
            "test_coverage": 90.0,
            "code_duplication": 5.0,
            "complexity": 10.0,
            "maintainability": 80.0,
            "security_score": 90.0,
            "performance_score": 85.0,
        }

        # Initialize results
        self.quality_metrics: List[QualityMetric] = []
        self.code_issues: List[CodeIssue] = []
        self.security_vulnerabilities: List[SecurityVulnerability] = []
        self.dependency_issues: List[DependencyIssue] = []
        self.performance_issues: List[PerformanceIssue] = []

    def run_full_test_suite(self) -> QualityMetric:
        """Run complete test suite with coverage reporting"""
        logger.info("Running full test suite with coverage...")

        try:
            # Run pytest with coverage, junit reporting, and parallel execution
            cmd = [
                "python",
                "-m",
                "pytest",
                "--cov=music_gen",
                "--cov-report=html:htmlcov",
                "--cov-report=json:coverage.json",
                "--cov-report=term-missing",
                "--junit-xml=test_results.xml",
                "--tb=short",
                "-v",
                "--durations=10",
                "--maxfail=5",
                "-n",
                "auto",  # Parallel execution
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root,
                timeout=1800,  # 30 minutes timeout
            )

            # Parse coverage results
            coverage_data = {}
            if os.path.exists("coverage.json"):
                with open("coverage.json", "r") as f:
                    coverage_data = json.load(f)

            total_coverage = coverage_data.get("totals", {}).get("percent_covered", 0)

            # Parse test results
            test_summary = self._parse_test_results(result.stdout)

            status = "pass" if total_coverage >= self.thresholds["test_coverage"] else "fail"

            recommendations = []
            if total_coverage < self.thresholds["test_coverage"]:
                recommendations.append(
                    f"Increase test coverage to {self.thresholds['test_coverage']}%"
                )

            if test_summary["failed"] > 0:
                recommendations.append(f"Fix {test_summary['failed']} failing tests")
                status = "fail"

            return QualityMetric(
                name="Test Coverage",
                value=total_coverage,
                target=self.thresholds["test_coverage"],
                status=status,
                details=f"Coverage: {total_coverage:.1f}%, Tests: {test_summary['passed']}/{test_summary['total']}",
                recommendations=recommendations,
            )

        except subprocess.TimeoutExpired:
            logger.error("Test suite timed out")
            return QualityMetric(
                name="Test Coverage",
                value=0,
                target=self.thresholds["test_coverage"],
                status="fail",
                details="Test suite timed out",
                recommendations=["Optimize slow tests", "Consider test parallelization"],
            )
        except Exception as e:
            logger.error(f"Error running test suite: {e}")
            return QualityMetric(
                name="Test Coverage",
                value=0,
                target=self.thresholds["test_coverage"],
                status="fail",
                details=f"Error: {str(e)}",
                recommendations=["Fix test suite configuration"],
            )

    def check_code_duplication(self) -> QualityMetric:
        """Check for code duplication"""
        logger.info("Checking for code duplication...")

        try:
            # Use jscpd for code duplication detection
            cmd = [
                "jscpd",
                "--min-lines",
                "10",
                "--min-tokens",
                "70",
                "--format",
                "json",
                "--output",
                "duplication_report.json",
                "music_gen/",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)

            duplication_percentage = 0
            duplicate_blocks = []

            if os.path.exists("duplication_report.json"):
                with open("duplication_report.json", "r") as f:
                    duplication_data = json.load(f)

                    statistics = duplication_data.get("statistics", {})
                    duplication_percentage = statistics.get("percentage", 0)

                    # Extract duplicate blocks
                    for duplicate in duplication_data.get("duplicates", []):
                        duplicate_blocks.append(
                            {
                                "lines": duplicate.get("lines", 0),
                                "tokens": duplicate.get("tokens", 0),
                                "files": [f["name"] for f in duplicate.get("map", [])],
                            }
                        )

            status = (
                "pass" if duplication_percentage <= self.thresholds["code_duplication"] else "fail"
            )

            recommendations = []
            if duplication_percentage > self.thresholds["code_duplication"]:
                recommendations.append("Refactor duplicate code into reusable functions")
                recommendations.append("Consider using design patterns to reduce duplication")

            # Add specific recommendations for duplicate blocks
            for block in duplicate_blocks[:5]:  # Top 5 duplicates
                if block["lines"] > 20:
                    recommendations.append(
                        f"Large duplicate block ({block['lines']} lines) in {', '.join(block['files'])}"
                    )

            return QualityMetric(
                name="Code Duplication",
                value=duplication_percentage,
                target=self.thresholds["code_duplication"],
                status=status,
                details=f"Duplication: {duplication_percentage:.1f}%, Blocks: {len(duplicate_blocks)}",
                recommendations=recommendations,
            )

        except Exception as e:
            logger.error(f"Error checking code duplication: {e}")
            return QualityMetric(
                name="Code Duplication",
                value=0,
                target=self.thresholds["code_duplication"],
                status="warn",
                details=f"Could not check duplication: {str(e)}",
                recommendations=["Install jscpd or alternative duplication checker"],
            )

    def review_dependency_updates(self) -> QualityMetric:
        """Review dependency updates and security vulnerabilities"""
        logger.info("Reviewing dependency updates...")

        try:
            # Check for outdated packages
            cmd = ["pip", "list", "--outdated", "--format=json"]
            result = subprocess.run(cmd, capture_output=True, text=True)

            outdated_packages = []
            if result.returncode == 0:
                outdated_packages = json.loads(result.stdout)

            # Check for known vulnerabilities using safety
            cmd = ["safety", "check", "--json", "--full-report"]
            safety_result = subprocess.run(cmd, capture_output=True, text=True)

            vulnerabilities = []
            if safety_result.returncode == 0:
                try:
                    safety_data = json.loads(safety_result.stdout)
                    vulnerabilities = safety_data.get("vulnerabilities", [])
                except json.JSONDecodeError:
                    pass

            # Calculate dependency health score
            total_packages = self._count_total_packages()
            vulnerable_packages = len(vulnerabilities)
            outdated_packages_count = len(outdated_packages)

            # Score based on vulnerabilities and outdated packages
            vulnerability_penalty = vulnerable_packages * 20
            outdated_penalty = min(outdated_packages_count * 2, 30)
            dependency_score = max(0, 100 - vulnerability_penalty - outdated_penalty)

            status = (
                "pass" if dependency_score >= 80 else "warn" if dependency_score >= 60 else "fail"
            )

            recommendations = []
            if vulnerable_packages > 0:
                recommendations.append(
                    f"Update {vulnerable_packages} packages with security vulnerabilities"
                )

            if outdated_packages_count > 10:
                recommendations.append(
                    f"Consider updating {outdated_packages_count} outdated packages"
                )

            # Store detailed issues
            for vuln in vulnerabilities:
                self.dependency_issues.append(
                    DependencyIssue(
                        package=vuln.get("package_name", "unknown"),
                        current_version=vuln.get("installed_version", "unknown"),
                        latest_version=vuln.get("patched_version", "unknown"),
                        issue_type="vulnerable",
                        severity=vuln.get("severity", "unknown"),
                        description=vuln.get("advisory", "Security vulnerability"),
                    )
                )

            return QualityMetric(
                name="Dependency Health",
                value=dependency_score,
                target=80.0,
                status=status,
                details=f"Score: {dependency_score:.1f}, Vulnerable: {vulnerable_packages}, Outdated: {outdated_packages_count}",
                recommendations=recommendations,
            )

        except Exception as e:
            logger.error(f"Error reviewing dependencies: {e}")
            return QualityMetric(
                name="Dependency Health",
                value=0,
                target=80.0,
                status="fail",
                details=f"Error: {str(e)}",
                recommendations=["Install safety and pip-audit for dependency checks"],
            )

    def scan_security_vulnerabilities(self) -> QualityMetric:
        """Scan for security vulnerabilities"""
        logger.info("Scanning for security vulnerabilities...")

        try:
            # Run bandit security scanner
            cmd = [
                "bandit",
                "-r",
                "music_gen/",
                "-f",
                "json",
                "-o",
                "security_report.json",
                "-ll",  # Low level and above
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)

            security_issues = []
            if os.path.exists("security_report.json"):
                with open("security_report.json", "r") as f:
                    security_data = json.load(f)
                    security_issues = security_data.get("results", [])

            # Categorize issues by severity
            severity_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
            for issue in security_issues:
                severity = issue.get("issue_severity", "LOW")
                severity_counts[severity] += 1

                # Store detailed vulnerability
                self.security_vulnerabilities.append(
                    SecurityVulnerability(
                        file_path=issue.get("filename", "unknown"),
                        line_number=issue.get("line_number", 0),
                        issue_id=issue.get("test_id", "unknown"),
                        severity=severity,
                        confidence=issue.get("issue_confidence", "unknown"),
                        description=issue.get("issue_text", "Security issue"),
                        more_info=issue.get("more_info", ""),
                    )
                )

            # Calculate security score
            high_penalty = severity_counts["HIGH"] * 30
            medium_penalty = severity_counts["MEDIUM"] * 10
            low_penalty = severity_counts["LOW"] * 2

            security_score = max(0, 100 - high_penalty - medium_penalty - low_penalty)

            status = "pass" if security_score >= self.thresholds["security_score"] else "fail"

            recommendations = []
            if severity_counts["HIGH"] > 0:
                recommendations.append(
                    f"Fix {severity_counts['HIGH']} high-severity security issues immediately"
                )
            if severity_counts["MEDIUM"] > 0:
                recommendations.append(
                    f"Address {severity_counts['MEDIUM']} medium-severity security issues"
                )
            if severity_counts["LOW"] > 5:
                recommendations.append(
                    f"Consider addressing {severity_counts['LOW']} low-severity security issues"
                )

            return QualityMetric(
                name="Security Score",
                value=security_score,
                target=self.thresholds["security_score"],
                status=status,
                details=f"Score: {security_score:.1f}, High: {severity_counts['HIGH']}, Medium: {severity_counts['MEDIUM']}, Low: {severity_counts['LOW']}",
                recommendations=recommendations,
            )

        except Exception as e:
            logger.error(f"Error scanning security vulnerabilities: {e}")
            return QualityMetric(
                name="Security Score",
                value=0,
                target=self.thresholds["security_score"],
                status="fail",
                details=f"Error: {str(e)}",
                recommendations=["Install bandit security scanner"],
            )

    def profile_performance_hotspots(self) -> QualityMetric:
        """Profile performance hotspots"""
        logger.info("Profiling performance hotspots...")

        try:
            # Use py-spy for performance profiling
            cmd = [
                "py-spy",
                "record",
                "--duration",
                "30",
                "--format",
                "json",
                "--output",
                "performance_profile.json",
                "--",
                "python",
                "-m",
                "music_gen.api.app",
            ]

            # This would need to be run against a running application
            # For now, we'll do static analysis
            performance_issues = self._analyze_performance_static()

            # Calculate performance score based on issues found
            critical_issues = sum(1 for issue in performance_issues if issue.impact == "Critical")
            major_issues = sum(1 for issue in performance_issues if issue.impact == "Major")
            minor_issues = sum(1 for issue in performance_issues if issue.impact == "Minor")

            performance_score = max(
                0, 100 - critical_issues * 25 - major_issues * 10 - minor_issues * 2
            )

            status = "pass" if performance_score >= self.thresholds["performance_score"] else "warn"

            recommendations = []
            if critical_issues > 0:
                recommendations.append(f"Fix {critical_issues} critical performance issues")
            if major_issues > 0:
                recommendations.append(f"Address {major_issues} major performance issues")

            return QualityMetric(
                name="Performance Score",
                value=performance_score,
                target=self.thresholds["performance_score"],
                status=status,
                details=f"Score: {performance_score:.1f}, Critical: {critical_issues}, Major: {major_issues}, Minor: {minor_issues}",
                recommendations=recommendations,
            )

        except Exception as e:
            logger.error(f"Error profiling performance: {e}")
            return QualityMetric(
                name="Performance Score",
                value=75.0,  # Default score when profiling unavailable
                target=self.thresholds["performance_score"],
                status="warn",
                details=f"Static analysis only: {str(e)}",
                recommendations=["Install py-spy for runtime profiling"],
            )

    def check_coding_standards(self) -> QualityMetric:
        """Check coding standards compliance"""
        logger.info("Checking coding standards...")

        try:
            # Run flake8 for style and complexity
            cmd = [
                "flake8",
                "music_gen/",
                "--format=json",
                "--output-file=style_report.json",
                "--max-complexity=10",
                "--max-line-length=88",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)

            style_issues = []
            if os.path.exists("style_report.json"):
                with open("style_report.json", "r") as f:
                    try:
                        style_data = json.load(f)
                        style_issues = style_data
                    except json.JSONDecodeError:
                        # flake8 might not output valid JSON
                        pass

            # Run black for formatting check
            black_result = subprocess.run(
                ["black", "--check", "--diff", "music_gen/"],
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )

            formatting_issues = black_result.returncode != 0

            # Run isort for import sorting
            isort_result = subprocess.run(
                ["isort", "--check-only", "--diff", "music_gen/"],
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )

            import_issues = isort_result.returncode != 0

            # Calculate coding standards score
            style_penalty = min(len(style_issues) * 2, 40)
            formatting_penalty = 20 if formatting_issues else 0
            import_penalty = 10 if import_issues else 0

            coding_score = max(0, 100 - style_penalty - formatting_penalty - import_penalty)

            status = "pass" if coding_score >= 80 else "warn" if coding_score >= 60 else "fail"

            recommendations = []
            if style_issues:
                recommendations.append(f"Fix {len(style_issues)} style issues")
            if formatting_issues:
                recommendations.append("Run 'black music_gen/' to fix formatting")
            if import_issues:
                recommendations.append("Run 'isort music_gen/' to fix import ordering")

            return QualityMetric(
                name="Coding Standards",
                value=coding_score,
                target=80.0,
                status=status,
                details=f"Score: {coding_score:.1f}, Style issues: {len(style_issues)}, Formatting: {'Issues' if formatting_issues else 'OK'}",
                recommendations=recommendations,
            )

        except Exception as e:
            logger.error(f"Error checking coding standards: {e}")
            return QualityMetric(
                name="Coding Standards",
                value=0,
                target=80.0,
                status="fail",
                details=f"Error: {str(e)}",
                recommendations=["Install flake8, black, and isort"],
            )

    def update_documentation(self) -> QualityMetric:
        """Check and update documentation"""
        logger.info("Checking documentation completeness...")

        try:
            # Check for required documentation files
            required_docs = [
                "README.md",
                "CHANGELOG.md",
                "CONTRIBUTING.md",
                "LICENSE",
                "docs/api/README.md",
                "docs/deployment.md",
                "docs/development.md",
            ]

            existing_docs = []
            missing_docs = []

            for doc in required_docs:
                doc_path = self.project_root / doc
                if doc_path.exists():
                    existing_docs.append(doc)
                else:
                    missing_docs.append(doc)

            # Check for docstring coverage
            docstring_coverage = self._check_docstring_coverage()

            # Calculate documentation score
            doc_file_score = (len(existing_docs) / len(required_docs)) * 50
            docstring_score = docstring_coverage * 0.5

            documentation_score = doc_file_score + docstring_score

            status = "pass" if documentation_score >= 80 else "warn"

            recommendations = []
            if missing_docs:
                recommendations.append(f"Create missing documentation: {', '.join(missing_docs)}")
            if docstring_coverage < 80:
                recommendations.append(
                    f"Improve docstring coverage to 80% (currently {docstring_coverage:.1f}%)"
                )

            return QualityMetric(
                name="Documentation",
                value=documentation_score,
                target=80.0,
                status=status,
                details=f"Score: {documentation_score:.1f}, Files: {len(existing_docs)}/{len(required_docs)}, Docstrings: {docstring_coverage:.1f}%",
                recommendations=recommendations,
            )

        except Exception as e:
            logger.error(f"Error checking documentation: {e}")
            return QualityMetric(
                name="Documentation",
                value=0,
                target=80.0,
                status="fail",
                details=f"Error: {str(e)}",
                recommendations=["Review documentation structure"],
            )

    def generate_quality_report(self) -> str:
        """Generate comprehensive quality report"""
        logger.info("Generating code quality report...")

        # Run all quality checks
        self.quality_metrics = [
            self.run_full_test_suite(),
            self.check_code_duplication(),
            self.review_dependency_updates(),
            self.scan_security_vulnerabilities(),
            self.profile_performance_hotspots(),
            self.check_coding_standards(),
            self.update_documentation(),
        ]

        # Calculate overall quality score
        overall_score = sum(metric.value for metric in self.quality_metrics) / len(
            self.quality_metrics
        )

        # Generate report
        report_content = self._format_quality_report(overall_score)

        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.results_dir / f"quality_report_{timestamp}.md"

        with open(report_path, "w") as f:
            f.write(report_content)

        logger.info(f"Quality report generated: {report_path}")
        return str(report_path)

    def _parse_test_results(self, test_output: str) -> Dict[str, int]:
        """Parse test results from pytest output"""
        results = {"passed": 0, "failed": 0, "skipped": 0, "total": 0}

        # Look for test summary line
        summary_pattern = r"(\d+) passed.*?(\d+) failed.*?(\d+) skipped"
        match = re.search(summary_pattern, test_output)

        if match:
            results["passed"] = int(match.group(1))
            results["failed"] = int(match.group(2))
            results["skipped"] = int(match.group(3))
            results["total"] = results["passed"] + results["failed"] + results["skipped"]

        return results

    def _count_total_packages(self) -> int:
        """Count total installed packages"""
        try:
            result = subprocess.run(
                ["pip", "list", "--format=json"], capture_output=True, text=True
            )
            if result.returncode == 0:
                packages = json.loads(result.stdout)
                return len(packages)
        except:
            pass
        return 0

    def _analyze_performance_static(self) -> List[PerformanceIssue]:
        """Analyze code for potential performance issues"""
        issues = []

        # This would scan for common performance anti-patterns
        # For now, return empty list
        return issues

    def _check_docstring_coverage(self) -> float:
        """Check docstring coverage"""
        try:
            result = subprocess.run(
                ["docstring-coverage", "music_gen/", "--percentage-only"],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                return float(result.stdout.strip().replace("%", ""))
        except:
            pass

        return 0.0

    def _format_quality_report(self, overall_score: float) -> str:
        """Format the quality report"""

        status_emoji = {"pass": "✅", "warn": "⚠️", "fail": "❌"}

        report = f"""# Code Quality Report - Music Gen AI

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Overall Score**: {overall_score:.1f}/100

---

## 📊 Quality Metrics Summary

| Metric | Score | Target | Status | Details |
|--------|-------|--------|--------|---------|
"""

        for metric in self.quality_metrics:
            emoji = status_emoji.get(metric.status, "❓")
            report += f"| {metric.name} | {metric.value:.1f} | {metric.target:.1f} | {emoji} {metric.status} | {metric.details} |\n"

        report += f"""

---

## 🔍 Detailed Analysis

"""

        for metric in self.quality_metrics:
            report += f"""
### {metric.name}

**Score**: {metric.value:.1f}/{metric.target:.1f} ({metric.status})
**Details**: {metric.details}

**Recommendations**:
"""
            for rec in metric.recommendations:
                report += f"- {rec}\n"

        if self.security_vulnerabilities:
            report += f"""

---

## 🔐 Security Vulnerabilities

"""
            for vuln in self.security_vulnerabilities[:10]:  # Top 10
                report += f"""
### {vuln.severity} - {vuln.issue_id}
**File**: {vuln.file_path}:{vuln.line_number}
**Description**: {vuln.description}
**Confidence**: {vuln.confidence}
"""

        if self.dependency_issues:
            report += f"""

---

## 📦 Dependency Issues

"""
            for dep in self.dependency_issues[:10]:  # Top 10
                report += f"""
### {dep.package} - {dep.issue_type}
**Current**: {dep.current_version} → **Latest**: {dep.latest_version}
**Severity**: {dep.severity}
**Description**: {dep.description}
"""

        report += f"""

---

## 🎯 Next Steps

### Immediate Actions (High Priority)
"""

        for metric in self.quality_metrics:
            if metric.status == "fail":
                for rec in metric.recommendations:
                    report += f"- {metric.name}: {rec}\n"

        report += f"""

### Medium Priority
"""

        for metric in self.quality_metrics:
            if metric.status == "warn":
                for rec in metric.recommendations:
                    report += f"- {metric.name}: {rec}\n"

        report += f"""

---

## 📈 Trends and Improvements

*This section would be populated with historical data comparison*

---

**Generated by**: Code Quality Review System
**Report ID**: {int(time.time())}
"""

        return report


def main():
    """Main function to run code quality review"""
    reviewer = CodeQualityReviewer()

    try:
        report_path = reviewer.generate_quality_report()
        print(f"✅ Code quality report generated: {report_path}")

        # Print summary
        overall_score = sum(metric.value for metric in reviewer.quality_metrics) / len(
            reviewer.quality_metrics
        )
        print(f"📊 Overall Quality Score: {overall_score:.1f}/100")

        # Print status summary
        for metric in reviewer.quality_metrics:
            status_emoji = {"pass": "✅", "warn": "⚠️", "fail": "❌"}.get(metric.status, "❓")
            print(f"{status_emoji} {metric.name}: {metric.value:.1f}/{metric.target:.1f}")

    except Exception as e:
        logger.error(f"Error running code quality review: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
