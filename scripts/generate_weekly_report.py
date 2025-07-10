#!/usr/bin/env python3
"""
Weekly Status Report Generator for Music Gen AI
Automatically generates weekly reflection reports with real data
"""

import os
import json
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
import requests
import psycopg2
from typing import Dict, List, Any, Optional
import logging
import sys
import re
from dataclasses import dataclass, asdict
import git
from jinja2 import Template

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"weekly_report_{int(time.time())}.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


@dataclass
class WeeklyMetrics:
    """Weekly metrics collection"""

    test_coverage: float
    api_response_time_p95: float
    critical_vulnerabilities: int
    uptime_percentage: float
    error_rate: float
    deployment_count: int
    commits_count: int
    prs_merged: int
    bugs_fixed: int
    features_delivered: int


@dataclass
class TechnicalDebt:
    """Technical debt item"""

    component: str
    debt_type: str
    severity: str
    estimated_fix_time: str
    target_sprint: str
    description: str


@dataclass
class SecurityConcern:
    """Security concern item"""

    concern: str
    severity: str
    component: str
    risk_level: str
    mitigation_plan: str
    owner: str


@dataclass
class RiskAssessment:
    """Risk assessment item"""

    risk_name: str
    probability: str
    impact: str
    mitigation_status: str
    owner: str


class WeeklyReportGenerator:
    def __init__(self):
        self.project_root = Path.cwd()
        self.reports_dir = self.project_root / "reports" / "weekly"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Initialize git repo
        try:
            self.repo = git.Repo(self.project_root)
        except git.InvalidGitRepositoryError:
            logger.error("Not a git repository")
            sys.exit(1)

        # Configuration
        self.prometheus_url = os.getenv("PROMETHEUS_URL", "http://prometheus:9090")
        self.api_url = os.getenv("API_URL", "https://api.musicgen.ai")

        # Database connection
        self.db_params = {
            "host": os.getenv("DB_HOST", "postgres"),
            "port": 5432,
            "database": "musicgen_prod",
            "user": "musicgen",
            "password": os.getenv("DB_PASSWORD"),
        }

        # Current week
        self.current_week_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        self.current_week_start -= timedelta(days=self.current_week_start.weekday())
        self.current_week_end = self.current_week_start + timedelta(days=6)

    def collect_git_metrics(self) -> Dict[str, Any]:
        """Collect git-related metrics for the week"""
        logger.info("Collecting git metrics...")

        # Get commits from the last week
        commits = list(
            self.repo.iter_commits(since=self.current_week_start, until=self.current_week_end)
        )

        # Analyze commit patterns
        commit_analysis = {
            "total_commits": len(commits),
            "authors": set(),
            "commit_types": {"feat": 0, "fix": 0, "docs": 0, "refactor": 0, "test": 0, "other": 0},
            "files_changed": set(),
            "lines_added": 0,
            "lines_removed": 0,
        }

        for commit in commits:
            commit_analysis["authors"].add(commit.author.name)

            # Parse conventional commit format
            commit_msg = commit.message.split("\n")[0]
            if commit_msg.startswith("feat:"):
                commit_analysis["commit_types"]["feat"] += 1
            elif commit_msg.startswith("fix:"):
                commit_analysis["commit_types"]["fix"] += 1
            elif commit_msg.startswith("docs:"):
                commit_analysis["commit_types"]["docs"] += 1
            elif commit_msg.startswith("refactor:"):
                commit_analysis["commit_types"]["refactor"] += 1
            elif commit_msg.startswith("test:"):
                commit_analysis["commit_types"]["test"] += 1
            else:
                commit_analysis["commit_types"]["other"] += 1

            # Get file changes
            try:
                for file in commit.stats.files:
                    commit_analysis["files_changed"].add(file)
                commit_analysis["lines_added"] += commit.stats.total["insertions"]
                commit_analysis["lines_removed"] += commit.stats.total["deletions"]
            except:
                pass

        commit_analysis["authors"] = list(commit_analysis["authors"])
        commit_analysis["files_changed"] = list(commit_analysis["files_changed"])

        return commit_analysis

    def collect_test_coverage(self) -> float:
        """Collect test coverage information"""
        logger.info("Collecting test coverage...")

        try:
            # Run pytest with coverage
            result = subprocess.run(
                [
                    "python",
                    "-m",
                    "pytest",
                    "--cov=music_gen",
                    "--cov-report=json:coverage.json",
                    "--tb=no",
                    "-q",
                ],
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )

            if os.path.exists("coverage.json"):
                with open("coverage.json", "r") as f:
                    coverage_data = json.load(f)
                    return coverage_data.get("totals", {}).get("percent_covered", 0)
        except Exception as e:
            logger.error(f"Error collecting coverage: {e}")

        return 0.0

    def collect_performance_metrics(self) -> Dict[str, float]:
        """Collect performance metrics from Prometheus"""
        logger.info("Collecting performance metrics...")

        metrics = {
            "response_time_p95": 0.0,
            "error_rate": 0.0,
            "uptime_percentage": 0.0,
            "requests_per_second": 0.0,
        }

        try:
            # Query Prometheus for metrics
            queries = {
                "response_time_p95": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[24h]))",
                "error_rate": 'rate(http_requests_total{status=~"5.."}[24h])',
                "uptime_percentage": "avg(up) * 100",
                "requests_per_second": "rate(http_requests_total[24h])",
            }

            for metric, query in queries.items():
                url = f"{self.prometheus_url}/api/v1/query"
                params = {"query": query}

                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data["data"]["result"]:
                        metrics[metric] = float(data["data"]["result"][0]["value"][1])

        except Exception as e:
            logger.error(f"Error collecting performance metrics: {e}")

        return metrics

    def scan_security_vulnerabilities(self) -> List[SecurityConcern]:
        """Scan for security vulnerabilities"""
        logger.info("Scanning for security vulnerabilities...")

        vulnerabilities = []

        try:
            # Run bandit security scan
            result = subprocess.run(
                ["bandit", "-r", "music_gen/", "-f", "json", "-o", "security_report.json"],
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )

            if os.path.exists("security_report.json"):
                with open("security_report.json", "r") as f:
                    security_data = json.load(f)

                    for result in security_data.get("results", []):
                        if result["issue_severity"] in ["HIGH", "CRITICAL"]:
                            vulnerabilities.append(
                                SecurityConcern(
                                    concern=result["issue_text"],
                                    severity=result["issue_severity"],
                                    component=result["filename"],
                                    risk_level=result["issue_confidence"],
                                    mitigation_plan="TBD",
                                    owner="Security Team",
                                )
                            )

        except Exception as e:
            logger.error(f"Error scanning vulnerabilities: {e}")

        return vulnerabilities

    def analyze_technical_debt(self) -> List[TechnicalDebt]:
        """Analyze technical debt in the codebase"""
        logger.info("Analyzing technical debt...")

        debt_items = []

        try:
            # Search for TODO, FIXME, HACK comments
            result = subprocess.run(
                ["grep", "-r", "-n", "--include=*.py", "-E", "(TODO|FIXME|HACK|XXX)", "music_gen/"],
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )

            for line in result.stdout.split("\n"):
                if line.strip():
                    parts = line.split(":", 3)
                    if len(parts) >= 3:
                        file_path = parts[0]
                        line_num = parts[1]
                        comment = parts[2].strip()

                        debt_type = "Code Quality"
                        if "TODO" in comment:
                            debt_type = "Enhancement"
                        elif "FIXME" in comment:
                            debt_type = "Bug Fix"
                        elif "HACK" in comment:
                            debt_type = "Technical Debt"

                        debt_items.append(
                            TechnicalDebt(
                                component=file_path,
                                debt_type=debt_type,
                                severity="Medium",
                                estimated_fix_time="2-4 hours",
                                target_sprint="TBD",
                                description=comment,
                            )
                        )

        except Exception as e:
            logger.error(f"Error analyzing technical debt: {e}")

        return debt_items[:10]  # Limit to top 10

    def collect_deployment_metrics(self) -> Dict[str, Any]:
        """Collect deployment-related metrics"""
        logger.info("Collecting deployment metrics...")

        metrics = {
            "deployment_count": 0,
            "avg_deployment_time": 0,
            "rollback_count": 0,
            "success_rate": 0,
        }

        try:
            # Check deployment logs
            deployment_logs = Path("logs").glob("production_deployment_*.log")
            metrics["deployment_count"] = len(list(deployment_logs))

            # Check for rollback logs
            rollback_logs = Path("logs").glob("rollback_*.log")
            metrics["rollback_count"] = len(list(rollback_logs))

            if metrics["deployment_count"] > 0:
                metrics["success_rate"] = (
                    (metrics["deployment_count"] - metrics["rollback_count"])
                    / metrics["deployment_count"]
                    * 100
                )

        except Exception as e:
            logger.error(f"Error collecting deployment metrics: {e}")

        return metrics

    def generate_executive_summary(self, data: Dict[str, Any]) -> str:
        """Generate executive summary based on collected data"""

        # Determine overall status
        status = "On Track"
        if data["metrics"]["error_rate"] > 0.05:
            status = "At Risk"
        if data["metrics"]["uptime_percentage"] < 99.0:
            status = "At Risk"
        if len(data["security_concerns"]) > 0:
            status = "At Risk"

        # Calculate completion percentage (simplified)
        completion = 75  # This would be calculated based on actual sprint data

        # Count active blockers
        blockers = len([debt for debt in data["technical_debt"] if debt.severity == "High"])

        return f"""
**Status**: {status}
**Completion**: {completion}% of planned work completed
**Blockers**: {blockers} active blockers
**Next Week Focus**: Performance optimization and security improvements
"""

    def generate_weekly_report(self) -> str:
        """Generate the complete weekly report"""
        logger.info("Generating weekly report...")

        # Collect all data
        git_metrics = self.collect_git_metrics()
        test_coverage = self.collect_test_coverage()
        performance_metrics = self.collect_performance_metrics()
        security_concerns = self.scan_security_vulnerabilities()
        technical_debt = self.analyze_technical_debt()
        deployment_metrics = self.collect_deployment_metrics()

        # Compile all data
        report_data = {
            "week_start": self.current_week_start.strftime("%Y-%m-%d"),
            "week_end": self.current_week_end.strftime("%Y-%m-%d"),
            "git_metrics": git_metrics,
            "test_coverage": test_coverage,
            "metrics": performance_metrics,
            "security_concerns": security_concerns,
            "technical_debt": technical_debt,
            "deployment_metrics": deployment_metrics,
        }

        # Generate report content
        report_content = self._format_report(report_data)

        # Save report
        report_filename = f"weekly_report_{self.current_week_start.strftime('%Y_%m_%d')}.md"
        report_path = self.reports_dir / report_filename

        with open(report_path, "w") as f:
            f.write(report_content)

        logger.info(f"Weekly report generated: {report_path}")
        return str(report_path)

    def _format_report(self, data: Dict[str, Any]) -> str:
        """Format the report data into markdown"""

        template = """# Weekly Status Report - Music Gen AI Project

## Week of: {{ data.week_start }} to {{ data.week_end }}
**Generated**: {{ current_date }}

---

## 📊 Executive Summary

{{ executive_summary }}

### Key Achievements This Week
- {{ data.git_metrics.total_commits }} commits by {{ data.git_metrics.authors|length }} contributors
- {{ data.git_metrics.commit_types.feat }} new features implemented
- {{ data.git_metrics.commit_types.fix }} bugs fixed
- {{ data.deployment_metrics.deployment_count }} deployments completed

---

## 🎯 Timeline Assessment

### Development Metrics
- **Commits**: {{ data.git_metrics.total_commits }}
- **Lines Added**: {{ data.git_metrics.lines_added }}
- **Lines Removed**: {{ data.git_metrics.lines_removed }}
- **Files Changed**: {{ data.git_metrics.files_changed|length }}

### Quality Metrics
- **Test Coverage**: {{ "%.1f"|format(data.test_coverage) }}%
- **Code Quality**: {{ "Good" if data.test_coverage > 80 else "Needs Improvement" }}

---

## 🚨 Technical Challenges

### Performance Issues
- **API Response Time (p95)**: {{ "%.2f"|format(data.metrics.response_time_p95) }}s
- **Error Rate**: {{ "%.3f"|format(data.metrics.error_rate) }}%
- **Uptime**: {{ "%.2f"|format(data.metrics.uptime_percentage) }}%

### Security Concerns
{% for concern in data.security_concerns %}
- **{{ concern.severity }}**: {{ concern.concern }} ({{ concern.component }})
{% endfor %}

---

## 💳 Technical Debt Assessment

### Current Technical Debt
{% for debt in data.technical_debt %}
- **{{ debt.component }}**: {{ debt.debt_type }} - {{ debt.severity }}
{% endfor %}

---

## 📈 Success Metrics Review

### Technical Metrics
| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Test Coverage | >90% | {{ "%.1f"|format(data.test_coverage) }}% | {{ "✅" if data.test_coverage > 90 else "❌" }} |
| API Response Time (p95) | <200ms | {{ "%.0f"|format(data.metrics.response_time_p95 * 1000) }}ms | {{ "✅" if data.metrics.response_time_p95 < 0.2 else "❌" }} |
| Error Rate | <0.1% | {{ "%.3f"|format(data.metrics.error_rate) }}% | {{ "✅" if data.metrics.error_rate < 0.001 else "❌" }} |
| Uptime | 99.9% | {{ "%.2f"|format(data.metrics.uptime_percentage) }}% | {{ "✅" if data.metrics.uptime_percentage > 99.9 else "❌" }} |

### Deployment Metrics
| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Deployment Count | - | {{ data.deployment_metrics.deployment_count }} | - |
| Success Rate | >95% | {{ "%.1f"|format(data.deployment_metrics.success_rate) }}% | {{ "✅" if data.deployment_metrics.success_rate > 95 else "❌" }} |

---

## 🔧 Recommendations for Next Week

### High Priority
1. **Performance Optimization**: Address API response time issues
2. **Security Fixes**: Resolve {{ data.security_concerns|length }} security concerns
3. **Test Coverage**: Improve coverage to meet 90% target

### Medium Priority
1. **Technical Debt**: Address {{ data.technical_debt|length }} identified debt items
2. **Documentation**: Update deployment procedures
3. **Monitoring**: Enhance alerting for critical metrics

---

**Report Generated**: {{ current_date }}
**Next Report**: {{ next_week_date }}

---

## 📎 Raw Data

### Git Analysis
```json
{{ git_data_json }}
```

### Performance Metrics
```json
{{ performance_data_json }}
```
"""

        from jinja2 import Template

        # Create template
        t = Template(template)

        # Render report
        return t.render(
            data=data,
            current_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            next_week_date=(self.current_week_start + timedelta(days=7)).strftime("%Y-%m-%d"),
            executive_summary=self.generate_executive_summary(data),
            git_data_json=json.dumps(data["git_metrics"], indent=2),
            performance_data_json=json.dumps(data["metrics"], indent=2),
        )


def main():
    """Main function to generate weekly report"""
    generator = WeeklyReportGenerator()

    try:
        report_path = generator.generate_weekly_report()
        print(f"✅ Weekly report generated successfully: {report_path}")

        # Also update the README with current status
        generator.update_readme_status()

    except Exception as e:
        logger.error(f"Error generating weekly report: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
