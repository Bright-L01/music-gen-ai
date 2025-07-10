#!/usr/bin/env python3
"""
Git Best Practices Manager for Music Gen AI
Implements git workflow automation, commit standards, and repository management
"""

import os
import json
import subprocess
import sys
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
import git
from semantic_version import Version
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("git_best_practices.log"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


@dataclass
class CommitAnalysis:
    """Commit analysis result"""

    commit_hash: str
    message: str
    author: str
    date: datetime
    follows_convention: bool
    commit_type: str
    scope: Optional[str]
    breaking_change: bool
    files_changed: int
    lines_added: int
    lines_removed: int


@dataclass
class BranchHealth:
    """Branch health assessment"""

    branch_name: str
    follows_naming_convention: bool
    has_pr: bool
    days_since_last_commit: int
    commits_ahead: int
    commits_behind: int
    is_stale: bool
    should_delete: bool


@dataclass
class ReleaseInfo:
    """Release information"""

    version: str
    tag: str
    date: datetime
    commits_since_last: int
    breaking_changes: int
    features: int
    fixes: int
    changelog_entry: str


class GitBestPracticesManager:
    def __init__(self):
        self.project_root = Path.cwd()

        # Initialize git repository
        try:
            self.repo = git.Repo(self.project_root)
        except git.InvalidGitRepositoryError:
            logger.error("Not a git repository")
            sys.exit(1)

        # Configuration
        self.config = {
            "branch_naming_patterns": [
                r"^feature/[a-z0-9-]+$",
                r"^fix/[a-z0-9-]+$",
                r"^hotfix/[a-z0-9-]+$",
                r"^refactor/[a-z0-9-]+$",
                r"^docs/[a-z0-9-]+$",
                r"^test/[a-z0-9-]+$",
            ],
            "commit_types": [
                "feat",
                "fix",
                "docs",
                "style",
                "refactor",
                "test",
                "chore",
                "perf",
                "ci",
                "build",
                "revert",
            ],
            "max_commit_message_length": 72,
            "max_days_stale_branch": 30,
            "require_pr_for_main": True,
            "require_review_approval": True,
            "protected_branches": ["main", "develop", "production"],
        }

        # GitHub API configuration
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.github_repo = self._get_github_repo()

    def setup_git_hooks(self):
        """Setup git hooks for enforcing standards"""
        logger.info("Setting up git hooks...")

        hooks_dir = self.project_root / ".git" / "hooks"
        hooks_dir.mkdir(exist_ok=True)

        # Pre-commit hook
        pre_commit_hook = hooks_dir / "pre-commit"
        pre_commit_content = """#!/bin/bash
# Pre-commit hook for code quality checks

echo "Running pre-commit checks..."

# Run code quality checks
python scripts/code_quality_review.py --quick-check

if [ $? -ne 0 ]; then
    echo "❌ Pre-commit checks failed. Please fix issues before committing."
    exit 1
fi

# Check for secrets
if command -v gitleaks &> /dev/null; then
    gitleaks detect --source . --verbose
    if [ $? -ne 0 ]; then
        echo "❌ Secrets detected. Please remove before committing."
        exit 1
    fi
fi

echo "✅ Pre-commit checks passed"
"""

        with open(pre_commit_hook, "w") as f:
            f.write(pre_commit_content)
        pre_commit_hook.chmod(0o755)

        # Commit-msg hook
        commit_msg_hook = hooks_dir / "commit-msg"
        commit_msg_content = """#!/bin/bash
# Commit message hook for enforcing conventional commits

commit_regex='^(feat|fix|docs|style|refactor|test|chore|perf|ci|build|revert)(\(.+\))?!?:.{1,50}'

if ! grep -qE "$commit_regex" "$1"; then
    echo "❌ Invalid commit message format!"
    echo "Please use conventional commit format: <type>(<scope>): <description>"
    echo "Types: feat, fix, docs, style, refactor, test, chore, perf, ci, build, revert"
    echo "Example: feat(api): add user authentication"
    exit 1
fi

echo "✅ Commit message format is valid"
"""

        with open(commit_msg_hook, "w") as f:
            f.write(commit_msg_content)
        commit_msg_hook.chmod(0o755)

        # Pre-push hook
        pre_push_hook = hooks_dir / "pre-push"
        pre_push_content = """#!/bin/bash
# Pre-push hook for final checks

echo "Running pre-push checks..."

# Check if pushing to protected branch
protected_branch=$(git rev-parse --abbrev-ref HEAD)
if [[ "$protected_branch" == "main" || "$protected_branch" == "develop" ]]; then
    echo "❌ Direct push to protected branch '$protected_branch' is not allowed"
    echo "Please create a pull request instead"
    exit 1
fi

# Run tests
python -m pytest tests/ -x --tb=short

if [ $? -ne 0 ]; then
    echo "❌ Tests failed. Please fix before pushing."
    exit 1
fi

echo "✅ Pre-push checks passed"
"""

        with open(pre_push_hook, "w") as f:
            f.write(pre_push_content)
        pre_push_hook.chmod(0o755)

        logger.info("✅ Git hooks setup complete")

    def create_commit_template(self):
        """Create commit message template"""
        logger.info("Creating commit message template...")

        template_content = """# <type>(<scope>): <subject>
#
# <body>
#
# <footer>

# Type should be one of the following:
# * feat: A new feature
# * fix: A bug fix
# * docs: Documentation only changes
# * style: Changes that do not affect the meaning of the code
# * refactor: A code change that neither fixes a bug nor adds a feature
# * test: Adding missing tests or correcting existing tests
# * chore: Changes to the build process or auxiliary tools
# * perf: A code change that improves performance
# * ci: Changes to our CI configuration files and scripts
# * build: Changes that affect the build system or external dependencies
# * revert: Reverts a previous commit
#
# Scope is optional and should be the name of the package affected
# Subject should be lowercase and not end with a period
# Body should include motivation for the change and contrast with previous behavior
# Footer should contain any information about Breaking Changes and reference issues
#
# Examples:
# feat(auth): add OAuth2 authentication
# fix(api): resolve timeout issue in music generation
# docs(readme): update installation instructions
# refactor(core): simplify config loading logic
# test(integration): add tests for user workflow
"""

        template_path = self.project_root / ".gitmessage"
        with open(template_path, "w") as f:
            f.write(template_content)

        # Configure git to use the template
        subprocess.run(["git", "config", "commit.template", ".gitmessage"], cwd=self.project_root)

        logger.info("✅ Commit template created")

    def analyze_commit_history(self, since_days: int = 30) -> List[CommitAnalysis]:
        """Analyze commit history for conventional commit compliance"""
        logger.info(f"Analyzing commit history for last {since_days} days...")

        since_date = datetime.now() - timedelta(days=since_days)
        commits = list(self.repo.iter_commits(since=since_date))

        analyses = []
        conventional_commit_pattern = (
            r"^(feat|fix|docs|style|refactor|test|chore|perf|ci|build|revert)(\(.+\))?!?:.+"
        )

        for commit in commits:
            message = commit.message.strip()
            subject = message.split("\n")[0]

            # Check if follows conventional commit
            follows_convention = bool(re.match(conventional_commit_pattern, subject))

            # Extract type and scope
            commit_type = "unknown"
            scope = None
            breaking_change = False

            if follows_convention:
                match = re.match(r"^(\w+)(\((.+)\))?(!)?: (.+)", subject)
                if match:
                    commit_type = match.group(1)
                    scope = match.group(3)
                    breaking_change = match.group(4) == "!" or "BREAKING CHANGE" in message

            # Get file changes
            files_changed = len(commit.stats.files)
            lines_added = commit.stats.total["insertions"]
            lines_removed = commit.stats.total["deletions"]

            analyses.append(
                CommitAnalysis(
                    commit_hash=commit.hexsha,
                    message=message,
                    author=commit.author.name,
                    date=commit.committed_datetime,
                    follows_convention=follows_convention,
                    commit_type=commit_type,
                    scope=scope,
                    breaking_change=breaking_change,
                    files_changed=files_changed,
                    lines_added=lines_added,
                    lines_removed=lines_removed,
                )
            )

        return analyses

    def audit_branch_health(self) -> List[BranchHealth]:
        """Audit branch health and identify stale branches"""
        logger.info("Auditing branch health...")

        branch_health = []

        # Get all branches
        branches = [
            ref.name.replace("refs/heads/", "")
            for ref in self.repo.references
            if ref.name.startswith("refs/heads/")
        ]

        for branch_name in branches:
            if branch_name in self.config["protected_branches"]:
                continue

            branch = self.repo.heads[branch_name]

            # Check naming convention
            follows_naming = any(
                re.match(pattern, branch_name) for pattern in self.config["branch_naming_patterns"]
            )

            # Get last commit date
            last_commit = branch.commit
            days_since_last = (
                datetime.now() - last_commit.committed_datetime.replace(tzinfo=None)
            ).days

            # Check if branch has PR
            has_pr = self._check_branch_has_pr(branch_name)

            # Check if branch is ahead/behind main
            main_branch = self.repo.heads["main"]
            commits_ahead = len(list(self.repo.iter_commits(f"main..{branch_name}")))
            commits_behind = len(list(self.repo.iter_commits(f"{branch_name}..main")))

            # Determine if stale
            is_stale = days_since_last > self.config["max_days_stale_branch"]
            should_delete = is_stale and not has_pr and commits_ahead == 0

            branch_health.append(
                BranchHealth(
                    branch_name=branch_name,
                    follows_naming_convention=follows_naming,
                    has_pr=has_pr,
                    days_since_last_commit=days_since_last,
                    commits_ahead=commits_ahead,
                    commits_behind=commits_behind,
                    is_stale=is_stale,
                    should_delete=should_delete,
                )
            )

        return branch_health

    def generate_semantic_version(self) -> str:
        """Generate next semantic version based on commits"""
        logger.info("Generating semantic version...")

        # Get current version from tags
        current_version = self._get_current_version()

        # Analyze commits since last release
        if current_version:
            commits_since_last = list(self.repo.iter_commits(f"{current_version}..HEAD"))
        else:
            commits_since_last = list(self.repo.iter_commits())

        # Analyze commit types
        has_breaking_changes = False
        has_features = False
        has_fixes = False

        for commit in commits_since_last:
            message = commit.message.strip()

            if (
                "BREAKING CHANGE" in message
                or message.startswith("feat!")
                or message.startswith("fix!")
            ):
                has_breaking_changes = True
            elif message.startswith("feat"):
                has_features = True
            elif message.startswith("fix"):
                has_fixes = True

        # Determine version bump
        if not current_version:
            next_version = "1.0.0"
        else:
            version = Version(current_version)

            if has_breaking_changes:
                next_version = str(version.next_major())
            elif has_features:
                next_version = str(version.next_minor())
            elif has_fixes:
                next_version = str(version.next_patch())
            else:
                next_version = str(version.next_patch())  # Default to patch

        return next_version

    def update_changelog(self, version: str) -> str:
        """Update CHANGELOG.md with new version"""
        logger.info(f"Updating CHANGELOG for version {version}...")

        changelog_path = self.project_root / "CHANGELOG.md"

        # Get commits since last release
        last_version = self._get_current_version()
        if last_version:
            commits_since_last = list(self.repo.iter_commits(f"{last_version}..HEAD"))
        else:
            commits_since_last = list(self.repo.iter_commits())

        # Categorize commits
        features = []
        fixes = []
        breaking_changes = []
        other = []

        for commit in commits_since_last:
            message = commit.message.strip()
            subject = message.split("\n")[0]

            if "BREAKING CHANGE" in message:
                breaking_changes.append(subject)
            elif subject.startswith("feat"):
                features.append(subject)
            elif subject.startswith("fix"):
                fixes.append(subject)
            else:
                other.append(subject)

        # Generate changelog entry
        changelog_entry = f"""
## [{version}] - {datetime.now().strftime('%Y-%m-%d')}

"""

        if breaking_changes:
            changelog_entry += "### 💥 BREAKING CHANGES\n"
            for change in breaking_changes:
                changelog_entry += f"- {change}\n"
            changelog_entry += "\n"

        if features:
            changelog_entry += "### ✨ Features\n"
            for feature in features:
                changelog_entry += f"- {feature}\n"
            changelog_entry += "\n"

        if fixes:
            changelog_entry += "### 🐛 Bug Fixes\n"
            for fix in fixes:
                changelog_entry += f"- {fix}\n"
            changelog_entry += "\n"

        if other:
            changelog_entry += "### 🔧 Other Changes\n"
            for change in other:
                changelog_entry += f"- {change}\n"
            changelog_entry += "\n"

        # Update CHANGELOG.md
        if changelog_path.exists():
            with open(changelog_path, "r") as f:
                existing_content = f.read()

            # Insert new entry after the header
            lines = existing_content.split("\n")
            header_end = 0
            for i, line in enumerate(lines):
                if line.startswith("## "):
                    header_end = i
                    break

            new_lines = lines[:header_end] + changelog_entry.split("\n") + lines[header_end:]
            new_content = "\n".join(new_lines)
        else:
            new_content = f"""# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

{changelog_entry}
"""

        with open(changelog_path, "w") as f:
            f.write(new_content)

        logger.info("✅ CHANGELOG updated")
        return changelog_entry

    def create_release(self, version: str, changelog_entry: str):
        """Create a new release with tag"""
        logger.info(f"Creating release {version}...")

        # Create and push tag
        tag_name = f"v{version}"
        self.repo.create_tag(tag_name, message=f"Release {version}")

        # Push tag
        subprocess.run(["git", "push", "origin", tag_name], cwd=self.project_root)

        # Create GitHub release if possible
        if self.github_token and self.github_repo:
            self._create_github_release(version, changelog_entry)

        logger.info(f"✅ Release {version} created")

    def cleanup_merged_branches(self):
        """Clean up merged branches"""
        logger.info("Cleaning up merged branches...")

        # Get merged branches
        result = subprocess.run(
            ["git", "branch", "--merged", "main"],
            capture_output=True,
            text=True,
            cwd=self.project_root,
        )

        merged_branches = [
            branch.strip().replace("* ", "")
            for branch in result.stdout.split("\n")
            if branch.strip()
            and not branch.strip().startswith("*")
            and branch.strip() not in self.config["protected_branches"]
        ]

        deleted_branches = []
        for branch in merged_branches:
            try:
                subprocess.run(["git", "branch", "-d", branch], cwd=self.project_root, check=True)
                deleted_branches.append(branch)
                logger.info(f"Deleted merged branch: {branch}")
            except subprocess.CalledProcessError:
                logger.warning(f"Could not delete branch: {branch}")

        # Clean up remote tracking branches
        subprocess.run(["git", "remote", "prune", "origin"], cwd=self.project_root)

        logger.info(f"✅ Cleaned up {len(deleted_branches)} merged branches")
        return deleted_branches

    def enforce_main_branch_protection(self):
        """Enforce main branch protection rules"""
        logger.info("Enforcing main branch protection...")

        if not self.github_token or not self.github_repo:
            logger.warning("GitHub token not configured, skipping branch protection")
            return

        # Configure branch protection via GitHub API
        protection_rules = {
            "required_status_checks": {"strict": True, "contexts": ["continuous-integration"]},
            "enforce_admins": True,
            "required_pull_request_reviews": {
                "required_approving_review_count": 1,
                "dismiss_stale_reviews": True,
                "require_code_owner_reviews": False,
            },
            "restrictions": None,
        }

        url = f"https://api.github.com/repos/{self.github_repo}/branches/main/protection"
        headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json",
        }

        try:
            response = requests.put(url, json=protection_rules, headers=headers)
            if response.status_code == 200:
                logger.info("✅ Main branch protection enabled")
            else:
                logger.error(f"Failed to enable branch protection: {response.status_code}")
        except Exception as e:
            logger.error(f"Error enabling branch protection: {e}")

    def generate_git_report(self) -> str:
        """Generate comprehensive git practices report"""
        logger.info("Generating git practices report...")

        # Analyze commit history
        commit_analyses = self.analyze_commit_history()

        # Audit branch health
        branch_health = self.audit_branch_health()

        # Generate report
        report_content = self._format_git_report(commit_analyses, branch_health)

        # Save report
        report_path = (
            self.project_root
            / "reports"
            / f"git_practices_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, "w") as f:
            f.write(report_content)

        logger.info(f"Git practices report generated: {report_path}")
        return str(report_path)

    def _get_current_version(self) -> Optional[str]:
        """Get current version from git tags"""
        try:
            # Get latest tag
            result = subprocess.run(
                ["git", "describe", "--tags", "--abbrev=0"],
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )

            if result.returncode == 0:
                return result.stdout.strip().replace("v", "")
        except:
            pass

        return None

    def _get_github_repo(self) -> Optional[str]:
        """Get GitHub repository name from git remote"""
        try:
            result = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )

            if result.returncode == 0:
                url = result.stdout.strip()
                # Extract owner/repo from GitHub URL
                if "github.com" in url:
                    match = re.search(r"github\.com[:/]([^/]+/[^/]+)", url)
                    if match:
                        return match.group(1).replace(".git", "")
        except:
            pass

        return None

    def _check_branch_has_pr(self, branch_name: str) -> bool:
        """Check if branch has an open PR"""
        if not self.github_token or not self.github_repo:
            return False

        url = f"https://api.github.com/repos/{self.github_repo}/pulls"
        headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json",
        }

        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                prs = response.json()
                return any(pr["head"]["ref"] == branch_name for pr in prs)
        except:
            pass

        return False

    def _create_github_release(self, version: str, changelog_entry: str):
        """Create GitHub release"""
        url = f"https://api.github.com/repos/{self.github_repo}/releases"
        headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json",
        }

        release_data = {
            "tag_name": f"v{version}",
            "name": f"Release {version}",
            "body": changelog_entry,
            "draft": False,
            "prerelease": False,
        }

        try:
            response = requests.post(url, json=release_data, headers=headers)
            if response.status_code == 201:
                logger.info(f"✅ GitHub release created for v{version}")
            else:
                logger.error(f"Failed to create GitHub release: {response.status_code}")
        except Exception as e:
            logger.error(f"Error creating GitHub release: {e}")

    def _format_git_report(
        self, commit_analyses: List[CommitAnalysis], branch_health: List[BranchHealth]
    ) -> str:
        """Format git practices report"""

        # Calculate statistics
        total_commits = len(commit_analyses)
        conventional_commits = sum(1 for c in commit_analyses if c.follows_convention)
        conventional_percentage = (
            (conventional_commits / total_commits * 100) if total_commits > 0 else 0
        )

        total_branches = len(branch_health)
        healthy_branches = sum(
            1 for b in branch_health if b.follows_naming_convention and not b.is_stale
        )
        stale_branches = sum(1 for b in branch_health if b.is_stale)

        report = f"""# Git Best Practices Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Summary

| Metric | Value | Status |
|--------|-------|--------|
| Conventional Commits | {conventional_commits}/{total_commits} ({conventional_percentage:.1f}%) | {'✅' if conventional_percentage >= 80 else '❌'} |
| Healthy Branches | {healthy_branches}/{total_branches} | {'✅' if healthy_branches == total_branches else '❌'} |
| Stale Branches | {stale_branches} | {'✅' if stale_branches == 0 else '⚠️'} |

---

## 📝 Commit Analysis

### Conventional Commit Compliance
- **Total Commits**: {total_commits}
- **Conventional Commits**: {conventional_commits} ({conventional_percentage:.1f}%)
- **Non-conventional Commits**: {total_commits - conventional_commits}

### Commit Types Distribution
"""

        # Count commit types
        commit_types = {}
        for commit in commit_analyses:
            commit_types[commit.commit_type] = commit_types.get(commit.commit_type, 0) + 1

        for commit_type, count in sorted(commit_types.items()):
            report += f"- **{commit_type}**: {count}\n"

        report += f"""

### Recent Non-Conventional Commits
"""

        non_conventional = [c for c in commit_analyses if not c.follows_convention][:10]
        for commit in non_conventional:
            report += f"- `{commit.commit_hash[:8]}` - {commit.message.split()[0][:50]}...\n"

        report += f"""

---

## 🌿 Branch Health

### Branch Status
- **Total Branches**: {total_branches}
- **Healthy Branches**: {healthy_branches}
- **Stale Branches**: {stale_branches}
- **Branches to Delete**: {sum(1 for b in branch_health if b.should_delete)}

### Branch Details
"""

        for branch in branch_health:
            status = "✅" if branch.follows_naming_convention and not branch.is_stale else "❌"
            report += f"- {status} **{branch.branch_name}** - {branch.days_since_last_commit} days old, {branch.commits_ahead} ahead, {branch.commits_behind} behind\n"

        report += f"""

---

## 🏷️ Version Management

### Current Version
- **Latest Tag**: {self._get_current_version() or 'None'}
- **Next Suggested Version**: {self.generate_semantic_version()}

### Release Readiness
- **Commits Since Last Release**: {len(commit_analyses)}
- **Breaking Changes**: {sum(1 for c in commit_analyses if c.breaking_change)}
- **Features**: {sum(1 for c in commit_analyses if c.commit_type == 'feat')}
- **Fixes**: {sum(1 for c in commit_analyses if c.commit_type == 'fix')}

---

## 🔧 Recommendations

### High Priority
"""

        if conventional_percentage < 80:
            report += "- Improve conventional commit compliance\n"

        if stale_branches > 0:
            report += f"- Clean up {stale_branches} stale branches\n"

        branches_to_delete = [b for b in branch_health if b.should_delete]
        if branches_to_delete:
            report += f"- Delete {len(branches_to_delete)} merged branches\n"

        report += f"""

### Medium Priority
- Set up automated branch protection
- Configure required PR reviews
- Enable automated dependency updates

### Low Priority
- Improve commit message documentation
- Set up commit message templates
- Configure automated changelog generation

---

## 📈 Historical Trends

*This section would be populated with historical data comparison*

---

**Generated by**: Git Best Practices Manager
"""

        return report


def main():
    """Main function to manage git best practices"""
    manager = GitBestPracticesManager()

    if len(sys.argv) < 2:
        print("Usage: python git_best_practices.py <command>")
        print("Commands:")
        print("  setup         - Set up git hooks and templates")
        print("  analyze       - Analyze commit history and branches")
        print("  cleanup       - Clean up merged branches")
        print("  release       - Create a new release")
        print("  report        - Generate git practices report")
        sys.exit(1)

    command = sys.argv[1]

    try:
        if command == "setup":
            manager.setup_git_hooks()
            manager.create_commit_template()
            manager.enforce_main_branch_protection()

        elif command == "analyze":
            commit_analyses = manager.analyze_commit_history()
            branch_health = manager.audit_branch_health()

            print(f"📊 Analyzed {len(commit_analyses)} commits")
            print(f"🌿 Analyzed {len(branch_health)} branches")

        elif command == "cleanup":
            deleted_branches = manager.cleanup_merged_branches()
            print(f"🧹 Cleaned up {len(deleted_branches)} branches")

        elif command == "release":
            version = manager.generate_semantic_version()
            changelog = manager.update_changelog(version)
            manager.create_release(version, changelog)
            print(f"🚀 Created release v{version}")

        elif command == "report":
            report_path = manager.generate_git_report()
            print(f"📄 Generated report: {report_path}")

        else:
            print(f"Unknown command: {command}")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Error executing command '{command}': {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
