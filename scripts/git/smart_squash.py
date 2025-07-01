#!/usr/bin/env python3
"""
Smart git squash - intelligently group and squash related commits.
"""

import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Tuple


def run_command(cmd: list, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command."""
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def get_commits() -> List[Dict[str, str]]:
    """Get all commits with their information."""
    result = run_command(["git", "log", "--pretty=format:%H|%ai|%s|%b", "--reverse"])

    commits = []
    for line in result.stdout.strip().split("\n"):
        if line:
            parts = line.split("|", 3)
            if len(parts) >= 3:
                commits.append(
                    {
                        "hash": parts[0],
                        "date": parts[1],
                        "subject": parts[2],
                        "body": parts[3] if len(parts) > 3 else "",
                    }
                )

    return commits


def categorize_commit(commit: Dict[str, str]) -> str:
    """Categorize a commit based on its message."""
    subject = commit["subject"].lower()

    # Common conventional commit patterns
    patterns = {
        "feature": r"^(feat|feature|add|implement)",
        "fix": r"^(fix|bugfix|hotfix|patch)",
        "docs": r"^(docs|documentation)",
        "style": r"^(style|format|prettier)",
        "refactor": r"^(refactor|refactoring)",
        "test": r"^(test|tests|testing)",
        "chore": r"^(chore|build|ci|perf)",
        "merge": r"^merge",
    }

    for category, pattern in patterns.items():
        if re.match(pattern, subject):
            return category

    # Additional categorization based on content
    if any(word in subject for word in ["initial", "init", "setup", "scaffold"]):
        return "initial"
    elif any(word in subject for word in ["api", "endpoint", "route"]):
        return "api"
    elif any(word in subject for word in ["model", "architecture", "network"]):
        return "model"
    elif any(word in subject for word in ["ui", "frontend", "interface"]):
        return "ui"

    return "other"


def clean_message(message: str) -> str:
    """Clean a commit message by removing emojis and excessive formatting."""
    # Remove emojis
    emoji_pattern = re.compile(
        "["
        "\U0001f600-\U0001f64f"  # emoticons
        "\U0001f300-\U0001f5ff"  # symbols & pictographs
        "\U0001f680-\U0001f6ff"  # transport & map symbols
        "\U0001f1e0-\U0001f1ff"  # flags (iOS)
        "\U00002702-\U000027b0"
        "\U000024c2-\U0001f251"
        "]+",
        flags=re.UNICODE,
    )

    message = emoji_pattern.sub("", message)

    # Remove multiple exclamation marks
    message = re.sub(r"!+", "!", message)

    # Remove excessive whitespace
    message = " ".join(message.split())

    # Ensure proper capitalization
    if message and message[0].islower():
        message = message[0].upper() + message[1:]

    return message.strip()


def group_commits(commits: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    """Group commits by category and time proximity."""
    groups = defaultdict(list)

    for commit in commits:
        category = categorize_commit(commit)
        groups[category].append(commit)

    return dict(groups)


def suggest_squash_plan(
    groups: Dict[str, List[Dict[str, str]]],
) -> List[Tuple[str, List[Dict[str, str]]]]:
    """Suggest a squash plan based on commit groups."""
    plan = []

    # Define the order we want categories to appear
    category_order = [
        "initial",
        "feature",
        "model",
        "api",
        "ui",
        "fix",
        "refactor",
        "test",
        "docs",
        "style",
        "chore",
        "other",
    ]

    for category in category_order:
        if category in groups and groups[category]:
            # For each category, we might want to further subdivide
            commits = groups[category]

            if len(commits) > 5 and category in ["feature", "fix"]:
                # Break large groups into smaller logical units
                # For now, just split by time (could be more sophisticated)
                subgroups = []
                current_group = [commits[0]]

                for i in range(1, len(commits)):
                    # You could add time-based splitting here
                    current_group.append(commits[i])

                    if len(current_group) >= 3:
                        subgroups.append((f"{category}_{len(subgroups)+1}", current_group))
                        current_group = []

                if current_group:
                    subgroups.append((f"{category}_{len(subgroups)+1}", current_group))

                plan.extend(subgroups)
            else:
                plan.append((category, commits))

    return plan


def generate_squash_message(category: str, commits: List[Dict[str, str]]) -> str:
    """Generate a clean squash commit message."""
    # Clean all commit messages
    cleaned_subjects = [clean_message(c["subject"]) for c in commits]

    # Create a summary based on category
    category_summaries = {
        "initial": "Initial project setup and scaffolding",
        "feature": "Implement core features",
        "model": "Add model architecture and components",
        "api": "Implement API endpoints",
        "ui": "Add user interface components",
        "fix": "Bug fixes and corrections",
        "refactor": "Code refactoring and improvements",
        "test": "Add tests and test infrastructure",
        "docs": "Documentation updates",
        "style": "Code style and formatting",
        "chore": "Build and maintenance tasks",
        "other": "Various improvements",
    }

    base_category = category.split("_")[0]  # Handle numbered categories
    summary = category_summaries.get(base_category, "Updates and improvements")

    # Build the commit message
    message_lines = [summary, ""]

    # Add individual commit summaries
    for commit in commits:
        cleaned_subject = clean_message(commit["subject"])
        # Remove common prefixes
        cleaned_subject = re.sub(
            r"^(feat|fix|docs|style|refactor|test|chore|build|ci|perf):\s*",
            "",
            cleaned_subject,
            flags=re.IGNORECASE,
        )
        if cleaned_subject:
            message_lines.append(f"- {cleaned_subject}")

    return "\n".join(message_lines)


def main():
    """Main function."""
    print("Smart Git History Squash")
    print("=" * 50)

    # Check if we're in a git repository
    result = run_command(["git", "rev-parse", "--git-dir"], check=False)
    if result.returncode != 0:
        print("Error: Not in a git repository")
        sys.exit(1)

    # Check for uncommitted changes
    result = run_command(["git", "status", "--porcelain"])
    if result.stdout.strip():
        print("Error: You have uncommitted changes. Please commit or stash them first.")
        sys.exit(1)

    # Get all commits
    commits = get_commits()
    print(f"Found {len(commits)} commits")

    # Group commits
    groups = group_commits(commits)

    # Generate squash plan
    plan = suggest_squash_plan(groups)

    print("\nSuggested squash plan:")
    print("-" * 50)

    for i, (category, category_commits) in enumerate(plan):
        print(f"\n{i+1}. {category.replace('_', ' ').title()} ({len(category_commits)} commits)")
        for commit in category_commits[:3]:  # Show first 3
            print(f"   - {clean_message(commit['subject'])[:60]}...")
        if len(category_commits) > 3:
            print(f"   ... and {len(category_commits) - 3} more")

    print("\n" + "-" * 50)
    print(f"This will squash {len(commits)} commits into {len(plan)} commits")

    response = input("\nProceed with squash? (y/n/m for manual): ").strip().lower()

    if response == "y":
        # Create backup branch
        current_branch = run_command(["git", "branch", "--show-current"]).stdout.strip()
        backup_branch = f"backup-{current_branch}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        print(f"\nCreating backup branch: {backup_branch}")
        run_command(["git", "checkout", "-b", backup_branch])
        run_command(["git", "checkout", current_branch])

        # Reset to first commit
        first_commit = commits[0]["hash"]
        run_command(["git", "reset", "--hard", first_commit])

        # Apply squashed commits
        for category, category_commits in plan:
            # Cherry-pick all commits in this group
            for commit in category_commits[1:]:  # Skip first as we're already there
                run_command(["git", "cherry-pick", "--no-commit", commit["hash"]], check=False)

            # Create squashed commit
            message = generate_squash_message(category, category_commits)
            run_command(["git", "commit", "-m", message], check=False)

        print("\nSquash complete!")
        print(f"Backup branch: {backup_branch}")
        print("\nTo undo: git reset --hard " + backup_branch)
        print("To push: git push --force origin " + current_branch)

    elif response == "m":
        print("\nManual mode - opening interactive rebase...")
        print("Use 'squash' or 's' to squash commits together")
        subprocess.run(["git", "rebase", "-i", commits[0]["hash"] + "^"])
    else:
        print("Squash cancelled")


if __name__ == "__main__":
    main()
