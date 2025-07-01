#!/usr/bin/env python3
"""
Clean git history to remove AI assistant references and standardize commits.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command."""
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def main():
    """Clean git history."""
    print("Git History Cleanup Script")
    print("=" * 50)

    # Check if we're in a git repository
    result = run_command(["git", "rev-parse", "--git-dir"], check=False)
    if result.returncode != 0:
        print("Error: Not in a git repository")
        sys.exit(1)

    # Get current branch
    result = run_command(["git", "branch", "--show-current"])
    current_branch = result.stdout.strip()
    print(f"Current branch: {current_branch}")

    # Create backup branch
    backup_branch = f"backup-{current_branch}-before-cleanup"
    print(f"\nCreating backup branch: {backup_branch}")
    run_command(["git", "checkout", "-b", backup_branch])
    run_command(["git", "checkout", current_branch])

    print("\nGit history cleanup options:")
    print("1. Interactive rebase to clean commit messages (recommended)")
    print("2. Squash all commits into one")
    print("3. Reset author information for all commits")
    print("4. Exit without changes")

    choice = input("\nSelect option (1-4): ").strip()

    if choice == "1":
        print("\nStarting interactive rebase...")
        print("In the editor that opens:")
        print("- Change 'pick' to 'reword' for commits you want to edit")
        print("- Save and close to continue")
        print("- Edit each commit message as prompted")
        print("\nPress Enter to continue...")
        input()

        # Get the initial commit
        result = run_command(["git", "rev-list", "--max-parents=0", "HEAD"])
        initial_commit = result.stdout.strip()

        # Start interactive rebase
        subprocess.run(["git", "rebase", "-i", f"{initial_commit}^"], check=False)

    elif choice == "2":
        print("\nSquashing all commits...")

        # Get the initial commit
        result = run_command(["git", "rev-list", "--max-parents=0", "HEAD"])
        initial_commit = result.stdout.strip()

        # Reset to initial commit
        run_command(["git", "reset", "--soft", initial_commit])

        # Create new commit
        commit_msg = input("Enter new commit message: ").strip()
        if not commit_msg:
            commit_msg = "feat: Music Generation AI System"

        run_command(["git", "commit", "-m", commit_msg])
        print(f"All commits squashed into: {commit_msg}")

    elif choice == "3":
        print("\nResetting author information...")

        # Get user info
        result = run_command(["git", "config", "user.name"])
        user_name = result.stdout.strip()
        result = run_command(["git", "config", "user.email"])
        user_email = result.stdout.strip()

        print(f"Current user: {user_name} <{user_email}>")

        # Confirm
        confirm = input("Reset all commits to this author? (y/n): ").strip().lower()
        if confirm == "y":
            # Use filter-branch to reset author
            cmd = [
                "git",
                "filter-branch",
                "-f",
                "--env-filter",
                f'export GIT_AUTHOR_NAME="{user_name}"; '
                f'export GIT_AUTHOR_EMAIL="{user_email}"; '
                f'export GIT_COMMITTER_NAME="{user_name}"; '
                f'export GIT_COMMITTER_EMAIL="{user_email}";',
                "--",
                "--all",
            ]
            run_command(cmd)
            print("Author information reset for all commits")
        else:
            print("Cancelled")

    else:
        print("\nNo changes made")
        return

    print("\n" + "=" * 50)
    print("Cleanup complete!")
    print(f"Backup branch created: {backup_branch}")
    print("\nTo undo changes:")
    print(f"  git reset --hard {backup_branch}")
    print("\nTo delete backup branch after verifying changes:")
    print(f"  git branch -D {backup_branch}")
    print("\nWARNING: If you're happy with the changes, you'll need to force push:")
    print(f"  git push --force origin {current_branch}")


if __name__ == "__main__":
    main()
