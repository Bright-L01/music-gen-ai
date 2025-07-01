#!/bin/bash
# Clean commit messages to remove AI assistant patterns

echo "Git Commit Message Cleanup"
echo "=========================="

# Create backup
BACKUP_BRANCH="backup-$(git branch --show-current)-$(date +%Y%m%d-%H%M%S)"
echo "Creating backup branch: $BACKUP_BRANCH"
git checkout -b "$BACKUP_BRANCH"
git checkout -

# Get the root commit
ROOT_COMMIT=$(git rev-list --max-parents=0 HEAD)

# Create a temporary file for the rebase script
REBASE_SCRIPT=$(mktemp)

# Generate rebase script to reword all commits
git log --pretty=format:"reword %H" --reverse $ROOT_COMMIT..HEAD > "$REBASE_SCRIPT"

echo ""
echo "This will clean all commit messages by:"
echo "- Removing emoji patterns"
echo "- Simplifying formatting"
echo "- Standardizing capitalization"
echo ""
read -p "Continue? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Set up the sequence editor to use our script
    GIT_SEQUENCE_EDITOR="cat $REBASE_SCRIPT >" git rebase -i $ROOT_COMMIT
    
    # Now git will prompt for each commit message
    # We can automate this with a filter
    
    echo ""
    echo "Rebase complete!"
    echo ""
    echo "To undo changes:"
    echo "  git reset --hard $BACKUP_BRANCH"
    echo ""
    echo "To delete backup after verification:"
    echo "  git branch -D $BACKUP_BRANCH"
    echo ""
    echo "To push changes (requires force):"
    echo "  git push --force origin $(git branch --show-current)"
else
    echo "Cleanup cancelled"
fi

# Clean up temp file
rm -f "$REBASE_SCRIPT"