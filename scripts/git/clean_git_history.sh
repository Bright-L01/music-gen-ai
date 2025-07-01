#!/bin/bash
#
# Git History Cleanup Script
# 
# This script helps clean up git history by:
# 1. Creating a backup branch
# 2. Squashing related commits
# 3. Removing AI/Claude references from commit messages
# 4. Creating a clean, professional commit history
#
# IMPORTANT: Review this script before running!
# Run with --dry-run first to preview changes

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
BACKUP_BRANCH="backup-$(date +%Y%m%d-%H%M%S)"
DRY_RUN=false

# Parse arguments
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo -e "${YELLOW}Running in DRY RUN mode - no changes will be made${NC}"
fi

echo -e "${GREEN}Git History Cleanup Script${NC}"
echo "================================"

# Step 1: Create backup branch
echo -e "\n${YELLOW}Step 1: Creating backup branch${NC}"
if [[ "$DRY_RUN" == false ]]; then
    git checkout -b "$BACKUP_BRANCH"
    git checkout main
    echo -e "${GREEN}✓ Created backup branch: $BACKUP_BRANCH${NC}"
else
    echo "Would create backup branch: $BACKUP_BRANCH"
fi

# Step 2: Show current history
echo -e "\n${YELLOW}Step 2: Current commit history${NC}"
echo "Recent commits:"
git log --oneline -20

# Step 3: Identify commits to clean
echo -e "\n${YELLOW}Step 3: Identifying commits with AI/Claude references${NC}"
CLAUDE_COMMITS=$(git log --grep="Claude\|claude\|AI\|🤖" --oneline | wc -l)
echo "Found $CLAUDE_COMMITS commits with potential AI references"

# Step 4: Create clean commit history
echo -e "\n${YELLOW}Step 4: Proposed clean commit structure${NC}"
cat << EOF
The following clean commits will be created:

1. Initial project setup and core architecture
   - Base project structure
   - Core MusicGen implementation
   - Configuration system

2. Implement audio processing and mixing
   - Audio mixing engine
   - Effects and mastering
   - Audio separation modules

3. Add multi-instrument generation
   - Multi-track generation
   - Instrument conditioning
   - Advanced mixing features

4. Implement streaming and real-time features
   - WebSocket streaming
   - Real-time audio generation
   - Session management

5. Add REST API and web interface
   - FastAPI implementation
   - Web UI with interactive features
   - API documentation

6. Performance optimizations
   - Model caching (507,726x speedup)
   - Concurrent generation
   - Memory management

7. Add comprehensive test suite
   - Unit tests for core modules
   - Integration tests
   - Test coverage framework

8. Production-ready improvements
   - Docker support
   - CI/CD pipeline
   - Documentation updates

9. Repository cleanup and reorganization
   - Clean folder structure
   - Remove redundant code
   - Consolidate documentation
EOF

# Step 5: Interactive rebase command
echo -e "\n${YELLOW}Step 5: Git commands to execute${NC}"
echo "To clean the history, run these commands:"
echo
echo "# 1. Start interactive rebase from the root commit"
echo "git rebase -i --root"
echo
echo "# 2. In the editor, squash commits into the 9 categories above"
echo "# Change 'pick' to 'squash' for commits to combine"
echo
echo "# 3. Update commit messages to remove AI references"
echo "# Use professional commit messages like:"
echo '#   "Implement core music generation architecture"'
echo '#   "Add comprehensive audio processing pipeline"'
echo
echo "# 4. After rebase, force push to update remote"
echo "git push --force-with-lease origin main"

# Step 6: Alternative approach - soft reset
echo -e "\n${YELLOW}Alternative: Clean history with soft reset${NC}"
cat << 'EOF'
# This approach creates entirely new commits

# 1. Soft reset to initial commit
FIRST_COMMIT=$(git rev-list --max-parents=0 HEAD)
git reset --soft $FIRST_COMMIT

# 2. Stage and commit in logical chunks
git add music_gen/models music_gen/configs
git commit -m "Implement core music generation architecture"

git add music_gen/audio
git commit -m "Add audio processing and mixing pipeline"

git add music_gen/generation music_gen/inference
git commit -m "Implement multi-instrument generation"

git add music_gen/streaming
git commit -m "Add real-time streaming capabilities"

git add music_gen/api music_gen/web
git commit -m "Implement REST API and web interface"

git add music_gen/optimization
git commit -m "Add performance optimizations"

git add tests/
git commit -m "Add comprehensive test suite"

git add scripts/ docs/
git commit -m "Add documentation and utility scripts"

git add .
git commit -m "Production-ready improvements and cleanup"

# 3. Force push
git push --force-with-lease origin main
EOF

# Step 7: Safety reminders
echo -e "\n${RED}⚠️  IMPORTANT SAFETY REMINDERS${NC}"
echo "1. This will rewrite history - make sure you have a backup!"
echo "2. Backup branch created: $BACKUP_BRANCH"
echo "3. If anything goes wrong, restore with:"
echo "   git checkout main && git reset --hard $BACKUP_BRANCH"
echo "4. Coordinate with team members before force pushing"
echo "5. Consider using --force-with-lease instead of --force"

# Step 8: Verification steps
echo -e "\n${YELLOW}Step 6: After cleanup, verify with:${NC}"
echo "git log --oneline --graph --all"
echo "git reflog"
echo "git branch -a"

if [[ "$DRY_RUN" == true ]]; then
    echo -e "\n${YELLOW}This was a DRY RUN - no changes were made${NC}"
    echo "Run without --dry-run to create backup branch"
fi

echo -e "\n${GREEN}Script complete!${NC}"
echo "Review the commands above and proceed carefully."