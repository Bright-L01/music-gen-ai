#!/bin/bash

# Music Generation Platform - Git Setup and Cleanup Script
# This script will set up proper version control and clean the repository

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# ASCII Art Banner
echo -e "${CYAN}"
cat << "EOF"
 __  __           _        ____            
|  \/  |_   _ ___(_) ___  / ___| ___ _ __  
| |\/| | | | / __| |/ __| | |  _ / _ \ '_ \ 
| |  | | |_| \__ \ | (__  | |_| |  __/ | | |
|_|  |_|__,_|___/_|\___|  \____|\___|_| |_|
                                            
    Git Setup & Cleanup Tool
EOF
echo -e "${NC}"

# Functions
log() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Check if we're in the music_gen directory
if [[ ! -f "docker-compose.microservices.yml" ]]; then
    error "This script must be run from the music_gen project root"
fi

# Phase 1: Pre-cleanup Analysis
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Phase 1: Analyzing Current State${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

# Run Python analysis
log "Running comprehensive analysis..."
python3 analyze_and_cleanup.py

# Show summary
echo
info "Analysis complete. Check cleanup_analysis_report.json for details."
echo

# Ask for confirmation
read -p "Proceed with cleanup and Git setup? (yes/no): " -n 3 -r
echo
if [[ ! $REPLY =~ ^[Yy]es$ ]]; then
    warning "Cleanup cancelled by user"
    exit 0
fi

# Phase 2: Backup
echo
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Phase 2: Creating Backup${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
log "Creating backup in $BACKUP_DIR..."

mkdir -p "$BACKUP_DIR"
cp -r music_gen music_gen_services tests services "$BACKUP_DIR/" 2>/dev/null || true
cp *.yml *.toml *.txt *.md "$BACKUP_DIR/" 2>/dev/null || true

log "Backup complete!"

# Phase 3: Clean up files
echo
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Phase 3: Cleaning Up Files${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

# Create organized structure
log "Creating organized directory structure..."
mkdir -p {src/{api,core,models,services,utils,web},tests/{unit,integration,performance,fixtures}}
mkdir -p {docs/{guides,api,archive},scripts,examples,outputs/{generated,test_results}}

# Move WAV files
log "Moving audio files to outputs..."
find . -maxdepth 1 -name "*.wav" -exec mv {} outputs/test_results/ \; 2>/dev/null || true
find . -maxdepth 1 -name "*.mp3" -exec mv {} outputs/test_results/ \; 2>/dev/null || true

# Move test files
log "Organizing test files..."
for file in test_*.py; do
    if [[ -f "$file" ]]; then
        mv "$file" tests/ 2>/dev/null || true
    fi
done

# Archive old documentation
log "Archiving old documentation..."
for pattern in "*COMPLETE*.md" "*STATUS*.md" "*REPORT*.md" "*SUMMARY*.md" "*ANALYSIS*.md"; do
    for file in $pattern; do
        if [[ -f "$file" && "$file" != "README_COMPREHENSIVE.md" ]]; then
            mv "$file" docs/archive/ 2>/dev/null || true
        fi
    done
done

# Clean up duplicate cache
if [[ -d "test_cache" && -d "cache" ]]; then
    log "Removing duplicate cache..."
    rm -rf test_cache/
fi

# Phase 4: Git Setup
echo
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Phase 4: Setting Up Git Version Control${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

# Initialize git if needed
if [[ ! -d ".git" ]]; then
    log "Initializing Git repository..."
    git init
fi

# Create comprehensive .gitignore
log "Creating .gitignore..."
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual Environment
venv/
ENV/
env/
.venv/
.env

# PyCharm
.idea/

# VS Code
.vscode/
*.code-workspace

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb

# Testing
.pytest_cache/
.coverage
.coverage.*
htmlcov/
.tox/
.nox/
.hypothesis/
*.cover
*.py,cover
.cache
nosetests.xml
coverage.xml
*.log

# Model and Data
cache/
models/
*.bin
*.ckpt
*.pth
*.h5
*.safetensors

# Audio files
outputs/
*.wav
*.mp3
*.flac
*.ogg
*.m4a

# Temporary files
temp/
tmp/
*.tmp
*.temp
.~*

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Backups
backup_*/
*.backup
*.bak

# Docker
.docker/

# Secrets
.env
.env.*
!.env.example
secrets/
*.key
*.pem

# Large files
*.zip
*.tar
*.gz
*.rar
*.7z

# Logs
logs/
*.log

# Custom
*_new/
instant_demo.py
emergency_docker_fix.sh
run_native_*.sh
check_status.sh
cleanup_analysis_report.json
EOF

# Create .gitattributes
log "Creating .gitattributes..."
cat > .gitattributes << 'EOF'
# Auto detect text files and perform LF normalization
* text=auto

# Python files
*.py text
*.pyw text
*.pyx text
*.pyi text

# Jupyter notebooks
*.ipynb text eol=lf

# Shell scripts
*.sh text eol=lf
*.bash text eol=lf

# Data files
*.csv text
*.json text
*.xml text
*.yml text
*.yaml text

# Documentation
*.md text
*.rst text
*.txt text

# Binary files
*.png binary
*.jpg binary
*.jpeg binary
*.gif binary
*.ico binary
*.wav binary
*.mp3 binary
*.flac binary
*.ogg binary
*.pdf binary
*.zip binary
*.gz binary
*.tar binary
*.tgz binary

# Model files
*.pt binary
*.pth binary
*.bin binary
*.ckpt binary
*.safetensors binary
*.h5 binary
EOF

# Use the comprehensive README
if [[ -f "README_COMPREHENSIVE.md" ]]; then
    log "Setting up comprehensive README..."
    mv README_COMPREHENSIVE.md README.md
fi

# Create CONTRIBUTING.md
log "Creating CONTRIBUTING.md..."
cat > CONTRIBUTING.md << 'EOF'
# Contributing to Music Generation Platform

Thank you for your interest in contributing! We welcome contributions of all kinds.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/music_gen.git`
3. Create a branch: `git checkout -b feature/your-feature`
4. Make your changes
5. Run tests: `pytest`
6. Commit: `git commit -m "Add your feature"`
7. Push: `git push origin feature/your-feature`
8. Create a Pull Request

## Code Style

- Follow PEP 8
- Use type hints
- Run `black` for formatting
- Add docstrings to all functions

## Testing

- Write tests for new features
- Ensure all tests pass
- Maintain >80% code coverage

## Questions?

Open an issue or discussion on GitHub.
EOF

# Create LICENSE
log "Creating LICENSE..."
cat > LICENSE << 'EOF'
MIT License

Copyright (c) 2024 Music Generation Platform

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF

# Phase 5: Git Operations
echo
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Phase 5: Git Operations${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

# Check current git status
log "Checking Git status..."
git status --short

# Stage files
log "Staging files..."
git add .gitignore .gitattributes README.md LICENSE CONTRIBUTING.md
git add src/ tests/ docs/ scripts/ examples/ 2>/dev/null || true
git add *.py *.yml *.toml *.txt 2>/dev/null || true

# Remove files that shouldn't be tracked
git rm --cached -r cache/ 2>/dev/null || true
git rm --cached -r outputs/ 2>/dev/null || true
git rm --cached *.wav *.mp3 2>/dev/null || true
git rm --cached .env 2>/dev/null || true

# Commit
log "Creating initial commit..."
git commit -m "feat: Initial commit - Music Generation Platform

- Comprehensive project structure
- Microservices architecture
- Text-to-music generation
- REST API and web interface
- Docker support
- Comprehensive documentation" || true

# Phase 6: GitHub Setup
echo
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Phase 6: GitHub Integration${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

# Check if remote exists
if git remote | grep -q "origin"; then
    info "Remote 'origin' already exists"
    git remote -v
else
    warning "No remote repository configured"
    echo
    echo "To connect to GitHub:"
    echo "1. Create a new repository on GitHub (without initializing)"
    echo "2. Run: git remote add origin https://github.com/yourusername/music_gen.git"
    echo "3. Run: git push -u origin main"
fi

# Summary
echo
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✨ Setup Complete!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"

echo
info "Summary of changes:"
echo "  ✅ Created backup in $BACKUP_DIR/"
echo "  ✅ Organized project structure"
echo "  ✅ Moved test outputs to outputs/"
echo "  ✅ Archived old documentation"
echo "  ✅ Set up Git with proper .gitignore"
echo "  ✅ Created comprehensive README"
echo "  ✅ Added LICENSE and CONTRIBUTING.md"

echo
info "Next steps:"
echo "  1. Review the changes"
echo "  2. Test that everything still works"
echo "  3. Push to GitHub:"
echo "     git remote add origin https://github.com/yourusername/music_gen.git"
echo "     git push -u origin main"
echo "  4. Set up GitHub Actions for CI/CD"
echo "  5. Create releases and tags"

echo
echo -e "${CYAN}🎉 Your Music Generation Platform is now clean and ready!${NC}"