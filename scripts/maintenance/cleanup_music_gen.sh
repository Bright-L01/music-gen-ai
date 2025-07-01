#!/bin/bash

# Music Generation Platform - Comprehensive Cleanup Script
# This script will safely clean and reorganize the project

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
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

# Check if we're in the right directory
if [[ ! -f "docker-compose.microservices.yml" ]]; then
    error "This script must be run from the music_gen project root directory"
fi

echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}    Music Generation Platform - Cleanup & Reorganization${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo

# Phase 1: Backup
log "Phase 1: Creating backup..."
if [[ ! -d "backup_$(date +%Y%m%d)" ]]; then
    mkdir -p "backup_$(date +%Y%m%d)"
    
    # Backup important directories
    for dir in music_gen music_gen_services tests services; do
        if [[ -d "$dir" ]]; then
            log "Backing up $dir..."
            cp -r "$dir" "backup_$(date +%Y%m%d)/" 2>/dev/null || true
        fi
    done
    
    # Backup important files
    log "Backing up configuration files..."
    cp *.yml *.toml *.txt *.md "backup_$(date +%Y%m%d)/" 2>/dev/null || true
    
    log "Backup complete in backup_$(date +%Y%m%d)/"
else
    warning "Backup already exists for today, skipping..."
fi

# Phase 2: Clean up redundant files
log "Phase 2: Cleaning up redundant files..."

# Create outputs directory for WAV files
mkdir -p outputs/test_results
mkdir -p outputs/generated

# Move WAV files to outputs
log "Moving WAV files to outputs directory..."
find . -maxdepth 1 -name "*.wav" -exec mv {} outputs/test_results/ \; 2>/dev/null || true

# Remove duplicate test files from root
log "Removing duplicate test files from root..."
for file in test_*.py; do
    if [[ -f "$file" && -f "tests/$(basename $file)" ]]; then
        log "  Removing duplicate: $file"
        rm "$file"
    elif [[ -f "$file" ]]; then
        log "  Moving to tests: $file"
        mv "$file" tests/
    fi
done

# Clean up *_new directories
log "Cleaning up *_new directories..."
for dir in api_new audio_new cli_new models_new utils_new web_new; do
    if [[ -d "$dir" && -z "$(ls -A $dir)" ]]; then
        log "  Removing empty directory: $dir"
        rmdir "$dir"
    elif [[ -d "$dir" ]]; then
        warning "  $dir is not empty, manual review needed"
    fi
done

# Remove duplicate cache
if [[ -d "test_cache" && -d "cache" ]]; then
    log "Removing duplicate test_cache..."
    rm -rf test_cache/
fi

# Phase 3: Reorganize structure
log "Phase 3: Reorganizing project structure..."

# Create new directory structure
mkdir -p src/{api,core,models,services,utils,web}
mkdir -p tests/{unit,integration,performance,fixtures}
mkdir -p docs
mkdir -p scripts
mkdir -p examples

# Move service code to src/services
if [[ -d "services" ]]; then
    log "Moving services to src/services..."
    cp -r services/* src/services/ 2>/dev/null || true
fi

# Move music_gen core to src/core
if [[ -d "music_gen" ]]; then
    log "Moving core generation code to src/core..."
    cp -r music_gen/* src/core/ 2>/dev/null || true
fi

# Move scripts
log "Organizing scripts..."
mv *.sh scripts/ 2>/dev/null || true

# Phase 4: Clean up documentation
log "Phase 4: Consolidating documentation..."

# Create docs directory structure
mkdir -p docs/{archive,guides,api}

# Archive old documentation
log "Archiving old documentation..."
for file in *COMPLETE*.md *STATUS*.md *REPORT*.md *SUMMARY*.md; do
    if [[ -f "$file" ]]; then
        mv "$file" docs/archive/
    fi
done

# Phase 5: Git cleanup
log "Phase 5: Preparing Git repository..."

# Create comprehensive .gitignore
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
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual Environment
venv/
ENV/
env/
.env

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/
.hypothesis/

# IDE
.idea/
.vscode/
*.swp
*.swo
*~

# Project specific
outputs/
cache/
*.wav
*.mp3
*.flac
logs/
temp/
backup_*/

# Docker
.docker/

# OS
.DS_Store
Thumbs.db
EOF

# Create README template
cat > README_NEW.md << 'EOF'
# Music Generation Platform

A modern text-to-music generation system powered by transformer models.

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/music_gen.git
cd music_gen

# Install dependencies
pip install -r requirements.txt

# Run the demo
python examples/basic_generation.py "Create upbeat electronic music"
```

## 📋 Features

- Text-to-music generation using state-of-the-art models
- Multiple music styles and genres
- REST API for integration
- Web interface for easy access
- Microservices architecture for scalability

## 🛠️ Installation

See [docs/getting-started.md](docs/getting-started.md) for detailed installation instructions.

## 📖 Documentation

- [Architecture Overview](docs/architecture.md)
- [API Reference](docs/api-reference.md)
- [Deployment Guide](docs/deployment.md)
- [Contributing](CONTRIBUTING.md)

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src
```

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.
EOF

log "✅ Cleanup complete!"
echo
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Review the changes in backup_$(date +%Y%m%d)/"
echo "2. Test that core functionality still works"
echo "3. Update imports in Python files if needed"
echo "4. Commit changes to Git"
echo "5. Push to GitHub repository"
echo
echo -e "${GREEN}Project is now cleaner and better organized!${NC}"

# Summary
echo
echo -e "${BLUE}Summary:${NC}"
echo "- Created backup in backup_$(date +%Y%m%d)/"
echo "- Moved test outputs to outputs/"
echo "- Consolidated test files in tests/"
echo "- Created organized src/ structure"
echo "- Archived old documentation"
echo "- Created new .gitignore and README template"