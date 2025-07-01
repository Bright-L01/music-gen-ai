#!/bin/bash

# Music21 MCP Server - Advanced Native Run Script
# Provides additional options and better error handling

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$PROJECT_ROOT/venv"
LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE="$LOG_DIR/music21-mcp-$(date +%Y%m%d-%H%M%S).log"

# Parse command line arguments
FRESH_INSTALL=false
VERBOSE=false
BACKGROUND=false
DEV_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --fresh)
            FRESH_INSTALL=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --background|-b)
            BACKGROUND=true
            shift
            ;;
        --dev)
            DEV_MODE=true
            shift
            ;;
        --help|-h)
            echo "Music21 MCP Server - Native Runner"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --fresh       Remove existing venv and do fresh install"
            echo "  --verbose     Show detailed output during installation"
            echo "  --background  Run server in background"
            echo "  --dev         Install development dependencies"
            echo "  --help        Show this help message"
            echo ""
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Function to check system requirements
check_requirements() {
    echo -e "${BLUE}Checking system requirements...${NC}"
    
    local errors=0
    
    # Check Python version
    if command -v python3 >/dev/null 2>&1; then
        local python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        local required_version="3.9"
        
        if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
            echo -e "${GREEN}✓ Python $python_version (>= $required_version)${NC}"
        else
            echo -e "${RED}✗ Python $python_version (requires >= $required_version)${NC}"
            ((errors++))
        fi
    else
        echo -e "${RED}✗ Python 3 not found${NC}"
        ((errors++))
    fi
    
    # Check for pip
    if python3 -m pip --version >/dev/null 2>&1; then
        echo -e "${GREEN}✓ pip is available${NC}"
    else
        echo -e "${RED}✗ pip not found${NC}"
        ((errors++))
    fi
    
    # Check for venv module
    if python3 -c "import venv" >/dev/null 2>&1; then
        echo -e "${GREEN}✓ venv module available${NC}"
    else
        echo -e "${YELLOW}! venv module not found (will try to install)${NC}"
    fi
    
    # Check disk space (need at least 500MB)
    local available_space=$(df -m "$PROJECT_ROOT" | awk 'NR==2 {print $4}')
    if [ "$available_space" -gt 500 ]; then
        echo -e "${GREEN}✓ Sufficient disk space (${available_space}MB available)${NC}"
    else
        echo -e "${RED}✗ Insufficient disk space (${available_space}MB available, need 500MB)${NC}"
        ((errors++))
    fi
    
    if [ $errors -gt 0 ]; then
        echo -e "\n${RED}System requirements not met. Please fix the issues above.${NC}"
        exit 1
    fi
    
    echo ""
}

# Function to setup virtual environment
setup_venv() {
    if [ "$FRESH_INSTALL" = true ] && [ -d "$VENV_DIR" ]; then
        echo -e "${YELLOW}Removing existing virtual environment...${NC}"
        rm -rf "$VENV_DIR"
    fi
    
    if [ ! -d "$VENV_DIR" ]; then
        echo -e "${BLUE}Creating virtual environment...${NC}"
        python3 -m venv "$VENV_DIR"
        echo -e "${GREEN}✓ Virtual environment created${NC}"
    else
        echo -e "${GREEN}✓ Using existing virtual environment${NC}"
    fi
    
    # Activate venv
    source "$VENV_DIR/bin/activate"
    
    # Upgrade pip
    echo -e "${BLUE}Upgrading pip...${NC}"
    if [ "$VERBOSE" = true ]; then
        pip install --upgrade pip setuptools wheel
    else
        pip install --upgrade pip setuptools wheel >/dev/null 2>&1
    fi
    echo -e "${GREEN}✓ pip upgraded${NC}"
    
    echo ""
}

# Function to install dependencies
install_dependencies() {
    echo -e "${BLUE}Installing dependencies...${NC}"
    
    cd "$PROJECT_ROOT"
    
    # Prepare installation command
    local install_cmd="pip install -e ."
    if [ "$DEV_MODE" = true ]; then
        install_cmd="pip install -e '.[dev]'"
    fi
    
    # Run installation
    if [ "$VERBOSE" = true ]; then
        eval $install_cmd
    else
        echo "This may take a few minutes..."
        if eval $install_cmd >/dev/null 2>&1; then
            echo -e "${GREEN}✓ All dependencies installed${NC}"
        else
            echo -e "${YELLOW}! Some optional dependencies failed${NC}"
            echo "  Installing core dependencies..."
            
            # Install core dependencies one by one
            local core_deps=(
                "music21>=9.1.0"
                "mcp>=0.1.0"
                "fastmcp>=0.1.0"
                "numpy>=1.24.0"
                "pydantic>=2.0.0"
            )
            
            for dep in "${core_deps[@]}"; do
                if [ "$VERBOSE" = true ]; then
                    pip install "$dep" || true
                else
                    pip install "$dep" >/dev/null 2>&1 || true
                fi
            done
            
            echo -e "${GREEN}✓ Core dependencies installed${NC}"
        fi
    fi
    
    echo ""
}

# Function to verify installation
verify_installation() {
    echo -e "${BLUE}Verifying installation...${NC}"
    
    # Test imports
    local test_script=$(cat <<EOF
import sys
try:
    import music21
    print(f"✓ music21 {music21.__version__}")
except ImportError as e:
    print(f"✗ music21: {e}")
    sys.exit(1)

try:
    from music21_mcp.server import mcp
    print("✓ MCP server module")
except ImportError as e:
    print(f"✗ MCP server: {e}")
    sys.exit(1)

try:
    import numpy
    print(f"✓ numpy {numpy.__version__}")
except ImportError:
    print("! numpy not available (optional)")

print("\n✓ All core components verified")
EOF
)
    
    if ! python3 -c "$test_script"; then
        echo -e "\n${RED}Installation verification failed${NC}"
        exit 1
    fi
    
    echo ""
}

# Function to create log directory
setup_logging() {
    if [ ! -d "$LOG_DIR" ]; then
        mkdir -p "$LOG_DIR"
        echo -e "${GREEN}✓ Created log directory${NC}"
    fi
}

# Function to start server
start_server() {
    cd "$PROJECT_ROOT"
    export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"
    
    if [ "$BACKGROUND" = true ]; then
        echo -e "${BLUE}Starting server in background...${NC}"
        setup_logging
        
        # Start server in background with logging
        nohup python3 -m music21_mcp.server > "$LOG_FILE" 2>&1 &
        local server_pid=$!
        
        # Save PID to file
        echo $server_pid > "$PROJECT_ROOT/.music21-mcp.pid"
        
        # Wait a moment to check if it started successfully
        sleep 2
        
        if kill -0 $server_pid 2>/dev/null; then
            echo -e "${GREEN}✓ Server started in background (PID: $server_pid)${NC}"
            echo -e "  Log file: $LOG_FILE"
            echo -e "  To stop: $0 --stop"
        else
            echo -e "${RED}✗ Server failed to start${NC}"
            echo -e "  Check log file: $LOG_FILE"
            exit 1
        fi
    else
        echo -e "${BLUE}Starting server...${NC}"
        echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
        echo ""
        
        # Run server in foreground
        python3 -m music21_mcp.server
    fi
}

# Function to stop server
stop_server() {
    if [ -f "$PROJECT_ROOT/.music21-mcp.pid" ]; then
        local pid=$(cat "$PROJECT_ROOT/.music21-mcp.pid")
        if kill -0 $pid 2>/dev/null; then
            echo -e "${BLUE}Stopping server (PID: $pid)...${NC}"
            kill $pid
            rm "$PROJECT_ROOT/.music21-mcp.pid"
            echo -e "${GREEN}✓ Server stopped${NC}"
        else
            echo -e "${YELLOW}Server not running (stale PID file)${NC}"
            rm "$PROJECT_ROOT/.music21-mcp.pid"
        fi
    else
        echo -e "${YELLOW}No server PID file found${NC}"
    fi
}

# Handle stop command
if [ "$1" = "--stop" ]; then
    stop_server
    exit 0
fi

# Main execution
echo -e "${GREEN}Music21 MCP Server - Native Runner${NC}"
echo "===================================="
echo ""

# Run setup steps
check_requirements
setup_venv
install_dependencies
verify_installation

# Start the server
start_server