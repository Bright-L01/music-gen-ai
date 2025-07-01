#!/bin/bash

# Music21 MCP Server - Native Run Script (No Docker)
# This script runs the music21 MCP server directly without Docker

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Project root directory (parent of music_gen)
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${GREEN}Music21 MCP Server - Native Runner${NC}"
echo "===================================="
echo ""

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to handle cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}Shutting down...${NC}"
    
    # Kill the server process if it's running
    if [ ! -z "$SERVER_PID" ]; then
        echo "Stopping MCP server (PID: $SERVER_PID)..."
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
    fi
    
    # Deactivate virtual environment if active
    if [ ! -z "$VIRTUAL_ENV" ]; then
        echo "Deactivating virtual environment..."
        deactivate 2>/dev/null || true
    fi
    
    echo -e "${GREEN}Cleanup complete.${NC}"
}

# Set up trap for cleanup
trap cleanup EXIT INT TERM

# Check for Python
echo "1. Checking Python installation..."
if ! command_exists python3; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    echo "Please install Python 3.9 or higher"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo -e "${GREEN}✓ Python $PYTHON_VERSION found${NC}"

# Check/create virtual environment
VENV_DIR="$PROJECT_ROOT/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "2. Creating Python virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo ""
    echo "2. Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "3. Activating virtual environment..."
source "$VENV_DIR/bin/activate"
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Upgrade pip
echo ""
echo "4. Upgrading pip..."
pip install --upgrade pip setuptools wheel >/dev/null 2>&1
echo -e "${GREEN}✓ pip upgraded${NC}"

# Install dependencies
echo ""
echo "5. Installing dependencies..."
echo "   This may take a few minutes on first run..."

# Install the package in development mode
if pip install -e "$PROJECT_ROOT" >/dev/null 2>&1; then
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo -e "${YELLOW}Warning: Some dependencies failed to install${NC}"
    echo "   Attempting to install core dependencies only..."
    
    # Try installing core dependencies individually
    pip install music21 mcp fastmcp numpy >/dev/null 2>&1 || true
fi

# Check if music21 is properly installed
echo ""
echo "6. Verifying music21 installation..."
if python3 -c "import music21; print(f'music21 version: {music21.__version__}')" 2>/dev/null; then
    echo -e "${GREEN}✓ music21 is ready${NC}"
else
    echo -e "${RED}Error: music21 installation failed${NC}"
    exit 1
fi

# Create a simple test to verify the server can import
echo ""
echo "7. Testing server imports..."
if python3 -c "from music21_mcp.server import mcp" 2>/dev/null; then
    echo -e "${GREEN}✓ Server modules can be imported${NC}"
else
    echo -e "${YELLOW}Warning: Server import test failed${NC}"
    echo "   The server may still work..."
fi

# Start the server
echo ""
echo "8. Starting Music21 MCP Server..."
echo "===================================="
echo ""
echo -e "${YELLOW}Server will start on the standard MCP port${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
echo ""

# Run the server
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

# Start the server and capture its PID
python3 -m music21_mcp.server &
SERVER_PID=$!

echo -e "${GREEN}✓ Server started with PID: $SERVER_PID${NC}"
echo ""
echo "Server is running. Use Ctrl+C to stop."
echo ""

# Wait for the server process
wait $SERVER_PID