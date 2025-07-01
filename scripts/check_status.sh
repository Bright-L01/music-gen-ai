#!/bin/bash

# Music21 MCP Server - Status Checker

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$PROJECT_ROOT/venv"
PID_FILE="$PROJECT_ROOT/.music21-mcp.pid"
LOG_DIR="$PROJECT_ROOT/logs"

echo -e "${BLUE}Music21 MCP Server Status${NC}"
echo "========================="
echo ""

# Check virtual environment
if [ -d "$VENV_DIR" ]; then
    echo -e "${GREEN}✓ Virtual environment exists${NC}"
    
    # Check installed packages
    if [ -f "$VENV_DIR/bin/python" ]; then
        source "$VENV_DIR/bin/activate" 2>/dev/null
        
        # Check key dependencies
        echo ""
        echo "Installed packages:"
        python3 -c "
import pkg_resources
packages = ['music21', 'mcp', 'fastmcp', 'numpy', 'pydantic']
for pkg in packages:
    try:
        version = pkg_resources.get_distribution(pkg).version
        print(f'  ✓ {pkg} {version}')
    except:
        print(f'  ✗ {pkg} not installed')
" 2>/dev/null || echo -e "${YELLOW}  Unable to check packages${NC}"
    fi
else
    echo -e "${RED}✗ Virtual environment not found${NC}"
    echo "  Run: ./run_native_no_docker.sh to set up"
fi

# Check server status
echo ""
echo "Server status:"
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 $PID 2>/dev/null; then
        echo -e "${GREEN}✓ Server is running (PID: $PID)${NC}"
        
        # Show process info
        echo ""
        echo "Process info:"
        ps -p $PID -o pid,ppid,user,%cpu,%mem,start,time,command | tail -n +1
    else
        echo -e "${YELLOW}! Server not running (stale PID file)${NC}"
        echo "  PID file exists but process $PID is not running"
    fi
else
    echo -e "${YELLOW}! Server not running${NC}"
    echo "  No PID file found"
fi

# Check logs
echo ""
echo "Recent logs:"
if [ -d "$LOG_DIR" ]; then
    latest_log=$(ls -t "$LOG_DIR"/music21-mcp-*.log 2>/dev/null | head -n1)
    if [ -n "$latest_log" ]; then
        echo "Latest log: $latest_log"
        echo "Last 10 lines:"
        echo "---"
        tail -n 10 "$latest_log" 2>/dev/null || echo "Unable to read log file"
        echo "---"
    else
        echo "No log files found"
    fi
else
    echo "Log directory not found"
fi

# System resources
echo ""
echo "System resources:"
echo -e "  CPU cores: $(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 'unknown')"
echo -e "  Memory: $(free -h 2>/dev/null | grep Mem | awk '{print $2}' || echo 'check manually')"
echo -e "  Disk space in $PROJECT_ROOT:"
df -h "$PROJECT_ROOT" | tail -n1 | awk '{print "    Used: " $3 " / " $2 " (" $5 " used)"}'

echo ""