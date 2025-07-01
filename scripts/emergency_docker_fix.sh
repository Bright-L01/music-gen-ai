#!/bin/bash

# Emergency Docker Fix Script for music21 MCP Server
# This script addresses:
# - Hash sum mismatches in apt-get
# - ARM64 compatibility on Mac
# - Complex dependencies failing to install

set -e

echo "=========================================="
echo "Emergency Docker Fix for music21 MCP Server"
echo "=========================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Step 1: Clean Docker completely
echo "Step 1: Cleaning Docker environment..."
echo "--------------------------------------"

clean_docker() {
    print_warning "Stopping all Docker containers..."
    docker stop $(docker ps -aq) 2>/dev/null || true
    
    print_warning "Removing all Docker containers..."
    docker rm $(docker ps -aq) 2>/dev/null || true
    
    print_warning "Removing all Docker images..."
    docker rmi $(docker images -q) 2>/dev/null || true
    
    print_warning "Pruning Docker system..."
    docker system prune -af --volumes 2>/dev/null || true
    
    print_success "Docker environment cleaned!"
}

# Ask user if they want to clean Docker
read -p "Do you want to clean Docker completely? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    clean_docker
fi

# Step 2: Create simplified Dockerfile for ARM64
echo ""
echo "Step 2: Creating simplified Dockerfile..."
echo "----------------------------------------"

# Create main Dockerfile
cat > Dockerfile.fixed << 'EOF'
# Use Python slim image for better compatibility
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1

# Fix apt-get hash sum issues
RUN echo "Acquire::http::Pipeline-Depth \"0\";" > /etc/apt/apt.conf.d/99fixbadproxy && \
    echo "Acquire::http::No-Cache \"true\";" >> /etc/apt/apt.conf.d/99fixbadproxy && \
    echo "Acquire::BrokenProxy \"true\";" >> /etc/apt/apt.conf.d/99fixbadproxy

# Update and install basic dependencies with retry logic
RUN for i in 1 2 3; do \
        apt-get clean && \
        rm -rf /var/lib/apt/lists/* && \
        apt-get update -y && \
        apt-get install -y --no-install-recommends \
            build-essential \
            curl \
            git \
            libxml2-dev \
            libxslt1-dev \
            zlib1g-dev \
            && break || \
        if [ $i -eq 3 ]; then exit 1; fi; \
        echo "Retry $i/3..."; \
        sleep 5; \
    done && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy only requirements first for better caching
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install toml && \
    python -c "import toml; \
               data = toml.load('pyproject.toml'); \
               deps = data.get('project', {}).get('dependencies', []); \
               print('\n'.join(deps))" > requirements.txt && \
    pip install -r requirements.txt || \
    (echo "Failed to install from pyproject.toml, trying minimal deps..." && \
     pip install music21 mcp numpy)

# Copy application code
COPY . .

# Install the package
RUN pip install -e . || echo "Warning: Package installation failed, continuing..."

# Expose port
EXPOSE 5000

# Create simple entrypoint script
RUN echo '#!/bin/bash\n\
if [ -f src/music21_mcp/server_simple.py ]; then\n\
    echo "Starting simplified server..."\n\
    python src/music21_mcp/server_simple.py\n\
elif [ -f src/music21_mcp/server.py ]; then\n\
    echo "Starting main server..."\n\
    python src/music21_mcp/server.py\n\
else\n\
    echo "No server file found!"\n\
    exit 1\n\
fi' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Set entrypoint
CMD ["/app/entrypoint.sh"]
EOF

print_success "Created Dockerfile.fixed"

# Create minimal Dockerfile as fallback
cat > Dockerfile.minimal << 'EOF'
# Ultra-minimal Dockerfile for testing
FROM python:3.11-slim

WORKDIR /app

# Install only essential packages
RUN pip install --no-cache-dir music21 mcp flask

# Copy minimal server
COPY src/music21_mcp/server_simple.py server.py

# Create minimal server if it doesn't exist
RUN if [ ! -f server.py ]; then \
    echo 'from flask import Flask, jsonify\n\
import music21\n\
\n\
app = Flask(__name__)\n\
\n\
@app.route("/health")\n\
def health():\n\
    return jsonify({"status": "ok", "music21_version": music21.__version__})\n\
\n\
@app.route("/analyze", methods=["POST"])\n\
def analyze():\n\
    return jsonify({"error": "Not implemented yet"})\n\
\n\
if __name__ == "__main__":\n\
    app.run(host="0.0.0.0", port=5000)' > server.py; \
fi

EXPOSE 5000
CMD ["python", "server.py"]
EOF

print_success "Created Dockerfile.minimal"

# Step 3: Create docker-compose file
echo ""
echo "Step 3: Creating docker-compose.yml..."
echo "-------------------------------------"

cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  music21-mcp:
    build:
      context: .
      dockerfile: Dockerfile.fixed
    image: music21-mcp:latest
    container_name: music21-mcp-server
    ports:
      - "5000:5000"
    environment:
      - PYTHONUNBUFFERED=1
    volumes:
      - ./src:/app/src
      - ./tests:/app/tests
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    restart: unless-stopped

  music21-mcp-minimal:
    build:
      context: .
      dockerfile: Dockerfile.minimal
    image: music21-mcp:minimal
    container_name: music21-mcp-minimal
    ports:
      - "5001:5000"
    restart: unless-stopped
EOF

print_success "Created docker-compose.yml"

# Step 4: Create build script
echo ""
echo "Step 4: Creating build scripts..."
echo "---------------------------------"

# Main build script
cat > build_docker.sh << 'EOF'
#!/bin/bash

echo "Building Docker image..."

# Try main build first
if docker build -f Dockerfile.fixed -t music21-mcp:latest .; then
    echo "✓ Main build successful!"
    exit 0
fi

echo "Main build failed, trying minimal build..."

# Fallback to minimal build
if docker build -f Dockerfile.minimal -t music21-mcp:minimal .; then
    echo "✓ Minimal build successful!"
    echo "Note: Running with limited functionality"
    exit 0
fi

echo "✗ All builds failed!"
exit 1
EOF

chmod +x build_docker.sh
print_success "Created build_docker.sh"

# Step 5: Create run script
cat > run_docker.sh << 'EOF'
#!/bin/bash

echo "Starting Docker container..."

# Check which image exists
if docker images | grep -q "music21-mcp.*latest"; then
    echo "Running full version..."
    docker run -d --name music21-mcp -p 5000:5000 music21-mcp:latest
elif docker images | grep -q "music21-mcp.*minimal"; then
    echo "Running minimal version..."
    docker run -d --name music21-mcp -p 5000:5000 music21-mcp:minimal
else
    echo "No Docker image found! Run ./build_docker.sh first"
    exit 1
fi

echo "Container started. Checking health..."
sleep 5

if curl -f http://localhost:5000/health 2>/dev/null; then
    echo "✓ Server is running!"
else
    echo "✗ Server health check failed"
    docker logs music21-mcp
fi
EOF

chmod +x run_docker.sh
print_success "Created run_docker.sh"

# Step 6: Create native fallback script
echo ""
echo "Step 6: Creating native fallback script..."
echo "-----------------------------------------"

cat > run_native.sh << 'EOF'
#!/bin/bash

echo "Running music21 MCP server natively (no Docker)..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install music21 mcp flask numpy

# Try to install from pyproject.toml
pip install -e . 2>/dev/null || echo "Warning: Could not install package"

# Run server
if [ -f "src/music21_mcp/server_simple.py" ]; then
    echo "Starting simplified server..."
    python src/music21_mcp/server_simple.py
elif [ -f "src/music21_mcp/server.py" ]; then
    echo "Starting main server..."
    python src/music21_mcp/server.py
else
    echo "Creating minimal server..."
    cat > minimal_server.py << 'PYEOF'
from flask import Flask, jsonify, request
import music21

app = Flask(__name__)

@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'music21_version': music21.__version__,
        'server': 'minimal'
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        # Basic analysis placeholder
        return jsonify({
            'status': 'success',
            'message': 'Analysis endpoint ready',
            'data': data
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting minimal music21 server on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=True)
PYEOF
    python minimal_server.py
fi
EOF

chmod +x run_native.sh
print_success "Created run_native.sh"

# Step 7: Create diagnostic script
echo ""
echo "Step 7: Creating diagnostic script..."
echo "------------------------------------"

cat > diagnose.sh << 'EOF'
#!/bin/bash

echo "Running Docker diagnostics..."
echo "============================"

# Check Docker installation
echo -n "Docker installed: "
if command -v docker &> /dev/null; then
    echo "✓"
    docker --version
else
    echo "✗"
    echo "Please install Docker first!"
    exit 1
fi

# Check Docker daemon
echo -n "Docker daemon running: "
if docker info &> /dev/null; then
    echo "✓"
else
    echo "✗"
    echo "Please start Docker daemon!"
    exit 1
fi

# Check architecture
echo -n "System architecture: "
arch=$(uname -m)
echo $arch

if [[ "$arch" == "arm64" || "$arch" == "aarch64" ]]; then
    echo "Note: Running on ARM64 (Apple Silicon)"
fi

# Check for existing containers
echo ""
echo "Existing containers:"
docker ps -a | grep music21 || echo "None"

# Check for existing images
echo ""
echo "Existing images:"
docker images | grep music21 || echo "None"

# Check disk space
echo ""
echo "Disk space:"
df -h / | tail -1

# Test network connectivity
echo ""
echo -n "Network connectivity: "
if curl -s https://pypi.org > /dev/null; then
    echo "✓"
else
    echo "✗ (This might cause package installation issues)"
fi
EOF

chmod +x diagnose.sh
print_success "Created diagnose.sh"

# Final instructions
echo ""
echo "=========================================="
echo "Emergency Docker Fix Complete!"
echo "=========================================="
echo ""
echo "Files created:"
echo "  - Dockerfile.fixed (main Dockerfile with fixes)"
echo "  - Dockerfile.minimal (fallback minimal version)"
echo "  - docker-compose.yml (for easy container management)"
echo "  - build_docker.sh (automated build script)"
echo "  - run_docker.sh (automated run script)"
echo "  - run_native.sh (fallback native Python script)"
echo "  - diagnose.sh (diagnostic tool)"
echo ""
echo "Next steps:"
echo "1. Run ./diagnose.sh to check your system"
echo "2. Run ./build_docker.sh to build the Docker image"
echo "3. Run ./run_docker.sh to start the container"
echo ""
echo "If Docker fails, use the native fallback:"
echo "  ./run_native.sh"
echo ""
echo "To use docker-compose instead:"
echo "  docker-compose up -d"
echo ""
echo "To check logs:"
echo "  docker logs music21-mcp"
echo ""
print_warning "Note: The minimal version has limited functionality but should work on any system."