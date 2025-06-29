#!/bin/bash

# Simple startup script for the Music Generation Platform

echo "🎵 Starting Music Generation Platform..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cat > .env << EOL
JWT_SECRET=music-gen-secret-key-$(date +%s)
SERVICE_API_KEY=internal-service-key
POSTGRES_PASSWORD=demo-password
EOL
fi

# Start services
echo "Starting all services..."
docker-compose -f docker-compose.microservices.yml up -d

echo "Waiting for services to start..."
sleep 20

# Test if API Gateway is responding
echo "Testing API Gateway..."
max_attempts=30
attempt=0

while [ $attempt -lt $max_attempts ]; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ API Gateway is responding!"
        break
    fi
    
    ((attempt++))
    echo "Waiting... (attempt $attempt/$max_attempts)"
    sleep 3
done

if [ $attempt -eq $max_attempts ]; then
    echo "❌ Services did not start properly"
    echo "Check logs with: docker-compose -f docker-compose.microservices.yml logs"
    exit 1
fi

# Install Python dependencies if needed
if [ ! -d "demo_venv" ]; then
    echo "Setting up Python environment..."
    python3 -m venv demo_venv
    source demo_venv/bin/activate
    pip install click httpx rich
else
    source demo_venv/bin/activate
fi

# Run simple test
echo "Running system test..."
python3 simple_test.py

echo ""
echo "🎉 System is ready!"
echo ""
echo "Next steps:"
echo "1. Try the full demo: python3 demo.py demo"
echo "2. Or test manually: python3 simple_test.py"
echo "3. View API docs: http://localhost:8000/docs"
echo "4. Check service status: docker-compose -f docker-compose.microservices.yml ps"
echo ""
echo "To stop: docker-compose -f docker-compose.microservices.yml down"