#!/bin/bash

echo "🔧 Music Generation Platform - Docker Fix Script"
echo "=============================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop first."
    exit 1
fi

echo "✅ Docker is running"

# Clean up any existing containers
echo "🧹 Cleaning up existing containers..."
docker-compose -f docker-compose.microservices.yml down 2>/dev/null

# Try to pull images with timeout
echo "📥 Pulling required images..."
echo "   This might take a few minutes on first run..."

# Pull with explicit platform for M1/M2 Macs
docker pull --platform linux/amd64 postgres:15-alpine || docker pull postgres:15
docker pull --platform linux/amd64 redis:7-alpine || docker pull redis:7

# Build services
echo "🏗️ Building microservices..."
docker-compose -f docker-compose.microservices.yml build --no-cache

# Start services
echo "🚀 Starting services..."
docker-compose -f docker-compose.microservices.yml up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

# Check service status
echo "📊 Checking service status..."
docker-compose -f docker-compose.microservices.yml ps

# Test API
echo "🧪 Testing API Gateway..."
curl -s http://localhost:8000/health > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ API Gateway is responding!"
    echo ""
    echo "🎉 System is ready! Run: python3 demo.py demo"
else
    echo "⚠️ API Gateway not ready yet. Wait a moment and try: python3 simple_test.py"
fi