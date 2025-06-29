#!/bin/bash

# Music Generation Platform Demo Startup Script
# 
# This script makes it easy to start the entire microservices platform
# and run the interactive demo.

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo -e "${BLUE}"
    echo "=================================================="
    echo "🎵 Music Generation Platform Demo"
    echo "=================================================="
    echo -e "${NC}"
}

print_step() {
    echo -e "${YELLOW}[STEP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Check dependencies
check_dependencies() {
    print_step "Checking dependencies..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed. Please install Python 3 first."
        exit 1
    fi
    
    print_success "All dependencies found!"
}

# Install demo CLI dependencies
install_demo_deps() {
    print_step "Installing demo CLI dependencies..."
    
    if [ ! -d "demo_venv" ]; then
        python3 -m venv demo_venv
    fi
    
    source demo_venv/bin/activate
    pip install -r demo-requirements.txt
    
    print_success "Demo CLI dependencies installed!"
}

# Start services
start_services() {
    print_step "Starting microservices..."
    
    # Create .env file if it doesn't exist
    if [ ! -f ".env" ]; then
        cat > .env << EOL
JWT_SECRET=music-gen-secret-key-for-demo-$(date +%s)
SERVICE_API_KEY=internal-service-key-for-demo
POSTGRES_PASSWORD=demo-password-$(date +%s)
EOL
        print_info "Created .env file with demo secrets"
    fi
    
    # Start services with docker-compose
    docker-compose -f docker-compose.microservices.yml up -d
    
    print_success "Services started!"
}

# Wait for services to be healthy
wait_for_services() {
    print_step "Waiting for services to be ready..."
    
    max_attempts=60
    attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            print_success "API Gateway is responding!"
            break
        fi
        
        ((attempt++))
        echo -n "."
        sleep 2
    done
    
    if [ $attempt -eq $max_attempts ]; then
        print_error "Services did not start within expected time"
        exit 1
    fi
    
    # Additional wait for all services
    print_step "Checking all services..."
    sleep 10
    
    # Test services health
    source demo_venv/bin/activate
    python3 demo.py setup --api-url http://localhost:8000
}

# Run demo
run_demo() {
    print_step "Starting interactive demo..."
    
    source demo_venv/bin/activate
    python3 demo.py demo --api-url http://localhost:8000
}

# Run tests
run_tests() {
    print_step "Running system tests..."
    
    # Install test dependencies
    source demo_venv/bin/activate
    pip install pytest pytest-asyncio
    
    # Run tests
    python3 -m pytest tests/test_complete_system.py -v
    
    print_success "Tests completed!"
}

# Stop services
stop_services() {
    print_step "Stopping services..."
    docker-compose -f docker-compose.microservices.yml down
    print_success "Services stopped!"
}

# Show logs
show_logs() {
    docker-compose -f docker-compose.microservices.yml logs -f
}

# Show status
show_status() {
    print_step "Service Status:"
    docker-compose -f docker-compose.microservices.yml ps
    
    print_step "Health Check:"
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        print_success "API Gateway: Healthy"
        
        # Check services health
        source demo_venv/bin/activate
        python3 demo.py quick-test --api-url http://localhost:8000
    else
        print_error "API Gateway: Not responding"
    fi
}

# Main menu
show_menu() {
    echo ""
    echo "Choose an option:"
    echo "1. Start full demo (recommended for first time)"
    echo "2. Start services only"
    echo "3. Run interactive demo"
    echo "4. Run tests"
    echo "5. Show status"
    echo "6. Show logs"
    echo "7. Stop services"
    echo "8. Exit"
    echo ""
}

# Main execution
main() {
    print_header
    
    # Check if services are already running
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        print_info "Services are already running!"
        show_menu
        read -p "Enter your choice (1-8): " choice
    else
        print_info "Services are not running. Starting full demo..."
        choice=1
    fi
    
    case $choice in
        1)
            check_dependencies
            install_demo_deps
            start_services
            wait_for_services
            run_demo
            ;;
        2)
            check_dependencies
            start_services
            wait_for_services
            ;;
        3)
            install_demo_deps
            run_demo
            ;;
        4)
            install_demo_deps
            run_tests
            ;;
        5)
            show_status
            ;;
        6)
            show_logs
            ;;
        7)
            stop_services
            ;;
        8)
            print_info "Goodbye!"
            exit 0
            ;;
        *)
            print_error "Invalid choice"
            exit 1
            ;;
    esac
}

# Handle script arguments
if [ $# -eq 0 ]; then
    main
else
    case $1 in
        start)
            check_dependencies
            install_demo_deps
            start_services
            wait_for_services
            ;;
        demo)
            install_demo_deps
            run_demo
            ;;
        test)
            install_demo_deps
            run_tests
            ;;
        stop)
            stop_services
            ;;
        status)
            show_status
            ;;
        logs)
            show_logs
            ;;
        *)
            echo "Usage: $0 [start|demo|test|stop|status|logs]"
            exit 1
            ;;
    esac
fi