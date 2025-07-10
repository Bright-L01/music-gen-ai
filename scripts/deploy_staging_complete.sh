#!/bin/bash
# Complete Staging Deployment Script for Music Gen AI
# Orchestrates full production-like staging environment deployment

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
STAGING_ENV_FILE=".env.staging"
LOG_FILE="staging_deployment.log"
RESULTS_DIR="staging_results"

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Colored output functions
info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
    log "INFO: $1"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
    log "SUCCESS: $1"
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
    log "WARNING: $1"
}

error() {
    echo -e "${RED}❌ $1${NC}"
    log "ERROR: $1"
}

# Error handling
handle_error() {
    error "Deployment failed at line $1"
    error "Check $LOG_FILE for details"
    cleanup_on_failure
    exit 1
}

trap 'handle_error $LINENO' ERR

# Cleanup function
cleanup_on_failure() {
    warning "Cleaning up failed deployment..."
    docker-compose -f docker-compose.staging.yml down --remove-orphans || true
    docker system prune -f || true
}

# Pre-deployment checks
pre_deployment_checks() {
    info "Running pre-deployment checks..."
    
    # Check Docker and Docker Compose
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed or not in PATH"
        exit 1
    fi
    
    # Check available disk space (need at least 20GB)
    available_space=$(df / | awk 'NR==2 {print $4}')
    if [ "$available_space" -lt 20971520 ]; then  # 20GB in KB
        error "Insufficient disk space. Need at least 20GB free."
        exit 1
    fi
    
    # Check available memory (need at least 8GB)
    available_memory=$(free -m | awk 'NR==2{printf "%.0f", $7/1024}')
    if [ "$available_memory" -lt 8 ]; then
        error "Insufficient memory. Need at least 8GB available."
        exit 1
    fi
    
    # Check if ports are available
    ports_to_check=(80 443 3000 5432 6379 8001 8002 9090)
    for port in "${ports_to_check[@]}"; do
        if ss -tuln | grep -q ":$port "; then
            error "Port $port is already in use"
            exit 1
        fi
    done
    
    success "Pre-deployment checks passed"
}

# Generate SSL certificates
generate_ssl_certificates() {
    info "Generating SSL certificates..."
    
    if [ ! -f "nginx/ssl/staging.crt" ]; then
        ./scripts/generate_ssl_certs.sh
        success "SSL certificates generated"
    else
        info "SSL certificates already exist"
    fi
}

# Create environment file
create_staging_environment() {
    info "Creating staging environment configuration..."
    
    # Generate random passwords if they don't exist
    if [ ! -f "$STAGING_ENV_FILE" ]; then
        cat > "$STAGING_ENV_FILE" << EOF
# Staging Environment Configuration
# Generated on $(date)

# Database Configuration
POSTGRES_PASSWORD=$(openssl rand -base64 32)
POSTGRES_HOST=postgres-staging
POSTGRES_PORT=5432
POSTGRES_DB=musicgen_staging
POSTGRES_USER=musicgen

# Redis Configuration
REDIS_PASSWORD=$(openssl rand -base64 32)
REDIS_HOST=redis-staging
REDIS_PORT=6379

# Application Secrets
JWT_SECRET=$(openssl rand -base64 48)
SESSION_SECRET=$(openssl rand -base64 32)
CSRF_SECRET=$(openssl rand -base64 32)
STAGING_API_KEY=staging_$(openssl rand -hex 16)

# Monitoring Passwords
GRAFANA_PASSWORD=$(openssl rand -base64 16)
FLOWER_PASSWORD=$(openssl rand -base64 16)

# External Services (optional)
WANDB_API_KEY=disabled
WANDB_ENTITY=staging

# Load Testing Configuration
TARGET_HOST=http://nginx-staging
TEST_DURATION=172800
CONCURRENT_USERS=10
RAMP_UP_TIME=300

# Staging Specific
ENVIRONMENT=staging
DEBUG=false
LOG_LEVEL=INFO
EOF
        success "Staging environment file created: $STAGING_ENV_FILE"
        warning "Please review and update $STAGING_ENV_FILE as needed"
    else
        info "Staging environment file already exists"
    fi
}

# Deploy infrastructure
deploy_infrastructure() {
    info "Deploying staging infrastructure..."
    
    # Export environment variables
    export $(cat "$STAGING_ENV_FILE" | grep -v '^#' | xargs)
    
    # Pull latest images
    info "Pulling Docker images..."
    docker-compose -f docker-compose.staging.yml pull
    
    # Build custom images
    info "Building custom images..."
    docker-compose -f docker-compose.staging.yml build
    
    # Start infrastructure services first
    info "Starting infrastructure services..."
    docker-compose -f docker-compose.staging.yml up -d postgres-staging redis-staging
    
    # Wait for databases to be ready
    info "Waiting for databases to be ready..."
    sleep 30
    
    # Start monitoring services
    info "Starting monitoring services..."
    docker-compose -f docker-compose.staging.yml up -d prometheus-staging grafana-staging
    docker-compose -f docker-compose.staging.yml up -d postgres-exporter-staging redis-exporter-staging
    docker-compose -f docker-compose.staging.yml up -d node-exporter-staging cadvisor-staging
    
    # Start logging infrastructure
    info "Starting logging infrastructure..."
    docker-compose -f docker-compose.staging.yml up -d elasticsearch-staging logstash-staging kibana-staging
    
    # Start application services
    info "Starting application services..."
    docker-compose -f docker-compose.staging.yml up -d musicgen-api-1 musicgen-api-2
    docker-compose -f docker-compose.staging.yml up -d musicgen-worker-1 musicgen-worker-2
    docker-compose -f docker-compose.staging.yml up -d flower-staging
    
    # Start load balancer
    info "Starting load balancer..."
    docker-compose -f docker-compose.staging.yml up -d nginx-staging
    
    success "Infrastructure deployment completed"
}

# Verify deployment
verify_deployment() {
    info "Verifying deployment..."
    
    # Wait for services to be fully up
    sleep 60
    
    # Check service health
    services_to_check=(
        "postgres-staging:PostgreSQL"
        "redis-staging:Redis"
        "musicgen-api-1:API Instance 1"
        "musicgen-api-2:API Instance 2"
        "nginx-staging:Load Balancer"
        "prometheus-staging:Prometheus"
        "grafana-staging:Grafana"
    )
    
    failed_services=()
    
    for service_info in "${services_to_check[@]}"; do
        service=$(echo "$service_info" | cut -d: -f1)
        name=$(echo "$service_info" | cut -d: -f2)
        
        if docker-compose -f docker-compose.staging.yml ps "$service" | grep -q "Up"; then
            success "$name is running"
        else
            error "$name is not running"
            failed_services+=("$name")
        fi
    done
    
    if [ ${#failed_services[@]} -gt 0 ]; then
        error "Some services failed to start: ${failed_services[*]}"
        return 1
    fi
    
    # Test HTTP endpoints
    info "Testing HTTP endpoints..."
    
    # Wait a bit more for services to be ready
    sleep 30
    
    endpoints=(
        "http://localhost/health:Health Check"
        "http://localhost:9090:Prometheus"
        "http://localhost:3000:Grafana"
    )
    
    for endpoint_info in "${endpoints[@]}"; do
        endpoint=$(echo "$endpoint_info" | cut -d: -f1)
        name=$(echo "$endpoint_info" | cut -d: -f2)
        
        if curl -s -o /dev/null -w "%{http_code}" "$endpoint" | grep -q "200\|302"; then
            success "$name endpoint is responding"
        else
            warning "$name endpoint is not responding (may still be starting up)"
        fi
    done
    
    success "Deployment verification completed"
}

# Run integration tests
run_integration_tests() {
    info "Running integration tests..."
    
    # Build load test container
    docker build -t musicgen-load-test -f Dockerfile.load-test .
    
    # Run integration tests
    docker run --rm --network staging-network \
        -v "$(pwd)/staging_results:/app/results" \
        -e TARGET_HOST=http://nginx-staging \
        -e STAGING_API_KEY="$(grep STAGING_API_KEY $STAGING_ENV_FILE | cut -d= -f2)" \
        musicgen-load-test python /app/scripts/integration_test_suite.py
    
    if [ $? -eq 0 ]; then
        success "Integration tests passed"
    else
        warning "Some integration tests failed - check results for details"
    fi
}

# Run load tests
run_load_tests() {
    info "Running performance load tests..."
    
    # Quick load test (5 minutes)
    docker run --rm --network staging-network \
        -v "$(pwd)/staging_results:/app/results" \
        -e TARGET_HOST=http://nginx-staging \
        -e CONCURRENT_USERS=5 \
        -e TEST_DURATION=300 \
        musicgen-load-test python /app/scripts/comprehensive_load_test.py http://nginx-staging 5 300
    
    if [ $? -eq 0 ]; then
        success "Load tests completed"
    else
        warning "Load tests had issues - check results for details"
    fi
}

# Run security tests
run_security_tests() {
    info "Running security penetration tests..."
    
    docker run --rm --network staging-network \
        -v "$(pwd)/staging_results:/app/results" \
        -e TARGET_HOST=http://nginx-staging \
        -e STAGING_API_KEY="$(grep STAGING_API_KEY $STAGING_ENV_FILE | cut -d= -f2)" \
        musicgen-load-test python /app/scripts/security_penetration_test.py
    
    if [ $? -eq 0 ]; then
        success "Security tests completed - no critical vulnerabilities"
    else
        warning "Security tests found vulnerabilities - check results for details"
    fi
}

# Test backup procedures
test_backup_procedures() {
    info "Testing backup and restore procedures..."
    
    docker run --rm --network staging-network \
        -v "$(pwd)/staging_results:/app/results" \
        -v /var/run/docker.sock:/var/run/docker.sock \
        -e POSTGRES_HOST=postgres-staging \
        -e REDIS_HOST=redis-staging \
        -e POSTGRES_PASSWORD="$(grep POSTGRES_PASSWORD $STAGING_ENV_FILE | cut -d= -f2)" \
        -e REDIS_PASSWORD="$(grep REDIS_PASSWORD $STAGING_ENV_FILE | cut -d= -f2)" \
        musicgen-load-test python /app/scripts/backup_restore_test.py
    
    if [ $? -eq 0 ]; then
        success "Backup/restore tests passed"
    else
        warning "Some backup/restore tests failed - check results for details"
    fi
}

# Generate deployment report
generate_deployment_report() {
    info "Generating deployment report..."
    
    mkdir -p "$RESULTS_DIR"
    
    # Get container status
    docker-compose -f docker-compose.staging.yml ps > "$RESULTS_DIR/container_status.txt"
    
    # Get system resources
    docker stats --no-stream > "$RESULTS_DIR/resource_usage.txt"
    
    # Get logs
    docker-compose -f docker-compose.staging.yml logs --tail=100 > "$RESULTS_DIR/deployment_logs.txt"
    
    # Create summary report
    cat > "$RESULTS_DIR/deployment_summary.md" << EOF
# Staging Deployment Summary

**Deployment Date:** $(date)
**Environment:** Staging
**Status:** $(if [ -f "$RESULTS_DIR/deployment_success" ]; then echo "✅ Success"; else echo "⚠️ Partial"; fi)

## Services Deployed

$(docker-compose -f docker-compose.staging.yml ps --format "table {{.Name}}\t{{.Status}}")

## Access URLs

- **Load Balancer:** http://localhost
- **Grafana:** http://localhost:3000 (admin/$(grep GRAFANA_PASSWORD $STAGING_ENV_FILE | cut -d= -f2))
- **Prometheus:** http://localhost:9090
- **Flower (Celery):** http://localhost:5555 (admin/$(grep FLOWER_PASSWORD $STAGING_ENV_FILE | cut -d= -f2))
- **Kibana:** http://localhost:5601

## Environment File

Configuration stored in: \`$STAGING_ENV_FILE\`

## Results Directory

Test results and logs: \`$RESULTS_DIR/\`

## Next Steps

1. Review test results in \`$RESULTS_DIR/\`
2. Monitor system performance via Grafana
3. Run extended load tests if needed:
   \`\`\`bash
   ./scripts/orchestrate_48h_load_test.py
   \`\`\`

## Cleanup

To stop and remove the staging environment:
\`\`\`bash
docker-compose -f docker-compose.staging.yml down -v
\`\`\`
EOF
    
    success "Deployment report generated: $RESULTS_DIR/deployment_summary.md"
}

# Print final status
print_final_status() {
    echo
    echo "=============================================================="
    echo "🎯 STAGING DEPLOYMENT COMPLETE"
    echo "=============================================================="
    echo
    echo "📊 Services Status:"
    docker-compose -f docker-compose.staging.yml ps --format "  {{.Name}}: {{.Status}}"
    echo
    echo "🌐 Access URLs:"
    echo "  Load Balancer:  http://localhost"
    echo "  Grafana:        http://localhost:3000"
    echo "  Prometheus:     http://localhost:9090"
    echo "  Flower:         http://localhost:5555"
    echo "  Kibana:         http://localhost:5601"
    echo
    echo "📁 Files Created:"
    echo "  Environment:    $STAGING_ENV_FILE"
    echo "  Results:        $RESULTS_DIR/"
    echo "  Logs:           $LOG_FILE"
    echo
    echo "🔧 Management Commands:"
    echo "  View logs:      docker-compose -f docker-compose.staging.yml logs -f"
    echo "  Stop services:  docker-compose -f docker-compose.staging.yml down"
    echo "  Clean volumes:  docker-compose -f docker-compose.staging.yml down -v"
    echo
    echo "🚀 Next Steps:"
    echo "  1. Review deployment summary: $RESULTS_DIR/deployment_summary.md"
    echo "  2. Monitor performance via Grafana: http://localhost:3000"
    echo "  3. Run 48-hour load test: ./scripts/orchestrate_48h_load_test.py"
    echo
    echo "=============================================================="
}

# Main deployment function
main() {
    echo "=============================================================="
    echo "🚀 MUSIC GEN AI - STAGING DEPLOYMENT"
    echo "=============================================================="
    
    # Create results directory
    mkdir -p "$RESULTS_DIR"
    
    # Run deployment steps
    pre_deployment_checks
    generate_ssl_certificates
    create_staging_environment
    deploy_infrastructure
    verify_deployment
    
    # Run tests
    run_integration_tests
    run_load_tests
    run_security_tests
    test_backup_procedures
    
    # Generate reports
    generate_deployment_report
    
    # Mark deployment as successful
    touch "$RESULTS_DIR/deployment_success"
    
    print_final_status
    
    success "Staging deployment completed successfully!"
    info "Check $RESULTS_DIR/deployment_summary.md for details"
}

# Handle script arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "test-only")
        info "Running tests only..."
        run_integration_tests
        run_load_tests
        run_security_tests
        test_backup_procedures
        ;;
    "cleanup")
        info "Cleaning up staging environment..."
        docker-compose -f docker-compose.staging.yml down -v --remove-orphans
        docker system prune -f
        success "Cleanup completed"
        ;;
    "status")
        info "Staging environment status:"
        docker-compose -f docker-compose.staging.yml ps
        ;;
    *)
        echo "Usage: $0 [deploy|test-only|cleanup|status]"
        echo "  deploy    - Full deployment (default)"
        echo "  test-only - Run tests only"
        echo "  cleanup   - Remove staging environment"
        echo "  status    - Show service status"
        exit 1
        ;;
esac