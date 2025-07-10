#!/bin/bash
# Production Deployment Script for Music Gen AI
# This script orchestrates the production deployment with safety checks and monitoring

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
DEPLOYMENT_ID=$(date +%Y%m%d_%H%M%S)
LOG_FILE="production_deployment_${DEPLOYMENT_ID}.log"
METRICS_FILE="deployment_metrics_${DEPLOYMENT_ID}.json"
ROLLBACK_POINT=""
DEPLOYMENT_START_TIME=$(date +%s)

# Deployment configuration
PROD_ENV_FILE=".env.production"
MAINTENANCE_MODE_ENABLED=false
HEALTH_CHECK_RETRIES=10
HEALTH_CHECK_DELAY=30
DEPLOYMENT_TIMEOUT=3600  # 1 hour

# Monitoring endpoints
PROMETHEUS_URL="${PROMETHEUS_URL:-http://prometheus:9090}"
GRAFANA_URL="${GRAFANA_URL:-http://grafana:3000}"
API_HEALTH_URL="${API_URL:-https://api.musicgen.ai}/health"

# Team contacts (should be encrypted/secured in production)
SLACK_WEBHOOK="${SLACK_WEBHOOK_URL}"
PAGERDUTY_TOKEN="${PAGERDUTY_TOKEN}"

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

critical() {
    echo -e "${RED}🚨 CRITICAL: $1${NC}"
    log "CRITICAL: $1"
    send_alert "CRITICAL" "$1"
}

# Progress tracking
progress() {
    echo -e "${PURPLE}🔄 $1${NC}"
    log "PROGRESS: $1"
}

# Send alerts to team
send_alert() {
    local severity=$1
    local message=$2
    
    # Slack notification
    if [ -n "$SLACK_WEBHOOK" ]; then
        curl -s -X POST "$SLACK_WEBHOOK" \
            -H 'Content-Type: application/json' \
            -d "{\"text\":\"[$severity] Production Deployment: $message\"}" || true
    fi
    
    # PagerDuty for critical alerts
    if [ "$severity" = "CRITICAL" ] && [ -n "$PAGERDUTY_TOKEN" ]; then
        curl -s -X POST "https://events.pagerduty.com/v2/enqueue" \
            -H 'Content-Type: application/json' \
            -H "Authorization: Token token=$PAGERDUTY_TOKEN" \
            -d "{
                \"routing_key\": \"$PAGERDUTY_TOKEN\",
                \"event_action\": \"trigger\",
                \"payload\": {
                    \"summary\": \"Production Deployment Critical: $message\",
                    \"severity\": \"critical\",
                    \"source\": \"deployment-script\"
                }
            }" || true
    fi
}

# Metrics collection
collect_metrics() {
    local phase=$1
    local metrics=$(cat <<EOF
{
    "deployment_id": "$DEPLOYMENT_ID",
    "phase": "$phase",
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "error_rate": $(get_error_rate),
    "response_time_p95": $(get_response_time_p95),
    "cpu_usage": $(get_cpu_usage),
    "memory_usage": $(get_memory_usage),
    "active_connections": $(get_active_connections)
}
EOF
    )
    
    echo "$metrics" >> "$METRICS_FILE"
}

# Metric collection functions
get_error_rate() {
    # Query Prometheus for error rate
    curl -s "$PROMETHEUS_URL/api/v1/query?query=rate(http_requests_total{status=~\"5..\"}[5m])" | \
        jq -r '.data.result[0].value[1] // 0' 2>/dev/null || echo "0"
}

get_response_time_p95() {
    # Query Prometheus for p95 response time
    curl -s "$PROMETHEUS_URL/api/v1/query?query=histogram_quantile(0.95,rate(http_request_duration_seconds_bucket[5m]))" | \
        jq -r '.data.result[0].value[1] // 0' 2>/dev/null || echo "0"
}

get_cpu_usage() {
    # Get average CPU usage across all nodes
    curl -s "$PROMETHEUS_URL/api/v1/query?query=avg(100-rate(node_cpu_seconds_total{mode=\"idle\"}[5m])*100)" | \
        jq -r '.data.result[0].value[1] // 0' 2>/dev/null || echo "0"
}

get_memory_usage() {
    # Get average memory usage
    curl -s "$PROMETHEUS_URL/api/v1/query?query=avg((1-node_memory_MemAvailable_bytes/node_memory_MemTotal_bytes)*100)" | \
        jq -r '.data.result[0].value[1] // 0' 2>/dev/null || echo "0"
}

get_active_connections() {
    # Get active database connections
    curl -s "$PROMETHEUS_URL/api/v1/query?query=pg_stat_database_numbackends" | \
        jq -r '.data.result | map(.value[1] | tonumber) | add // 0' 2>/dev/null || echo "0"
}

# Pre-deployment checks
pre_deployment_checks() {
    info "Running pre-deployment checks..."
    
    # Check if maintenance window is active
    current_hour=$(date +%H)
    if [ $current_hour -ge 9 ] && [ $current_hour -le 17 ]; then
        warning "Deployment during business hours detected!"
        read -p "Are you sure you want to continue? (yes/no): " confirm
        if [ "$confirm" != "yes" ]; then
            error "Deployment cancelled"
            exit 1
        fi
    fi
    
    # Verify deployment permissions
    if [ "$USER" != "deploy" ] && [ "$USER" != "root" ]; then
        warning "Not running as deploy user"
    fi
    
    # Check system resources
    info "Checking system resources..."
    
    # CPU check
    cpu_usage=$(get_cpu_usage)
    if (( $(echo "$cpu_usage > 80" | bc -l) )); then
        error "CPU usage too high: ${cpu_usage}%"
        exit 1
    fi
    
    # Memory check
    memory_usage=$(get_memory_usage)
    if (( $(echo "$memory_usage > 85" | bc -l) )); then
        error "Memory usage too high: ${memory_usage}%"
        exit 1
    fi
    
    # Disk space check
    disk_usage=$(df -h / | awk 'NR==2 {print $5}' | sed 's/%//')
    if [ "$disk_usage" -gt 80 ]; then
        error "Disk usage too high: ${disk_usage}%"
        exit 1
    fi
    
    # Verify backup completed recently
    info "Checking recent backups..."
    last_backup=$(find /backups -name "prod_backup_*.tar.gz" -mtime -1 | head -1)
    if [ -z "$last_backup" ]; then
        error "No recent backup found (less than 24 hours old)"
        exit 1
    fi
    success "Recent backup found: $last_backup"
    
    # Test connectivity to critical services
    info "Testing connectivity..."
    
    services=(
        "Database:5432"
        "Redis:6379"
        "Elasticsearch:9200"
    )
    
    for service in "${services[@]}"; do
        host=$(echo $service | cut -d: -f1)
        port=$(echo $service | cut -d: -f2)
        
        if ! nc -z "$host" "$port" 2>/dev/null; then
            error "Cannot connect to $host on port $port"
            exit 1
        fi
    done
    
    success "All pre-deployment checks passed"
}

# Enable maintenance mode
enable_maintenance_mode() {
    info "Enabling maintenance mode..."
    
    # Update load balancer to serve maintenance page
    kubectl set image deployment/nginx-ingress \
        nginx-ingress=nginx-maintenance:latest \
        --namespace=production \
        --record
    
    # Wait for rollout
    kubectl rollout status deployment/nginx-ingress \
        --namespace=production \
        --timeout=300s
    
    MAINTENANCE_MODE_ENABLED=true
    success "Maintenance mode enabled"
    
    # Verify maintenance page is being served
    response=$(curl -s -o /dev/null -w "%{http_code}" "$API_HEALTH_URL" || echo "000")
    if [ "$response" != "503" ]; then
        warning "Maintenance page may not be properly configured (expected 503, got $response)"
    fi
    
    # Wait for existing requests to complete
    info "Waiting for existing requests to complete..."
    sleep 30
}

# Disable maintenance mode
disable_maintenance_mode() {
    if [ "$MAINTENANCE_MODE_ENABLED" = true ]; then
        info "Disabling maintenance mode..."
        
        # Restore normal load balancer configuration
        kubectl set image deployment/nginx-ingress \
            nginx-ingress=nginx:stable \
            --namespace=production \
            --record
        
        # Wait for rollout
        kubectl rollout status deployment/nginx-ingress \
            --namespace=production \
            --timeout=300s
        
        MAINTENANCE_MODE_ENABLED=false
        success "Maintenance mode disabled"
    fi
}

# Create rollback point
create_rollback_point() {
    info "Creating rollback point..."
    
    ROLLBACK_POINT="rollback_${DEPLOYMENT_ID}"
    
    # Tag current Docker images
    docker tag musicgen-api:current musicgen-api:$ROLLBACK_POINT
    docker tag musicgen-worker:current musicgen-worker:$ROLLBACK_POINT
    
    # Export current Kubernetes configurations
    kubectl get deployments,services,configmaps,secrets \
        --namespace=production \
        -o yaml > "k8s_backup_${ROLLBACK_POINT}.yaml"
    
    # Backup current database schema version
    kubectl exec -it postgres-primary-0 --namespace=production -- \
        pg_dump -s musicgen_prod > "schema_backup_${ROLLBACK_POINT}.sql"
    
    success "Rollback point created: $ROLLBACK_POINT"
}

# Database migration
run_database_migrations() {
    info "Running database migrations..."
    
    # Create migration backup
    progress "Creating pre-migration backup..."
    kubectl exec -it postgres-primary-0 --namespace=production -- \
        pg_dump musicgen_prod | gzip > "pre_migration_backup_${DEPLOYMENT_ID}.sql.gz"
    
    # Run migrations
    progress "Applying database migrations..."
    kubectl run migration-job-${DEPLOYMENT_ID} \
        --image=musicgen-api:latest \
        --namespace=production \
        --restart=Never \
        --command -- python manage.py migrate --no-input
    
    # Wait for migration to complete
    kubectl wait --for=condition=complete \
        job/migration-job-${DEPLOYMENT_ID} \
        --namespace=production \
        --timeout=600s
    
    # Verify migration success
    migration_status=$(kubectl get job migration-job-${DEPLOYMENT_ID} \
        --namespace=production \
        -o jsonpath='{.status.succeeded}')
    
    if [ "$migration_status" != "1" ]; then
        error "Database migration failed"
        return 1
    fi
    
    success "Database migrations completed successfully"
    
    # Cleanup migration job
    kubectl delete job migration-job-${DEPLOYMENT_ID} --namespace=production
}

# Deploy application services
deploy_application() {
    info "Deploying application services..."
    
    # Deploy API services (blue-green deployment)
    progress "Deploying API services..."
    kubectl set image deployment/musicgen-api \
        musicgen-api=musicgen-api:${VERSION:-latest} \
        --namespace=production \
        --record
    
    # Wait for rollout with progress
    kubectl rollout status deployment/musicgen-api \
        --namespace=production \
        --timeout=600s \
        --watch
    
    # Deploy worker services
    progress "Deploying worker services..."
    kubectl set image deployment/musicgen-worker \
        musicgen-worker=musicgen-worker:${VERSION:-latest} \
        --namespace=production \
        --record
    
    kubectl rollout status deployment/musicgen-worker \
        --namespace=production \
        --timeout=600s \
        --watch
    
    # Deploy scheduler services
    progress "Deploying scheduler services..."
    kubectl set image deployment/musicgen-scheduler \
        musicgen-scheduler=musicgen-scheduler:${VERSION:-latest} \
        --namespace=production \
        --record
    
    kubectl rollout status deployment/musicgen-scheduler \
        --namespace=production \
        --timeout=300s \
        --watch
    
    success "All application services deployed"
}

# Health checks
run_health_checks() {
    info "Running health checks..."
    
    local retries=0
    local all_healthy=false
    
    while [ $retries -lt $HEALTH_CHECK_RETRIES ] && [ "$all_healthy" = false ]; do
        all_healthy=true
        
        # Check API health
        progress "Checking API health (attempt $((retries + 1))/$HEALTH_CHECK_RETRIES)..."
        api_health=$(curl -s "$API_HEALTH_URL" | jq -r '.status' 2>/dev/null || echo "unknown")
        
        if [ "$api_health" != "healthy" ]; then
            warning "API health check failed: $api_health"
            all_healthy=false
        else
            success "API is healthy"
        fi
        
        # Check database connectivity
        progress "Checking database connectivity..."
        db_health=$(kubectl exec postgres-primary-0 --namespace=production -- \
            pg_isready -h localhost -U musicgen 2>&1)
        
        if [[ ! "$db_health" =~ "accepting connections" ]]; then
            warning "Database health check failed"
            all_healthy=false
        else
            success "Database is accepting connections"
        fi
        
        # Check Redis connectivity
        progress "Checking Redis connectivity..."
        redis_health=$(kubectl exec redis-master-0 --namespace=production -- \
            redis-cli ping 2>&1)
        
        if [ "$redis_health" != "PONG" ]; then
            warning "Redis health check failed"
            all_healthy=false
        else
            success "Redis is responding"
        fi
        
        # Check worker health
        progress "Checking worker health..."
        worker_ready=$(kubectl get deployment musicgen-worker \
            --namespace=production \
            -o jsonpath='{.status.readyReplicas}')
        worker_desired=$(kubectl get deployment musicgen-worker \
            --namespace=production \
            -o jsonpath='{.spec.replicas}')
        
        if [ "$worker_ready" != "$worker_desired" ]; then
            warning "Worker health check failed: $worker_ready/$worker_desired ready"
            all_healthy=false
        else
            success "All workers are ready"
        fi
        
        if [ "$all_healthy" = false ]; then
            retries=$((retries + 1))
            if [ $retries -lt $HEALTH_CHECK_RETRIES ]; then
                info "Waiting $HEALTH_CHECK_DELAY seconds before retry..."
                sleep $HEALTH_CHECK_DELAY
            fi
        fi
    done
    
    if [ "$all_healthy" = false ]; then
        error "Health checks failed after $HEALTH_CHECK_RETRIES attempts"
        return 1
    fi
    
    success "All health checks passed"
}

# Run smoke tests
run_smoke_tests() {
    info "Running smoke tests..."
    
    # Test authentication
    progress "Testing authentication..."
    auth_test=$(curl -s -X POST "$API_URL/api/v1/auth/test" \
        -H "Content-Type: application/json" \
        -d '{"email":"test@example.com","password":"test"}' \
        -w "\n%{http_code}" | tail -1)
    
    if [ "$auth_test" != "200" ] && [ "$auth_test" != "401" ]; then
        error "Authentication test failed with status: $auth_test"
        return 1
    fi
    success "Authentication endpoint responding correctly"
    
    # Test music generation (small request)
    progress "Testing music generation..."
    gen_response=$(curl -s -X POST "$API_URL/api/v1/generate" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $TEST_API_TOKEN" \
        -d '{"prompt":"test","duration":5}' \
        -w "\n%{http_code}" | tail -1)
    
    if [ "$gen_response" != "200" ] && [ "$gen_response" != "202" ]; then
        error "Generation test failed with status: $gen_response"
        return 1
    fi
    success "Music generation endpoint working"
    
    # Test file operations
    progress "Testing file operations..."
    file_test=$(curl -s "$API_URL/api/v1/files/test" \
        -H "Authorization: Bearer $TEST_API_TOKEN" \
        -w "\n%{http_code}" | tail -1)
    
    if [ "$file_test" != "200" ] && [ "$file_test" != "404" ]; then
        error "File operations test failed with status: $file_test"
        return 1
    fi
    success "File operations endpoint working"
    
    # Test WebSocket connectivity
    progress "Testing WebSocket connectivity..."
    ws_test=$(curl -s -o /dev/null -w "%{http_code}" \
        --header "Connection: Upgrade" \
        --header "Upgrade: websocket" \
        --header "Sec-WebSocket-Version: 13" \
        --header "Sec-WebSocket-Key: test" \
        "$API_URL/ws")
    
    if [ "$ws_test" != "101" ] && [ "$ws_test" != "426" ]; then
        warning "WebSocket test returned unexpected status: $ws_test"
    else
        success "WebSocket endpoint responding"
    fi
    
    success "All smoke tests passed"
}

# Performance validation
validate_performance() {
    info "Validating performance metrics..."
    
    # Collect current metrics
    collect_metrics "performance_validation"
    
    # Check response time
    response_time=$(get_response_time_p95)
    if (( $(echo "$response_time > 2.0" | bc -l) )); then
        warning "Response time p95 above threshold: ${response_time}s > 2.0s"
    else
        success "Response time within SLA: ${response_time}s"
    fi
    
    # Check error rate
    error_rate=$(get_error_rate)
    if (( $(echo "$error_rate > 0.01" | bc -l) )); then
        warning "Error rate above threshold: ${error_rate} > 1%"
    else
        success "Error rate within SLA: ${error_rate}"
    fi
    
    # Check CPU usage
    cpu_usage=$(get_cpu_usage)
    if (( $(echo "$cpu_usage > 70" | bc -l) )); then
        warning "CPU usage high: ${cpu_usage}%"
    else
        success "CPU usage normal: ${cpu_usage}%"
    fi
    
    # Check memory usage
    memory_usage=$(get_memory_usage)
    if (( $(echo "$memory_usage > 80" | bc -l) )); then
        warning "Memory usage high: ${memory_usage}%"
    else
        success "Memory usage normal: ${memory_usage}%"
    fi
    
    # Run quick load test
    progress "Running quick load test..."
    ab -n 1000 -c 10 -H "Authorization: Bearer $TEST_API_TOKEN" \
        "$API_URL/api/v1/health" > load_test_results.txt 2>&1
    
    requests_per_second=$(grep "Requests per second" load_test_results.txt | awk '{print $4}')
    if (( $(echo "$requests_per_second < 100" | bc -l) )); then
        warning "Low throughput detected: ${requests_per_second} req/s"
    else
        success "Throughput acceptable: ${requests_per_second} req/s"
    fi
}

# Enable production monitoring
enable_production_monitoring() {
    info "Enabling production monitoring..."
    
    # Update Prometheus alerts
    progress "Updating Prometheus alert rules..."
    kubectl apply -f monitoring/prometheus-alerts-production.yaml \
        --namespace=monitoring
    
    # Enable detailed logging
    progress "Enabling detailed logging..."
    kubectl set env deployment/musicgen-api \
        LOG_LEVEL=INFO \
        ENABLE_METRICS=true \
        ENABLE_TRACING=true \
        --namespace=production
    
    # Configure Grafana dashboards
    progress "Configuring Grafana dashboards..."
    curl -X POST "$GRAFANA_URL/api/dashboards/db" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $GRAFANA_API_KEY" \
        -d @monitoring/grafana-dashboard-production.json
    
    # Enable APM
    progress "Enabling Application Performance Monitoring..."
    kubectl set env deployment/musicgen-api \
        ELASTIC_APM_ENABLED=true \
        ELASTIC_APM_ENVIRONMENT=production \
        --namespace=production
    
    # Set up log aggregation rules
    progress "Configuring log aggregation..."
    kubectl apply -f monitoring/fluentd-config-production.yaml \
        --namespace=logging
    
    success "Production monitoring enabled"
}

# Rollback procedure
perform_rollback() {
    critical "Initiating rollback procedure..."
    
    # Re-enable maintenance mode if not already enabled
    if [ "$MAINTENANCE_MODE_ENABLED" = false ]; then
        enable_maintenance_mode
    fi
    
    # Rollback application deployments
    info "Rolling back application deployments..."
    
    kubectl rollout undo deployment/musicgen-api --namespace=production
    kubectl rollout undo deployment/musicgen-worker --namespace=production
    kubectl rollout undo deployment/musicgen-scheduler --namespace=production
    
    # Wait for rollback to complete
    kubectl rollout status deployment/musicgen-api --namespace=production --timeout=600s
    kubectl rollout status deployment/musicgen-worker --namespace=production --timeout=600s
    kubectl rollout status deployment/musicgen-scheduler --namespace=production --timeout=300s
    
    # Restore database if migrations were applied
    if [ -f "pre_migration_backup_${DEPLOYMENT_ID}.sql.gz" ]; then
        warning "Restoring database from pre-migration backup..."
        kubectl exec -i postgres-primary-0 --namespace=production -- \
            psql musicgen_prod < <(gunzip -c "pre_migration_backup_${DEPLOYMENT_ID}.sql.gz")
    fi
    
    # Run health checks on rolled back version
    if run_health_checks; then
        success "Rollback completed successfully"
        disable_maintenance_mode
        return 0
    else
        critical "Rollback health checks failed - manual intervention required!"
        return 1
    fi
}

# Monitor deployment
monitor_deployment() {
    local duration=$1
    info "Monitoring deployment for $duration seconds..."
    
    local start_time=$(date +%s)
    local end_time=$((start_time + duration))
    
    while [ $(date +%s) -lt $end_time ]; do
        # Collect metrics
        collect_metrics "monitoring"
        
        # Check error rate
        error_rate=$(get_error_rate)
        if (( $(echo "$error_rate > 0.05" | bc -l) )); then
            critical "Error rate exceeded 5%: ${error_rate}"
            return 1
        fi
        
        # Check response time
        response_time=$(get_response_time_p95)
        if (( $(echo "$response_time > 5.0" | bc -l) )); then
            critical "Response time p95 exceeded 5s: ${response_time}s"
            return 1
        fi
        
        # Check pod restarts
        restarts=$(kubectl get pods --namespace=production \
            -o jsonpath='{range .items[*]}{.status.containerStatuses[*].restartCount}{"\n"}{end}' | \
            awk '{sum+=$1} END {print sum}')
        
        if [ "$restarts" -gt 5 ]; then
            critical "Excessive pod restarts detected: $restarts"
            return 1
        fi
        
        # Display current status
        echo -ne "\r⏱️  Monitoring: Error rate: ${error_rate}%, Response time: ${response_time}s, Time remaining: $((end_time - $(date +%s)))s"
        
        sleep 10
    done
    
    echo # New line after monitoring
    success "Monitoring period completed successfully"
}

# Main deployment function
main() {
    echo "=========================================="
    echo "🚀 PRODUCTION DEPLOYMENT - $DEPLOYMENT_ID"
    echo "=========================================="
    
    # Start deployment timer
    DEPLOYMENT_START_TIME=$(date +%s)
    
    # Send deployment start notification
    send_alert "INFO" "Production deployment started - ID: $DEPLOYMENT_ID"
    
    # Phase 1: Pre-deployment checks
    progress "PHASE 1: Pre-deployment checks"
    if ! pre_deployment_checks; then
        error "Pre-deployment checks failed"
        exit 1
    fi
    
    # Phase 2: Create rollback point
    progress "PHASE 2: Creating rollback point"
    create_rollback_point
    
    # Phase 3: Enable maintenance mode
    progress "PHASE 3: Enabling maintenance mode"
    enable_maintenance_mode
    
    # Phase 4: Database migrations
    progress "PHASE 4: Database migrations"
    if ! run_database_migrations; then
        error "Database migrations failed"
        perform_rollback
        exit 1
    fi
    
    # Phase 5: Deploy application
    progress "PHASE 5: Deploying application"
    if ! deploy_application; then
        error "Application deployment failed"
        perform_rollback
        exit 1
    fi
    
    # Phase 6: Health checks
    progress "PHASE 6: Health checks"
    if ! run_health_checks; then
        error "Health checks failed"
        perform_rollback
        exit 1
    fi
    
    # Phase 7: Smoke tests
    progress "PHASE 7: Smoke tests"
    if ! run_smoke_tests; then
        error "Smoke tests failed"
        perform_rollback
        exit 1
    fi
    
    # Phase 8: Performance validation
    progress "PHASE 8: Performance validation"
    validate_performance
    
    # Phase 9: Enable monitoring
    progress "PHASE 9: Enabling production monitoring"
    enable_production_monitoring
    
    # Phase 10: Disable maintenance mode
    progress "PHASE 10: Disabling maintenance mode"
    disable_maintenance_mode
    
    # Phase 11: Post-deployment monitoring
    progress "PHASE 11: Post-deployment monitoring (5 minutes)"
    if ! monitor_deployment 300; then
        error "Post-deployment monitoring detected issues"
        perform_rollback
        exit 1
    fi
    
    # Calculate deployment duration
    DEPLOYMENT_END_TIME=$(date +%s)
    DEPLOYMENT_DURATION=$((DEPLOYMENT_END_TIME - DEPLOYMENT_START_TIME))
    
    # Final metrics collection
    collect_metrics "deployment_complete"
    
    # Success notification
    success "Production deployment completed successfully!"
    info "Deployment ID: $DEPLOYMENT_ID"
    info "Duration: $((DEPLOYMENT_DURATION / 60)) minutes $((DEPLOYMENT_DURATION % 60)) seconds"
    info "Logs: $LOG_FILE"
    info "Metrics: $METRICS_FILE"
    
    send_alert "SUCCESS" "Production deployment completed - Duration: $((DEPLOYMENT_DURATION / 60))m"
    
    echo "=========================================="
    echo "📋 NEXT STEPS:"
    echo "1. Monitor dashboards closely for 24 hours"
    echo "2. Check user feedback channels"
    echo "3. Review deployment metrics"
    echo "4. Keep team on standby"
    echo "=========================================="
}

# Handle script arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "rollback")
        warning "Manual rollback requested"
        perform_rollback
        ;;
    "status")
        info "Checking deployment status..."
        run_health_checks
        validate_performance
        ;;
    "monitor")
        duration="${2:-3600}"
        monitor_deployment "$duration"
        ;;
    *)
        echo "Usage: $0 [deploy|rollback|status|monitor [duration]]"
        echo "  deploy   - Run full production deployment"
        echo "  rollback - Perform manual rollback"
        echo "  status   - Check current system status"
        echo "  monitor  - Monitor system for specified duration (default: 1 hour)"
        exit 1
        ;;
esac