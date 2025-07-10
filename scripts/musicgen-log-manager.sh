#!/bin/bash

# Music Gen AI Log Management Script
# Handles log cleanup, archival, monitoring, and health checks
# Follows 2024 best practices for production log management

set -euo pipefail

# Configuration from environment variables
LOG_DIR="${LOG_DIR:-/var/log/musicgen}"
ARCHIVE_DIR="${ARCHIVE_DIR:-/var/archive/musicgen-logs}"
MAX_LOG_AGE_DAYS="${MAX_LOG_AGE_DAYS:-90}"
MAX_ARCHIVE_AGE_DAYS="${MAX_ARCHIVE_AGE_DAYS:-2555}"
DISK_USAGE_THRESHOLD="${DISK_USAGE_THRESHOLD:-85}"
MONITORING_WEBHOOK="${MONITORING_WEBHOOK:-}"
ALERT_WEBHOOK="${ALERT_WEBHOOK:-}"
SERVICE_NAME="${SERVICE_NAME:-musicgen-api}"
ENVIRONMENT="${ENVIRONMENT:-production}"

# Logging functions
log_info() {
    echo "$(date -Iseconds) [INFO] $*" | tee -a "${LOG_DIR}/management.log"
}

log_error() {
    echo "$(date -Iseconds) [ERROR] $*" | tee -a "${LOG_DIR}/management.log" >&2
}

log_warn() {
    echo "$(date -Iseconds) [WARN] $*" | tee -a "${LOG_DIR}/management.log"
}

# Send alert function
send_alert() {
    local severity="$1"
    local message="$2"
    local details="${3:-}"
    
    if [[ -n "$ALERT_WEBHOOK" ]]; then
        curl -s -X POST "$ALERT_WEBHOOK" \
            -H "Content-Type: application/json" \
            -d "{
                \"service\": \"$SERVICE_NAME\",
                \"environment\": \"$ENVIRONMENT\",
                \"severity\": \"$severity\",
                \"message\": \"$message\",
                \"details\": \"$details\",
                \"timestamp\": \"$(date -Iseconds)\",
                \"hostname\": \"$(hostname)\"
            }" || log_error "Failed to send alert: $message"
    fi
}

# Check disk usage
check_disk_usage() {
    log_info "Checking disk usage..."
    
    local usage
    usage=$(df "$LOG_DIR" | awk 'NR==2 {print $5}' | sed 's/%//')
    
    log_info "Disk usage: ${usage}%"
    
    if [[ $usage -ge $DISK_USAGE_THRESHOLD ]]; then
        log_error "Disk usage ${usage}% exceeds threshold ${DISK_USAGE_THRESHOLD}%"
        send_alert "critical" "Log disk usage critical" "Usage: ${usage}%, Threshold: ${DISK_USAGE_THRESHOLD}%"
        return 1
    elif [[ $usage -ge $((DISK_USAGE_THRESHOLD - 10)) ]]; then
        log_warn "Disk usage ${usage}% approaching threshold ${DISK_USAGE_THRESHOLD}%"
        send_alert "warning" "Log disk usage warning" "Usage: ${usage}%, Threshold: ${DISK_USAGE_THRESHOLD}%"
    fi
    
    return 0
}

# Clean old log files
cleanup_old_logs() {
    log_info "Cleaning up old log files..."
    
    local cleaned_count=0
    local cleaned_size=0
    
    # Clean application logs older than MAX_LOG_AGE_DAYS
    while IFS= read -r -d '' file; do
        if [[ -f "$file" ]]; then
            local size
            size=$(stat -c%s "$file" 2>/dev/null || echo 0)
            cleaned_size=$((cleaned_size + size))
            cleaned_count=$((cleaned_count + 1))
            log_info "Removing old log file: $file ($(numfmt --to=iec $size))"
            rm -f "$file"
        fi
    done < <(find "$LOG_DIR" -name "*.log.*" -type f -mtime "+$MAX_LOG_AGE_DAYS" -print0 2>/dev/null || true)
    
    # Clean compressed logs
    while IFS= read -r -d '' file; do
        if [[ -f "$file" ]]; then
            local size
            size=$(stat -c%s "$file" 2>/dev/null || echo 0)
            cleaned_size=$((cleaned_size + size))
            cleaned_count=$((cleaned_count + 1))
            log_info "Removing old compressed log: $file ($(numfmt --to=iec $size))"
            rm -f "$file"
        fi
    done < <(find "$LOG_DIR" -name "*.gz" -type f -mtime "+$MAX_LOG_AGE_DAYS" -print0 2>/dev/null || true)
    
    if [[ $cleaned_count -gt 0 ]]; then
        log_info "Cleaned $cleaned_count old log files, freed $(numfmt --to=iec $cleaned_size)"
        send_alert "info" "Log cleanup completed" "Files: $cleaned_count, Size freed: $(numfmt --to=iec $cleaned_size)"
    fi
}

# Archive logs to long-term storage
archive_logs() {
    log_info "Archiving logs to long-term storage..."
    
    # Create archive directory structure
    mkdir -p "$ARCHIVE_DIR/$(date +%Y/%m)"
    
    local archived_count=0
    local archived_size=0
    
    # Archive logs older than 7 days but newer than MAX_LOG_AGE_DAYS
    while IFS= read -r -d '' file; do
        if [[ -f "$file" && "$file" == *.gz ]]; then
            local basename
            basename=$(basename "$file")
            local archive_path="$ARCHIVE_DIR/$(date +%Y/%m)/$basename"
            
            if [[ ! -f "$archive_path" ]]; then
                local size
                size=$(stat -c%s "$file" 2>/dev/null || echo 0)
                archived_size=$((archived_size + size))
                archived_count=$((archived_count + 1))
                
                log_info "Archiving log file: $file -> $archive_path"
                cp "$file" "$archive_path"
                
                # Verify archive integrity
                if gzip -t "$archive_path" 2>/dev/null; then
                    rm -f "$file"
                else
                    log_error "Archive verification failed for $archive_path"
                    rm -f "$archive_path"
                fi
            fi
        fi
    done < <(find "$LOG_DIR" -name "*.log.*.gz" -type f -mtime +7 -mtime -"$MAX_LOG_AGE_DAYS" -print0 2>/dev/null || true)
    
    if [[ $archived_count -gt 0 ]]; then
        log_info "Archived $archived_count log files, total size $(numfmt --to=iec $archived_size)"
    fi
}

# Clean old archives
cleanup_old_archives() {
    log_info "Cleaning up old archives..."
    
    local cleaned_count=0
    local cleaned_size=0
    
    # Clean archives older than MAX_ARCHIVE_AGE_DAYS
    while IFS= read -r -d '' file; do
        if [[ -f "$file" ]]; then
            local size
            size=$(stat -c%s "$file" 2>/dev/null || echo 0)
            cleaned_size=$((cleaned_size + size))
            cleaned_count=$((cleaned_count + 1))
            log_info "Removing old archive: $file ($(numfmt --to=iec $size))"
            rm -f "$file"
        fi
    done < <(find "$ARCHIVE_DIR" -name "*.gz" -type f -mtime "+$MAX_ARCHIVE_AGE_DAYS" -print0 2>/dev/null || true)
    
    # Remove empty archive directories
    find "$ARCHIVE_DIR" -type d -empty -delete 2>/dev/null || true
    
    if [[ $cleaned_count -gt 0 ]]; then
        log_info "Cleaned $cleaned_count old archives, freed $(numfmt --to=iec $cleaned_size)"
    fi
}

# Check log file health
check_log_health() {
    log_info "Checking log file health..."
    
    local issues=0
    
    # Check if log files exist and are writable
    for logfile in app.log audit.log performance.log error.log; do
        local filepath="$LOG_DIR/$logfile"
        
        if [[ ! -f "$filepath" ]]; then
            log_warn "Log file does not exist: $filepath"
            touch "$filepath"
            chown musicgen:musicgen "$filepath" 2>/dev/null || true
            chmod 644 "$filepath"
        fi
        
        if [[ ! -w "$filepath" ]]; then
            log_error "Log file not writable: $filepath"
            issues=$((issues + 1))
        fi
        
        # Check file size
        local size
        size=$(stat -c%s "$filepath" 2>/dev/null || echo 0)
        if [[ $size -gt 1073741824 ]]; then  # 1GB
            log_warn "Log file is large ($(numfmt --to=iec $size)): $filepath"
            send_alert "warning" "Large log file detected" "File: $filepath, Size: $(numfmt --to=iec $size)"
        fi
    done
    
    return $issues
}

# Monitor log activity
monitor_log_activity() {
    log_info "Monitoring log activity..."
    
    local activity_file="$LOG_DIR/.last_activity"
    local current_time
    current_time=$(date +%s)
    
    # Check if logs have been written recently (last 10 minutes)
    local recent_activity=false
    for logfile in app.log audit.log performance.log error.log; do
        local filepath="$LOG_DIR/$logfile"
        if [[ -f "$filepath" ]]; then
            local mtime
            mtime=$(stat -c%Y "$filepath" 2>/dev/null || echo 0)
            if [[ $((current_time - mtime)) -lt 600 ]]; then  # 10 minutes
                recent_activity=true
                break
            fi
        fi
    done
    
    if [[ "$recent_activity" == "false" ]]; then
        if [[ -f "$activity_file" ]]; then
            local last_alert
            last_alert=$(cat "$activity_file" 2>/dev/null || echo 0)
            if [[ $((current_time - last_alert)) -gt 3600 ]]; then  # 1 hour
                log_warn "No recent log activity detected"
                send_alert "warning" "No recent log activity" "No logs written in the last 10 minutes"
                echo "$current_time" > "$activity_file"
            fi
        else
            echo "$current_time" > "$activity_file"
        fi
    else
        rm -f "$activity_file"
    fi
}

# Generate log statistics
generate_log_stats() {
    log_info "Generating log statistics..."
    
    local stats_file="$LOG_DIR/stats.json"
    local temp_stats
    temp_stats=$(mktemp)
    
    {
        echo "{"
        echo "  \"timestamp\": \"$(date -Iseconds)\","
        echo "  \"hostname\": \"$(hostname)\","
        echo "  \"log_directory\": \"$LOG_DIR\","
        echo "  \"files\": {"
        
        local first_file=true
        for logfile in app.log audit.log performance.log error.log; do
            local filepath="$LOG_DIR/$logfile"
            
            if [[ "$first_file" == "false" ]]; then
                echo ","
            fi
            first_file=false
            
            echo -n "    \"$logfile\": {"
            
            if [[ -f "$filepath" ]]; then
                local size
                local lines
                local mtime
                size=$(stat -c%s "$filepath" 2>/dev/null || echo 0)
                lines=$(wc -l < "$filepath" 2>/dev/null || echo 0)
                mtime=$(stat -c%Y "$filepath" 2>/dev/null || echo 0)
                
                echo "\"exists\": true,"
                echo "      \"size_bytes\": $size,"
                echo "      \"size_human\": \"$(numfmt --to=iec $size)\","
                echo "      \"lines\": $lines,"
                echo "      \"last_modified\": \"$(date -d @$mtime -Iseconds 2>/dev/null || echo 'unknown')\""
            else
                echo "\"exists\": false"
            fi
            
            echo -n "    }"
        done
        
        echo ""
        echo "  },"
        
        # Disk usage
        local usage
        usage=$(df "$LOG_DIR" | awk 'NR==2 {print $5}' | sed 's/%//')
        echo "  \"disk_usage_percent\": $usage,"
        
        # Archive statistics
        local archive_count=0
        local archive_size=0
        if [[ -d "$ARCHIVE_DIR" ]]; then
            while IFS= read -r -d '' file; do
                if [[ -f "$file" ]]; then
                    archive_count=$((archive_count + 1))
                    local size
                    size=$(stat -c%s "$file" 2>/dev/null || echo 0)
                    archive_size=$((archive_size + size))
                fi
            done < <(find "$ARCHIVE_DIR" -name "*.gz" -type f -print0 2>/dev/null || true)
        fi
        
        echo "  \"archive\": {"
        echo "    \"count\": $archive_count,"
        echo "    \"size_bytes\": $archive_size,"
        echo "    \"size_human\": \"$(numfmt --to=iec $archive_size)\""
        echo "  }"
        echo "}"
    } > "$temp_stats"
    
    mv "$temp_stats" "$stats_file"
    chmod 644 "$stats_file"
    
    log_info "Log statistics updated: $stats_file"
}

# Send health report
send_health_report() {
    if [[ -n "$MONITORING_WEBHOOK" ]]; then
        local stats_file="$LOG_DIR/stats.json"
        if [[ -f "$stats_file" ]]; then
            curl -s -X POST "$MONITORING_WEBHOOK" \
                -H "Content-Type: application/json" \
                --data-binary "@$stats_file" || log_error "Failed to send health report"
        fi
    fi
}

# Main execution
main() {
    log_info "Starting log management cycle..."
    
    # Create necessary directories
    mkdir -p "$LOG_DIR" "$ARCHIVE_DIR"
    
    # Check permissions
    if [[ ! -w "$LOG_DIR" ]]; then
        log_error "Log directory not writable: $LOG_DIR"
        exit 1
    fi
    
    # Run health checks
    if ! check_log_health; then
        log_error "Log health check failed"
        send_alert "error" "Log health check failed" "See management.log for details"
    fi
    
    # Monitor activity
    monitor_log_activity
    
    # Check disk usage
    if ! check_disk_usage; then
        log_error "Disk usage check failed"
        # Continue with cleanup to try to free space
    fi
    
    # Perform maintenance
    archive_logs
    cleanup_old_logs
    cleanup_old_archives
    
    # Generate statistics
    generate_log_stats
    
    # Send health report
    send_health_report
    
    log_info "Log management cycle completed successfully"
}

# Handle signals
trap 'log_error "Log management interrupted"; exit 130' INT TERM

# Run main function
main "$@"