-- PostgreSQL Staging Database Initialization
-- Creates schema, users, and initial data for staging environment

-- Create staging-specific extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create staging application user with limited privileges
CREATE USER musicgen_app WITH ENCRYPTED PASSWORD 'staging_app_password_change_me';

-- Create database schema
CREATE SCHEMA IF NOT EXISTS musicgen AUTHORIZATION musicgen;
CREATE SCHEMA IF NOT EXISTS monitoring AUTHORIZATION musicgen;
CREATE SCHEMA IF NOT EXISTS audit AUTHORIZATION musicgen;

-- Grant schema permissions
GRANT USAGE ON SCHEMA musicgen TO musicgen_app;
GRANT USAGE ON SCHEMA monitoring TO musicgen_app;
GRANT USAGE ON SCHEMA audit TO musicgen_app;

-- Users table
CREATE TABLE IF NOT EXISTS musicgen.users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    api_key VARCHAR(255) UNIQUE,
    is_active BOOLEAN DEFAULT true,
    is_verified BOOLEAN DEFAULT false,
    subscription_tier VARCHAR(50) DEFAULT 'free',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login TIMESTAMP WITH TIME ZONE,
    login_count INTEGER DEFAULT 0,
    rate_limit_tier VARCHAR(50) DEFAULT 'standard'
);

-- Generation requests table
CREATE TABLE IF NOT EXISTS musicgen.generation_requests (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES musicgen.users(id) ON DELETE CASCADE,
    prompt TEXT NOT NULL,
    duration DECIMAL(5,2) NOT NULL,
    temperature DECIMAL(3,2) DEFAULT 0.8,
    model_name VARCHAR(100) NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    error_message TEXT,
    file_path VARCHAR(500),
    file_size_bytes BIGINT,
    generation_time_seconds DECIMAL(10,3),
    queue_time_seconds DECIMAL(10,3),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB,
    request_ip INET,
    user_agent TEXT
);

-- Audio files metadata table
CREATE TABLE IF NOT EXISTS musicgen.audio_files (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    generation_request_id UUID REFERENCES musicgen.generation_requests(id) ON DELETE CASCADE,
    file_path VARCHAR(500) NOT NULL,
    file_name VARCHAR(255) NOT NULL,
    file_size_bytes BIGINT NOT NULL,
    duration_seconds DECIMAL(10,3) NOT NULL,
    sample_rate INTEGER NOT NULL,
    format VARCHAR(10) NOT NULL,
    checksum VARCHAR(64),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    download_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP WITH TIME ZONE
);

-- Model cache table
CREATE TABLE IF NOT EXISTS musicgen.model_cache (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100) NOT NULL,
    model_path VARCHAR(500) NOT NULL,
    model_size_bytes BIGINT,
    cache_key VARCHAR(255) UNIQUE NOT NULL,
    last_used TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    use_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB
);

-- Task queue table (for tracking Celery tasks)
CREATE TABLE IF NOT EXISTS musicgen.task_queue (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    task_id VARCHAR(255) UNIQUE NOT NULL,
    task_name VARCHAR(255) NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    queue_name VARCHAR(100) NOT NULL,
    priority INTEGER DEFAULT 5,
    args JSONB,
    kwargs JSONB,
    result JSONB,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    worker_id VARCHAR(255),
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3
);

-- API usage tracking table
CREATE TABLE IF NOT EXISTS monitoring.api_usage (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES musicgen.users(id) ON DELETE SET NULL,
    endpoint VARCHAR(255) NOT NULL,
    method VARCHAR(10) NOT NULL,
    status_code INTEGER NOT NULL,
    response_time_ms INTEGER NOT NULL,
    request_size_bytes INTEGER,
    response_size_bytes INTEGER,
    ip_address INET,
    user_agent TEXT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    api_key_used VARCHAR(255),
    rate_limited BOOLEAN DEFAULT FALSE
);

-- System metrics table
CREATE TABLE IF NOT EXISTS monitoring.system_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(255) NOT NULL,
    metric_value DECIMAL(15,6) NOT NULL,
    metric_type VARCHAR(50) NOT NULL, -- counter, gauge, histogram
    labels JSONB,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    instance VARCHAR(255) NOT NULL
);

-- Error logs table
CREATE TABLE IF NOT EXISTS monitoring.error_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    level VARCHAR(20) NOT NULL,
    message TEXT NOT NULL,
    module VARCHAR(255),
    function_name VARCHAR(255),
    line_number INTEGER,
    stack_trace TEXT,
    user_id UUID REFERENCES musicgen.users(id) ON DELETE SET NULL,
    request_id VARCHAR(255),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    additional_context JSONB
);

-- Audit logs table
CREATE TABLE IF NOT EXISTS audit.audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES musicgen.users(id) ON DELETE SET NULL,
    action VARCHAR(255) NOT NULL,
    resource_type VARCHAR(100) NOT NULL,
    resource_id VARCHAR(255),
    old_values JSONB,
    new_values JSONB,
    ip_address INET,
    user_agent TEXT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_users_email ON musicgen.users(email);
CREATE INDEX IF NOT EXISTS idx_users_api_key ON musicgen.users(api_key);
CREATE INDEX IF NOT EXISTS idx_users_created_at ON musicgen.users(created_at);

CREATE INDEX IF NOT EXISTS idx_generation_requests_user_id ON musicgen.generation_requests(user_id);
CREATE INDEX IF NOT EXISTS idx_generation_requests_status ON musicgen.generation_requests(status);
CREATE INDEX IF NOT EXISTS idx_generation_requests_created_at ON musicgen.generation_requests(created_at);
CREATE INDEX IF NOT EXISTS idx_generation_requests_model_name ON musicgen.generation_requests(model_name);

CREATE INDEX IF NOT EXISTS idx_audio_files_generation_request_id ON musicgen.audio_files(generation_request_id);
CREATE INDEX IF NOT EXISTS idx_audio_files_expires_at ON musicgen.audio_files(expires_at);
CREATE INDEX IF NOT EXISTS idx_audio_files_created_at ON musicgen.audio_files(created_at);

CREATE INDEX IF NOT EXISTS idx_model_cache_model_name ON musicgen.model_cache(model_name);
CREATE INDEX IF NOT EXISTS idx_model_cache_cache_key ON musicgen.model_cache(cache_key);
CREATE INDEX IF NOT EXISTS idx_model_cache_last_used ON musicgen.model_cache(last_used);

CREATE INDEX IF NOT EXISTS idx_task_queue_task_id ON musicgen.task_queue(task_id);
CREATE INDEX IF NOT EXISTS idx_task_queue_status ON musicgen.task_queue(status);
CREATE INDEX IF NOT EXISTS idx_task_queue_queue_name ON musicgen.task_queue(queue_name);
CREATE INDEX IF NOT EXISTS idx_task_queue_created_at ON musicgen.task_queue(created_at);

CREATE INDEX IF NOT EXISTS idx_api_usage_user_id ON monitoring.api_usage(user_id);
CREATE INDEX IF NOT EXISTS idx_api_usage_endpoint ON monitoring.api_usage(endpoint);
CREATE INDEX IF NOT EXISTS idx_api_usage_timestamp ON monitoring.api_usage(timestamp);
CREATE INDEX IF NOT EXISTS idx_api_usage_status_code ON monitoring.api_usage(status_code);

CREATE INDEX IF NOT EXISTS idx_system_metrics_metric_name ON monitoring.system_metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp ON monitoring.system_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_system_metrics_instance ON monitoring.system_metrics(instance);

CREATE INDEX IF NOT EXISTS idx_error_logs_level ON monitoring.error_logs(level);
CREATE INDEX IF NOT EXISTS idx_error_logs_timestamp ON monitoring.error_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_error_logs_module ON monitoring.error_logs(module);

CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON audit.audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_action ON audit.audit_logs(action);
CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON audit.audit_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_logs_resource_type ON audit.audit_logs(resource_type);

-- Grant table permissions to application user
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA musicgen TO musicgen_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA monitoring TO musicgen_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA audit TO musicgen_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA musicgen TO musicgen_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA monitoring TO musicgen_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA audit TO musicgen_app;

-- Create staging test data
INSERT INTO musicgen.users (email, username, password_hash, api_key, subscription_tier) VALUES
('admin@staging.com', 'staging_admin', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewmDBgWDhBBXhHpu', 'staging_api_key_change_me', 'premium'),
('test@staging.com', 'staging_test', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewmDBgWDhBBXhHpu', 'staging_test_key', 'standard'),
('load@staging.com', 'staging_load', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewmDBgWDhBBXhHpu', 'staging_load_key', 'free')
ON CONFLICT (email) DO NOTHING;

-- Create functions for cleanup and maintenance
CREATE OR REPLACE FUNCTION musicgen.cleanup_expired_files()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM musicgen.audio_files 
    WHERE expires_at < NOW();
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    INSERT INTO monitoring.system_metrics (metric_name, metric_value, metric_type, instance)
    VALUES ('files_cleaned_up', deleted_count, 'counter', 'database_cleanup');
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for updated_at timestamp
CREATE OR REPLACE FUNCTION musicgen.update_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_users_timestamp
    BEFORE UPDATE ON musicgen.users
    FOR EACH ROW
    EXECUTE FUNCTION musicgen.update_timestamp();

-- Create view for user statistics
CREATE OR REPLACE VIEW musicgen.user_stats AS
SELECT 
    u.id,
    u.email,
    u.username,
    u.subscription_tier,
    u.created_at,
    COUNT(gr.id) as total_generations,
    COALESCE(SUM(gr.generation_time_seconds), 0) as total_generation_time,
    COALESCE(AVG(gr.generation_time_seconds), 0) as avg_generation_time,
    MAX(gr.created_at) as last_generation
FROM musicgen.users u
LEFT JOIN musicgen.generation_requests gr ON u.id = gr.user_id
GROUP BY u.id, u.email, u.username, u.subscription_tier, u.created_at;

-- Create view for system health
CREATE OR REPLACE VIEW monitoring.system_health AS
SELECT 
    COUNT(CASE WHEN status = 'pending' THEN 1 END) as pending_tasks,
    COUNT(CASE WHEN status = 'processing' THEN 1 END) as processing_tasks,
    COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_tasks,
    COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_tasks,
    AVG(CASE WHEN completed_at IS NOT NULL THEN 
        EXTRACT(EPOCH FROM (completed_at - started_at)) END) as avg_processing_time
FROM musicgen.task_queue
WHERE created_at > NOW() - INTERVAL '1 hour';

-- Grant view permissions
GRANT SELECT ON musicgen.user_stats TO musicgen_app;
GRANT SELECT ON monitoring.system_health TO musicgen_app;

-- Create staging-specific configuration table
CREATE TABLE IF NOT EXISTS musicgen.staging_config (
    key VARCHAR(255) PRIMARY KEY,
    value TEXT NOT NULL,
    description TEXT,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

INSERT INTO musicgen.staging_config (key, value, description) VALUES
('load_testing_enabled', 'true', 'Enable load testing features'),
('max_concurrent_generations', '10', 'Maximum concurrent generation requests'),
('file_retention_days', '7', 'Days to retain generated files'),
('rate_limit_requests_per_minute', '100', 'API rate limit per user per minute'),
('enable_debug_logging', 'true', 'Enable debug level logging'),
('monitoring_interval_seconds', '30', 'System monitoring collection interval')
ON CONFLICT (key) DO UPDATE SET 
    value = EXCLUDED.value,
    updated_at = NOW();

GRANT SELECT, UPDATE ON musicgen.staging_config TO musicgen_app;

-- Create materialized view for performance metrics
CREATE MATERIALIZED VIEW IF NOT EXISTS monitoring.hourly_metrics AS
SELECT 
    DATE_TRUNC('hour', timestamp) as hour,
    endpoint,
    COUNT(*) as request_count,
    AVG(response_time_ms) as avg_response_time,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY response_time_ms) as p95_response_time,
    COUNT(*) FILTER (WHERE status_code >= 500) as error_count,
    COUNT(DISTINCT user_id) as unique_users
FROM monitoring.api_usage
GROUP BY DATE_TRUNC('hour', timestamp), endpoint;

CREATE UNIQUE INDEX IF NOT EXISTS idx_hourly_metrics_hour_endpoint 
ON monitoring.hourly_metrics(hour, endpoint);

GRANT SELECT ON monitoring.hourly_metrics TO musicgen_app;

-- Create automatic refresh for materialized view
CREATE OR REPLACE FUNCTION monitoring.refresh_hourly_metrics()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY monitoring.hourly_metrics;
END;
$$ LANGUAGE plpgsql;

-- Log successful initialization
INSERT INTO monitoring.system_metrics (metric_name, metric_value, metric_type, instance)
VALUES ('database_initialized', 1, 'gauge', 'staging_init');

COMMIT;