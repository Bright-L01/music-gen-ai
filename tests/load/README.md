# Music Gen AI Load Testing Suite

Comprehensive load testing framework for the Music Gen AI system, including API performance testing, WebSocket streaming tests, database connection pooling analysis, Redis caching performance, and bottleneck identification.

## 🚀 Quick Start

```bash
# Install dependencies
pip install locust websockets aiohttp psutil

# Run basic load test
locust -f tests/load/locustfile.py --host=http://localhost:8000 -u 50 -r 5 -t 300s

# Run all load tests
cd tests/load
python run_all_tests.py
```

## 📁 Test Suite Components

### 1. Locust Load Testing (`locustfile.py`)
**Primary API load testing with multiple user types:**

- **MusicGenerationUser**: Tests music generation endpoints
  - Single and batch generation requests
  - Task status polling
  - File downloads
  - Authentication flows

- **WebSocketStreamingUser**: Tests real-time streaming
  - Streaming session creation
  - WebSocket connections
  - Session management

- **DatabaseStressUser**: Tests database performance
  - Heavy read/write operations
  - Connection pool stress testing
  - Rapid query execution

- **RedisStressUser**: Tests caching layer
  - Rate limiting stress tests
  - Session storage operations
  - CSRF token generation

**Usage:**
```bash
# Basic test (50 users, 5 per second spawn rate, 5 minutes)
locust -f locustfile.py --host=http://localhost:8000 -u 50 -r 5 -t 300s

# Heavy load test
locust -f locustfile.py --host=http://localhost:8000 -u 200 -r 20 -t 600s

# Component-specific tests
locust -f locustfile.py --host=http://localhost:8000 --tags websocket
locust -f locustfile.py --host=http://localhost:8000 --tags database
```

### 2. WebSocket Load Testing (`websocket_load_test.py`)
**Specialized WebSocket streaming performance tests:**

- Connection lifecycle testing
- Concurrent connection limits
- Message throughput analysis
- Reconnection behavior
- Connection failure handling

**Usage:**
```bash
python websocket_load_test.py
```

**Features:**
- Tests up to 100 concurrent WebSocket connections
- Measures connection establishment time
- Tracks message throughput and latency
- Identifies connection limits and failures
- Generates detailed streaming performance report

### 3. Database Pool Testing (`database_pool_test.py`)
**Database connection pool optimization and stress testing:**

- Pool size optimization
- Connection lifecycle management
- Concurrent scaling analysis
- Query performance under load
- Pool exhaustion testing

**Usage:**
```bash
python database_pool_test.py
```

**Test Scenarios:**
- Optimal pool size determination (5-50 connections)
- Connection lifecycle with variable load
- Concurrent scaling (10-100 users)
- Query patterns (read-heavy, write-heavy, mixed)

### 4. Redis Pool Testing (`redis_pool_test.py`)
**Redis connection pool and caching performance analysis:**

- Cache performance optimization
- Rate limiting effectiveness
- Session management performance
- Memory usage scaling
- Burst load handling

**Usage:**
```bash
python redis_pool_test.py
```

**Test Areas:**
- Cache hit rate optimization
- Rate limiting under stress
- Session storage performance
- Memory usage patterns
- Connection pool efficiency

### 5. Performance Monitoring (`performance_monitor.py`)
**Real-time performance metrics collection:**

- Response time tracking
- Resource utilization monitoring
- Error rate analysis
- Throughput measurements
- Bottleneck detection

**Features:**
- System resource monitoring (CPU, memory, disk, network)
- Application metrics tracking
- Concurrent request monitoring
- Automated bottleneck identification

### 6. Metrics Collection (`metrics_collector.py`)
**Baseline performance metrics establishment:**

- Performance threshold definition
- Baseline statistics calculation
- System health monitoring
- Capacity planning metrics
- SLA recommendation generation

### 7. Bottleneck Analysis (`bottleneck_analyzer.py`)
**Comprehensive performance bottleneck identification:**

- Cross-component analysis
- Performance issue prioritization
- Optimization recommendations
- Capacity planning insights
- Production readiness assessment

**Usage:**
```bash
python bottleneck_analyzer.py
```

### 8. Baseline Metrics Report (`baseline_metrics_report.py`)
**Performance baseline establishment and monitoring threshold generation:**

- Industry standard benchmarking
- SLA recommendations
- Monitoring threshold calculation
- Production readiness assessment
- Capacity planning metrics

**Usage:**
```bash
python baseline_metrics_report.py
```

## 📊 Generated Reports

### Performance Reports
- `performance_report.json` - Comprehensive Locust test results
- `websocket_load_report.json` - WebSocket performance analysis
- `database_pool_report.json` - Database optimization results
- `redis_pool_report.json` - Redis performance analysis
- `bottleneck_analysis_report.json` - System bottleneck identification
- `baseline_metrics_report.json` - Performance baselines and SLA recommendations

### Quick Reference
- `baseline_summary.json` - Executive summary of performance
- `load_test_report.html` - Locust HTML report (if generated)

## 🎯 Key Performance Metrics

### API Performance
- **Response Time**: P50, P90, P95, P99 percentiles
- **Throughput**: Requests per second capacity
- **Error Rate**: Percentage of failed requests
- **Concurrent Capacity**: Maximum concurrent users

### WebSocket Performance
- **Connection Time**: WebSocket establishment latency
- **Success Rate**: Connection success percentage
- **Message Throughput**: Messages per second
- **Concurrent Connections**: Maximum simultaneous connections

### Database Performance
- **Query Latency**: Average and percentile query times
- **Throughput**: Queries per second capacity
- **Connection Pool**: Optimal pool size and utilization
- **Success Rate**: Query success percentage

### Redis Performance
- **Operation Latency**: Cache operation response times
- **Cache Hit Rate**: Percentage of successful cache hits
- **Throughput**: Commands per second capacity
- **Memory Usage**: Cache memory utilization

### System Resources
- **CPU Utilization**: Processor usage under load
- **Memory Usage**: RAM consumption patterns
- **Disk I/O**: Read/write performance
- **Network I/O**: Network bandwidth utilization

## 🔧 Configuration

### Environment Variables
```bash
# API Configuration
export API_HOST=http://localhost:8000
export API_TIMEOUT=30

# Load Test Configuration
export MAX_USERS=100
export SPAWN_RATE=10
export TEST_DURATION=300

# Database Configuration
export DB_POOL_MIN=5
export DB_POOL_MAX=20

# Redis Configuration
export REDIS_POOL_MIN=5
export REDIS_POOL_MAX=50
```

### Customizing Tests

#### Locust Configuration
Edit `locustfile.py` to modify:
- User behavior patterns
- Wait times between requests
- Task weights and priorities
- Authentication patterns

#### WebSocket Testing
Modify `websocket_load_test.py` for:
- Connection patterns
- Message sizes
- Streaming duration
- Concurrency levels

#### Database Testing
Adjust `database_pool_test.py` for:
- Pool size ranges
- Query patterns
- Load scenarios
- Test duration

## 📈 Performance Thresholds

### Production Readiness Criteria
- **API Response Time P95**: < 2000ms
- **API Error Rate**: < 1%
- **Database Query P90**: < 500ms
- **Redis Operation P90**: < 10ms
- **Cache Hit Rate**: > 80%
- **WebSocket Success Rate**: > 95%

### Warning Thresholds
- **API Response Time P95**: 2000ms
- **API Error Rate**: 5%
- **CPU Usage**: 70%
- **Memory Usage**: 80%
- **Database Pool Usage**: 70%

### Critical Thresholds
- **API Response Time P95**: 5000ms
- **API Error Rate**: 15%
- **CPU Usage**: 90%
- **Memory Usage**: 95%
- **Database Pool Usage**: 90%

## 🚨 Troubleshooting

### Common Issues

#### High Response Times
```bash
# Check system resources
python performance_monitor.py

# Analyze bottlenecks
python bottleneck_analyzer.py

# Review database performance
python database_pool_test.py
```

#### Connection Failures
```bash
# Test WebSocket connections
python websocket_load_test.py

# Check connection pools
python database_pool_test.py
python redis_pool_test.py
```

#### Memory Issues
```bash
# Monitor system resources
python metrics_collector.py

# Check Redis memory usage
python redis_pool_test.py
```

### Performance Optimization

#### API Performance
1. **Enable Response Caching**
   ```python
   # Add caching middleware
   @cache(expire=300)
   def expensive_endpoint():
       pass
   ```

2. **Optimize Database Queries**
   ```sql
   -- Add indexes for frequently queried columns
   CREATE INDEX idx_tasks_status ON tasks(status);
   CREATE INDEX idx_tasks_user_id ON tasks(user_id);
   ```

3. **Implement Connection Pooling**
   ```python
   # Optimize pool configuration
   DATABASE_POOL_SIZE = 20
   DATABASE_MAX_OVERFLOW = 10
   ```

#### WebSocket Performance
1. **Connection Keep-Alive**
   ```javascript
   // Implement ping/pong for connection health
   websocket.ping();
   ```

2. **Message Batching**
   ```python
   # Batch multiple messages
   await websocket.send_batch(messages)
   ```

### Scaling Recommendations

#### Horizontal Scaling
- **API Servers**: Scale to 3+ instances with load balancer
- **Database**: Implement read replicas
- **Redis**: Set up Redis cluster
- **WebSocket**: Use WebSocket clustering

#### Vertical Scaling
- **CPU**: Upgrade to higher core count
- **Memory**: Increase RAM for caching
- **Storage**: Use SSD for database storage

## 📋 Monitoring Setup

### Production Monitoring
```bash
# Set up continuous monitoring
python metrics_collector.py --continuous

# Configure alerts
python setup_alerts.py --config production_thresholds.json
```

### Dashboard Creation
```bash
# Generate performance dashboard
python create_dashboard.py --type grafana
python create_dashboard.py --type datadog
```

## 🔄 CI/CD Integration

### GitHub Actions Example
```yaml
name: Performance Testing
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  performance-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run Load Tests
        run: |
          pip install locust
          locust -f tests/load/locustfile.py --headless -u 50 -r 5 -t 60s
      - name: Performance Analysis
        run: python tests/load/bottleneck_analyzer.py
```

## 🎯 Best Practices

### Load Testing
1. **Start Small**: Begin with low user counts
2. **Gradual Ramp-up**: Increase load gradually
3. **Monitor Resources**: Watch CPU, memory, and I/O
4. **Test Realistic Scenarios**: Use production-like data
5. **Regular Testing**: Include in CI/CD pipeline

### Performance Optimization
1. **Profile First**: Identify bottlenecks before optimizing
2. **Cache Strategically**: Cache expensive operations
3. **Optimize Queries**: Use indexes and query optimization
4. **Scale Appropriately**: Choose horizontal vs vertical scaling
5. **Monitor Continuously**: Set up real-time monitoring

## 📚 Additional Resources

- [Locust Documentation](https://docs.locust.io/)
- [WebSocket Performance Testing](https://github.com/websockets/websockets)
- [Database Performance Tuning](https://use-the-index-luke.com/)
- [Redis Performance Guide](https://redis.io/topics/latency)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your load tests
4. Include documentation
5. Submit a pull request

For questions or issues, please open a GitHub issue or contact the performance engineering team.