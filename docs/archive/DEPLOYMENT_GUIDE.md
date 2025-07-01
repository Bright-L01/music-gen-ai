# 🚀 MusicGen AI Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying MusicGen AI in production environments, covering Docker containerization, Kubernetes orchestration, cloud deployment, and monitoring setup.

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Docker Deployment](#docker-deployment)
3. [Kubernetes Deployment](#kubernetes-deployment)
4. [Cloud Deployment](#cloud-deployment)
5. [Monitoring & Observability](#monitoring--observability)
6. [Security Configuration](#security-configuration)
7. [Performance Tuning](#performance-tuning)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements
- **CPU**: 8+ cores recommended
- **RAM**: 32GB minimum
- **GPU**: NVIDIA GPU with 8GB+ VRAM (24GB recommended)
- **Storage**: 100GB+ SSD
- **OS**: Ubuntu 20.04+ or similar

### Software Requirements
```bash
# Check versions
docker --version  # 20.10+
kubectl version   # 1.21+
python --version  # 3.8+
nvidia-smi       # CUDA 11.8+
```

---

## 🐳 Docker Deployment

### 1. Production Dockerfile

```dockerfile
# Dockerfile.prod
FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install package
RUN pip3 install --no-cache-dir -e .

# Create non-root user
RUN useradd -m -u 1000 musicgen && \
    chown -R musicgen:musicgen /app

USER musicgen

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV MODEL_CACHE_DIR=/app/models
ENV LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:8000/health')"

# Expose ports
EXPOSE 8000

# Start command
CMD ["uvicorn", "music_gen.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

### 2. Docker Compose Configuration

```yaml
# docker-compose.yml
version: '3.8'

services:
  musicgen-api:
    build:
      context: .
      dockerfile: Dockerfile.prod
    image: musicgen:latest
    container_name: musicgen-api
    ports:
      - "8000:8000"
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - MODEL_SIZE=base
      - MAX_BATCH_SIZE=4
      - CACHE_SIZE=1000
      - LOG_LEVEL=INFO
    volumes:
      - ./models:/app/models
      - ./outputs:/app/outputs
      - ./logs:/app/logs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    container_name: musicgen-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - musicgen-api
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    container_name: musicgen-redis
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    command: redis-server --appendonly yes
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    container_name: musicgen-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    restart: unless-stopped

volumes:
  redis-data:
  prometheus-data:
```

### 3. Build and Run

```bash
# Build production image
docker build -f Dockerfile.prod -t musicgen:latest .

# Run with Docker Compose
docker-compose up -d

# Check status
docker-compose ps
docker-compose logs -f musicgen-api

# Scale horizontally
docker-compose up -d --scale musicgen-api=3
```

---

## ☸️ Kubernetes Deployment

### 1. Kubernetes Manifests

```yaml
# musicgen-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: musicgen-api
  labels:
    app: musicgen
spec:
  replicas: 3
  selector:
    matchLabels:
      app: musicgen
  template:
    metadata:
      labels:
        app: musicgen
    spec:
      containers:
      - name: musicgen
        image: musicgen:latest
        ports:
        - containerPort: 8000
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        - name: MODEL_SIZE
          value: "base"
        resources:
          requests:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: 1
          limits:
            memory: "16Gi"
            cpu: "8"
            nvidia.com/gpu: 1
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: model-cache
          mountPath: /app/models
        - name: output-storage
          mountPath: /app/outputs
      volumes:
      - name: model-cache
        persistentVolumeClaim:
          claimName: musicgen-models-pvc
      - name: output-storage
        persistentVolumeClaim:
          claimName: musicgen-outputs-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: musicgen-service
spec:
  selector:
    app: musicgen
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: musicgen-models-pvc
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 100Gi
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: musicgen-outputs-pvc
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 50Gi
```

### 2. Horizontal Pod Autoscaler

```yaml
# musicgen-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: musicgen-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: musicgen-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: gpu_utilization
      target:
        type: AverageValue
        averageValue: "80"
```

### 3. Ingress Configuration

```yaml
# musicgen-ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: musicgen-ingress
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/proxy-body-size: "100m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
spec:
  tls:
  - hosts:
    - api.musicgen.ai
    secretName: musicgen-tls
  rules:
  - host: api.musicgen.ai
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: musicgen-service
            port:
              number: 80
```

### 4. Deploy to Kubernetes

```bash
# Create namespace
kubectl create namespace musicgen

# Deploy application
kubectl apply -f musicgen-deployment.yaml -n musicgen
kubectl apply -f musicgen-hpa.yaml -n musicgen
kubectl apply -f musicgen-ingress.yaml -n musicgen

# Check status
kubectl get pods -n musicgen
kubectl get svc -n musicgen
kubectl get hpa -n musicgen

# View logs
kubectl logs -f deployment/musicgen-api -n musicgen
```

---

## ☁️ Cloud Deployment

### AWS Deployment

#### 1. ECS Task Definition
```json
{
  "family": "musicgen-task",
  "taskRoleArn": "arn:aws:iam::123456789:role/musicgen-task-role",
  "executionRoleArn": "arn:aws:iam::123456789:role/musicgen-execution-role",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["EC2"],
  "cpu": "8192",
  "memory": "32768",
  "placementConstraints": [
    {
      "type": "memberOf",
      "expression": "attribute:ecs.instance-type =~ p3.*"
    }
  ],
  "containerDefinitions": [
    {
      "name": "musicgen",
      "image": "123456789.dkr.ecr.us-east-1.amazonaws.com/musicgen:latest",
      "cpu": 8192,
      "memory": 32768,
      "resourceRequirements": [
        {
          "type": "GPU",
          "value": "1"
        }
      ],
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "MODEL_SIZE",
          "value": "base"
        },
        {
          "name": "AWS_REGION",
          "value": "us-east-1"
        }
      ],
      "mountPoints": [
        {
          "sourceVolume": "model-cache",
          "containerPath": "/app/models"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/musicgen",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ],
  "volumes": [
    {
      "name": "model-cache",
      "efsVolumeConfiguration": {
        "fileSystemId": "fs-12345678",
        "rootDirectory": "/models"
      }
    }
  ]
}
```

#### 2. Terraform Configuration
```hcl
# main.tf
provider "aws" {
  region = "us-east-1"
}

resource "aws_ecs_cluster" "musicgen" {
  name = "musicgen-cluster"
  
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

resource "aws_ecs_service" "musicgen" {
  name            = "musicgen-service"
  cluster         = aws_ecs_cluster.musicgen.id
  task_definition = aws_ecs_task_definition.musicgen.arn
  desired_count   = 3

  deployment_configuration {
    maximum_percent         = 200
    minimum_healthy_percent = 100
  }

  network_configuration {
    subnets         = aws_subnet.private.*.id
    security_groups = [aws_security_group.musicgen.id]
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.musicgen.arn
    container_name   = "musicgen"
    container_port   = 8000
  }
}

resource "aws_autoscaling_policy" "musicgen" {
  name                   = "musicgen-autoscaling"
  policy_type            = "TargetTrackingScaling"
  resource_id            = aws_ecs_service.musicgen.id
  scalable_dimension     = "ecs:service:DesiredCount"
  service_namespace      = "ecs"

  target_tracking_scaling_policy_configuration {
    target_value = 70.0

    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }

    scale_out_cooldown = 60
    scale_in_cooldown  = 180
  }
}
```

### GCP Deployment

#### Cloud Run Configuration
```yaml
# service.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: musicgen-api
  annotations:
    run.googleapis.com/cpu-throttling: "false"
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/minScale: "2"
        autoscaling.knative.dev/maxScale: "100"
        run.googleapis.com/execution-environment: gen2
    spec:
      containerConcurrency: 1
      timeoutSeconds: 300
      serviceAccountName: musicgen-sa
      containers:
      - image: gcr.io/project-id/musicgen:latest
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_SIZE
          value: "base"
        resources:
          requests:
            cpu: "4"
            memory: "16Gi"
            nvidia.com/gpu: "1"
          limits:
            cpu: "8"
            memory: "32Gi"
            nvidia.com/gpu: "1"
```

---

## 📊 Monitoring & Observability

### 1. Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'musicgen'
    static_configs:
      - targets: ['musicgen-api:8000']
    metrics_path: '/metrics'

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

  - job_name: 'nvidia-gpu'
    static_configs:
      - targets: ['nvidia-gpu-exporter:9445']

rule_files:
  - 'alerts.yml'

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']
```

### 2. Grafana Dashboard

```json
{
  "dashboard": {
    "title": "MusicGen AI Monitoring",
    "panels": [
      {
        "title": "Generation Latency",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, musicgen_generation_duration_seconds_bucket)"
          }
        ]
      },
      {
        "title": "GPU Utilization",
        "targets": [
          {
            "expr": "nvidia_gpu_utilization_percent"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "targets": [
          {
            "expr": "musicgen_memory_usage_bytes / 1024 / 1024 / 1024"
          }
        ]
      },
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "rate(musicgen_requests_total[5m])"
          }
        ]
      }
    ]
  }
}
```

### 3. Logging Configuration

```python
# logging_config.py
import logging
import json
from pythonjsonlogger import jsonlogger

# Configure structured logging
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
logHandler.setFormatter(formatter)

logger = logging.getLogger()
logger.addHandler(logHandler)
logger.setLevel(logging.INFO)

# Log format
log_format = {
    "timestamp": "%(asctime)s",
    "level": "%(levelname)s",
    "service": "musicgen",
    "trace_id": "%(trace_id)s",
    "message": "%(message)s",
    "duration_ms": "%(duration_ms)s",
    "model_size": "%(model_size)s",
    "gpu_memory_mb": "%(gpu_memory_mb)s"
}
```

---

## 🔒 Security Configuration

### 1. API Authentication

```python
# auth_middleware.py
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
```

### 2. Rate Limiting

```python
# rate_limiter.py
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["100 per hour", "10 per minute"]
)

@app.post("/generate")
@limiter.limit("5 per minute")
async def generate_music(request: GenerationRequest):
    # Generation logic
    pass
```

### 3. Network Security

```yaml
# network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: musicgen-network-policy
spec:
  podSelector:
    matchLabels:
      app: musicgen
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - namespaceSelector: {}
    ports:
    - protocol: TCP
      port: 443
    - protocol: TCP
      port: 80
```

---

## 🚀 Performance Tuning

### 1. NGINX Configuration

```nginx
# nginx.conf
upstream musicgen_backend {
    least_conn;
    server musicgen-api-1:8000 max_fails=3 fail_timeout=30s;
    server musicgen-api-2:8000 max_fails=3 fail_timeout=30s;
    server musicgen-api-3:8000 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name api.musicgen.ai;

    client_max_body_size 100M;
    proxy_read_timeout 300s;
    proxy_connect_timeout 75s;

    location / {
        proxy_pass http://musicgen_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        
        # Enable caching for generated audio
        location ~ \.(wav|mp3)$ {
            proxy_cache audio_cache;
            proxy_cache_valid 200 1h;
            proxy_cache_key "$request_uri";
            add_header X-Cache-Status $upstream_cache_status;
        }
    }

    # Health check endpoint
    location /health {
        access_log off;
        proxy_pass http://musicgen_backend/health;
    }
}
```

### 2. Database Optimization

```sql
-- Create indexes for performance
CREATE INDEX idx_generations_user_id ON generations(user_id);
CREATE INDEX idx_generations_created_at ON generations(created_at);
CREATE INDEX idx_generations_prompt_hash ON generations(prompt_hash);

-- Partition table by date
CREATE TABLE generations_2024_01 PARTITION OF generations
FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Out of Memory Errors
```bash
# Check GPU memory
nvidia-smi

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# Reduce batch size
export MAX_BATCH_SIZE=2
```

#### 2. Slow Generation
```bash
# Check GPU utilization
watch -n 1 nvidia-smi

# Profile performance
python -m torch.profiler.profile --activities cpu,cuda

# Enable mixed precision
export ENABLE_MIXED_PRECISION=true
```

#### 3. Model Loading Issues
```bash
# Verify model files
ls -la /app/models/

# Check model checksums
md5sum /app/models/*.pt

# Re-download models
python scripts/download_models.py --model-size base
```

### Health Checks

```python
# health_check.py
@app.get("/health")
async def health_check():
    checks = {
        "status": "healthy",
        "model_loaded": model is not None,
        "gpu_available": torch.cuda.is_available(),
        "gpu_memory_free": torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else 0,
        "cache_size": len(generation_cache),
        "uptime": time.time() - start_time
    }
    
    if not checks["model_loaded"] or not checks["gpu_available"]:
        checks["status"] = "unhealthy"
        
    return checks
```

---

## 📝 Deployment Checklist

- [ ] Docker image built and tested
- [ ] Environment variables configured
- [ ] SSL certificates installed
- [ ] Database migrations completed
- [ ] Model files downloaded and cached
- [ ] Health checks passing
- [ ] Monitoring dashboards set up
- [ ] Alerts configured
- [ ] Backup strategy implemented
- [ ] Security scan completed
- [ ] Load testing performed
- [ ] Documentation updated
- [ ] Rollback plan ready

---

## 🎯 Next Steps

1. **Staging Deployment**: Test full deployment in staging environment
2. **Load Testing**: Run stress tests to validate performance
3. **Security Audit**: Complete security assessment
4. **Monitoring Setup**: Configure all dashboards and alerts
5. **Production Release**: Deploy with canary or blue-green strategy

For additional support, consult the [Performance Optimization Guide](PERFORMANCE_OPTIMIZATION_GUIDE.md) and [Production Readiness Report](PRODUCTION_READINESS_REPORT.md).