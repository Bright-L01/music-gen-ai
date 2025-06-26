# Multi-stage build for MusicGen API
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    pkg-config \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r musicgen && useradd -r -g musicgen musicgen

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Development stage
FROM base as development

# Install development dependencies
RUN pip install jupyter ipywidgets notebook

# Copy source code
COPY . .

# Install package in development mode
RUN pip install -e .

# Set ownership
RUN chown -R musicgen:musicgen /app

# Switch to non-root user
USER musicgen

# Expose port
EXPOSE 8000

# Development command
CMD ["uvicorn", "music_gen.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Production stage
FROM base as production

# Copy source code
COPY . .

# Install package
RUN pip install .

# Create directories
RUN mkdir -p /app/models /app/data /app/logs /tmp/musicgen && \
    chown -R musicgen:musicgen /app /tmp/musicgen

# Switch to non-root user
USER musicgen

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production command
CMD ["gunicorn", "music_gen.api.main:app", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--workers", "4", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "300", \
     "--keep-alive", "2", \
     "--max-requests", "1000", \
     "--max-requests-jitter", "100", \
     "--preload"]

# GPU-enabled production stage
FROM nvidia/cuda:11.8-runtime-ubuntu20.04 as gpu-production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    build-essential \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    pkg-config \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Create non-root user
RUN groupadd -r musicgen && useradd -r -g musicgen musicgen

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt pyproject.toml ./
RUN pip install --upgrade pip && \
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    pip install -r requirements.txt

# Copy source code and install
COPY . .
RUN pip install .

# Create directories and set permissions
RUN mkdir -p /app/models /app/data /app/logs /tmp/musicgen && \
    chown -R musicgen:musicgen /app /tmp/musicgen

# Switch to non-root user
USER musicgen

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# GPU production command
CMD ["gunicorn", "music_gen.api.main:app", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--workers", "2", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "600", \
     "--keep-alive", "2", \
     "--max-requests", "100", \
     "--max-requests-jitter", "10", \
     "--preload"]