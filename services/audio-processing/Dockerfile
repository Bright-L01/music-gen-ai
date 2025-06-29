# Multi-stage build for optimized image
FROM python:3.10-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.10-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 audioprocessor

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /home/audioprocessor/.local

# Copy application code
COPY src/ ./src/

# Create directories for storage
RUN mkdir -p /app/storage/raw /app/storage/processed /app/storage/waveforms /app/storage/temp && \
    chown -R audioprocessor:audioprocessor /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PATH=/home/audioprocessor/.local/bin:$PATH
ENV STORAGE_PATH=/app/storage

# Switch to non-root user
USER audioprocessor

# Expose port
EXPOSE 8002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

# Run the service
CMD ["python", "-m", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8002"]