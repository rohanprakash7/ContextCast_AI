# Multi-stage build for optimized image size
# Target: <500MB final image

# Stage 1: Builder
FROM python:3.9-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .

# Install dependencies to user directory
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime
FROM python:3.9-slim

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY api ./api
COPY chains ./chains
COPY indexing ./indexing
COPY retrieval ./retrieval
COPY augmentation ./augmentation
COPY generation ./generation
COPY config ./config
COPY utils ./utils
COPY run.py .

# Set Python path
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run application
CMD ["python", "run.py"]