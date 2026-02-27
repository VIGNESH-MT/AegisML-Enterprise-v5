# AegisML Enterprise — Production Docker Image
# Multi-stage build for minimal image size

# ── Stage 1: Builder ─────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Runtime ─────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="AegisML Enterprise <enterprise@aegisml.io>"
LABEL version="5.0.0"
LABEL description="AegisML Production ML Reliability & Drift Assurance Engine"
LABEL org.opencontainers.image.source="https://github.com/your-org/aegisml"

# Security: run as non-root user
RUN useradd --create-home --shell /bin/bash aegisml

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY --chown=aegisml:aegisml . .

# Create output directories with correct permissions
RUN mkdir -p /app/verification /app/examples && \
    chown -R aegisml:aegisml /app

USER aegisml

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()" || exit 1

EXPOSE 8000

# Default: run FastAPI server
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4", "--access-log"]
