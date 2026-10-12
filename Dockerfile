# ==============================================================================
# Nexvia Sentinel — Production Dockerfile for Render / Cloud Deployment
# ==============================================================================
# Multi-stage-clean single image using Python 3.10 slim Linux base.
# Configured for single-worker stateful graph snapshot and CPU-only PyTorch.
# ==============================================================================

FROM python:3.10-slim

# Prevent bytecode compilation and enable unbuffered logging for cloud streaming
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000 \
    LEDGER_DB_PATH=/data/sentinel_ledger.db \
    ENVIRONMENT=production

WORKDIR /app

# Install minimal OS packages for building native extensions (psutil, confluent-kafka) and health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install pinned CPU-only PyTorch for Python 3.10
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "torch==2.2.2+cpu" --index-url https://download.pytorch.org/whl/cpu

# Copy backend dependency specification and install Python libraries
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r /app/backend/requirements.txt

# Copy backend source code, ML model artifacts, and project definition
COPY backend /app/backend
COPY ml /app/ml
COPY pyproject.toml /app/pyproject.toml

# Prepare default persistent data directory for SQLite ledger
RUN mkdir -p /data /app/backend/data

# Expose default HTTP port
EXPOSE 8000

# Container healthcheck targeting authoritative Sentinel health endpoint via Python urllib
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request, os; port = os.environ.get('PORT', '8000'); urllib.request.urlopen(f'http://localhost:{port}/api/v1/health', timeout=4)" || exit 1

# Start Uvicorn: bind to 0.0.0.0, dynamic Render PORT, strictly 1 worker for graph state
CMD ["sh", "-c", "exec uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
