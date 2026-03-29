# Data Cleaning Environment — Dockerfile
# Builds a containerised OpenEnv-compliant HF Space on port 7860.

FROM python:3.11-slim

# Install OS deps (needed for some numpy/uvicorn builds)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# HF Spaces runs as a non-root user; ensure write access if needed
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Expose HF Spaces default port
EXPOSE 7860

# Health-check so the Space is marked live
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
