# SentinelVision — Industrial Agentic Vision Pipeline Container
# Base Image: Lightweight Python 3.12 Linux
FROM python:3.12-slim

# Install system libraries for OpenCV & media handling + container healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy pinned requirements
COPY requirements.txt .

# Install CPU-optimized dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code, config, models, and dashboard
COPY src/ /app/src/
COPY config/ /app/config/
COPY dashboard/ /app/dashboard/
COPY yolov8n.pt /app/yolov8n.pt
COPY data/ /app/data/

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PORT=8000

EXPOSE 8000

# Healthcheck for AWS ECS / Container monitoring
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/api/health || exit 1

# Default Entrypoint: FastAPI Production Server
CMD ["python", "-m", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
