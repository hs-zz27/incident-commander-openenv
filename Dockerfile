FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml ./
COPY server/ ./server/
COPY openenv.yaml ./
COPY inference.py ./
COPY evaluate.py ./
COPY README.md ./
COPY __init__.py ./
COPY client.py ./
COPY models.py ./

# Install the package and all dependencies via pyproject.toml
RUN pip install --no-cache-dir -e .

# Environment variables (HF Spaces compatible)
ENV HOST=0.0.0.0
ENV PORT=7860
ENV WORKERS=1

# Expose both ports (8000 for local, 7860 for HF Spaces)
EXPOSE 7860 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')" || exit 1

# Run the FastAPI server — HF Spaces sets PORT=7860
CMD uvicorn server.app:app --host $HOST --port $PORT --workers $WORKERS
