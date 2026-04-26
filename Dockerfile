# Incident Commander — FastAPI backend (OpenEnv-compatible)
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Installable package metadata
COPY pyproject.toml README.md ./

# Application code (root modules are imported by server.app and live_inference)
COPY server/ ./server/
COPY openenv.yaml ./
COPY __init__.py ./

COPY orchestrator.py evaluate_trained.py evaluate.py train_grpo.py ./
COPY inference.py multi_agent_inference.py live_inference.py ./
COPY run_baselines.py plot_baselines.py plot_training.py sft_warmstart.py ./
COPY client.py models.py ./

ARG INSTALL_TRAIN=0
RUN pip install --no-cache-dir --upgrade pip && \
    if [ "$INSTALL_TRAIN" = "1" ]; then \
      pip install --no-cache-dir -e ".[train]"; \
    else \
      pip install --no-cache-dir -e .; \
    fi

# Default matches docker-compose port mapping; override for HF Spaces (e.g. PORT=7860)
ENV HOST=0.0.0.0
ENV PORT=8000
ENV WORKERS=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8000')+'/health', timeout=4)"

CMD uvicorn server.app:app --host "$HOST" --port "$PORT" --workers "$WORKERS"
