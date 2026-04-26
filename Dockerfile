# ── Stage 1: Build Next.js frontend ──────────────────────────────
FROM node:22-alpine AS frontend-builder
RUN apk add --no-cache libc6-compat
WORKDIR /frontend

COPY frontend_app/package.json frontend_app/package-lock.json ./
RUN npm ci

COPY frontend_app/ .

# At runtime the browser talks to the same origin (/api is proxied)
ENV NEXT_PUBLIC_API_BASE_URL=""
ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build

# ── Stage 2: Final image (Python + Node) ────────────────────────
FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl nginx nodejs npm supervisor && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python backend ──────────────────────────────────────────────
COPY pyproject.toml README.md ./
COPY server/ ./server/
COPY openenv.yaml __init__.py ./
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

# ── Next.js standalone output ──────────────────────────────────
COPY --from=frontend-builder /frontend/public /app/frontend/public
COPY --from=frontend-builder /frontend/.next/standalone /app/frontend
COPY --from=frontend-builder /frontend/.next/static /app/frontend/.next/static

# ── Nginx: reverse-proxy port 7860 → frontend:3000 + API:8000 ─
COPY nginx.conf /etc/nginx/nginx.conf

# ── Supervisord: run both processes ─────────────────────────────
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

ENV HOST=0.0.0.0
ENV PORT=7860
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsSL http://127.0.0.1:7860/health || exit 1

CMD ["supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
