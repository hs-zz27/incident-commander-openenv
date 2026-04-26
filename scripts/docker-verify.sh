#!/usr/bin/env sh
# Build and smoke-test the stack. Requires a running Docker daemon.
set -e
cd "$(dirname "$0")/.."

echo "==> docker compose build"
docker compose build

echo "==> docker compose up -d"
docker compose up -d

echo "==> wait for API health"
i=0
while ! curl -fsS "http://127.0.0.1:8000/health" >/dev/null 2>&1; do
  i=$((i + 1))
  if [ "$i" -gt 60 ]; then
    echo "API health check timed out" >&2
    docker compose logs api
    exit 1
  fi
  sleep 1
done
echo "    API OK"

echo "==> frontend HTTP"
curl -fsS -o /dev/null "http://127.0.0.1:3000/" || {
  echo "Frontend check failed" >&2
  docker compose logs frontend
  exit 1
}
echo "    frontend OK"

echo "==> docker compose down"
docker compose down

echo "All checks passed."
