#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/deploy_to_hf.sh <username_or_org> <repo_name> [space|model] [public|private]

Examples:
  ./scripts/deploy_to_hf.sh alice incident-commander space public
  ./scripts/deploy_to_hf.sh my-org incident-commander model private

Notes:
  - This script deploys the current git repository to Hugging Face.
  - model_training.ipynb must be tracked in git (it is checked before push).
  - For "space", the repo is created with Docker SDK to match this project.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 2 || $# -gt 4 ]]; then
  usage
  exit 1
fi

OWNER="$1"
REPO_NAME="$2"
REPO_TYPE="${3:-space}"      # space | model
VISIBILITY="${4:-public}"    # public | private

if [[ "$REPO_TYPE" != "space" && "$REPO_TYPE" != "model" ]]; then
  echo "Error: repo type must be 'space' or 'model'"
  exit 1
fi

if [[ "$VISIBILITY" != "public" && "$VISIBILITY" != "private" ]]; then
  echo "Error: visibility must be 'public' or 'private'"
  exit 1
fi

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Error: run this inside a git repository."
  exit 1
fi

if ! git ls-files --error-unmatch "model_training.ipynb" >/dev/null 2>&1; then
  echo "Error: model_training.ipynb is not tracked in git."
  echo "Run: git add model_training.ipynb && git commit -m 'Track notebook'"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Prefer project venv if available; fall back to system python3
if [[ -x "${SCRIPT_DIR}/.venv312/bin/python3" ]]; then
  PY="${SCRIPT_DIR}/.venv312/bin/python3"
  HF_CLI="${SCRIPT_DIR}/.venv312/bin/hf"
elif [[ -x "${SCRIPT_DIR}/.venv/bin/python3" ]]; then
  PY="${SCRIPT_DIR}/.venv/bin/python3"
  HF_CLI="${SCRIPT_DIR}/.venv/bin/hf"
elif command -v python3 >/dev/null 2>&1; then
  PY="python3"
  HF_CLI="hf"
else
  echo "Error: python3 is required."
  exit 1
fi

if ! "$PY" -c "import huggingface_hub" >/dev/null 2>&1; then
  echo "Installing huggingface_hub..."
  "$PY" -m pip install --upgrade "huggingface_hub[cli]"
fi

if ! "$HF_CLI" auth whoami >/dev/null 2>&1; then
  echo "Not logged in to Hugging Face. Starting login..."
  "$HF_CLI" auth login
fi

HF_REPO_ID="${OWNER}/${REPO_NAME}"
PRIVATE_FLAG="False"
if [[ "$VISIBILITY" == "private" ]]; then
  PRIVATE_FLAG="True"
fi

echo "Creating Hugging Face repo (or reusing if it exists): ${HF_REPO_ID}"
"$PY" - <<PY
from huggingface_hub import HfApi

repo_id = "${HF_REPO_ID}"
repo_type = "${REPO_TYPE}"
private = ${PRIVATE_FLAG}

api = HfApi()

if repo_type == "space":
    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        private=private,
        exist_ok=True,
        space_sdk="docker",
    )
else:
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
    )
print(f"Ready: {repo_type} -> {repo_id}")
PY

HF_TOKEN=$("$HF_CLI" auth token 2>/dev/null | tr -d '[:space:]')
if [[ -z "$HF_TOKEN" ]]; then
  echo "Error: could not retrieve HF token. Run: $HF_CLI auth login"
  exit 1
fi

REMOTE_NAME="huggingface"
if [[ "$REPO_TYPE" == "space" ]]; then
  REMOTE_URL="https://hf_user:${HF_TOKEN}@huggingface.co/spaces/${HF_REPO_ID}"
  DISPLAY_URL="https://huggingface.co/spaces/${HF_REPO_ID}"
else
  REMOTE_URL="https://hf_user:${HF_TOKEN}@huggingface.co/${HF_REPO_ID}"
  DISPLAY_URL="https://huggingface.co/${HF_REPO_ID}"
fi

if git remote get-url "$REMOTE_NAME" >/dev/null 2>&1; then
  git remote set-url "$REMOTE_NAME" "$REMOTE_URL"
else
  git remote add "$REMOTE_NAME" "$REMOTE_URL"
fi

echo "Pushing current branch to Hugging Face main..."
git push "$REMOTE_NAME" HEAD:main

# Reset remote URL to strip token so it's not stored in .git/config long-term
if [[ "$REPO_TYPE" == "space" ]]; then
  git remote set-url "$REMOTE_NAME" "https://huggingface.co/spaces/${HF_REPO_ID}"
else
  git remote set-url "$REMOTE_NAME" "https://huggingface.co/${HF_REPO_ID}"
fi

echo
echo "Done. URL: ${DISPLAY_URL}"
