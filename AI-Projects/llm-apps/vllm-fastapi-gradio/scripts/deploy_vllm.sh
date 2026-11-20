#!/usr/bin/env bash
set -euo pipefail

load_env_file() {
  local env_file="$1"
  local line=""
  local key=""
  local value=""
  local first_char=""
  local last_char=""

  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%$'\r'}"
    [[ -z "$line" ]] && continue
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    [[ "$line" != *=* ]] && continue

    key="${line%%=*}"
    value="${line#*=}"

    key="${key#export }"
    key="${key#${key%%[![:space:]]*}}"
    key="${key%${key##*[![:space:]]}}"
    value="${value#${value%%[![:space:]]*}}"

    first_char="${value:0:1}"
    last_char="${value: -1}"
    if [[ "$first_char" == '"' && "$last_char" == '"' ]]; then
      value="${value:1:${#value}-2}"
    elif [[ "$first_char" == "'" && "$last_char" == "'" ]]; then
      value="${value:1:${#value}-2}"
    fi

    export "$key=$value"
  done < "$env_file"
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from .env.example. Update values if needed."
fi

load_env_file .env

MODEL_REF="${1:-${MODEL_NAME}}"
MODEL_ALIAS="$(echo "$MODEL_REF" | tr '/:' '__')"
HF_HOME_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$HF_HOME_DIR}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$HF_HOME_DIR/checkpoints}"
MODEL_DIR="$CHECKPOINT_ROOT/$MODEL_ALIAS"

mkdir -p "$CHECKPOINT_ROOT"
mkdir -p "$HF_CACHE_DIR"

uv sync
uv run python -m src.download_model --model "$MODEL_REF" --output-dir "$MODEL_DIR"

docker build -t "${VLLM_IMAGE_NAME}" .
docker rm -f "${VLLM_CONTAINER_NAME}" >/dev/null 2>&1 || true

docker run -d \
  --name "${VLLM_CONTAINER_NAME}" \
  --gpus all \
  -p "${VLLM_PORT}:8000" \
  -e HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}" \
  -e HF_HOME=/root/.cache/huggingface \
  -v "$MODEL_DIR:/models/current:ro" \
  -v "$HF_CACHE_DIR:/root/.cache/huggingface" \
  "${VLLM_IMAGE_NAME}" \
  --model /models/current \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype "${MODEL_DTYPE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --trust-remote-code

echo "vLLM is starting at http://localhost:${VLLM_PORT}"
echo "Model source: ${MODEL_REF}"
echo "Mounted model: ${MODEL_DIR}"
echo "Shared Hugging Face cache: ${HF_CACHE_DIR}"