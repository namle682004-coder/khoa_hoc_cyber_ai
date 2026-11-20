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

export VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:${VLLM_PORT}/v1}"

uv sync
uv run uvicorn src.api:app --host "${FASTAPI_HOST}" --port "${FASTAPI_PORT}"