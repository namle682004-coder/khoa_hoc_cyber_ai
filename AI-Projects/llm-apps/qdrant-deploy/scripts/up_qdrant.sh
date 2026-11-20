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

command -v docker >/dev/null 2>&1 || { echo "docker is required"; exit 1; }

echo "[1/3] Starting Qdrant container"
docker compose up -d

echo "[2/3] Waiting for health endpoint"
TIMEOUT="${QDRANT_STARTUP_TIMEOUT:-60}"
POLL="${QDRANT_STARTUP_POLL_INTERVAL:-2}"
HEALTH_URL="${QDRANT_HEALTH_URL:-http://localhost:6333/healthz}"
START_TS="$(date +%s)"

until curl -fsS "$HEALTH_URL" >/dev/null 2>&1; do
  NOW_TS="$(date +%s)"
  ELAPSED="$((NOW_TS - START_TS))"
  if [[ "$ELAPSED" -ge "$TIMEOUT" ]]; then
    echo "Qdrant not ready within ${TIMEOUT}s"
    docker compose logs --tail 120 qdrant || true
    exit 1
  fi
  echo "... waiting (${ELAPSED}s/${TIMEOUT}s)"
  sleep "$POLL"
done

echo "[3/3] Qdrant is healthy"
echo "HTTP: ${QDRANT_BASE_URL:-http://localhost:6333}"
echo "gRPC: localhost:${QDRANT_GRPC_PORT:-6334}"
