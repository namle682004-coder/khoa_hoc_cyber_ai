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
fi
load_env_file .env

COLLECTION_NAME="${1:-${DEFAULT_COLLECTION:-rag_documents}}"
VECTOR_SIZE="${2:-${VECTOR_SIZE:-384}}"
DISTANCE="${3:-${DISTANCE:-Cosine}}"
BASE_URL="${QDRANT_BASE_URL:-http://localhost:6333}"
API_KEY="${QDRANT_API_KEY:-}"

AUTH_HEADER=()
if [[ -n "$API_KEY" ]]; then
  AUTH_HEADER=(-H "api-key: ${API_KEY}")
fi

curl -fsS -X PUT "${BASE_URL}/collections/${COLLECTION_NAME}" \
  -H "Content-Type: application/json" \
  "${AUTH_HEADER[@]}" \
  -d "{\"vectors\": {\"size\": ${VECTOR_SIZE}, \"distance\": \"${DISTANCE}\"}}"

echo
printf 'Collection ready: %s (size=%s, distance=%s)\n' "$COLLECTION_NAME" "$VECTOR_SIZE" "$DISTANCE"
