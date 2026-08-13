#!/usr/bin/env bash
# Start two real llama.cpp servers that differ in exactly one flag, then run the
# live control tests against them.
#
# The pair is the whole point. Both servers run the same image and the same
# model; only the physical batch differs. The first is capped by the default
# n_ubatch far below the context it advertises on /props -- the incident this
# tool was written for -- and the second is configured correctly. A detector
# that flags everything passes the first and fails the second, so the pair is
# falsifiable in both directions.
#
# Usage:  scripts/live_control.sh [path/to/embedding-model.gguf]
#
# With no argument the script downloads a small public embedding model.
set -euo pipefail

IMAGE="${LLMPROBE_LLAMACPP_IMAGE:-ghcr.io/ggml-org/llama.cpp:server-vulkan-b9049}"
MODEL="${1:-}"
CTX=8192
BROKEN_PORT="${LLMPROBE_BROKEN_PORT:-18081}"
HEALTHY_PORT="${LLMPROBE_HEALTHY_PORT:-18082}"
BROKEN_NAME=llmprobe-control-broken
HEALTHY_NAME=llmprobe-control-healthy

cleanup() { docker rm -f "$BROKEN_NAME" "$HEALTHY_NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

if [[ -z "$MODEL" ]]; then
  MODEL="${TMPDIR:-/tmp}/llmprobe-control-model.gguf"
  if [[ ! -f "$MODEL" ]]; then
    echo "downloading a small embedding model to $MODEL"
    curl -fsSL -o "$MODEL" \
      "https://huggingface.co/second-state/All-MiniLM-L6-v2-Embedding-GGUF/resolve/main/all-MiniLM-L6-v2-Q4_K_M.gguf"
  fi
fi
MODEL="$(cd "$(dirname "$MODEL")" && pwd)/$(basename "$MODEL")"

cleanup
# No GPU device is passed: the control must be reproducible on any machine, and
# a 30 MB embedding model does not need one.
docker run -d --name "$BROKEN_NAME" -p "127.0.0.1:$BROKEN_PORT:8080" \
  -v "$MODEL:/m.gguf:ro" "$IMAGE" \
  -m /m.gguf --embeddings --host 0.0.0.0 --port 8080 --ctx-size "$CTX" --pooling cls >/dev/null
docker run -d --name "$HEALTHY_NAME" -p "127.0.0.1:$HEALTHY_PORT:8080" \
  -v "$MODEL:/m.gguf:ro" "$IMAGE" \
  -m /m.gguf --embeddings --host 0.0.0.0 --port 8080 --ctx-size "$CTX" --pooling cls \
  -b "$CTX" -ub "$CTX" >/dev/null

for port in "$BROKEN_PORT" "$HEALTHY_PORT"; do
  for _ in $(seq 1 90); do
    curl -sf -m 5 "http://127.0.0.1:$port/health" >/dev/null 2>&1 && break
    sleep 2
  done
done

export LLMPROBE_LIVE_BROKEN_URL="http://127.0.0.1:$BROKEN_PORT"
export LLMPROBE_LIVE_HEALTHY_URL="http://127.0.0.1:$HEALTHY_PORT"
echo "broken:  $LLMPROBE_LIVE_BROKEN_URL (default n_ubatch)"
echo "healthy: $LLMPROBE_LIVE_HEALTHY_URL (-b $CTX -ub $CTX)"

python -m pytest tests/test_live_llamacpp.py -v "${@:2}"
