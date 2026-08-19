#!/bin/bash
# Relaunch the vLLM container at 128k, wait for readiness, verify config, run cline-bench.
# Usage: bash scripts/serve_and_bench.sh
set -e

MODEL_PATH=/models/Qwen3.8-27B
SERVED_NAME=Qwen/Qwen3.8-27B
MAX_LEN=131072
TIMEOUT_S=900   # generous: first start may capture CUDA graphs

echo "[1/4] relaunching vllm container..."
docker rm -f vllm 2>/dev/null || true
docker run -d --name vllm --gpus '"device=2"' \
  -v /mnt/raid5/models:/models -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model "$MODEL_PATH" --served-model-name "$SERVED_NAME" \
  --max-model-len "$MAX_LEN" --max-num-seqs 32 --port 8000 >/dev/null

echo "[2/4] waiting for server (timeout ${TIMEOUT_S}s)..."
t0=$(date +%s)
until curl -sf localhost:8000/health >/dev/null 2>&1; do
  if ! docker ps -q -f name=^vllm$ | grep -q .; then
    echo "FATAL: container exited. Last logs:"
    docker logs vllm 2>&1 | tail -30
    exit 1
  fi
  if [ $(( $(date +%s) - t0 )) -gt "$TIMEOUT_S" ]; then
    echo "FATAL: timed out waiting for /health. Last logs:"
    docker logs vllm 2>&1 | tail -30
    exit 1
  fi
  sleep 5
done
echo "server healthy after $(( $(date +%s) - t0 ))s"

echo "[3/4] verifying served config..."
GOT=$(curl -s localhost:8000/v1/models | grep -o '"max_model_len":[0-9]*' | head -1)
echo "  $GOT (expected \"max_model_len\":$MAX_LEN)"
[ "$GOT" = "\"max_model_len\":$MAX_LEN" ] || { echo "FATAL: config mismatch"; exit 1; }

echo "[4/4] launching cline-bench @ 128k episode budget..."
time uv run python scripts/bench.py --tasks clinebench.jsonl --n-tasks 11 --k 1 \
  --concurrency 4 --max-turns 50 --temperature 0 --max-context "$MAX_LEN" \
  --model "$SERVED_NAME" --tokenizer /mnt/raid5/models/Qwen3.8-27B
