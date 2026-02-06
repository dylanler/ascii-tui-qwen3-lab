#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-0.6B}"
ADAPTER_PATH="${ADAPTER_PATH:-artifacts/qwen3_ascii_tui_lora}"
LORA_NAME="${LORA_NAME:-qwen3-ascii-tui-lora}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-ascii-tui-base}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_LORA_RANK="${MAX_LORA_RANK:-64}"
DTYPE="${DTYPE:-auto}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"

if [[ ! -d "${ADAPTER_PATH}" ]]; then
  echo "ERROR: Adapter directory not found: ${ADAPTER_PATH}"
  exit 1
fi

if [[ ! -f "${ADAPTER_PATH}/adapter_config.json" ]]; then
  echo "ERROR: adapter_config.json is missing in: ${ADAPTER_PATH}"
  exit 1
fi

echo "Starting vLLM endpoint..."
echo "  base model:        ${BASE_MODEL}"
echo "  served model name: ${SERVED_MODEL_NAME}"
echo "  lora model name:   ${LORA_NAME}"
echo "  adapter path:      ${ADAPTER_PATH}"
echo "  tensor parallel:   ${TENSOR_PARALLEL_SIZE}"
echo "  endpoint:          http://${HOST}:${PORT}/v1"
echo ""
echo "Use this model in requests to apply LoRA:"
echo "  ${LORA_NAME}"
echo ""

exec uv run vllm serve "${BASE_MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --enable-lora \
  --lora-modules "${LORA_NAME}=${ADAPTER_PATH}" \
  --max-lora-rank "${MAX_LORA_RANK}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --dtype "${DTYPE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}"
