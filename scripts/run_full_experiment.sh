#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-0.6B}"
TARGET_ROWS="${TARGET_ROWS:-1000}"
GENERATION_MODE="${GENERATION_MODE:-local}"

if [[ "${GENERATION_MODE}" == "openai" && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY is required when GENERATION_MODE=openai."
  exit 1
fi

if [[ "${GENERATION_MODE}" == "local" ]]; then
  echo "Using direct local synthetic generation (no API calls)."
fi

echo "[1/5] Syncing UV environment"
uv sync

echo "[2/5] Generating synthetic dataset with 100 parallel local agents"
uv run python scripts/generate_synthetic_dataset.py \
  --model gpt-5.3 \
  --generation-mode "${GENERATION_MODE}" \
  --target-rows "${TARGET_ROWS}" \
  --agent-count 100 \
  --concurrency 100 \
  --rows-per-call 5 \
  --raw-output data/raw/synthetic_ascii_tui.jsonl \
  --train-output data/processed/train.jsonl \
  --eval-output data/processed/eval.jsonl

echo "[3/5] Fine-tuning Qwen3 0.6B on 4 GPUs"
uv run torchrun --nproc_per_node=4 scripts/train_qwen3_ascii_tui.py \
  --model-name "${MODEL_NAME}" \
  --train-file data/processed/train.jsonl \
  --eval-file data/processed/eval.jsonl \
  --output-dir artifacts/qwen3_ascii_tui_lora \
  --num-train-epochs 3 \
  --logging-steps 1 \
  --eval-steps 5 \
  --save-steps 10 \
  --gradient-accumulation-steps 4 \
  --per-device-train-batch-size 4 \
  --per-device-eval-batch-size 4

echo "[4/5] Plotting loss curve"
uv run python scripts/plot_loss_curve.py \
  --trainer-state artifacts/qwen3_ascii_tui_lora/trainer_state.json \
  --output artifacts/qwen3_ascii_tui_lora/loss_curve.png

echo "[5/5] Generating qualitative samples"
CUDA_VISIBLE_DEVICES="${SAMPLE_GPU:-0}" uv run python scripts/generate_samples.py \
  --base-model "${MODEL_NAME}" \
  --adapter-path artifacts/qwen3_ascii_tui_lora \
  --output artifacts/qwen3_ascii_tui_lora/sample_generations.md

echo "Experiment complete."
