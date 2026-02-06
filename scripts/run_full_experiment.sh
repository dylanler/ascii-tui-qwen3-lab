#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-0.6B}"
TARGET_ROWS="${TARGET_ROWS:-1000}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is required for synthetic data generation."
  exit 1
fi

echo "[1/4] Syncing UV environment"
uv sync

echo "[2/4] Generating synthetic dataset with 100 GPT-5.3 agents"
uv run python scripts/generate_synthetic_dataset.py \
  --model gpt-5.3 \
  --target-rows "${TARGET_ROWS}" \
  --agent-count 100 \
  --concurrency 100 \
  --rows-per-call 5 \
  --raw-output data/raw/synthetic_ascii_tui.jsonl \
  --train-output data/processed/train.jsonl \
  --eval-output data/processed/eval.jsonl

echo "[3/4] Fine-tuning Qwen3 0.6B on 4 GPUs"
uv run torchrun --nproc_per_node=4 scripts/train_qwen3_ascii_tui.py \
  --model-name "${MODEL_NAME}" \
  --train-file data/processed/train.jsonl \
  --eval-file data/processed/eval.jsonl \
  --output-dir artifacts/qwen3_ascii_tui_lora

echo "[4/4] Plotting loss curve"
uv run python scripts/plot_loss_curve.py \
  --trainer-state artifacts/qwen3_ascii_tui_lora/trainer_state.json \
  --output artifacts/qwen3_ascii_tui_lora/loss_curve.png

echo "Experiment complete."

