# Qwen3 ASCII/TUI Fine-Tuning Experiment

This project defines a full experiment to fine-tune `Qwen3 0.6B` for generating high-quality ASCII art and TUI-based learning diagrams across broad topics, including:

- lifecycle of a volcano
- how does gravity work
- double-slit experiment
- and many related science/engineering topics

## Goals

- Generate a synthetic instruction dataset of about `1,000` rows.
- Use `100` GPT-5.3 agents in parallel for dataset generation.
- Fine-tune on local GPUs using all `4` GPUs.
- Track and visualize the loss curve.
- Keep environment management clean with `uv`.

## Project Layout

```text
ascii-tui-qwen3-lab/
├── configs/
│   └── topics.yaml
├── data/
│   ├── raw/
│   └── processed/
├── scripts/
│   ├── generate_synthetic_dataset.py
│   ├── train_qwen3_ascii_tui.py
│   ├── plot_loss_curve.py
│   └── run_full_experiment.sh
├── PROGRESS.md
├── README.md
└── pyproject.toml
```

## Environment Setup (UV)

```bash
uv sync
```

Required env vars:

```bash
export OPENAI_API_KEY=...
```

## Experiment Design

### 1) Synthetic data generation

Script: `scripts/generate_synthetic_dataset.py`

- Creates `100` async agent tasks in parallel (`--agent-count 100 --concurrency 100`).
- Uses model `gpt-5.3` by default.
- Assigns topic coverage across required and supplemental topics from `configs/topics.yaml`.
- Validates JSON rows and removes duplicates.
- Writes:
  - `data/raw/synthetic_ascii_tui.jsonl`
  - `data/processed/train.jsonl`
  - `data/processed/eval.jsonl`

Default command:

```bash
uv run python scripts/generate_synthetic_dataset.py \
  --model gpt-5.3 \
  --target-rows 1000 \
  --agent-count 100 \
  --concurrency 100
```

### 2) Fine-tuning on 4 GPUs

Script: `scripts/train_qwen3_ascii_tui.py`

- Base model default: `Qwen/Qwen3-0.6B`.
- QLoRA setup with 4-bit loading and LoRA adapters.
- Enforces 4-GPU launch by default (`WORLD_SIZE` must be 4).
- Uses DDP via:

```bash
uv run torchrun --nproc_per_node=4 scripts/train_qwen3_ascii_tui.py \
  --model-name Qwen/Qwen3-0.6B \
  --train-file data/processed/train.jsonl \
  --eval-file data/processed/eval.jsonl \
  --output-dir artifacts/qwen3_ascii_tui_lora
```

### 3) Loss curve visualization

Script: `scripts/plot_loss_curve.py`

- Reads `trainer_state.json` from HF Trainer output.
- Plots train/eval losses over step.
- Saves:
  - `artifacts/qwen3_ascii_tui_lora/loss_curve.png`

Command:

```bash
uv run python scripts/plot_loss_curve.py \
  --trainer-state artifacts/qwen3_ascii_tui_lora/trainer_state.json \
  --output artifacts/qwen3_ascii_tui_lora/loss_curve.png
```

## One-command run

```bash
bash scripts/run_full_experiment.sh
```

This executes:

1. `uv sync`
2. 100-agent synthetic generation (~1,000 rows)
3. 4-GPU fine-tuning
4. loss curve plotting

## Notes

- If your GPU topology differs, keep `--nproc_per_node=4` and set `CUDA_VISIBLE_DEVICES` to the desired four devices.
- If your account exposes a different model ID than `gpt-5.3`, pass `--model <id>` to generation.
- Generated data and artifacts are git-ignored by default.

