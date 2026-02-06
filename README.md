# Qwen3 ASCII/TUI Fine-Tuning Experiment

This project defines a full experiment to fine-tune `Qwen3 0.6B` for generating high-quality ASCII art and TUI-based learning diagrams across broad topics, including:

- lifecycle of a volcano
- how does gravity work
- double-slit experiment
- and many related science/engineering topics

## Goals

- Generate a synthetic instruction dataset of about `1,000` rows.
- Use `100` parallel local agents for direct synthetic generation.
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
│   ├── generate_samples.py
│   └── run_full_experiment.sh
├── PROGRESS.md
├── README.md
└── pyproject.toml
```

## Environment Setup (UV)

```bash
uv sync
```

## Experiment Design

### 1) Synthetic data generation

Script: `scripts/generate_synthetic_dataset.py`

- Creates `100` async agent tasks in parallel (`--agent-count 100 --concurrency 100`).
- Runs in direct local mode by default (`--generation-mode local`), so no API is required.
- Keeps `gpt-5.3` model metadata for compatibility with an optional API path.
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
  --generation-mode local \
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
  --output-dir artifacts/qwen3_ascii_tui_lora \
  --num-train-epochs 3 \
  --logging-steps 1 \
  --eval-steps 5 \
  --save-steps 10
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

### 4) Qualitative sample checks

Script: `scripts/generate_samples.py`

- Loads base model + LoRA adapter and generates topic samples.
- Saves:
  - `artifacts/qwen3_ascii_tui_lora/sample_generations.md`

Command:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/generate_samples.py \
  --base-model Qwen/Qwen3-0.6B \
  --adapter-path artifacts/qwen3_ascii_tui_lora
```

### 5) Serve the fine-tuned adapter with vLLM

Scripts:
- `scripts/start_vllm_endpoint.sh`
- `scripts/test_vllm_endpoint.py`

Start an OpenAI-compatible endpoint on one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/start_vllm_endpoint.sh
```

If port `8000` is busy:

```bash
CUDA_VISIBLE_DEVICES=0 PORT=8010 bash scripts/start_vllm_endpoint.sh
```

Defaults:
- endpoint: `http://127.0.0.1:8000/v1`
- base model alias: `qwen3-ascii-tui-base`
- LoRA model name (use this for inference): `qwen3-ascii-tui-lora`

In another terminal, run a test request:

```bash
uv run python scripts/test_vllm_endpoint.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model qwen3-ascii-tui-lora
```

## One-command run

```bash
bash scripts/run_full_experiment.sh
```

This executes:

1. `uv sync`
2. 100-agent direct local synthetic generation (~1,000 rows)
3. 4-GPU fine-tuning
4. loss curve plotting
5. qualitative sample generation

## Latest Run Snapshot

- Dataset rows: `1000` (`950 train`, `50 eval`)
- Generation mode: `local` (direct generation, no API)
- 4-GPU training: completed with `45` train steps over `3` epochs
- Train loss: `4.12497 -> 0.10813`
- Eval loss: `2.06111 -> 0.11857`
- Artifacts:
  - `artifacts/qwen3_ascii_tui_lora/trainer_state.json`
  - `artifacts/qwen3_ascii_tui_lora/loss_curve.png`
  - `artifacts/qwen3_ascii_tui_lora/sample_generations.md`

## Notes

- If your GPU topology differs, keep `--nproc_per_node=4` and set `CUDA_VISIBLE_DEVICES` to the desired four devices.
- If you want API-backed generation later, pass `--generation-mode openai` and set `OPENAI_API_KEY`.
- Generated data and artifacts are git-ignored by default.
