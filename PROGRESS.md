# Progress Log

## 2026-02-06

- [x] Created new project directory: `ascii-tui-qwen3-lab` (separate from existing qwen folders).
- [x] Initialized a new Git repository.
- [x] Set up UV-managed Python project with training/data dependencies in `pyproject.toml`.
- [x] Added topic coverage config including required prompts:
  - lifecycle of a volcano
  - how does gravity work
  - double-slit experiment
- [x] Implemented synthetic dataset generation script:
  - 100 parallel agents
  - default model `gpt-5.3`
  - target ~1,000 rows
  - schema validation, retry logic, deduplication, train/eval split
- [x] Implemented 4-GPU fine-tuning script for Qwen3 0.6B with QLoRA.
- [x] Implemented loss-curve plotting from trainer logs.
- [x] Added one-command orchestration script for the full run.
- [x] Added complete experiment README with commands and design details.
- [x] Generated `uv.lock` and synced dependencies with `uv sync`.
- [x] Validated script CLIs and Python syntax compilation.
- [x] Ran a local smoke test for loss plotting output.
- [x] Added compatibility patches for `transformers` v5 API changes (`TrainingArguments` and `Trainer` signatures).
- [x] Added local generation fallback mode when `OPENAI_API_KEY` is not available.
- [x] Set full-run orchestrator to default `GENERATION_MODE=local` for direct non-API generation.
- [x] Executed full run end-to-end on local 4x A100 hardware.
- [x] Generated 1,000-row dataset (`950 train`, `50 eval`) with 100 agents in parallel.
- [x] Completed 4-GPU QLoRA training for 3 epochs.
- [x] Plotted loss curve and generated qualitative sample outputs.
- [x] Confidence gate passed:
  - strong train/eval loss reduction over full run
  - required topics present in dataset and qualitative output file
- [x] Added vLLM OpenAI-compatible endpoint scripts for LoRA serving + endpoint smoke test client.
- [x] Validated vLLM endpoint startup and live inference on `http://127.0.0.1:8010/v1`.
- [x] Added vLLM launcher support for `TENSOR_PARALLEL_SIZE` (multi-GPU serving, e.g. GPUs `2,3`).
- [x] Added README examples for `curl` usage and token-budget guardrails (`prompt_tokens + max_tokens <= max_model_len`).
- [x] Updated and pushed Hugging Face model card for `mr-dee/qwen3-ascii-tui-lora`.

## Run Metrics (2026-02-06)

- Train points: `45`
- Eval points: `9`
- First train loss: `4.12497`
- Final train loss: `0.10813`
- First eval loss: `2.06111`
- Best/final eval loss: `0.11857`
- Best eval epoch: `3.0`
- Artifacts:
  - `artifacts/qwen3_ascii_tui_lora/trainer_state.json`
  - `artifacts/qwen3_ascii_tui_lora/loss_curve.png`
  - `artifacts/qwen3_ascii_tui_lora/sample_generations.md`
