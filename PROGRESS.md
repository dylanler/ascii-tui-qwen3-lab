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
- [ ] Execute full run end-to-end on local hardware (requires API key, model access, and runtime budget).
- [ ] Record final metrics and attach `loss_curve.png` snapshot.
