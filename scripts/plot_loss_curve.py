#!/usr/bin/env python3
"""Plot training/eval loss from Hugging Face trainer_state.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot loss curve from trainer_state.json")
    parser.add_argument(
        "--trainer-state",
        type=Path,
        default=Path("artifacts/qwen3_ascii_tui_lora/trainer_state.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/qwen3_ascii_tui_lora/loss_curve.png"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.trainer_state.open("r", encoding="utf-8") as f:
        state = json.load(f)

    train_steps: list[int] = []
    train_losses: list[float] = []
    eval_steps: list[int] = []
    eval_losses: list[float] = []

    for entry in state.get("log_history", []):
        step = entry.get("step")
        if step is None:
            continue
        if "loss" in entry:
            train_steps.append(int(step))
            train_losses.append(float(entry["loss"]))
        if "eval_loss" in entry:
            eval_steps.append(int(step))
            eval_losses.append(float(entry["eval_loss"]))

    if not train_losses:
        raise RuntimeError("No training loss entries found in trainer_state.json")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(train_steps, train_losses, label="train_loss", linewidth=2)
    if eval_losses:
        plt.plot(eval_steps, eval_losses, label="eval_loss", linewidth=2)
    plt.title("Qwen3 ASCII/TUI Fine-Tuning Loss Curve")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=160)
    print(f"Saved loss curve: {args.output}")


if __name__ == "__main__":
    main()

