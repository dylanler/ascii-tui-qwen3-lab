#!/usr/bin/env python3
"""Fine-tune Qwen3 0.6B for ASCII/TUI educational diagrams on 4 local GPUs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


SYSTEM_INSTRUCTION = (
    "You are an expert educational assistant that teaches with precise ASCII diagrams and "
    "terminal-friendly layouts."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Qwen3 0.6B with LoRA on synthetic ASCII/TUI data.")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--train-file", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument("--eval-file", type=Path, default=Path("data/processed/eval.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/qwen3_ascii_tui_lora"))
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-non-4gpu", action="store_true")
    return parser.parse_args()


def verify_gpu_setup(allow_non_4gpu: bool) -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    cuda_count = torch.cuda.device_count()
    if allow_non_4gpu:
        return
    if world_size != 4:
        raise RuntimeError(
            f"WORLD_SIZE={world_size}. Launch with torchrun --nproc_per_node=4 to use all 4 GPUs."
        )
    if cuda_count < 4:
        raise RuntimeError(f"Detected {cuda_count} CUDA devices, but 4 are required.")


def build_prompt(tokenizer: AutoTokenizer, instruction: str, response: str) -> str:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": response},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except Exception:
            pass
    return (
        f"<|system|>\n{SYSTEM_INSTRUCTION}\n"
        f"<|user|>\n{instruction}\n"
        f"<|assistant|>\n{response}"
    )


def tokenize_dataset(args: argparse.Namespace, tokenizer: AutoTokenizer) -> Any:
    dataset = load_dataset(
        "json",
        data_files={"train": str(args.train_file), "eval": str(args.eval_file)},
    )

    def _tokenize(batch: dict[str, list[Any]]) -> dict[str, Any]:
        texts = [
            build_prompt(tokenizer, instruction=inst, response=resp)
            for inst, resp in zip(batch["instruction"], batch["response"])
        ]
        tokens = tokenizer(
            texts,
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
        )
        tokens["labels"] = [ids.copy() for ids in tokens["input_ids"]]
        return tokens

    columns_to_remove = dataset["train"].column_names
    tokenized = dataset.map(
        _tokenize,
        batched=True,
        remove_columns=columns_to_remove,
        desc="Tokenizing dataset",
    )
    return tokenized


def build_model(args: argparse.Namespace) -> Any:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map={"": local_rank},
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    return model


def main() -> None:
    args = parse_args()
    verify_gpu_setup(allow_non_4gpu=args.allow_non_4gpu)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized = tokenize_dataset(args, tokenizer)
    model = build_model(args)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=True,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        report_to=["tensorboard"],
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["eval"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    train_result = trainer.train()
    trainer.save_model()
    trainer.save_state()

    metrics = train_result.metrics
    metrics_path = args.output_dir / "train_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Training complete. Adapter/model outputs: {args.output_dir}")
    print(f"Trainer state: {args.output_dir / 'trainer_state.json'}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()

