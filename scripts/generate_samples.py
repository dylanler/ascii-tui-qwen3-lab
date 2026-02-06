#!/usr/bin/env python3
"""Generate qualitative sample outputs from the fine-tuned adapter."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate sample markdown from fine-tuned Qwen3 adapter.")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--adapter-path", type=Path, default=Path("artifacts/qwen3_ascii_tui_lora"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/qwen3_ascii_tui_lora/sample_generations.md"),
    )
    parser.add_argument("--max-new-tokens", type=int, default=260)
    parser.add_argument(
        "--topics",
        nargs="+",
        default=[
            "lifecycle of a volcano",
            "how does gravity work",
            "double-slit experiment",
            "photosynthesis",
        ],
    )
    return parser.parse_args()


def clean_output(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def main() -> None:
    args = parse_args()
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(str(args.adapter_path), trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, str(args.adapter_path))
    model.eval()

    system = (
        "You are an expert educational assistant that teaches with precise ASCII diagrams and "
        "terminal-friendly layouts."
    )
    sections = ["# Fine-Tuned Sample Generations", ""]
    for topic in args.topics:
        user = (
            f"Teach '{topic}' with a terminal-friendly ASCII/TUI diagram. "
            "Keep width <= 90 chars and include key takeaways."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.35,
                top_p=0.9,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated = outputs[0][inputs["input_ids"].shape[1] :]
        text = clean_output(tokenizer.decode(generated, skip_special_tokens=True))

        sections.append(f"## {topic}")
        sections.append("")
        sections.append("```text")
        sections.append(text)
        sections.append("```")
        sections.append("")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(sections), encoding="utf-8")
    print(f"Wrote sample generations: {args.output}")


if __name__ == "__main__":
    main()

