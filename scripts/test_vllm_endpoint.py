#!/usr/bin/env python3
"""Send a test request to a local vLLM OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import sys

from openai import OpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test vLLM endpoint with a chat completion request.")
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", type=str, default="local-dev-key")
    parser.add_argument("--model", type=str, default="qwen3-ascii-tui-lora")
    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "Teach the double-slit experiment with a compact ASCII/TUI diagram. "
            "Include 3 key takeaways."
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=450)
    parser.add_argument("--temperature", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    models = client.models.list()
    available = [m.id for m in models.data]
    if args.model not in available:
        print("Requested model is not currently exposed by the endpoint.", file=sys.stderr)
        print(f"requested: {args.model}", file=sys.stderr)
        print("available:", file=sys.stderr)
        for model_id in available:
            print(f"  - {model_id}", file=sys.stderr)
        raise SystemExit(1)

    completion = client.chat.completions.create(
        model=args.model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You generate accurate terminal-first educational diagrams "
                    "using high-quality ASCII layouts."
                ),
            },
            {"role": "user", "content": args.prompt},
        ],
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    content = completion.choices[0].message.content or ""
    print(content.strip())


if __name__ == "__main__":
    main()
