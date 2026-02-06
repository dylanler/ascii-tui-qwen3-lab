#!/usr/bin/env python3
"""Generate a synthetic instruction dataset with 100 parallel GPT agents.

The output is JSONL for training/evaluation:
- data/raw/synthetic_ascii_tui.jsonl
- data/processed/train.jsonl
- data/processed/eval.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import math
import os
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import orjson
import yaml
from jsonschema import ValidationError, validate
from openai import AsyncOpenAI
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter


SYSTEM_PROMPT = """You create exceptional educational text diagrams.

Output ONLY JSON, no prose.
Each example must:
1) Teach the requested science/engineering topic.
2) Include both plain-language explanation and high-quality ASCII/TUI-style diagram blocks.
3) Be readable in a terminal (max width ~90 chars).
4) Use varied structures: timelines, flowcharts, stacked layers, side-by-side comparisons, and causal chains.
5) Be accurate, concise, and visually impressive.
"""


JSON_RESPONSE_SPEC = """Return this exact top-level JSON object:
{
  "examples": [
    {
      "topic": "string",
      "instruction": "string",
      "response": "string",
      "diagram_style": "ascii_art|tui_flow|hybrid",
      "difficulty": "beginner|intermediate|advanced",
      "tags": ["string", "string"]
    }
  ]
}
"""


EXAMPLE_SCHEMA = {
    "type": "object",
    "required": ["topic", "instruction", "response", "diagram_style", "difficulty", "tags"],
    "properties": {
        "topic": {"type": "string", "minLength": 3},
        "instruction": {"type": "string", "minLength": 20},
        "response": {"type": "string", "minLength": 80},
        "diagram_style": {"type": "string", "enum": ["ascii_art", "tui_flow", "hybrid"]},
        "difficulty": {"type": "string", "enum": ["beginner", "intermediate", "advanced"]},
        "tags": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 8},
    },
}


AGENT_PERSONAS = [
    "terminal-native science teacher",
    "systems engineer who teaches with box-drawing diagrams",
    "physics explainer focused on intuition-first visuals",
    "biology educator who uses lifecycle timelines",
    "math-first tutor who compresses concepts into flow states",
    "visual storyteller for command-line learners",
    "careful experimental scientist with stepwise diagrams",
    "educational UX writer for text-mode interfaces",
]


@dataclass
class GenerationConfig:
    model: str
    target_rows: int
    agent_count: int
    rows_per_call: int
    concurrency: int
    temperature: float
    max_output_tokens: int
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic ASCII/TUI training data.")
    parser.add_argument("--topics-config", type=Path, default=Path("configs/topics.yaml"))
    parser.add_argument("--raw-output", type=Path, default=Path("data/raw/synthetic_ascii_tui.jsonl"))
    parser.add_argument("--train-output", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument("--eval-output", type=Path, default=Path("data/processed/eval.jsonl"))
    parser.add_argument("--target-rows", type=int, default=1000)
    parser.add_argument("--agent-count", type=int, default=100)
    parser.add_argument("--rows-per-call", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=100)
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    parser.add_argument("--model", type=str, default="gpt-5.3")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max-output-tokens", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_topics(path: Path) -> tuple[list[str], list[str]]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    required = list(dict.fromkeys(data.get("required_topics", [])))
    supplemental = list(dict.fromkeys(data.get("supplemental_topics", [])))
    if not required:
        raise ValueError("At least one required topic is needed.")
    if not supplemental:
        raise ValueError("At least one supplemental topic is needed.")
    return required, supplemental


def build_topic_plan(required: list[str], supplemental: list[str], target_rows: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    # Guarantee broad coverage of required topics.
    guaranteed_per_required = max(20, target_rows // (len(required) * 4))
    plan = [topic for topic in required for _ in range(guaranteed_per_required)]
    remaining = max(0, target_rows - len(plan))
    topic_pool = required + supplemental
    plan.extend(rng.choices(topic_pool, k=remaining))
    rng.shuffle(plan)
    return plan[:target_rows]


def assign_topics_to_agents(topic_plan: list[str], agent_count: int, seed: int) -> list[list[str]]:
    rng = random.Random(seed + 1)
    assignments = [[] for _ in range(agent_count)]
    for idx, topic in enumerate(topic_plan):
        assignments[idx % agent_count].append(topic)
    for topics in assignments:
        rng.shuffle(topics)
    return assignments


def extract_json_payload(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return orjson.loads(cleaned)
    except orjson.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in model response.") from None
        return orjson.loads(cleaned[start : end + 1])


def extract_output_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text
    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                chunks.append(text)
    if chunks:
        return "\n".join(chunks)
    raise ValueError("Empty model response.")


def normalize_row(row: dict[str, Any], topic_hint: str, agent_id: int) -> dict[str, Any]:
    out = {
        "topic": str(row.get("topic") or topic_hint).strip(),
        "instruction": str(row.get("instruction", "")).strip(),
        "response": str(row.get("response", "")).strip(),
        "diagram_style": str(row.get("diagram_style", "hybrid")).strip(),
        "difficulty": str(row.get("difficulty", "intermediate")).strip(),
        "tags": [str(tag).strip() for tag in row.get("tags", []) if str(tag).strip()],
        "agent_id": agent_id,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    validate(instance=out, schema=EXAMPLE_SCHEMA)
    if "\n" not in out["response"]:
        raise ValidationError("Response must include multiline terminal content.")
    return out


async def request_examples(
    client: AsyncOpenAI,
    cfg: GenerationConfig,
    agent_id: int,
    topics: list[str],
) -> list[dict[str, Any]]:
    persona = AGENT_PERSONAS[agent_id % len(AGENT_PERSONAS)]
    user_prompt = (
        f"Agent #{agent_id} persona: {persona}.\n"
        f"Create {len(topics)} examples, one per topic in this exact order:\n"
        + "\n".join(f"{idx + 1}. {topic}" for idx, topic in enumerate(topics))
        + "\n\n"
        + JSON_RESPONSE_SPEC
        + "\nDo not include markdown fences."
    )

    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(5),
        wait=wait_exponential_jitter(initial=1, max=20),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    ):
        with attempt:
            response = await client.responses.create(
                model=cfg.model,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=cfg.temperature,
                max_output_tokens=cfg.max_output_tokens,
            )
            text = extract_output_text(response)
            payload = extract_json_payload(text)
            examples_raw = payload.get("examples", payload if isinstance(payload, list) else [])
            if not isinstance(examples_raw, list):
                raise ValueError("Model output does not include an examples list.")

            rows: list[dict[str, Any]] = []
            for idx, item in enumerate(examples_raw):
                if not isinstance(item, dict):
                    continue
                topic_hint = topics[min(idx, len(topics) - 1)]
                try:
                    rows.append(normalize_row(item, topic_hint=topic_hint, agent_id=agent_id))
                except ValidationError:
                    continue
            if not rows:
                raise ValueError("No valid rows produced in response.")
            return rows


async def run_agent(
    client: AsyncOpenAI,
    cfg: GenerationConfig,
    agent_id: int,
    topics: list[str],
    sem: asyncio.Semaphore,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i in range(0, len(topics), cfg.rows_per_call):
        batch_topics = topics[i : i + cfg.rows_per_call]
        async with sem:
            rows = await request_examples(client, cfg, agent_id=agent_id, topics=batch_topics)
        out.extend(rows)
    return out


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    uniq: list[dict[str, Any]] = []
    for row in rows:
        key = (
            re.sub(r"\s+", " ", row["instruction"].lower()).strip()
            + " || "
            + re.sub(r"\s+", " ", row["response"].lower()).strip()
        )
        if key in seen:
            continue
        seen.add(key)
        uniq.append(row)
    return uniq


def split_rows(rows: list[dict[str, Any]], eval_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed + 7)
    shuffled = rows.copy()
    rng.shuffle(shuffled)
    eval_size = max(1, int(len(shuffled) * eval_ratio))
    eval_rows = shuffled[:eval_size]
    train_rows = shuffled[eval_size:]
    return train_rows, eval_rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        for idx, row in enumerate(rows):
            row = dict(row)
            row["id"] = f"ascii_tui_{idx:05d}"
            f.write(orjson.dumps(row))
            f.write(b"\n")


async def async_main(args: argparse.Namespace) -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is required.")
    if args.agent_count != 100:
        raise ValueError("Use --agent-count 100 to satisfy the 100-agent requirement.")

    required_topics, supplemental_topics = load_topics(args.topics_config)
    topic_plan = build_topic_plan(required_topics, supplemental_topics, args.target_rows, args.seed)
    assignments = assign_topics_to_agents(topic_plan, args.agent_count, args.seed)

    cfg = GenerationConfig(
        model=args.model,
        target_rows=args.target_rows,
        agent_count=args.agent_count,
        rows_per_call=args.rows_per_call,
        concurrency=args.concurrency,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        seed=args.seed,
    )

    sem = asyncio.Semaphore(args.concurrency)
    client = AsyncOpenAI()

    tasks = [
        run_agent(client, cfg=cfg, agent_id=agent_id, topics=topics, sem=sem)
        for agent_id, topics in enumerate(assignments)
        if topics
    ]
    per_agent_rows = await asyncio.gather(*tasks)
    rows = [row for chunk in per_agent_rows for row in chunk]
    rows = dedupe_rows(rows)
    if len(rows) < args.target_rows:
        raise RuntimeError(
            f"Generated {len(rows)} rows after dedupe, below target {args.target_rows}. "
            "Increase retries or run again."
        )
    rows = rows[: args.target_rows]

    train_rows, eval_rows = split_rows(rows, eval_ratio=args.eval_ratio, seed=args.seed)

    write_jsonl(args.raw_output, rows)
    write_jsonl(args.train_output, train_rows)
    write_jsonl(args.eval_output, eval_rows)

    print(f"Generated raw rows: {len(rows)} -> {args.raw_output}")
    print(f"Train rows: {len(train_rows)} -> {args.train_output}")
    print(f"Eval rows: {len(eval_rows)} -> {args.eval_output}")

    topic_counts: dict[str, int] = {}
    for row in rows:
        topic_counts[row["topic"]] = topic_counts.get(row["topic"], 0) + 1
    print("Top topic coverage:")
    for topic, count in sorted(topic_counts.items(), key=lambda kv: kv[1], reverse=True)[:15]:
        print(f"  {topic}: {count}")


def main() -> None:
    args = parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()

