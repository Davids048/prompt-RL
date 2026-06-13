#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any


def load_prompts(task: dict[str, Any]) -> list[dict[str, Any]]:
    source_path = Path(task["source_path"])
    rows: list[tuple[int, str]] = []
    with source_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            prompt = line.strip()
            if prompt:
                rows.append((idx, prompt))

    selected = select_subset(rows, task.get("subset", {"mode": "full"}))
    prompt_rows: list[dict[str, Any]] = []
    for source_idx, prompt in selected:
        prompt_rows.append({
            "prompt_index": int(source_idx),
            "prompt_source": "txt",
            "prompt_source_path": str(source_path),
            "prompt_source_index": int(source_idx),
            "original_prompt": prompt,
            "prompt_metadata": {
                "line_index": int(source_idx),
            },
        })
    return prompt_rows


def select_subset(
    rows: list[tuple[int, str]],
    subset: dict[str, Any],
) -> list[tuple[int, str]]:
    mode = subset.get("mode", "full")
    if mode == "full":
        return rows
    if mode == "first_n":
        return rows[:int(subset["n"])]
    raise ValueError(f"Unsupported txt subset mode: {mode!r}")
