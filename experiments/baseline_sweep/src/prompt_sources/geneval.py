#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_prompts(task: dict[str, Any]) -> list[dict[str, Any]]:
    source_path = Path(task["source_path"])
    rows: list[tuple[int, dict[str, Any]]] = []
    with source_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rows.append((idx, json.loads(line)))

    selected = select_subset(rows, task.get("subset", {"mode": "full"}))
    prompt_rows: list[dict[str, Any]] = []
    for source_idx, metadata in selected:
        prompt_rows.append({
            "prompt_index": int(source_idx),
            "prompt_source": "geneval",
            "prompt_source_path": str(source_path),
            "prompt_source_index": int(source_idx),
            "original_prompt": metadata["prompt"],
            "prompt_metadata": metadata,
        })
    return prompt_rows


def select_subset(
    rows: list[tuple[int, dict[str, Any]]],
    subset: dict[str, Any],
) -> list[tuple[int, dict[str, Any]]]:
    mode = subset.get("mode", "full")
    if mode == "full":
        return rows
    if mode == "first_n":
        return rows[:int(subset["n"])]
    if mode == "balanced_by_tag":
        per_tag = int(subset["per_tag"])
        wanted_tags = subset.get("tags")
        wanted = set(wanted_tags) if wanted_tags else None
        counts: dict[str, int] = {}
        selected: list[tuple[int, dict[str, Any]]] = []
        for idx, metadata in rows:
            tag = metadata.get("tag", "")
            if wanted is not None and tag not in wanted:
                continue
            if counts.get(tag, 0) >= per_tag:
                continue
            selected.append((idx, metadata))
            counts[tag] = counts.get(tag, 0) + 1
        return selected
    raise ValueError(f"Unsupported GenEval subset mode: {mode!r}")
