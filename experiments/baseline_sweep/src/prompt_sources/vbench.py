#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_prompts(task: dict[str, Any]) -> list[dict[str, Any]]:
    source_path = Path(task["source_path"])
    subset = task.get("subset", {"mode": "official_dimensions"})
    mode = subset.get("mode", "official_dimensions")
    if mode != "official_dimensions":
        raise ValueError(f"Unsupported VBench subset mode: {mode!r}")

    dimensions = subset.get("dimensions")
    if not dimensions:
        raise ValueError("VBench official_dimensions subset requires dimensions.")

    requested_dimensions = set(str(dimension) for dimension in dimensions)
    full_info_path = vbench_full_info_path(task, source_path)
    full_info = load_json(full_info_path)
    per_dimension_n = subset.get("per_dimension_n")
    dimension_counts = {dimension: 0 for dimension in requested_dimensions}
    prompt_rows: list[dict[str, Any]] = []
    rows_by_prompt: dict[str, dict[str, Any]] = {}
    for source_idx, prompt_info in enumerate(full_info):
        source_dimensions = [str(dimension) for dimension in prompt_info.get("dimension", [])]
        selected_dimensions = [
            dimension
            for dimension in source_dimensions
            if dimension in requested_dimensions
        ]
        if per_dimension_n is not None:
            limit = int(per_dimension_n)
            selected_dimensions = [
                dimension
                for dimension in selected_dimensions
                if dimension_counts[dimension] < limit
            ]
        if not selected_dimensions:
            continue
        original_prompt = str(prompt_info["prompt_en"])
        existing = rows_by_prompt.get(original_prompt)
        if existing is None:
            existing = {
                "prompt_index": len(prompt_rows),
                "prompt_source": "vbench",
                "prompt_source_path": str(full_info_path),
                "prompt_source_index": int(source_idx),
                "original_prompt": original_prompt,
                "prompt_metadata": {
                    "dimensions": [],
                    "source_dimensions": [],
                    "prompt_source_indices": [],
                },
            }
            rows_by_prompt[original_prompt] = existing
            prompt_rows.append(existing)
        metadata = existing["prompt_metadata"]
        metadata["prompt_source_indices"].append(int(source_idx))
        for dimension in selected_dimensions:
            if dimension not in metadata["dimensions"]:
                metadata["dimensions"].append(dimension)
                dimension_counts[dimension] += 1
        for dimension in source_dimensions:
            if dimension not in metadata["source_dimensions"]:
                metadata["source_dimensions"].append(dimension)
        if subset.get("n") is not None and len(prompt_rows) >= int(subset["n"]):
            break
    return prompt_rows


def vbench_full_info_path(task: dict[str, Any], prompt_root: Path) -> Path:
    params = task.get("eval_params", {}).get("vbench", {})
    configured = (
        task.get("full_info_path")
        or task.get("full_json_dir")
        or params.get("full_info_path")
        or params.get("full_json_dir")
    )
    if configured:
        full_info_path = Path(configured)
    else:
        full_info_path = prompt_root.parent / "vbench" / "VBench_full_info.json"
    if not full_info_path.exists():
        raise FileNotFoundError(f"VBench full-info file not found: {full_info_path}")
    return full_info_path


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list.")
    return data
