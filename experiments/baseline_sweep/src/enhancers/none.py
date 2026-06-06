#!/usr/bin/env python3
from __future__ import annotations

from typing import Any


def enhance(
    prompt_rows: list[dict[str, Any]],
    enhancer: dict[str, Any],
    trial: dict[str, Any],
    run_dir: Any,
) -> list[dict[str, Any]]:
    del enhancer, trial, run_dir
    rows: list[dict[str, Any]] = []
    for row in prompt_rows:
        item = dict(row)
        item.update({
            "enhancer_model": None,
            "enhancer_alias": "none",
            "enhancer_backend": "none",
            "enhancer_template": None,
            "enhancer_params": {},
            "enhanced_prompt": None,
            "generation_prompt": row["original_prompt"],
            "eval_prompt": row["original_prompt"],
            "enhancement_error": None,
        })
        rows.append(item)
    return rows
