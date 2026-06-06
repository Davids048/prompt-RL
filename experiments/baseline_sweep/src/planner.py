#!/usr/bin/env python3
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def expand_trials(config: dict[str, Any]) -> list[dict[str, Any]]:
    sweep = config["sweep"]
    enhancers = as_list(sweep.get("enhancer"))
    generators = as_list(sweep.get("generator"))
    tasks = as_list(sweep.get("task"))
    trials: list[dict[str, Any]] = []
    counter = 1
    for task_name in tasks:
        if task_name not in config["tasks"]:
            raise ValueError(f"Unknown task in sweep: {task_name!r}")
        for generator_name in generators:
            if generator_name not in config["generators"]:
                raise ValueError(f"Unknown generator in sweep: {generator_name!r}")
            for enhancer_name in enhancers:
                if enhancer_name not in config["enhancers"]:
                    raise ValueError(f"Unknown enhancer in sweep: {enhancer_name!r}")
                trials.append({
                    "trial_id": f"trial_{counter:06d}",
                    "run": config["run"],
                    "seed_base": int(config["seed"]),
                    "task": deepcopy(config["tasks"][task_name]),
                    "generator": deepcopy(config["generators"][generator_name]),
                    "enhancer": deepcopy(config["enhancers"][enhancer_name]),
                })
                counter += 1
    return trials


def write_run_plan(path: Path, trials: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in trials:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_run_plan(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
