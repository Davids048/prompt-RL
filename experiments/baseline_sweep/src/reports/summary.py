#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def write_summary(run_dir: Path) -> None:
    records = read_records(run_dir / "records.jsonl")
    metrics: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    counts = {
        "records": len(records),
        "successful": 0,
        "failed_generation": 0,
        "failed_eval": 0,
        "failed_enhancement": 0,
        "failed_other": 0,
    }
    for row in records:
        error = row.get("error")
        if error:
            stage = error.get("stage")
            if stage == "generation":
                counts["failed_generation"] += 1
            elif stage == "eval":
                counts["failed_eval"] += 1
            elif stage == "enhancement":
                counts["failed_enhancement"] += 1
            else:
                counts["failed_other"] += 1
            continue
        if not row.get("eval"):
            continue
        counts["successful"] += 1
        key = trial_key(row)
        for eval_name, eval_result in row.get("eval", {}).items():
            for metric, value in eval_result.items():
                if metric == "source":
                    continue
                if isinstance(value, int | float):
                    metrics[key][f"{eval_name}_{metric}"].append(float(value))

    summary = {
        "counts": counts,
        "trials": {
            key: {
                metric: {
                    "mean": mean(values),
                    "count": len(values),
                }
                for metric, values in sorted(metric_values.items())
            }
            for key, metric_values in sorted(metrics.items())
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (run_dir / "summary.md").write_text(summary_markdown(summary), encoding="utf-8")


def read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def trial_key(row: dict[str, Any]) -> str:
    enhancer = row.get("enhancer_alias") or "none"
    return f"{row.get('task')}/{row.get('generator_alias')}/{enhancer}"


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summary_markdown(summary: dict[str, Any]) -> str:
    lines = ["# Baseline Sweep Summary", ""]
    lines.append("## Counts")
    lines.append("")
    lines.append("| Field | Count |")
    lines.append("| ----- | ----: |")
    for key, value in summary["counts"].items():
        lines.append(f"| `{key}` | {value} |")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    if not summary["trials"]:
        lines.append("No successful eval records yet.")
        lines.append("")
        return "\n".join(lines)
    lines.append("| Trial | Metric | Mean | Count |")
    lines.append("| ----- | ------ | ---: | ----: |")
    for trial, metrics in summary["trials"].items():
        for metric, stat in metrics.items():
            mean_value = stat["mean"]
            mean_cell = "" if mean_value is None else f"{mean_value:.6f}"
            lines.append(f"| `{trial}` | `{metric}` | {mean_cell} | {stat['count']} |")
    lines.append("")
    return "\n".join(lines)
