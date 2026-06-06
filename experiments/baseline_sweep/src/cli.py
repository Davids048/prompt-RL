#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import config as config_mod
import planner
import runner
from reports.summary import write_summary


def cmd_run(args: argparse.Namespace) -> None:
    config_path = Path(args.config)
    resolved = config_mod.resolve_config(config_path)
    run_dir = config_mod.run_dir(resolved)
    assert_existing_run_compatible(run_dir, resolved)
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, run_dir / "config.yaml")
    config_mod.write_yaml(run_dir / "resolved_config.yaml", resolved)
    trials = planner.expand_trials(resolved)
    planner.write_run_plan(run_dir / "run_plan.jsonl", trials)
    print(f"[baseline_sweep] run_dir={run_dir}")
    print(f"[baseline_sweep] planned_trials={len(trials)}")
    for trial in trials:
        runner.run_trial(trial, run_dir)
    write_summary(run_dir)
    print(f"[baseline_sweep] summary={run_dir / 'summary.md'}")


def assert_existing_run_compatible(run_dir: Path, resolved: dict) -> None:
    old_resolved_path = run_dir / "resolved_config.yaml"
    records_path = run_dir / "records.jsonl"
    if not old_resolved_path.exists() or not records_path.exists():
        return
    old_resolved = config_mod.load_yaml(old_resolved_path)
    if config_fingerprint(old_resolved) == config_fingerprint(resolved):
        return
    raise SystemExit(
        "Existing records were found for this run name, but the resolved config changed. "
        f"Use a new run name or move the old run directory first: {run_dir}"
    )


def config_fingerprint(value: dict) -> str:
    normalized = normalize_for_fingerprint(value)
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_for_fingerprint(value):
    if isinstance(value, dict):
        return {
            key: normalize_for_fingerprint(item)
            for key, item in sorted(value.items())
            if key not in {"resolved_at_utc", "config_source"}
        }
    if isinstance(value, list):
        return [normalize_for_fingerprint(item) for item in value]
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Baseline Sweep experiments.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("config")
    run.set_defaults(func=cmd_run)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
