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
from reports.prompt_video import write_html_report
from reports.summary import write_summary


def cmd_run(args: argparse.Namespace) -> None:
    run_dir, trials = prepare_run(Path(args.config))
    for trial in trials:
        runner.run_trial(trial, run_dir)
    write_summary(run_dir)
    print(f"[baseline_sweep] summary={run_dir / 'summary.md'}")


def cmd_prepare(args: argparse.Namespace) -> None:
    prepare_run(Path(args.config))


def cmd_list_trials(args: argparse.Namespace) -> None:
    for trial in planner.read_run_plan(Path(args.run_dir) / "run_plan.jsonl"):
        print(trial["trial_id"])


def cmd_run_trial(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    trial = find_trial(run_dir, args.trial_id)
    runner.run_trial(trial, run_dir)


def cmd_summary(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    write_summary(run_dir)
    print(f"[baseline_sweep] summary={run_dir / 'summary.md'}")


def cmd_html_report(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    output = Path(args.output) if args.output else None
    report_path = write_html_report(run_dir, output)
    print(f"[baseline_sweep] html_report={report_path}")


def prepare_run(config_path: Path) -> tuple[Path, list[dict]]:
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
    return run_dir, trials


def find_trial(run_dir: Path, trial_id: str) -> dict:
    run_plan = run_dir / "run_plan.jsonl"
    for trial in planner.read_run_plan(run_plan):
        if trial["trial_id"] == trial_id:
            return trial
    raise SystemExit(f"Trial {trial_id!r} not found in {run_plan}")


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
    prepare = sub.add_parser("prepare")
    prepare.add_argument("config")
    prepare.set_defaults(func=cmd_prepare)

    list_trials = sub.add_parser("list-trials")
    list_trials.add_argument("run_dir")
    list_trials.set_defaults(func=cmd_list_trials)

    run_trial = sub.add_parser("run-trial")
    run_trial.add_argument("run_dir")
    run_trial.add_argument("trial_id")
    run_trial.set_defaults(func=cmd_run_trial)

    summary = sub.add_parser("summary")
    summary.add_argument("run_dir")
    summary.set_defaults(func=cmd_summary)

    html_report = sub.add_parser("html-report")
    html_report.add_argument("run_dir")
    html_report.add_argument("--output", default=None)
    html_report.set_defaults(func=cmd_html_report)

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
