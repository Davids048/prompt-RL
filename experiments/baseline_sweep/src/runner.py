#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import re
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import registry


def run_trial(trial: dict[str, Any], run_dir: Path) -> None:
    drop_closed_log_handlers()
    log_path = run_dir / "logs" / trial_log_name(trial)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        with redirect_stdout(log_file), redirect_stderr(log_file):
            print(f"[runner] start {trial['trial_id']}")
            _run_trial_inner(trial, run_dir, str(log_path.relative_to(run_dir)))
            print(f"[runner] done {trial['trial_id']}")
    drop_closed_log_handlers()


def drop_closed_log_handlers() -> None:
    loggers = [logging.getLogger()]
    loggers.extend(
        logger
        for logger in logging.Logger.manager.loggerDict.values()
        if isinstance(logger, logging.Logger)
    )
    for logger in loggers:
        for handler in list(logger.handlers):
            stream = getattr(handler, "stream", None)
            if getattr(stream, "closed", False):
                logger.removeHandler(handler)
                handler.close()


def _run_trial_inner(trial: dict[str, Any], run_dir: Path, log_rel_path: str) -> None:
    task = trial["task"]
    prompt_loader = registry.PROMPT_SOURCES[task["prompt_source"]]
    enhancer_fn = registry.ENHANCERS[trial["enhancer"]["backend"]]
    generator_fn = registry.GENERATORS[trial["generator"]["backend"]]

    compatibility_error = validate_trial_compatibility(trial)
    if compatibility_error:
        append_final_records(run_dir / "records.jsonl", [trial_error_record(trial, "config", compatibility_error, log_rel_path)])
        return

    try:
        prompt_rows = prompt_loader(task)
    except Exception as exc:  # noqa: BLE001
        append_final_records(run_dir / "records.jsonl", [trial_error_record(trial, "prompt_source", repr(exc), log_rel_path)])
        return

    finished = finished_attempts(run_dir, trial)
    expected = expected_attempts(prompt_rows, task)
    if expected and expected.issubset(finished):
        print(f"[runner] skip_completed_attempts={len(expected)}")
        return

    try:
        enhanced_rows = apply_enhancement_reuse(prompt_rows, trial, run_dir, enhancer_fn)
    except Exception as exc:  # noqa: BLE001
        enhanced_rows = failed_enhancement_rows(prompt_rows, trial, repr(exc))
    generation_items = build_generation_items(enhanced_rows, trial, log_rel_path, finished)
    if finished:
        print(f"[runner] existing_final_attempts={len(finished)}")
    generation_ready = [row for row in generation_items if row.get("generation_prompt") and not row.get("error")]
    pre_failed = [row for row in generation_items if row.get("error")]
    existing_artifacts, generation_missing = split_existing_artifacts(generation_ready, run_dir)
    if existing_artifacts:
        print(f"[runner] reused_artifacts_for_eval={len(existing_artifacts)}")
    try:
        generated = generator_fn(generation_missing, trial["generator"], trial, run_dir) if generation_missing else []
    except Exception as exc:  # noqa: BLE001
        generated = failed_generation_rows(generation_missing, repr(exc))
    records = pre_failed + existing_artifacts + generated

    for eval_name in task.get("eval", []):
        eval_fn = registry.EVALS[eval_name]
        try:
            records = eval_fn(records, trial, run_dir)
        except Exception as exc:  # noqa: BLE001
            records = failed_eval_rows(records, eval_name, repr(exc))

    records = attach_log_path(records, log_rel_path)
    if records:
        append_final_records(run_dir / "records.jsonl", records)


def expected_attempts(prompt_rows: list[dict[str, Any]], task: dict[str, Any]) -> set[tuple[int, int]]:
    samples_per_prompt = int(task.get("samples_per_prompt", 1))
    return {
        (int(row["prompt_index"]), sample_idx)
        for row in prompt_rows
        for sample_idx in range(samples_per_prompt)
    }


def finished_attempts(run_dir: Path, trial: dict[str, Any]) -> set[tuple[int, int]]:
    path = run_dir / "records.jsonl"
    if not path.exists():
        return set()

    finished: set[tuple[int, int]] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not same_trial_settings(row, trial):
                continue
            if row.get("prompt_index") is None or row.get("sample_index") is None:
                continue
            key = (int(row["prompt_index"]), int(row["sample_index"]))
            if artifact_and_eval_exist(row, trial, run_dir):
                finished.add(key)
    return finished


def same_trial_settings(row: dict[str, Any], trial: dict[str, Any]) -> bool:
    task = trial["task"]
    generator = trial["generator"]
    enhancer = trial["enhancer"]
    if row.get("trial_id") != trial["trial_id"]:
        return False
    if row.get("task") != task["name"]:
        return False
    if row.get("generator_model") != generator["name"]:
        return False
    if row.get("generator_backend") != generator["backend"]:
        return False
    if row.get("generator_params") != generator.get("params", {}):
        return False
    expected_enhancer_model = None if enhancer["backend"] == "none" else enhancer["name"]
    if row.get("enhancer_model") != expected_enhancer_model:
        return False
    if row.get("enhancer_backend") != enhancer["backend"]:
        return False
    if row.get("enhancer_template") != enhancer.get("template"):
        return False
    if row.get("enhancer_params") != enhancer.get("params", {}):
        return False
    if row.get("prompt_index") is None or row.get("sample_index") is None:
        return False
    samples_per_prompt = int(task.get("samples_per_prompt", 1))
    expected_seed = int(trial["seed_base"]) + int(row["prompt_index"]) * samples_per_prompt + int(row["sample_index"])
    return row.get("seed") == expected_seed


def artifact_and_eval_exist(row: dict[str, Any], trial: dict[str, Any], run_dir: Path) -> bool:
    artifact_path = row.get("artifact_path")
    if not artifact_path:
        return False
    artifact = Path(artifact_path)
    if not artifact.is_absolute():
        artifact = run_dir / artifact
    if not artifact.exists():
        return False
    for eval_name in trial["task"].get("eval", []):
        eval_result = row.get("eval", {}).get(eval_name)
        if not eval_result:
            return False
        if not eval_source_has_record(row, eval_result, run_dir):
            return False
    return True


def eval_source_has_record(row: dict[str, Any], eval_result: dict[str, Any], run_dir: Path) -> bool:
    source = eval_result.get("source")
    if not source:
        return True
    source_path = Path(source)
    if not source_path.is_absolute():
        source_path = run_dir / source_path
    if not source_path.exists():
        return False
    try:
        with source_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                if int(raw.get("prompt_index", -1)) != int(row["prompt_index"]):
                    continue
                if int(raw.get("sample_index", -1)) != int(row["sample_index"]):
                    continue
                if raw.get("artifact_path") != row.get("artifact_path"):
                    continue
                return True
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return False


def apply_enhancement_reuse(
    prompt_rows: list[dict[str, Any]],
    trial: dict[str, Any],
    run_dir: Path,
    enhancer_fn: Any,
) -> list[dict[str, Any]]:
    enhancer = trial["enhancer"]
    if enhancer["backend"] == "none":
        return enhancer_fn(prompt_rows, enhancer, trial, run_dir)

    reusable = existing_enhancements(run_dir, enhancer)
    reused: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for row in prompt_rows:
        key = row["original_prompt"]
        if key in reusable:
            item = dict(row)
            item.update(reusable[key])
            reused.append(item)
        else:
            missing.append(row)
    return reused + enhancer_fn(missing, enhancer, trial, run_dir)


def existing_enhancements(run_dir: Path, enhancer: dict[str, Any]) -> dict[str, dict[str, Any]]:
    path = run_dir / "records.jsonl"
    if not path.exists():
        return {}
    reusable: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("enhancer_model") != enhancer.get("name"):
                continue
            if row.get("enhancer_backend") != enhancer.get("backend"):
                continue
            if row.get("enhancer_template") != enhancer.get("template"):
                continue
            if row.get("enhancer_params") != enhancer.get("params", {}):
                continue
            if row.get("enhanced_prompt"):
                reusable[row["original_prompt"]] = {
                    "enhancer_model": row.get("enhancer_model"),
                    "enhancer_alias": row.get("enhancer_alias"),
                    "enhancer_backend": row.get("enhancer_backend"),
                    "enhancer_template": row.get("enhancer_template"),
                    "enhancer_params": row.get("enhancer_params", {}),
                    "enhanced_prompt": row.get("enhanced_prompt"),
                    "generation_prompt": row.get("enhanced_prompt"),
                    "eval_prompt": row.get("original_prompt"),
                    "enhancement_error": None,
                }
    return reusable


def failed_enhancement_rows(
    prompt_rows: list[dict[str, Any]],
    trial: dict[str, Any],
    message: str,
) -> list[dict[str, Any]]:
    enhancer = trial["enhancer"]
    rows: list[dict[str, Any]] = []
    for row in prompt_rows:
        item = dict(row)
        item.update({
            "enhancer_model": None if enhancer["backend"] == "none" else enhancer["name"],
            "enhancer_alias": enhancer["alias"],
            "enhancer_backend": enhancer["backend"],
            "enhancer_template": enhancer.get("template"),
            "enhancer_params": enhancer.get("params", {}),
            "enhanced_prompt": None,
            "generation_prompt": None,
            "eval_prompt": row["original_prompt"],
            "enhancement_error": message,
        })
        rows.append(item)
    return rows


def build_generation_items(
    enhanced_rows: list[dict[str, Any]],
    trial: dict[str, Any],
    log_rel_path: str,
    finished: set[tuple[int, int]],
) -> list[dict[str, Any]]:
    task = trial["task"]
    generator = trial["generator"]
    enhancer = trial["enhancer"]
    samples_per_prompt = int(task.get("samples_per_prompt", 1))
    artifact_ext = artifact_extension(generator)
    items: list[dict[str, Any]] = []
    for row in enhanced_rows:
        for sample_idx in range(samples_per_prompt):
            key = (int(row["prompt_index"]), sample_idx)
            if key in finished:
                continue
            seed = int(trial["seed_base"]) + int(row["prompt_index"]) * samples_per_prompt + sample_idx
            artifact_path = (
                Path("artifacts")
                / trial["trial_id"]
                / f"{int(row['prompt_index']):06d}_{sample_idx:04d}_{safe_name(task['name'])}{artifact_ext}"
            )
            item = {
                "trial_id": trial["trial_id"],
                "task": task["name"],
                "prompt_index": int(row["prompt_index"]),
                "sample_index": sample_idx,
                "prompt_source": row["prompt_source"],
                "prompt_source_path": row["prompt_source_path"],
                "prompt_source_index": row["prompt_source_index"],
                "original_prompt": row["original_prompt"],
                "prompt_metadata": row["prompt_metadata"],
                "enhancer_model": None if enhancer["backend"] == "none" else enhancer["name"],
                "enhancer_alias": enhancer["alias"],
                "enhancer_backend": enhancer["backend"],
                "enhancer_template": enhancer.get("template"),
                "enhancer_params": enhancer.get("params", {}),
                "enhanced_prompt": row.get("enhanced_prompt"),
                "generation_prompt": row.get("generation_prompt"),
                "eval_prompt": row["original_prompt"],
                "generator_model": generator["name"],
                "generator_alias": generator["alias"],
                "generator_backend": generator["backend"],
                "generator_params": generator.get("params", {}),
                "seed": seed,
                "artifact_path": str(artifact_path),
                "eval": {},
                "error": None,
            }
            if row.get("enhancement_error"):
                item["artifact_path"] = None
                item["error"] = {
                    "stage": "enhancement",
                    "message": row["enhancement_error"],
                    "log_path": log_rel_path,
                }
            items.append(item)
    return items


def split_existing_artifacts(
    rows: list[dict[str, Any]],
    run_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    existing: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for row in rows:
        artifact_path = row.get("artifact_path")
        if artifact_path and (run_dir / artifact_path).exists():
            item = dict(row)
            item["artifact_reused"] = True
            item["generation_wall_time_sec"] = 0.0
            existing.append(item)
        else:
            missing.append(row)
    return existing, missing


def failed_generation_rows(rows: list[dict[str, Any]], message: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["artifact_path"] = None
        item["error"] = {
            "stage": "generation",
            "message": message,
        }
        out.append(item)
    return out


def failed_eval_rows(rows: list[dict[str, Any]], eval_name: str, message: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        if item.get("artifact_path") and not item.get("error"):
            item["eval"] = {}
            item["error"] = {
                "stage": "eval",
                "eval": eval_name,
                "message": message,
            }
        out.append(item)
    return out


def trial_error_record(
    trial: dict[str, Any],
    stage: str,
    message: str,
    log_rel_path: str,
) -> dict[str, Any]:
    task = trial["task"]
    generator = trial["generator"]
    enhancer = trial["enhancer"]
    return {
        "trial_id": trial["trial_id"],
        "task": task["name"],
        "prompt_index": None,
        "sample_index": None,
        "prompt_source": task.get("prompt_source"),
        "prompt_source_path": task.get("source_path"),
        "prompt_source_index": None,
        "original_prompt": None,
        "prompt_metadata": {},
        "enhancer_model": None if enhancer["backend"] == "none" else enhancer["name"],
        "enhancer_alias": enhancer["alias"],
        "enhancer_backend": enhancer["backend"],
        "enhancer_template": enhancer.get("template"),
        "enhancer_params": enhancer.get("params", {}),
        "enhanced_prompt": None,
        "generation_prompt": None,
        "eval_prompt": None,
        "generator_model": generator["name"],
        "generator_alias": generator["alias"],
        "generator_backend": generator["backend"],
        "generator_params": generator.get("params", {}),
        "seed": None,
        "artifact_path": None,
        "eval": {},
        "error": {
            "stage": stage,
            "message": message,
            "log_path": log_rel_path,
        },
    }


def attach_log_path(rows: list[dict[str, Any]], log_rel_path: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        error = item.get("error")
        if error and "log_path" not in error:
            error = dict(error)
            error["log_path"] = log_rel_path
            item["error"] = error
        out.append(item)
    return out


def validate_trial_compatibility(trial: dict[str, Any]) -> str | None:
    task = trial["task"]
    generator = trial["generator"]
    workload_type = generator_workload_type(generator)
    if workload_type not in {"t2i", "i2i", "t2v", "i2v"}:
        return f"Unsupported FastVideo workload_type={workload_type!r}; expected one of t2i, i2i, t2v, i2v."
    if "geneval" in task.get("eval", []) and not is_image_workload(workload_type):
        return f"GenEval requires an image workload, but generator workload_type={workload_type!r}."
    return None


def artifact_extension(generator: dict[str, Any]) -> str:
    workload_type = generator_workload_type(generator)
    if is_image_workload(workload_type):
        return ".png"
    if workload_type.endswith("2v"):
        return ".mp4"
    raise ValueError(f"Unsupported FastVideo workload_type={workload_type!r}.")


def generator_workload_type(generator: dict[str, Any]) -> str:
    return str(generator.get("params", {}).get("workload_type", "t2i")).lower()


def is_image_workload(workload_type: str) -> bool:
    return workload_type.endswith("2i")


def append_final_records(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def trial_log_name(trial: dict[str, Any]) -> str:
    parts = [
        trial["trial_id"],
        trial["task"]["alias"],
        trial["generator"]["alias"],
        trial["enhancer"]["alias"],
    ]
    return "_".join(safe_name(part) for part in parts) + ".log"


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "unnamed"
