#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]


def run(
    records: list[dict[str, Any]],
    trial: dict[str, Any],
    run_dir: Path,
) -> list[dict[str, Any]]:
    task = trial["task"]
    params = task.get("eval_params", {}).get("vbench", {})
    dimensions = [str(dimension) for dimension in params.get("dimensions", task.get("subset", {}).get("dimensions", []))]
    if not dimensions:
        raise ValueError("VBench eval requires a non-empty dimensions list.")
    mode = str(params.get("mode", "vbench_standard"))
    if mode == "vbench_standard" and int(task.get("samples_per_prompt", 1)) != 5:
        raise ValueError("VBench standard mode requires samples_per_prompt=5 for official prompt-0..4 videos.")

    eval_dir = run_dir / "eval" / trial["trial_id"] / "vbench"
    stage_dir = eval_dir / "videos"
    output_dir = eval_dir / "official_results"
    eval_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = eval_dir / "results.jsonl"
    raw_rel = str(raw_path.relative_to(run_dir))
    name = trial["trial_id"]
    results_path = output_dir / f"{name}_eval_results.json"
    full_info_path = output_dir / f"{name}_full_info.json"

    out = [dict(row) for row in records]
    candidates = [row for row in out if row.get("artifact_path") and not row.get("error")]
    if not candidates:
        return out

    staged_by_record = stage_videos(candidates, run_dir, stage_dir)
    repo = resolve_repo_path(params.get("repo", "FastVideo/fastvideo/third_party/eval/vbench"))
    full_json_dir = resolve_repo_path(params.get("full_json_dir", repo / "vbench" / "VBench_full_info.json"))
    old_path = list(sys.path)
    saved_vbench = pop_module_tree("vbench")
    try:
        sys.path.insert(0, str(repo))
        patch_openai_clip_import()
        import torch
        from vbench import VBench

        device_name = str(params.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
        # Official VBench passes a torch.device. The GRiT-backed object_class
        # evaluator calls device.type, so a plain "cuda"/"cpu" string crashes.
        device = torch.device(device_name)
        benchmark = VBench(device, str(full_json_dir), str(output_dir))
        benchmark.evaluate(
            videos_path=str(stage_dir),
            name=name,
            prompt_list=[],
            dimension_list=dimensions,
            local=bool(params.get("local", False)),
            read_frame=bool(params.get("read_frame", False)),
            mode=mode,
            imaging_quality_preprocessing_mode=str(params.get("imaging_quality_preprocessing_mode", "longer")),
        )
    finally:
        torch_mod = sys.modules.get("torch")
        if torch_mod is not None and torch_mod.cuda.is_available():
            torch_mod.cuda.empty_cache()
        sys.path[:] = old_path
        restore_module_tree("vbench", saved_vbench)

    scores_by_stage_path = load_scores_by_stage_path(results_path)
    for row, stage_path in staged_by_record:
        score_by_dimension = scores_by_stage_path.get(str(stage_path.resolve()), {})
        expected_dimensions = record_dimensions(row, dimensions)
        selected_scores = {
            dimension: score_by_dimension[dimension]
            for dimension in expected_dimensions
            if dimension in score_by_dimension
        }
        if not selected_scores:
            row["eval"] = {}
            row["error"] = {
                "stage": "eval",
                "eval": "vbench",
                "message": f"No VBench per-video score found for staged artifact {stage_path}",
            }
            continue
        result = {
            "trial_id": row["trial_id"],
            "prompt_index": row["prompt_index"],
            "sample_index": row["sample_index"],
            "artifact_path": row["artifact_path"],
            "eval_prompt": row["eval_prompt"],
            "staged_path": str(stage_path),
            "dimensions": expected_dimensions,
            "scores": selected_scores,
            "official_results_path": str(results_path.relative_to(run_dir)),
            "official_full_info_path": str(full_info_path.relative_to(run_dir)),
            "source": raw_rel,
        }
        append_jsonl(raw_path, result)
        row.setdefault("eval", {})
        row["eval"]["vbench"] = {
            **selected_scores,
            "source": raw_rel,
        }

    return out


def artifact_path(row: dict[str, Any], run_dir: Path) -> Path:
    path = Path(row["artifact_path"])
    return path if path.is_absolute() else run_dir / path


def resolve_repo_path(value: Any) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def patch_openai_clip_import() -> None:
    import pkg_resources
    import packaging
    import packaging.version

    if not hasattr(pkg_resources, "packaging"):
        pkg_resources.packaging = packaging


def stage_videos(
    rows: list[dict[str, Any]],
    run_dir: Path,
    stage_dir: Path,
) -> list[tuple[dict[str, Any], Path]]:
    stage_dir.mkdir(parents=True, exist_ok=True)
    staged: list[tuple[dict[str, Any], Path]] = []
    for row in rows:
        src = artifact_path(row, run_dir)
        name = f"{row['original_prompt']}-{int(row['sample_index'])}{src.suffix}"
        if os.sep in name or (os.altsep and os.altsep in name):
            raise ValueError(f"VBench official prompt cannot be staged as a filename: {row['original_prompt']!r}")
        dst = stage_dir / name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        try:
            dst.symlink_to(src)
        except OSError:
            shutil.copy2(src, dst)
        staged.append((row, dst))
    return staged


def record_dimensions(row: dict[str, Any], fallback: list[str]) -> list[str]:
    metadata = row.get("prompt_metadata", {})
    dimensions = metadata.get("dimensions") or metadata.get("dimension")
    if isinstance(dimensions, str):
        return [dimensions]
    if isinstance(dimensions, list):
        return [str(dimension) for dimension in dimensions]
    return fallback


def load_scores_by_stage_path(results_path: Path) -> dict[str, dict[str, float]]:
    with results_path.open("r", encoding="utf-8") as f:
        results = json.load(f)
    scores: dict[str, dict[str, float]] = defaultdict(dict)
    for dimension, payload in results.items():
        video_results = []
        if isinstance(payload, list) and len(payload) > 1:
            video_results = payload[1]
        for item in video_results:
            if not isinstance(item, dict) or "video_path" not in item:
                continue
            if "video_results" not in item:
                continue
            scores[str(Path(item["video_path"]).resolve())][dimension] = float(item["video_results"])
    return scores


def pop_module_tree(root: str) -> dict[str, Any]:
    saved: dict[str, Any] = {}
    for name in list(sys.modules):
        if name == root or name.startswith(root + "."):
            saved[name] = sys.modules.pop(name)
    return saved


def restore_module_tree(root: str, saved: dict[str, Any]) -> None:
    for name in list(sys.modules):
        if name == root or name.startswith(root + "."):
            del sys.modules[name]
    sys.modules.update(saved)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
