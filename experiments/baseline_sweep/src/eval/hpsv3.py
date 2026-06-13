#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]


def run(
    records: list[dict[str, Any]],
    trial: dict[str, Any],
    run_dir: Path,
) -> list[dict[str, Any]]:
    task = trial["task"]
    params = task.get("eval_params", {}).get("hpsv3", {})
    eval_dir = run_dir / "eval" / trial["trial_id"] / "hpsv3"
    frame_dir = eval_dir / "frames"
    eval_dir.mkdir(parents=True, exist_ok=True)
    raw_path = eval_dir / "frame_results.jsonl"
    raw_rel = str(raw_path.relative_to(run_dir))

    out = [dict(row) for row in records]
    candidates = [row for row in out if row.get("artifact_path") and not row.get("error")]
    if not candidates:
        return out

    extracted: list[tuple[dict[str, Any], list[Path]]] = []
    for row in candidates:
        try:
            frames = extract_uniform_frames(
                video_path=artifact_path(row, run_dir),
                output_dir=frame_dir / row["trial_id"] / f"{int(row['prompt_index']):06d}_{int(row['sample_index']):04d}",
                count=int(params.get("num_frames", 8)),
            )
            extracted.append((row, frames))
        except Exception as exc:  # noqa: BLE001
            row["eval"] = {}
            row["error"] = {
                "stage": "eval",
                "eval": "hpsv3",
                "message": repr(exc),
            }

    if not extracted:
        return out

    repo = resolve_repo_path(params.get("repo", "evaluations/HPSv3"))
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    prepare_hpsv3_compat()

    import torch
    from hpsv3.inference import HPSv3RewardInferencer

    device = str(params.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    inferencer = HPSv3RewardInferencer(
        config_path=params.get("config_path"),
        checkpoint_path=params.get("checkpoint_path"),
        device=device,
    )
    try:
        frame_prompts: list[str] = []
        frame_paths: list[str] = []
        spans: list[tuple[dict[str, Any], int, int]] = []
        for row, frames in extracted:
            start = len(frame_paths)
            frame_paths.extend(str(path) for path in frames)
            frame_prompts.extend([row["eval_prompt"]] * len(frames))
            spans.append((row, start, len(frame_paths)))

        scores = reward_in_batches(
            inferencer=inferencer,
            prompts=frame_prompts,
            paths=frame_paths,
            batch_size=int(params.get("batch_size", 32)),
        )
        for row, start, stop in spans:
            frame_scores = scores[start:stop]
            score = aggregate(frame_scores, str(params.get("aggregate", "mean")))
            result = {
                "trial_id": row["trial_id"],
                "prompt_index": row["prompt_index"],
                "sample_index": row["sample_index"],
                "artifact_path": row["artifact_path"],
                "eval_prompt": row["eval_prompt"],
                "score": score,
                "aggregate": str(params.get("aggregate", "mean")),
                "frame_scores": frame_scores,
                "source": raw_rel,
            }
            append_jsonl(raw_path, result)
            row.setdefault("eval", {})
            row["eval"]["hpsv3"] = {
                "score": score,
                "source": raw_rel,
            }
    finally:
        del inferencer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return out


def artifact_path(row: dict[str, Any], run_dir: Path) -> Path:
    path = Path(row["artifact_path"])
    return path if path.is_absolute() else run_dir / path


def resolve_repo_path(value: Any) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def prepare_hpsv3_compat() -> None:
    """Patch HPSv3 to run in the shared FastVideo environment.

    HPSv3 pins transformers==4.45.2, but the baseline sweep process uses the
    repo-wide FastVideo venv with a newer transformers. Keep these shims here
    instead of editing the vendored HPSv3 repo, so the benchmark code remains
    inspectable and the workaround is scoped to this wrapper.
    """
    patch_transformers_video_input_alias()

    import hpsv3.train as hps_train
    from hpsv3.model.qwen2vl_trainer import Qwen2VLRewardModelBT

    # The current venv has a flash_attn namespace from flash-attn-4 but not the
    # flash-attn package metadata expected by transformers' FlashAttention-2
    # availability check. HPSv3 already falls back to SDPA when flash_attn is
    # None, so force that fallback rather than compiling/downgrading packages.
    hps_train.flash_attn = None

    if getattr(Qwen2VLRewardModelBT, "_baseline_sweep_hpsv3_compat", False):
        return

    original_load_state_dict = Qwen2VLRewardModelBT.load_state_dict
    original_forward = Qwen2VLRewardModelBT.forward

    def load_state_dict_with_hpsv3_remap(self, state_dict, *args, **kwargs):
        state_dict = remap_hpsv3_state_dict_for_new_transformers(state_dict)
        return original_load_state_dict(self, state_dict, *args, **kwargs)

    def forward_with_hpsv3_layout_aliases(self, *args, **kwargs):
        patch_hpsv3_model_layout_aliases(self)
        return original_forward(self, *args, **kwargs)

    Qwen2VLRewardModelBT.load_state_dict = load_state_dict_with_hpsv3_remap
    Qwen2VLRewardModelBT.forward = forward_with_hpsv3_layout_aliases
    Qwen2VLRewardModelBT._baseline_sweep_hpsv3_compat = True


def patch_hpsv3_model_layout_aliases(model: Any) -> None:
    # HPSv3's copied reward-model forward still uses the older transformers
    # layout, where token embeddings lived at self.model.embed_tokens. Newer
    # Qwen2-VL wraps the language model under self.model.language_model.
    # Recreate only the missing alias needed by that copied forward.
    inner_model = getattr(model, "model", None)
    language_model = getattr(inner_model, "language_model", None)
    if language_model is not None and not hasattr(inner_model, "embed_tokens"):
        inner_model.embed_tokens = language_model.embed_tokens


def patch_transformers_video_input_alias() -> None:
    from transformers import image_utils

    # HPSv3's copied Qwen2-VL image processor imports VideoInput from
    # transformers.image_utils. Newer transformers removed that alias; it is
    # only used for typing here, so reusing ImageInput preserves runtime behavior.
    if not hasattr(image_utils, "VideoInput") and hasattr(image_utils, "ImageInput"):
        image_utils.VideoInput = image_utils.ImageInput


def remap_hpsv3_state_dict_for_new_transformers(state_dict: Any) -> Any:
    if not isinstance(state_dict, dict) or not needs_hpsv3_key_remap(state_dict):
        return state_dict

    return {
        remap_hpsv3_state_key_for_new_transformers(key): value
        for key, value in state_dict.items()
    }


def needs_hpsv3_key_remap(state_dict: dict[str, Any]) -> bool:
    has_old_qwen_keys = any(
        key.startswith("visual.")
        or key.startswith("model.layers.")
        or key.startswith("model.embed_tokens.")
        or key.startswith("model.norm.")
        for key in state_dict
    )
    has_new_qwen_keys = any(
        key.startswith("model.visual.") or key.startswith("model.language_model.")
        for key in state_dict
    )
    return has_old_qwen_keys and not has_new_qwen_keys


def remap_hpsv3_state_key_for_new_transformers(key: str) -> str:
    # HPSv3's released checkpoint was saved against the older Qwen2-VL module
    # layout used by transformers==4.45.2:
    #   visual.*       -> model.visual.*
    #   model.layers.* -> model.language_model.layers.*
    #   model.norm.*   -> model.language_model.norm.*
    # The newer transformers in this environment wraps those same components
    # under model.visual and model.language_model.
    if key.startswith("visual."):
        return f"model.{key}"
    if key.startswith("model."):
        return f"model.language_model.{key.removeprefix('model.')}"
    return key


def extract_uniform_frames(video_path: Path, output_dir: Path, count: int) -> list[Path]:
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for HPSv3 frame sampling: {video_path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise RuntimeError(f"Video reports no frames: {video_path}")
        indices = uniform_indices(total, max(1, count))
        frame_paths: list[Path] = []
        for out_idx, frame_idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")
            frame_path = output_dir / f"{out_idx:04d}.png"
            if not cv2.imwrite(str(frame_path), frame):
                raise RuntimeError(f"Could not write sampled frame: {frame_path}")
            frame_paths.append(frame_path)
        return frame_paths
    finally:
        cap.release()


def uniform_indices(total: int, count: int) -> list[int]:
    if count == 1:
        return [total // 2]
    if count >= total:
        return list(range(total))
    return [round(i * (total - 1) / (count - 1)) for i in range(count)]


def reward_to_float(value: Any) -> float:
    item = value
    if hasattr(item, "detach"):
        item = item.detach().cpu()
    if hasattr(item, "tolist"):
        item = item.tolist()
    while isinstance(item, list):
        item = item[0]
    return float(item)


def reward_in_batches(
    inferencer: Any,
    prompts: list[str],
    paths: list[str],
    batch_size: int,
) -> list[float]:
    batch_size = max(1, batch_size)
    scores: list[float] = []
    for start in range(0, len(paths), batch_size):
        stop = min(start + batch_size, len(paths))
        rewards = inferencer.reward(prompts[start:stop], paths[start:stop])
        scores.extend(rewards_to_scores(rewards, stop - start))
    return scores


def rewards_to_scores(rewards: Any, expected_count: int) -> list[float]:
    if expected_count == 1:
        try:
            return [reward_to_float(rewards[0])]
        except (IndexError, TypeError):
            return [reward_to_float(rewards)]
    return [reward_to_float(rewards[idx]) for idx in range(expected_count)]


def aggregate(values: list[float], mode: str) -> float:
    if not values:
        raise ValueError("Cannot aggregate an empty HPSv3 score list.")
    if mode == "mean":
        return sum(values) / len(values)
    if mode == "min":
        return min(values)
    if mode == "max":
        return max(values)
    raise ValueError(f"Unsupported HPSv3 aggregate mode: {mode!r}")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
