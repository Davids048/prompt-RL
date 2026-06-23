"""W&B media logging for prompt-enhancer validation rollouts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from slime.utils.metric_utils import compute_rollout_step


def _metadata(sample: Any) -> dict[str, Any]:
    return getattr(sample, "metadata", None) or {}


def _first_existing_path(paths: Any) -> Path | None:
    if isinstance(paths, str):
        paths = [paths]
    for raw_path in paths or []:
        path = Path(raw_path)
        if path.exists():
            return path
    return None


def _caption(sample: Any, metadata: dict[str, Any]) -> str:
    request = metadata.get("fastvideo_request") or {}
    reward = metadata.get("fastvideo_reward") or getattr(sample, "reward", None) or {}
    request_id = request.get("request_id") or "validation_image"
    avg_reward = reward.get("avg") if isinstance(reward, dict) else None
    if avg_reward is None:
        return str(request_id)
    return f"{request_id} | avg={avg_reward:.4f}"


def _wandb_images(wandb: Any, samples: list[Any]) -> list[Any]:
    """Convert FastVideo artifact paths on validation samples into W&B images."""
    images = []
    for sample in samples:
        metadata = _metadata(sample)
        generation = metadata.get("fastvideo_generation") or {}
        image_path = _first_existing_path(generation.get("artifact_paths"))
        if image_path is None:
            continue
        images.append(wandb.Image(str(image_path), caption=_caption(sample, metadata)))
    return images


def log_eval_images(
    rollout_id: int,
    args: Any,
    data: dict[str, dict[str, Any]],
    extra_metrics: dict[str, Any] | None = None,
) -> bool:
    """Log validation images to W&B media and let Slime log its scalar metrics."""
    if not getattr(args, "use_wandb", False):
        return False

    import wandb

    if wandb.run is None:
        return False

    payload: dict[str, Any] = {"eval/step": compute_rollout_step(args, rollout_id)}
    for dataset_name, dataset_data in (data or {}).items():
        images = _wandb_images(wandb, dataset_data.get("samples") or [])
        if images:
            payload[f"eval/{dataset_name}/images"] = images

    if len(payload) > 1:
        wandb.log(payload, commit=False)
    return False
