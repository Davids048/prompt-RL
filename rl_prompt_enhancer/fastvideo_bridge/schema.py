"""Request validation for the Slime-to-FastVideo generate-and-score bridge."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, "
    "works, paintings, images, static, overall gray, worst quality, low "
    "quality, JPEG compression residue, ugly, incomplete, extra fingers, "
    "poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen "
    "limbs, fused fingers, still picture, messy background, three legs, many "
    "people in the background, walking backwards"
)

DEFAULT_GENERATOR: dict[str, Any] = {
    "model": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "height": 480,
    "width": 832,
    "fps": 16,
    "num_frames": 1,
    "num_inference_steps": 50,
    "guidance_scale": 6.0,
    "flow_shift": 8.0,
    "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
}

REWARD_WEIGHTS: dict[str, float] = {
    "pickscore": 1.0,
    "clipscore": 1.0,
}


def default_generator() -> dict[str, Any]:
    return deepcopy(DEFAULT_GENERATOR)


def _coerce_text(generator: dict[str, Any], field: str) -> str:
    value = str(generator.get(field, "")).strip()
    if not value:
        raise ValueError(f"generator.{field} must be non-empty")
    return value


def _coerce_int(generator: dict[str, Any], field: str, *, minimum: int) -> int:
    try:
        value = int(generator[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"generator.{field} must be an integer") from exc
    if value < minimum:
        raise ValueError(f"generator.{field} must be >= {minimum}")
    return value


def _coerce_float(generator: dict[str, Any], field: str, *, minimum: float | None = None) -> float:
    try:
        value = float(generator[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"generator.{field} must be a number") from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"generator.{field} must be >= {minimum}")
    return value


def _validate_generator(generator: dict[str, Any]) -> dict[str, Any]:
    resolved_generator = default_generator()
    resolved_generator.update(generator)

    validated = {
        "model": _coerce_text(resolved_generator, "model"),
        "height": _coerce_int(resolved_generator, "height", minimum=1),
        "width": _coerce_int(resolved_generator, "width", minimum=1),
        "fps": _coerce_int(resolved_generator, "fps", minimum=1),
        "num_frames": _coerce_int(resolved_generator, "num_frames", minimum=1),
        "num_inference_steps": _coerce_int(resolved_generator, "num_inference_steps", minimum=1),
        "guidance_scale": _coerce_float(resolved_generator, "guidance_scale", minimum=0.0),
        "flow_shift": _coerce_float(resolved_generator, "flow_shift"),
        "negative_prompt": str(resolved_generator.get("negative_prompt", "")),
    }
    if validated["num_frames"] != 1:
        raise ValueError("image generation requires generator.num_frames to be 1")
    return validated


def validate_generate_and_score_request(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")

    required = [
        "request_id",
        "original_prompt",
        "enhanced_prompt",
        "artifact_kind",
        "comparison_group_id",
        "seed",
        "generator",
    ]
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError(f"missing required field(s): {missing}")

    request_id = str(payload["request_id"]).strip()
    original_prompt = str(payload["original_prompt"]).strip()
    enhanced_prompt = str(payload["enhanced_prompt"]).strip()
    artifact_kind = str(payload["artifact_kind"]).strip()
    comparison_group_id = str(payload["comparison_group_id"]).strip()
    generator = payload["generator"]

    if not request_id:
        raise ValueError("request_id must be non-empty")
    if not original_prompt:
        raise ValueError("original_prompt must be non-empty")
    if not enhanced_prompt:
        raise ValueError("enhanced_prompt must be non-empty")
    if artifact_kind != "image":
        raise ValueError("artifact_kind must be image")
    if not comparison_group_id:
        raise ValueError("comparison_group_id must be non-empty")
    if not isinstance(generator, dict):
        raise ValueError("generator must be a JSON object")

    try:
        seed = int(payload["seed"])
    except (TypeError, ValueError) as exc:
        raise ValueError("seed must be an integer") from exc

    return {
        "request_id": request_id,
        "original_prompt": original_prompt,
        "enhanced_prompt": enhanced_prompt,
        "artifact_kind": artifact_kind,
        "comparison_group_id": comparison_group_id,
        "seed": seed,
        "generator": _validate_generator(generator),
    }
