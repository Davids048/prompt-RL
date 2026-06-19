"""FastVideo generate-and-score service for RL prompt-enhancer GRPO runs."""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import re
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import imageio
from fastapi import FastAPI, HTTPException

from .schema import REWARD_WEIGHTS, validate_generate_and_score_request

app = FastAPI(title="RL Prompt Enhancer FastVideo Image Service")

_FASTVIDEO_RUNTIME: dict[str, Any] | None = None
_GENERATOR: Any | None = None
_GENERATOR_FLOW_SHIFT: float | None = None
_SCORER: Any | None = None
_LOCK = asyncio.Lock()


def _fastvideo_runtime() -> dict[str, Any]:
    # Delay heavy FastVideo imports until the service actually handles work.
    global _FASTVIDEO_RUNTIME
    if _FASTVIDEO_RUNTIME is None:
        from fastvideo.api.schema import (
            EngineConfig,
            GenerationRequest,
            GeneratorConfig,
            OutputConfig,
            ParallelismConfig,
            PipelineSelection,
            SamplingConfig,
        )
        from fastvideo.entrypoints.video_generator import VideoGenerator
        from fastvideo.train.methods.rl.rewards import build_multi_reward_scorer

        _FASTVIDEO_RUNTIME = {
            "EngineConfig": EngineConfig,
            "GenerationRequest": GenerationRequest,
            "GeneratorConfig": GeneratorConfig,
            "OutputConfig": OutputConfig,
            "ParallelismConfig": ParallelismConfig,
            "PipelineSelection": PipelineSelection,
            "SamplingConfig": SamplingConfig,
            "VideoGenerator": VideoGenerator,
            "build_multi_reward_scorer": build_multi_reward_scorer,
        }
    return _FASTVIDEO_RUNTIME


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _reward_weights() -> dict[str, float]:
    """Load reward weights from a readable snapshot path or inline JSON env."""
    path = os.environ.get("RLPE_REWARD_WEIGHTS_PATH")
    if path:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    else:
        raw = os.environ.get("RLPE_REWARD_WEIGHTS_JSON")
        if not raw:
            return dict(REWARD_WEIGHTS)
        data = json.loads(raw)
    if not isinstance(data, dict) or not data:
        raise RuntimeError("reward weights must be a non-empty JSON object")
    return {str(key): float(value) for key, value in data.items()}


def _scalar_reward_key() -> str:
    return os.environ.get("RLPE_SCALAR_REWARD_KEY", "avg")


def _output_root() -> Path:
    return Path(
        os.environ.get(
            "RLPE_FASTVIDEO_OUTPUT_ROOT",
            "/tmp/rl_prompt_enhancer_fastvideo/artifacts",
        )
    )


def _ledger_root() -> Path:
    return Path(
        os.environ.get(
            "RLPE_LEDGER_ROOT",
            str(_output_root().parent / "ledgers"),
        )
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    # Keep append-only ledgers so Slime samples can be traced back to artifacts.
    with _ledger_lock(path.stem):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


@contextmanager
def _ledger_lock(name: str):
    # Cross-process FastVideo workers share ledgers and seed validation state.
    lock_path = _ledger_root() / "locks" / f"{name}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _seed_index_path() -> Path:
    return _ledger_root() / "seed_index.json"


def _load_seed_index() -> dict[str, int]:
    path = _seed_index_path()
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    return {str(key): int(value) for key, value in data.items()}


def _store_seed_index(index: dict[str, int]) -> None:
    path = _seed_index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(index, handle, sort_keys=True, indent=2)
        handle.write("\n")
    tmp_path.replace(path)


def _validate_same_seed(request: dict[str, Any]) -> bool:
    # Enforce the comparison policy: prompt variants share one seed per group.
    with _ledger_lock("seed_index"):
        index = _load_seed_index()
        comparison_group_id = request["comparison_group_id"]
        seed = int(request["seed"])
        existing_seed = index.get(comparison_group_id)
        if existing_seed is not None and existing_seed != seed:
            raise RuntimeError(
                f"comparison_group_id {comparison_group_id!r} already used seed "
                f"{existing_seed}, got {seed}"
            )
        index[comparison_group_id] = seed
        _store_seed_index(index)
        _append_jsonl(
            _ledger_root() / "seed_policy.jsonl",
            {
                "timestamp": _utc_now(),
                "request_id": request["request_id"],
                "comparison_group_id": comparison_group_id,
                "seed": seed,
                "same_seed_validated": True,
                "worker": _worker_info(),
            },
        )
    return True


def _worker_info() -> dict[str, Any]:
    return {
        "worker_id": os.environ.get("RLPE_FASTVIDEO_WORKER_ID", "0"),
        "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "num_gpus": _env_int("RLPE_FASTVIDEO_NUM_GPUS", 1),
        "service_port": os.environ.get("RLPE_FASTVIDEO_SERVICE_PORT", ""),
    }


def _record_request(request: dict[str, Any]) -> None:
    _append_jsonl(
        _ledger_root() / "requests.jsonl",
        {
            "timestamp": _utc_now(),
            "request_id": request["request_id"],
            "comparison_group_id": request["comparison_group_id"],
            "artifact_kind": request["artifact_kind"],
            "seed": request["seed"],
            "original_prompt": request["original_prompt"],
            "enhanced_prompt": request["enhanced_prompt"],
            "generator": request["generator"],
            "worker": _worker_info(),
        },
    )


def _record_success(request: dict[str, Any], response: dict[str, Any]) -> None:
    scalar_reward_key = _scalar_reward_key()
    _append_jsonl(
        _ledger_root() / "artifacts.jsonl",
        {
            "timestamp": _utc_now(),
            "request_id": request["request_id"],
            "comparison_group_id": request["comparison_group_id"],
            "artifact_kind": response["artifact_kind"],
            "artifact_paths": response["artifact_paths"],
            "generator_config_used": response["generator_config_used"],
            "seed_used": response["seed_used"],
            "same_seed_validated": response["same_seed_validated"],
            "timing": response["timing"],
            "worker": response["worker"],
        },
    )
    _append_jsonl(
        _ledger_root() / "rewards.jsonl",
        {
            "timestamp": _utc_now(),
            "request_id": request["request_id"],
            "comparison_group_id": request["comparison_group_id"],
            "rewards": response["rewards"],
            "scalar_reward_key": scalar_reward_key,
            "scalar_reward": response["rewards"][scalar_reward_key],
            "worker": response["worker"],
        },
    )


def _record_error(payload: dict[str, Any], error: Exception) -> None:
    _append_jsonl(
        _ledger_root() / "errors.jsonl",
        {
            "timestamp": _utc_now(),
            "request_id": payload.get("request_id"),
            "comparison_group_id": payload.get("comparison_group_id"),
            "error_type": type(error).__name__,
            "error": str(error),
            "worker": _worker_info(),
        },
    )


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned[:180] or "request"


def _generator_config(flow_shift: float) -> Any:
    # Construct the frozen Wan2.1 one-frame generator used as the experiment tool.
    runtime = _fastvideo_runtime()
    num_gpus = _env_int("RLPE_FASTVIDEO_NUM_GPUS", 1)
    return runtime["GeneratorConfig"](
        model_path=os.environ.get("RLPE_FASTVIDEO_MODEL", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"),
        engine=runtime["EngineConfig"](
            num_gpus=num_gpus,
            execution_backend=os.environ.get("RLPE_FASTVIDEO_EXECUTION_BACKEND", "mp"),
            parallelism=runtime["ParallelismConfig"](
                tp_size=_env_int("RLPE_FASTVIDEO_TP_SIZE", 1),
                sp_size=_env_int("RLPE_FASTVIDEO_SP_SIZE", 1),
                hsdp_replicate_dim=_env_int("RLPE_FASTVIDEO_HSDP_REPLICATE_DIM", 1),
                hsdp_shard_dim=_env_int("RLPE_FASTVIDEO_HSDP_SHARD_DIM", num_gpus),
            ),
        ),
        pipeline=runtime["PipelineSelection"](
            workload_type=os.environ.get("RLPE_FASTVIDEO_WORKLOAD_TYPE", "t2v"),
            experimental={"flow_shift": flow_shift},
        ),
    )


def _get_generator(flow_shift: float) -> Any:
    # Reuse one generator instance so requests do not repeatedly reload the model.
    global _GENERATOR, _GENERATOR_FLOW_SHIFT
    if _GENERATOR is None:
        _GENERATOR = _fastvideo_runtime()["VideoGenerator"].from_config(_generator_config(flow_shift))
        _GENERATOR_FLOW_SHIFT = flow_shift
    elif _GENERATOR_FLOW_SHIFT != flow_shift:
        raise RuntimeError(
            "FastVideo generator is already initialized with flow_shift="
            f"{_GENERATOR_FLOW_SHIFT}, got {flow_shift}"
        )
    return _GENERATOR


def _get_scorer():
    # Reuse the FastVideo PickScore+CLIPScore scorer across all requests.
    global _SCORER
    if _SCORER is None:
        _SCORER = _fastvideo_runtime()["build_multi_reward_scorer"](
            _reward_weights(),
            device=os.environ.get("RLPE_REWARD_DEVICE", "cuda"),
        )
    return _SCORER


def _tensor_scores_to_floats(scores: dict[str, Any]) -> dict[str, float]:
    import torch

    result: dict[str, float] = {}
    for name, value in scores.items():
        if not torch.is_tensor(value):
            result[name] = float(value)
            continue
        if value.numel() != 1:
            raise RuntimeError(f"reward {name!r} returned {value.numel()} values for one request")
        result[name] = float(value.detach().float().cpu().reshape(-1)[0].item())
    return result


def _write_image_artifact(result: Any, artifact_dir: Path) -> list[str]:
    # Persist a concrete artifact path for the JSON response and run ledgers.
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / "image.png"

    frames = getattr(result, "frames", None)
    if frames is not None and len(frames) > 0:
        imageio.imwrite(artifact_path, frames[0])
        return [str(artifact_path)]

    video_path = getattr(result, "video_path", None)
    if video_path:
        return [str(video_path)]

    raise RuntimeError("FastVideo returned no frames and no artifact path")


def _generate_and_score(payload: dict[str, Any]) -> dict[str, Any]:
    # Single-request adapter: validate, generate one image, score it, then respond.
    request = validate_generate_and_score_request(payload)
    _record_request(request)
    same_seed_validated = _validate_same_seed(request)
    generator_config = request["generator"]
    flow_shift = float(generator_config["flow_shift"])
    artifact_dir = _output_root() / _safe_name(request["request_id"])
    start = time.perf_counter()

    runtime = _fastvideo_runtime()
    generator = _get_generator(flow_shift)
    generation_request = runtime["GenerationRequest"](
        prompt=request["enhanced_prompt"],
        negative_prompt=str(generator_config["negative_prompt"]),
        sampling=runtime["SamplingConfig"](
            seed=request["seed"],
            num_frames=1,
            height=int(generator_config["height"]),
            width=int(generator_config["width"]),
            fps=int(generator_config["fps"]),
            num_inference_steps=int(generator_config["num_inference_steps"]),
            guidance_scale=float(generator_config["guidance_scale"]),
        ),
        output=runtime["OutputConfig"](
            output_path=str(artifact_dir),
            output_video_name="fastvideo_internal.mp4",
            save_video=False,
            return_frames=True,
        ),
    )

    result = generator.generate(generation_request)
    if isinstance(result, list):
        if len(result) != 1:
            raise RuntimeError(f"expected one FastVideo result, got {len(result)}")
        result = result[0]

    artifact_paths = _write_image_artifact(result, artifact_dir)
    samples = getattr(result, "samples", None)
    if samples is None:
        raise RuntimeError("FastVideo result has no samples tensor for reward scoring")

    # Score the generated image tensor against the enhanced prompt.
    scorer = _get_scorer()
    rewards = _tensor_scores_to_floats(scorer(samples, [request["enhanced_prompt"]]))
    scalar_reward_key = _scalar_reward_key()
    if scalar_reward_key not in rewards:
        raise RuntimeError(f"reward scorer did not return scalar key {scalar_reward_key!r}: {sorted(rewards)}")
    elapsed = time.perf_counter() - start
    extra = getattr(result, "extra", {}) or {}

    response = {
        "request_id": request["request_id"],
        "fastvideo_request_id": request["request_id"],
        "status": "completed",
        "artifact_kind": "image",
        "artifact_paths": artifact_paths,
        "generator_config_used": generator_config,
        "seed_used": request["seed"],
        "comparison_group_id": request["comparison_group_id"],
        "same_seed_validated": same_seed_validated,
        "rewards": rewards,
        "timing": {
            "service_e2e_seconds": elapsed,
            "fastvideo_generation_seconds": getattr(result, "generation_time", None),
            "fastvideo_e2e_seconds": extra.get("e2e_latency"),
        },
        "worker": _worker_info(),
        "error": None,
    }
    _record_success(request, response)
    return response


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "artifact_kind": "image",
        "num_frames": 1,
        "reward_weights": _reward_weights(),
        "scalar_reward_key": _scalar_reward_key(),
        "worker": _worker_info(),
    }


@app.post("/generate_and_score")
async def generate_and_score(payload: dict[str, Any]) -> dict[str, Any]:
    # Serialize requests so one FastVideo generator owns the configured GPUs.
    async with _LOCK:
        try:
            return await asyncio.to_thread(_generate_and_score, payload)
        except Exception as exc:
            _record_error(payload, exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
