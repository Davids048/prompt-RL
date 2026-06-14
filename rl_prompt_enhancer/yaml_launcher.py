"""YAML-backed launcher for FastVideo image GRPO prompt-enhancer experiments."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when the YAML config is incomplete or internally inconsistent."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require(config: dict[str, Any], path: str) -> Any:
    current: Any = config
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ConfigError(f"missing required config field: {path}")
        current = current[part]
    if current is None or current == "":
        raise ConfigError(f"empty required config field: {path}")
    return current


def _get(config: dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = config
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _as_path(value: Any, field: str) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        raise ConfigError(f"{field} must be an absolute path: {path}")
    return path


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _quote(argv: list[str]) -> str:
    return shlex.join(str(item) for item in argv)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ConfigError(f"config must be a YAML mapping: {path}")
    return data


def _validate_config(config: dict[str, Any]) -> None:
    required = [
        "run.name",
        "run.output_dir",
        "paths.repo_root",
        "paths.python",
        "paths.slime_root",
        "paths.fastvideo_root",
        "paths.megatron_root",
        "compute.mode",
        "compute.slime.job_id",
        "compute.slime.nodelist",
        "compute.slime.cuda_visible_devices",
        "compute.slime.num_gpus",
        "compute.fastvideo.job_id",
        "compute.fastvideo.nodelist",
        "compute.fastvideo.cuda_visible_devices",
        "compute.fastvideo.num_gpus",
        "ray.address",
        "ray.port",
        "ray.dashboard_port",
        "ray.client_server_port",
        "ray.temp_dir",
        "fastvideo_service.host",
        "fastvideo_service.port",
        "fastvideo_service.public_host",
        "fastvideo_service.output_root",
        "fastvideo_service.ledger_root",
        "prompt_data.source_path",
        "prompt_data.jsonl_path",
        "prompt_data.prompt_id_prefix",
        "prompt_data.prompt_template",
        "prompt_enhancer.hf_checkpoint",
        "prompt_enhancer.torch_dist_checkpoint",
        "prompt_enhancer.slime_model_script",
        "prompt_enhancer.custom_generate_function_path",
        "generation.model",
        "generation.height",
        "generation.width",
        "generation.fps",
        "generation.num_frames",
        "generation.num_inference_steps",
        "generation.guidance_scale",
        "generation.flow_shift",
        "seed_policy.base_seed",
        "reward.metrics",
        "reward.scalar_key",
        "slime_grpo.reward_key",
        "slime_grpo.num_rollout",
        "slime_grpo.n_samples_per_prompt",
        "slime_grpo.global_batch_size",
        "wandb.project",
        "wandb.group",
    ]
    for field in required:
        _require(config, field)

    if str(_require(config, "generation.artifact_kind")) != "image":
        raise ConfigError("generation.artifact_kind must be image")
    if int(_require(config, "generation.num_frames")) != 1:
        raise ConfigError("image GRPO runs require generation.num_frames to be 1")
    if str(_require(config, "reward.scalar_key")) != str(_require(config, "slime_grpo.reward_key")):
        raise ConfigError("reward.scalar_key must match slime_grpo.reward_key")
    if str(_require(config, "compute.mode")) not in {"slurm_overlap", "local"}:
        raise ConfigError("compute.mode must be slurm_overlap or local")
    fastvideo_worker_count = _fastvideo_worker_count(config)
    if fastvideo_worker_count < 1:
        raise ConfigError("fastvideo_service.data_parallel_workers must be >= 1")
    fastvideo_devices = _fastvideo_cuda_devices(config)
    if fastvideo_worker_count > len(fastvideo_devices):
        raise ConfigError(
            "fastvideo_service.data_parallel_workers cannot exceed the number of "
            "compute.fastvideo.cuda_visible_devices entries"
        )
    _fastvideo_worker_ports(config)
    if _validation_enabled(config):
        for field in [
            "validation.source_path",
            "validation.jsonl_path",
            "validation.eval_config_path",
            "validation.dataset_name",
            "validation.prompt_id_prefix",
            "validation.eval_interval",
            "validation.input_key",
            "validation.metadata_key",
            "validation.n_samples_per_eval_prompt",
            "validation.max_response_len",
            "validation.temperature",
        ]:
            _require(config, field)


def _path_status(label: str, path: Path) -> str:
    status = "ok" if path.exists() else "missing"
    return f"{status}\t{label}\t{path}"


def _validation_enabled(config: dict[str, Any]) -> bool:
    return _bool(_get(config, "validation.enabled", False))


def _check_required_paths(config: dict[str, Any], *, strict: bool) -> list[str]:
    paths = {
        "repo_root": _as_path(_require(config, "paths.repo_root"), "paths.repo_root"),
        "python": _as_path(_require(config, "paths.python"), "paths.python"),
        "slime_root": _as_path(_require(config, "paths.slime_root"), "paths.slime_root"),
        "fastvideo_root": _as_path(_require(config, "paths.fastvideo_root"), "paths.fastvideo_root"),
        "megatron_root": _as_path(_require(config, "paths.megatron_root"), "paths.megatron_root"),
        "prompt_source": _as_path(_require(config, "prompt_data.source_path"), "prompt_data.source_path"),
        "prompt_template": _as_path(_require(config, "prompt_data.prompt_template"), "prompt_data.prompt_template"),
        "hf_checkpoint": _as_path(_require(config, "prompt_enhancer.hf_checkpoint"), "prompt_enhancer.hf_checkpoint"),
        "slime_model_script": _as_path(
            _require(config, "prompt_enhancer.slime_model_script"),
            "prompt_enhancer.slime_model_script",
        ),
    }
    prompt_jsonl = _as_path(_require(config, "prompt_data.jsonl_path"), "prompt_data.jsonl_path")
    torch_dist_checkpoint = _as_path(
        _require(config, "prompt_enhancer.torch_dist_checkpoint"),
        "prompt_enhancer.torch_dist_checkpoint",
    )
    statuses = [_path_status(label, path) for label, path in paths.items()]
    statuses.append(_path_status("torch_dist_checkpoint_local_view", torch_dist_checkpoint))
    statuses.append(_path_status("prompt_jsonl", prompt_jsonl))
    if _validation_enabled(config):
        validation_source = _as_path(_require(config, "validation.source_path"), "validation.source_path")
        validation_jsonl = _as_path(_require(config, "validation.jsonl_path"), "validation.jsonl_path")
        validation_eval_config = _as_path(
            _require(config, "validation.eval_config_path"),
            "validation.eval_config_path",
        )
        statuses.append(_path_status("validation_source", validation_source))
        statuses.append(_path_status("validation_jsonl", validation_jsonl))
        statuses.append(_path_status("validation_eval_config", validation_eval_config))
        paths["validation_source"] = validation_source

    missing = [f"{label}: {path}" for label, path in paths.items() if not path.exists()]
    if missing and strict:
        raise ConfigError("required path(s) are missing:\n" + "\n".join(missing))
    if not prompt_jsonl.exists() and not paths["prompt_source"].exists() and strict:
        raise ConfigError("prompt_data.jsonl_path is missing and prompt_data.source_path is unavailable")
    return statuses


def _check_node_local_paths(config: dict[str, Any]) -> None:
    if str(_require(config, "compute.mode")) == "local":
        return
    checkpoint = str(_require(config, "prompt_enhancer.torch_dist_checkpoint"))
    command = [
        *_slurm_prefix(config, "slime"),
        "bash",
        "-lc",
        "test -d \"$1\" && find \"$1\" -mindepth 1 -print -quit | grep -q .",
        "bash",
        checkpoint,
    ]
    result = subprocess.run(command, check=False, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Slime torch-dist checkpoint is not available on the assigned Slime node "
            f"or Slurm is unreachable: {checkpoint}\n{result.stderr}{result.stdout}"
        )


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _script(path: Path, body: str) -> None:
    _write_text(path, body)
    path.chmod(0o755)


def _git_snapshot(label: str, path: Path) -> str:
    lines = [f"## {label}", f"path={path}"]
    if not (path / ".git").exists():
        return "\n".join([*lines, "git=missing", ""])
    for key, cmd in [
        ("head", ["git", "-C", str(path), "rev-parse", "HEAD"]),
        ("branch", ["git", "-C", str(path), "rev-parse", "--abbrev-ref", "HEAD"]),
    ]:
        result = subprocess.run(cmd, check=False, text=True, capture_output=True)
        lines.append(f"{key}={result.stdout.strip()}")
    lines.append("status_short_begin")
    status = subprocess.run(["git", "-C", str(path), "status", "--short"], check=False, text=True, capture_output=True)
    lines.extend(status.stdout.rstrip("\n").splitlines())
    lines.append("status_short_end")
    lines.append("")
    return "\n".join(lines)


def _load_model_args(model_script: Path) -> list[str]:
    # The YAML chooses the model script; this helper extracts Slime's canonical Qwen args.
    command = 'source "$1"; printf "%s\\0" "${MODEL_ARGS[@]}"'
    result = subprocess.run(
        ["bash", "-lc", command, "bash", str(model_script)],
        check=True,
        capture_output=True,
    )
    return [item.decode("utf-8") for item in result.stdout.split(b"\0") if item]


def _service_url(config: dict[str, Any]) -> str:
    return _service_urls(config)[0]


def _fastvideo_worker_count(config: dict[str, Any]) -> int:
    return int(_get(config, "fastvideo_service.data_parallel_workers", 1))


def _fastvideo_worker_ports(config: dict[str, Any]) -> list[int]:
    count = _fastvideo_worker_count(config)
    raw_ports = _get(config, "fastvideo_service.worker_ports")
    if raw_ports is not None:
        ports = [int(port) for port in raw_ports]
        if len(ports) != count:
            raise ConfigError("fastvideo_service.worker_ports length must match data_parallel_workers")
        return ports
    base_port = int(_require(config, "fastvideo_service.port"))
    return [base_port + index for index in range(count)]


def _fastvideo_cuda_devices(config: dict[str, Any]) -> list[str]:
    devices = [
        item.strip()
        for item in str(_require(config, "compute.fastvideo.cuda_visible_devices")).split(",")
        if item.strip()
    ]
    if not devices:
        raise ConfigError("compute.fastvideo.cuda_visible_devices must contain at least one device")
    return devices


def _service_urls(config: dict[str, Any]) -> list[str]:
    host = str(_require(config, "fastvideo_service.public_host"))
    return [f"http://{host}:{port}" for port in _fastvideo_worker_ports(config)]


def _pythonpath(config: dict[str, Any], *, include_megatron: bool) -> str:
    parts = [
        str(_require(config, "paths.repo_root")),
        str(_require(config, "paths.slime_root")),
        str(_require(config, "paths.fastvideo_root")),
    ]
    if include_megatron:
        parts.append(str(_require(config, "paths.megatron_root")))
    return ":".join(parts)


def _reward_weights_json(config: dict[str, Any]) -> str:
    metrics = _require(config, "reward.metrics")
    if not isinstance(metrics, dict) or not metrics:
        raise ConfigError("reward.metrics must be a non-empty mapping")
    return json.dumps({str(key): float(value) for key, value in metrics.items()}, sort_keys=True, separators=(",", ":"))


def _generator_config(config: dict[str, Any]) -> dict[str, Any]:
    generation = _require(config, "generation")
    if not isinstance(generation, dict):
        raise ConfigError("generation must be a mapping")
    keys = [
        "model",
        "height",
        "width",
        "fps",
        "num_frames",
        "num_inference_steps",
        "guidance_scale",
        "flow_shift",
        "negative_prompt",
    ]
    return {key: generation[key] for key in keys}


def _make_dirs(run_dir: Path) -> None:
    for relative in [
        "snapshot",
        "snapshot/validation",
        "commands",
        "logs",
        "checkpoints",
        "wandb",
        "summaries",
        "fastvideo_service/artifacts",
        "fastvideo_service/ledgers",
    ]:
        (run_dir / relative).mkdir(parents=True, exist_ok=True)


def _write_contract(config: dict[str, Any], run_dir: Path) -> None:
    contract = _get(config, "run.contract", {})
    lines = [
        f"Goal: {contract.get('goal', 'Run the RL prompt-enhancer experiment.')}",
        f"Setup: {contract.get('setup', 'Use the YAML config as setup source of truth.')}",
        f"Launch: {contract.get('launch', 'python -m rl_prompt_enhancer.yaml_launcher --config <yaml>')}",
        f"Result: {run_dir}",
        f"Class: {contract.get('class', 'real experiment')}",
    ]
    _write_text(run_dir / "summaries" / "experiment_contract.txt", "\n".join(lines) + "\n")


def _convert_fastvideo_validation_json(config: dict[str, Any]) -> int:
    source_path = _as_path(_require(config, "validation.source_path"), "validation.source_path")
    output_path = _as_path(_require(config, "validation.jsonl_path"), "validation.jsonl_path")
    prompt_id_prefix = str(_require(config, "validation.prompt_id_prefix"))
    with source_path.open(encoding="utf-8") as handle:
        source_data = json.load(handle)
    rows = source_data.get("data") if isinstance(source_data, dict) else None
    if not isinstance(rows, list):
        raise ConfigError(f"validation.source_path must contain a JSON object with a data list: {source_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as output:
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            prompt = str(row.get("caption") or "").strip()
            if not prompt:
                continue
            count += 1
            prompt_id = f"{prompt_id_prefix}_{count:06d}"
            record = {
                "prompt_id": prompt_id,
                "source_path": str(source_path),
                "source_index": index,
                "original_prompt": prompt,
                "metadata": {
                    "prompt_id": prompt_id,
                    "source_path": str(source_path),
                    "source_index": index,
                    "original_prompt": prompt,
                    "split": "validation",
                },
            }
            output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    if count == 0:
        raise ConfigError(f"validation.source_path produced no usable captions: {source_path}")
    return count


def _write_validation_eval_config(config: dict[str, Any]) -> Path | None:
    if not _validation_enabled(config):
        return None

    jsonl_count = _convert_fastvideo_validation_json(config)
    eval_config_path = _as_path(_require(config, "validation.eval_config_path"), "validation.eval_config_path")
    eval_config = {
        "eval": {
            "defaults": {
                "input_key": _require(config, "validation.input_key"),
                "metadata_key": _require(config, "validation.metadata_key"),
                "n_samples_per_eval_prompt": int(_require(config, "validation.n_samples_per_eval_prompt")),
                "max_response_len": int(_require(config, "validation.max_response_len")),
                "temperature": float(_require(config, "validation.temperature")),
            },
            "datasets": [
                {
                    "name": _require(config, "validation.dataset_name"),
                    "path": _require(config, "validation.jsonl_path"),
                    "custom_generate_function_path": _require(
                        config,
                        "prompt_enhancer.custom_generate_function_path",
                    ),
                    "metadata_overrides": {
                        "validation_source": _require(config, "validation.source_path"),
                        "validation_prompt_count": jsonl_count,
                    },
                }
            ],
        }
    }
    eval_config_path.parent.mkdir(parents=True, exist_ok=True)
    eval_config_path.write_text(yaml.safe_dump(eval_config, sort_keys=False), encoding="utf-8")
    return eval_config_path


def _write_snapshot(config_path: Path, config: dict[str, Any], run_dir: Path, *, render_only: bool) -> dict[str, Path]:
    snapshot_dir = run_dir / "snapshot"
    config_snapshot = snapshot_dir / "config.yaml"
    runtime_snapshot = snapshot_dir / "runtime.json"
    source_versions = snapshot_dir / "source_versions.txt"
    path_status = snapshot_dir / "path_status.tsv"

    shutil.copyfile(config_path, config_snapshot)
    _write_text(path_status, "\n".join(_check_required_paths(config, strict=not render_only)) + "\n")
    _write_text(
        source_versions,
        "\n".join(
            [
                _git_snapshot("UniRL", _as_path(_require(config, "paths.repo_root"), "paths.repo_root")),
                _git_snapshot("Slime", _as_path(_require(config, "paths.slime_root"), "paths.slime_root")),
                _git_snapshot("FastVideo", _as_path(_require(config, "paths.fastvideo_root"), "paths.fastvideo_root")),
                _git_snapshot("Megatron-LM", _as_path(_require(config, "paths.megatron_root"), "paths.megatron_root")),
            ]
        ),
    )
    _write_json(
        runtime_snapshot,
        {
            "timestamp_utc": _utc_now(),
            "config_source": str(config_path),
            "config_snapshot": str(config_snapshot),
            "render_only": render_only,
            "run_name": _require(config, "run.name"),
            "run_dir": str(run_dir),
            "fastvideo_service_url": _service_url(config),
            "fastvideo_service_urls": _service_urls(config),
            "health_url": f"{_service_url(config)}/health",
            "health_urls": [f"{url}/health" for url in _service_urls(config)],
        },
    )
    return {
        "config": config_snapshot,
        "runtime": runtime_snapshot,
        "source_versions": source_versions,
        "path_status": path_status,
    }


def _render_ray_script(config: dict[str, Any], run_dir: Path) -> Path:
    python = _as_path(_require(config, "paths.python"), "paths.python")
    ray_bin = python.parent / "ray"
    command = [
        "/usr/bin/env",
        f"CUDA_VISIBLE_DEVICES={_require(config, 'compute.slime.cuda_visible_devices')}",
        str(ray_bin),
        "start",
        "--head",
        "--node-ip-address",
        str(_require(config, "ray.master_addr")),
        "--num-gpus",
        str(_require(config, "compute.slime.num_gpus")),
        "--port",
        str(_require(config, "ray.port")),
        "--dashboard-port",
        str(_require(config, "ray.dashboard_port")),
        "--ray-client-server-port",
        str(_require(config, "ray.client_server_port")),
        "--temp-dir",
        str(_require(config, "ray.temp_dir")),
        "--disable-usage-stats",
    ]
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Start the isolated Ray head for Slime rollout and training workers.",
        f"export RAY_ADDRESS={shlex.quote(str(_require(config, 'ray.address')))}",
        f"export RUN_DIR={shlex.quote(str(run_dir))}",
        _quote(command),
    ]
    if _bool(_get(config, "ray.keep_alive", True)):
        lines.extend(
            [
                'echo "Ray head started; keeping this Slurm step alive."',
                "while true; do sleep 3600; done",
            ]
        )
    script_path = run_dir / "commands" / "start_isolated_ray_head.sh"
    _script(script_path, "\n".join(lines) + "\n")
    return script_path


def _render_fastvideo_script(config: dict[str, Any], run_dir: Path) -> Path:
    python = _as_path(_require(config, "paths.python"), "paths.python")
    uvicorn = python.parent / "uvicorn"
    worker_count = _fastvideo_worker_count(config)
    worker_ports = _fastvideo_worker_ports(config)
    cuda_devices = _fastvideo_cuda_devices(config)
    output_root = str(_require(config, "fastvideo_service.output_root"))
    common_env = [
        f"PYTHONPATH={_pythonpath(config, include_megatron=False)}",
        "PYTHONNOUSERSITE=1",
        f"RLPE_LEDGER_ROOT={_require(config, 'fastvideo_service.ledger_root')}",
        f"RLPE_FASTVIDEO_MODEL={_require(config, 'generation.model')}",
        f"PICKSCORE_PROCESSOR_MODEL={_require(config, 'fastvideo_service.pickscore_processor_model')}",
        f"PICKSCORE_MODEL={_require(config, 'fastvideo_service.pickscore_model')}",
        f"CLIPSCORE_MODEL={_require(config, 'fastvideo_service.clipscore_model')}",
        f"RLPE_FASTVIDEO_EXECUTION_BACKEND={_require(config, 'fastvideo_service.execution_backend')}",
        f"RLPE_FASTVIDEO_WORKLOAD_TYPE={_require(config, 'fastvideo_service.workload_type')}",
        f"RLPE_REWARD_DEVICE={_require(config, 'fastvideo_service.reward_device')}",
        f"RLPE_REWARD_WEIGHTS_JSON={_reward_weights_json(config)}",
        f"RLPE_SCALAR_REWARD_KEY={_require(config, 'reward.scalar_key')}",
    ]
    worker_specs: list[dict[str, Any]] = []
    if worker_count == 1:
        worker_specs.append(
            {
                "worker_id": 0,
                "cuda_visible_devices": str(_require(config, "compute.fastvideo.cuda_visible_devices")),
                "port": worker_ports[0],
                "output_root": output_root,
                "num_gpus": int(_require(config, "compute.fastvideo.num_gpus")),
                "tp_size": int(_require(config, "fastvideo_service.tp_size")),
                "sp_size": int(_require(config, "fastvideo_service.sp_size")),
                "hsdp_replicate_dim": int(_require(config, "fastvideo_service.hsdp_replicate_dim")),
                "hsdp_shard_dim": int(_require(config, "fastvideo_service.hsdp_shard_dim")),
            }
        )
    else:
        for worker_id in range(worker_count):
            worker_specs.append(
                {
                    "worker_id": worker_id,
                    "cuda_visible_devices": cuda_devices[worker_id],
                    "port": worker_ports[worker_id],
                    "output_root": str(Path(output_root) / f"worker_{worker_id}"),
                    "num_gpus": 1,
                    "tp_size": 1,
                    "sp_size": 1,
                    "hsdp_replicate_dim": 1,
                    "hsdp_shard_dim": 1,
                }
            )

    script_path = run_dir / "commands" / "launch_service.sh"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Start FastVideo image generation and reward workers.",
        "pids=()",
        "cleanup() {",
        "  for pid in \"${pids[@]:-}\"; do",
        "    kill \"$pid\" 2>/dev/null || true",
        "  done",
        "  wait 2>/dev/null || true",
        "}",
        "trap cleanup EXIT INT TERM",
    ]
    for spec in worker_specs:
        service_env = [
            *common_env,
            f"CUDA_VISIBLE_DEVICES={spec['cuda_visible_devices']}",
            f"RLPE_FASTVIDEO_WORKER_ID={spec['worker_id']}",
            f"RLPE_FASTVIDEO_SERVICE_PORT={spec['port']}",
            f"RLPE_FASTVIDEO_OUTPUT_ROOT={spec['output_root']}",
            f"RLPE_FASTVIDEO_NUM_GPUS={spec['num_gpus']}",
            f"RLPE_FASTVIDEO_TP_SIZE={spec['tp_size']}",
            f"RLPE_FASTVIDEO_SP_SIZE={spec['sp_size']}",
            f"RLPE_FASTVIDEO_HSDP_REPLICATE_DIM={spec['hsdp_replicate_dim']}",
            f"RLPE_FASTVIDEO_HSDP_SHARD_DIM={spec['hsdp_shard_dim']}",
        ]
        command = [
            "/usr/bin/env",
            *service_env,
            str(uvicorn),
            "rl_prompt_enhancer.fastvideo_bridge.generate_and_score_server:app",
            "--host",
            str(_require(config, "fastvideo_service.host")),
            "--port",
            str(spec["port"]),
            "--log-level",
            "info",
        ]
        lines.extend(
            [
                "",
                f"# Worker {spec['worker_id']}: GPU {spec['cuda_visible_devices']}, port {spec['port']}.",
                f"{_quote(command)} &",
                "pids+=(\"$!\")",
            ]
        )
    lines.extend(
        [
            "",
            f"echo \"started {worker_count} FastVideo worker(s)\"",
            "wait -n",
        ]
    )
    _script(
        script_path,
        "\n".join(lines) + "\n",
    )
    return script_path


def _flag(args: list[str], name: str, value: Any) -> None:
    args.extend([name, str(value)])


def _render_slime_script(config: dict[str, Any], run_dir: Path) -> Path:
    python = _as_path(_require(config, "paths.python"), "paths.python")
    ray_bin = python.parent / "ray"
    slime_root = _as_path(_require(config, "paths.slime_root"), "paths.slime_root")
    model_args = _load_model_args(_as_path(_require(config, "prompt_enhancer.slime_model_script"), "prompt_enhancer.slime_model_script"))

    train_args: list[str] = [*model_args]
    _flag(train_args, "--hf-checkpoint", _require(config, "prompt_enhancer.hf_checkpoint"))
    _flag(train_args, "--ref-load", _require(config, "prompt_enhancer.torch_dist_checkpoint"))
    _flag(train_args, "--save", run_dir / "checkpoints")
    _flag(train_args, "--save-interval", _require(config, "slime_grpo.save_interval"))

    _flag(train_args, "--prompt-data", _require(config, "prompt_data.jsonl_path"))
    _flag(train_args, "--input-key", _require(config, "slime_grpo.input_key"))
    _flag(train_args, "--metadata-key", _require(config, "slime_grpo.metadata_key"))
    if _bool(_require(config, "slime_grpo.apply_chat_template")):
        train_args.append("--apply-chat-template")
    _flag(train_args, "--custom-generate-function-path", _require(config, "prompt_enhancer.custom_generate_function_path"))
    _flag(train_args, "--reward-key", _require(config, "slime_grpo.reward_key"))
    _flag(train_args, "--num-rollout", _require(config, "slime_grpo.num_rollout"))
    _flag(train_args, "--rollout-batch-size", _require(config, "slime_grpo.rollout_batch_size"))
    _flag(train_args, "--n-samples-per-prompt", _require(config, "slime_grpo.n_samples_per_prompt"))
    _flag(train_args, "--rollout-max-response-len", _require(config, "slime_grpo.rollout_max_response_len"))
    _flag(train_args, "--rollout-temperature", _require(config, "slime_grpo.rollout_temperature"))
    _flag(train_args, "--global-batch-size", _require(config, "slime_grpo.global_batch_size"))
    if _validation_enabled(config):
        _flag(train_args, "--eval-interval", _require(config, "validation.eval_interval"))
        _flag(train_args, "--eval-config", _require(config, "validation.eval_config_path"))
        _flag(train_args, "--eval-reward-key", _require(config, "reward.scalar_key"))
        if _bool(_get(config, "validation.skip_eval_before_train", False)):
            train_args.append("--skip-eval-before-train")

    optimizer = _require(config, "slime_grpo.optimizer")
    _flag(train_args, "--optimizer", optimizer["name"])
    _flag(train_args, "--lr", optimizer["lr"])
    _flag(train_args, "--lr-decay-style", optimizer["lr_decay_style"])
    _flag(train_args, "--weight-decay", optimizer["weight_decay"])
    _flag(train_args, "--adam-beta1", optimizer["adam_beta1"])
    _flag(train_args, "--adam-beta2", optimizer["adam_beta2"])

    grpo = _require(config, "slime_grpo.grpo")
    _flag(train_args, "--advantage-estimator", grpo["advantage_estimator"])
    if _bool(grpo["use_kl_loss"]):
        train_args.append("--use-kl-loss")
    _flag(train_args, "--kl-loss-coef", grpo["kl_loss_coef"])
    _flag(train_args, "--kl-loss-type", grpo["kl_loss_type"])
    _flag(train_args, "--entropy-coef", grpo["entropy_coef"])
    _flag(train_args, "--eps-clip", grpo["eps_clip"])
    _flag(train_args, "--eps-clip-high", grpo["eps_clip_high"])

    parallel = _require(config, "slime_grpo.model_parallel")
    _flag(train_args, "--tensor-model-parallel-size", parallel["tensor_model_parallel_size"])
    if _bool(parallel["sequence_parallel"]):
        train_args.append("--sequence-parallel")
    _flag(train_args, "--pipeline-model-parallel-size", parallel["pipeline_model_parallel_size"])
    _flag(train_args, "--context-parallel-size", parallel["context_parallel_size"])
    _flag(train_args, "--expert-model-parallel-size", parallel["expert_model_parallel_size"])
    _flag(train_args, "--expert-tensor-parallel-size", parallel["expert_tensor_parallel_size"])
    _flag(train_args, "--qkv-format", parallel["qkv_format"])
    _flag(train_args, "--micro-batch-size", parallel["micro_batch_size"])

    sglang = _require(config, "slime_grpo.sglang")
    _flag(train_args, "--rollout-num-gpus-per-engine", sglang["rollout_num_gpus_per_engine"])
    _flag(train_args, "--sglang-mem-fraction-static", sglang["mem_fraction_static"])
    _flag(train_args, "--sglang-cuda-graph-max-bs", sglang["cuda_graph_max_bs"])
    if _bool(sglang["enable_metrics"]):
        train_args.append("--sglang-enable-metrics")

    misc = _require(config, "slime_grpo.misc")
    _flag(train_args, "--attention-dropout", misc["attention_dropout"])
    _flag(train_args, "--hidden-dropout", misc["hidden_dropout"])
    if _bool(misc["accumulate_allreduce_grads_in_fp32"]):
        train_args.append("--accumulate-allreduce-grads-in-fp32")
    if _bool(misc["attention_softmax_in_fp32"]):
        train_args.append("--attention-softmax-in-fp32")
    _flag(train_args, "--attention-backend", misc["attention_backend"])
    _flag(train_args, "--loss-mask-type", misc["loss_mask_type"])
    _flag(train_args, "--actor-num-nodes", misc["actor_num_nodes"])
    _flag(train_args, "--actor-num-gpus-per-node", _require(config, "compute.slime.num_gpus"))
    if _bool(misc["colocate"]):
        train_args.append("--colocate")

    if _bool(_require(config, "wandb.enabled")):
        train_args.append("--use-wandb")
        _flag(train_args, "--wandb-project", _require(config, "wandb.project"))
        _flag(train_args, "--wandb-group", _require(config, "wandb.group"))
        _flag(train_args, "--wandb-dir", run_dir / "wandb")
        if _bool(_require(config, "wandb.disable_random_suffix")):
            train_args.append("--disable-wandb-random-suffix")

    runtime_env = {
        "PYTHONPATH": _pythonpath(config, include_megatron=True),
        **_require(config, "ray.runtime_env"),
        "CUDA_VISIBLE_DEVICES": str(_require(config, "compute.slime.cuda_visible_devices")),
        "WANDB_DIR": str(run_dir / "wandb"),
        "RLPE_FASTVIDEO_SERVICE_URL": _service_url(config),
        "RLPE_FASTVIDEO_SERVICE_URLS": ",".join(_service_urls(config)),
        "RLPE_PROMPT_TEMPLATE_PATH": str(_require(config, "prompt_data.prompt_template")),
        "RLPE_SEED_BASE": str(_require(config, "seed_policy.base_seed")),
        "RLPE_GENERATOR_CONFIG_JSON": json.dumps(_generator_config(config), sort_keys=True, separators=(",", ":")),
        "RLPE_REWARD_WEIGHTS_JSON": _reward_weights_json(config),
        "RLPE_SCALAR_REWARD_KEY": str(_require(config, "reward.scalar_key")),
        "CC": str(_require(config, "slime_grpo.jit.cc")),
        "CXX": str(_require(config, "slime_grpo.jit.cxx")),
        "NVCC_PREPEND_FLAGS": str(_require(config, "slime_grpo.jit.nvcc_prepend_flags")),
    }
    runtime_env_json = json.dumps({"env_vars": runtime_env}, separators=(",", ":"))
    command = [
        str(ray_bin),
        "job",
        "submit",
        "--address",
        str(_require(config, "ray.address")),
        "--runtime-env-json",
        runtime_env_json,
        "--",
        str(python),
        str(slime_root / "train.py"),
        *train_args,
    ]
    script_path = run_dir / "commands" / "slime_ray_job_submit.sh"
    _script(
        script_path,
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "",
                "# Submit the Slime GRPO job rendered from snapshot/config.yaml.",
                f"cd {shlex.quote(str(slime_root))}",
                _quote(command),
            ]
        )
        + "\n",
    )
    return script_path


def _slurm_prefix(config: dict[str, Any], role: str) -> list[str]:
    if str(_require(config, "compute.mode")) == "local":
        return []
    node_key = f"compute.{role}.nodelist"
    job_key = f"compute.{role}.job_id"
    command = [
        "srun",
        "--overlap",
        "--jobid",
        str(_require(config, job_key)),
        "--nodes=1",
        "--ntasks=1",
        "--export=ALL",
    ]
    nodelist = str(_require(config, node_key))
    if nodelist:
        command.extend(["--nodelist", nodelist])
    return command


def _prepare_prompt_data(config: dict[str, Any], run_dir: Path) -> None:
    prompt_jsonl = _as_path(_require(config, "prompt_data.jsonl_path"), "prompt_data.jsonl_path")
    if prompt_jsonl.exists() and prompt_jsonl.stat().st_size > 0:
        return
    source = _as_path(_require(config, "prompt_data.source_path"), "prompt_data.source_path")
    python = _as_path(_require(config, "paths.python"), "paths.python")
    command = [
        str(python),
        "-m",
        "rl_prompt_enhancer.data.convert_pickscore_text_to_slime_jsonl",
        "--source",
        str(source),
        "--output",
        str(prompt_jsonl),
        "--prompt-id-prefix",
        str(_require(config, "prompt_data.prompt_id_prefix")),
    ]
    with (run_dir / "logs" / "prepare_prompt_data.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)


def _stream_command(command: list[str], log_path: Path) -> int:
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
            log.flush()
        return process.wait()


def _wait_for_health(config: dict[str, Any], run_dir: Path) -> None:
    urls = [f"{url}/health" for url in _service_urls(config)]
    retries = int(_require(config, "ray.health_retries"))
    interval = int(_require(config, "ray.health_interval_seconds"))
    for attempt in range(1, retries + 1):
        responses = []
        failures = []
        for url in urls:
            command = [*_slurm_prefix(config, "slime"), "curl", "-fsS", url]
            result = subprocess.run(command, check=False, capture_output=True, text=True)
            if result.returncode == 0:
                responses.append({"url": url, "response": result.stdout})
            else:
                failures.append({"url": url, "stderr": result.stderr, "stdout": result.stdout})
        if not failures:
            _write_text(
                run_dir / "logs" / "fastvideo_health.jsonl",
                "\n".join(json.dumps(record, sort_keys=True) for record in responses) + "\n",
            )
            print(f"fastvideo service healthy: {', '.join(urls)}")
            return
        if attempt == retries:
            _write_text(
                run_dir / "logs" / "fastvideo_health_error.json",
                json.dumps(failures, indent=2, sort_keys=True) + "\n",
            )
            raise RuntimeError(f"FastVideo service did not become healthy at all worker URLs: {urls}")
        time.sleep(interval)


def _launch(config: dict[str, Any], run_dir: Path, commands: dict[str, Path]) -> int:
    _check_required_paths(config, strict=True)
    _check_node_local_paths(config)
    _prepare_prompt_data(config, run_dir)
    prompt_count = sum(1 for _ in _as_path(_require(config, "prompt_data.jsonl_path"), "prompt_data.jsonl_path").open())
    print(f"prompt data: {_require(config, 'prompt_data.jsonl_path')} ({prompt_count} records)")

    ray_log = (run_dir / "logs" / "ray_head_srun.log").open("w", encoding="utf-8")
    service_log = (run_dir / "logs" / "fastvideo_service_srun.log").open("w", encoding="utf-8")
    ray_process: subprocess.Popen[str] | None = None
    service_process: subprocess.Popen[str] | None = None
    try:
        ray_process = subprocess.Popen(
            [*_slurm_prefix(config, "slime"), "bash", str(commands["ray"])],
            stdout=ray_log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        service_process = subprocess.Popen(
            [*_slurm_prefix(config, "fastvideo"), "bash", str(commands["fastvideo"])],
            stdout=service_log,
            stderr=subprocess.STDOUT,
            text=True,
        )

        time.sleep(int(_require(config, "ray.startup_seconds")))
        _wait_for_health(config, run_dir)
        slime_command = [*_slurm_prefix(config, "slime"), "bash", str(commands["slime"])]
        return_code = _stream_command(slime_command, run_dir / "logs" / "ray_job_submit.log")
        if return_code != 0:
            raise RuntimeError(f"Slime Ray submit failed with exit code {return_code}")
        return 0
    finally:
        for process in [service_process, ray_process]:
            if process is not None and process.poll() is None:
                process.terminate()
        ray_log.close()
        service_log.close()


def render(config_path: Path, *, render_only: bool) -> tuple[dict[str, Any], Path, dict[str, Path]]:
    config_path = config_path.resolve()
    config = _load_yaml(config_path)
    _validate_config(config)
    run_dir = _as_path(_require(config, "run.output_dir"), "run.output_dir")
    allow_existing = _bool(_get(config, "run.allow_existing", False))
    if run_dir.exists() and any(run_dir.iterdir()) and not allow_existing:
        raise ConfigError(f"run.output_dir already exists and is non-empty: {run_dir}")
    _make_dirs(run_dir)
    _write_contract(config, run_dir)
    validation_eval_config = _write_validation_eval_config(config)
    snapshot_paths = _write_snapshot(config_path, config, run_dir, render_only=render_only)
    commands = {
        "ray": _render_ray_script(config, run_dir),
        "fastvideo": _render_fastvideo_script(config, run_dir),
        "slime": _render_slime_script(config, run_dir),
    }
    runtime = json.loads(snapshot_paths["runtime"].read_text(encoding="utf-8"))
    runtime["rendered_commands"] = {key: str(path) for key, path in commands.items()}
    if validation_eval_config is not None:
        runtime["validation_eval_config"] = str(validation_eval_config)
    _write_json(snapshot_paths["runtime"], runtime)
    return config, run_dir, commands


def _record_launcher_failure(config_path: Path, error: BaseException) -> None:
    try:
        config = _load_yaml(config_path.resolve())
        run_dir = _as_path(_require(config, "run.output_dir"), "run.output_dir")
    except Exception:
        return
    _make_dirs(run_dir)
    message = f"{_utc_now()} {type(error).__name__}: {error}\n"
    with (run_dir / "logs" / "launcher_error.log").open("a", encoding="utf-8") as handle:
        handle.write(message)
    runtime_path = run_dir / "snapshot" / "runtime.json"
    runtime: dict[str, Any] = {}
    if runtime_path.exists():
        try:
            runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            runtime = {}
    runtime["last_failure"] = {
        "timestamp_utc": _utc_now(),
        "error_type": type(error).__name__,
        "error": str(error),
    }
    _write_json(runtime_path, runtime)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="YAML source of truth for this run.")
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Validate config and render run snapshots/commands without launching Ray, FastVideo, or Slime.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config, run_dir, commands = render(args.config, render_only=args.render_only)
        print((run_dir / "summaries" / "experiment_contract.txt").read_text(encoding="utf-8"), end="")
        print(f"config snapshot: {run_dir / 'snapshot' / 'config.yaml'}")
        print(f"rendered command snapshots: {run_dir / 'commands'}")
        if args.render_only:
            print("render-only: launch not executed")
            return 0
        return _launch(config, run_dir, commands)
    except (ConfigError, RuntimeError, subprocess.CalledProcessError) as exc:
        _record_launcher_failure(args.config, exc)
        print(f"yaml launch failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
