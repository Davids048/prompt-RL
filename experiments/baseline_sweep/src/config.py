#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
SUITE_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return data


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def resolve_path(value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    candidate = REPO_ROOT / path
    if candidate.exists() or str(value).startswith(("evaluations/", "outputs/", "FastVideo/")):
        return str(candidate)
    return str(SUITE_ROOT / path)


def short_alias(name: str) -> str:
    if name == "none":
        return "none"
    tail = name.rstrip("/").split("/")[-1]
    alias = re.sub(r"[^A-Za-z0-9_.-]+", "_", tail).strip("_")
    return alias or "unnamed"


def resolve_config(config_path: Path) -> dict[str, Any]:
    raw = load_yaml(config_path)
    required = ("run", "enhancers", "generators", "tasks", "sweep")
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    config = deepcopy(raw)
    config["repo_root"] = str(REPO_ROOT)
    config["suite_root"] = str(SUITE_ROOT)
    config["config_source"] = str(config_path.resolve())
    config["resolved_at_utc"] = dt.datetime.now(dt.UTC).isoformat()
    config["seed"] = int(config.get("seed", 0))
    config["output_root"] = resolve_path(str(config.get("output_root", "outputs/baseline_sweep")))

    resolved_enhancers: dict[str, Any] = {}
    for name, spec in config["enhancers"].items():
        if name == "none":
            resolved_enhancers[name] = {
                "name": "none",
                "alias": "none",
                "backend": "none",
                "template": None,
                "params": {},
            }
            continue
        if not isinstance(spec, dict):
            raise ValueError(f"Enhancer {name!r} must be null or a mapping.")
        item = deepcopy(spec)
        item["name"] = name
        item["alias"] = short_alias(name)
        item.setdefault("params", {})
        if "template" in item and item["template"]:
            item["template"] = resolve_path(str(item["template"]))
        resolved_enhancers[name] = item
    config["enhancers"] = resolved_enhancers

    resolved_generators: dict[str, Any] = {}
    for name, spec in config["generators"].items():
        if not isinstance(spec, dict):
            raise ValueError(f"Generator {name!r} must be a mapping.")
        item = deepcopy(spec)
        item["name"] = name
        item["alias"] = short_alias(name)
        item.setdefault("params", {})
        item["param_provenance"] = generator_param_provenance(name, item.get("params", {}))
        resolved_generators[name] = item
    config["generators"] = resolved_generators

    resolved_tasks: dict[str, Any] = {}
    for name, spec in config["tasks"].items():
        if not isinstance(spec, dict):
            raise ValueError(f"Task {name!r} must be a mapping.")
        item = deepcopy(spec)
        item["name"] = name
        item["alias"] = short_alias(name)
        item["samples_per_prompt"] = int(item.get("samples_per_prompt", 1))
        if "source_path" in item and item["source_path"]:
            item["source_path"] = resolve_path(str(item["source_path"]))
        item.setdefault("eval", [])
        item.setdefault("eval_params", {})
        resolved_tasks[name] = item
    config["tasks"] = resolved_tasks
    return config


def generator_param_provenance(model_name: str, params: dict[str, Any]) -> dict[str, str]:
    provenance: dict[str, str] = {}
    for key in params:
        provenance[key] = "config"
    if "stable-diffusion-3.5" in model_name:
        if params.get("width") == 1024:
            provenance["width"] = "PromptRL local notes: SD3 Table 1 at 1024 resolution"
        if params.get("height") == 1024:
            provenance["height"] = "PromptRL local notes: SD3 Table 1 at 1024 resolution"
        if params.get("steps") == 20:
            provenance["steps"] = "PromptRL local notes: 20 denoising steps"
        if params.get("guidance") == 6.0 or params.get("guidance_scale") == 6.0 or params.get("cfg") == 6.0:
            provenance["guidance"] = "FastVideo sd35_medium preset default"
        elif "cfg" in params or "guidance" in params or "guidance_scale" in params:
            provenance["guidance"] = "config; verify against PromptRL release before claiming reproduction"
    return provenance


def run_dir(config: dict[str, Any]) -> Path:
    return Path(config["output_root"]) / str(config["run"])
