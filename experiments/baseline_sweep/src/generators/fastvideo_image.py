#!/usr/bin/env python3
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any


def generate(
    items: list[dict[str, Any]],
    generator: dict[str, Any],
    trial: dict[str, Any],
    run_dir: Path,
) -> list[dict[str, Any]]:
    del trial
    import torch
    from fastvideo import VideoGenerator

    params = generator.get("params", {})
    attention_backend = params.get("attention_backend")
    if attention_backend:
        os.environ["FASTVIDEO_ATTENTION_BACKEND"] = str(attention_backend)

    init_kwargs = {
        "num_gpus": int(params.get("num_gpus", 1)),
        "workload_type": "t2i",
        "sp_size": 1,
        "tp_size": 1,
        "dit_cpu_offload": bool(params.get("cpu_offload", False)),
        "dit_layerwise_offload": False,
        "text_encoder_cpu_offload": bool(params.get("cpu_offload", False)),
        "vae_cpu_offload": bool(params.get("cpu_offload", False)),
        "image_encoder_cpu_offload": False,
        "pin_cpu_memory": False,
        "use_fsdp_inference": False,
    }
    model = VideoGenerator.from_pretrained(model_path=generator["name"], **init_kwargs)
    results: list[dict[str, Any]] = []
    try:
        for item in items:
            row = dict(item)
            if row.get("error"):
                results.append(row)
                continue
            output_path = run_dir / row["artifact_path"]
            output_path.parent.mkdir(parents=True, exist_ok=True)
            kwargs: dict[str, Any] = {
                "output_path": str(output_path),
                "num_frames": 1,
                "fps": 1,
                "seed": int(row["seed"]),
                "save_video": True,
                "return_frames": False,
                "negative_prompt": str(params.get("negative_prompt", "")),
            }
            if "height" in params:
                kwargs["height"] = int(params["height"])
            if "width" in params:
                kwargs["width"] = int(params["width"])
            if "steps" in params:
                kwargs["num_inference_steps"] = int(params["steps"])
            guidance = params.get("guidance", params.get("guidance_scale", params.get("cfg")))
            if guidance is not None:
                kwargs["guidance_scale"] = float(guidance)
            start = time.perf_counter()
            try:
                generated = model.generate_video(row["generation_prompt"], **kwargs)
                artifact_path = output_path
                if isinstance(generated, dict) and generated.get("video_path"):
                    artifact_path = Path(generated["video_path"])
                    if artifact_path.is_absolute():
                        try:
                            artifact_path = artifact_path.relative_to(run_dir)
                        except ValueError:
                            pass
                row.update({
                    "artifact_path": str(artifact_path),
                    "generation_wall_time_sec": time.perf_counter() - start,
                    "error": None,
                })
            except Exception as exc:  # noqa: BLE001
                row.update({
                    "artifact_path": None,
                    "generation_wall_time_sec": time.perf_counter() - start,
                    "error": {
                        "stage": "generation",
                        "message": repr(exc),
                    },
                })
            results.append(row)
    finally:
        shutdown = getattr(model, "shutdown", None)
        if callable(shutdown):
            shutdown()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results
