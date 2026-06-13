#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


METRIC_KEYS = {
    "VQ": "visual_quality",
    "MQ": "motion_quality",
    "TA": "text_alignment",
    "Overall": "overall",
}
REPO_ROOT = Path(__file__).resolve().parents[4]


def run(
    records: list[dict[str, Any]],
    trial: dict[str, Any],
    run_dir: Path,
) -> list[dict[str, Any]]:
    task = trial["task"]
    params = task.get("eval_params", {}).get("videoalign", {})
    if params.get("fps") is not None and params.get("num_frames") is not None:
        raise ValueError("VideoAlign params cannot set both fps and num_frames.")

    eval_dir = run_dir / "eval" / trial["trial_id"] / "videoalign"
    eval_dir.mkdir(parents=True, exist_ok=True)
    raw_path = eval_dir / "results.jsonl"
    raw_rel = str(raw_path.relative_to(run_dir))

    out = [dict(row) for row in records]
    candidates = [row for row in out if row.get("artifact_path") and not row.get("error")]
    if not candidates:
        return out

    repo = resolve_repo_path(params.get("repo", "evaluations/DanceGRPO"))
    checkpoint_dir = resolve_repo_path(params.get("checkpoint_dir", repo / "videoalign_ckpt"))
    old_path = list(sys.path)
    saved_fastvideo = pop_module_tree("fastvideo")
    inferencer = None
    try:
        sys.path.insert(0, str(repo))
        prepare_videoalign_compat()
        import torch
        from fastvideo.models.videoalign.inference import VideoVLMRewardInference

        device = str(params.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
        dtype = torch_dtype(str(params.get("torch_dtype", "bfloat16")), torch)
        inferencer = VideoVLMRewardInference(
            load_from_pretrained=str(checkpoint_dir),
            load_from_pretrained_step=int(params.get("checkpoint_step", -1)),
            device=device,
            dtype=dtype,
        )

        batch_size = int(params.get("batch_size", 1))
        for chunk_start in range(0, len(candidates), batch_size):
            chunk = candidates[chunk_start:chunk_start + batch_size]
            rewards = inferencer.reward(
                video_paths=[str(artifact_path(row, run_dir)) for row in chunk],
                prompts=[row["eval_prompt"] for row in chunk],
                fps=params.get("fps"),
                num_frames=params.get("num_frames"),
                max_pixels=params.get("max_pixels"),
                use_norm=bool(params.get("use_norm", True)),
            )
            for row, reward in zip(chunk, rewards):
                result = {
                    "trial_id": row["trial_id"],
                    "prompt_index": row["prompt_index"],
                    "sample_index": row["sample_index"],
                    "artifact_path": row["artifact_path"],
                    "eval_prompt": row["eval_prompt"],
                    "source": raw_rel,
                }
                result.update({name: float(reward[key]) for key, name in METRIC_KEYS.items() if key in reward})
                append_jsonl(raw_path, result)
                row.setdefault("eval", {})
                row["eval"]["videoalign"] = {
                    key: value
                    for key, value in result.items()
                    if key in set(METRIC_KEYS.values()) | {"source"}
                }
    finally:
        if inferencer is not None:
            del inferencer
        torch_mod = sys.modules.get("torch")
        if torch_mod is not None and torch_mod.cuda.is_available():
            torch_mod.cuda.empty_cache()
        sys.path[:] = old_path
        restore_module_tree("fastvideo", saved_fastvideo)

    return out


def artifact_path(row: dict[str, Any], run_dir: Path) -> Path:
    path = Path(row["artifact_path"])
    return path if path.is_absolute() else run_dir / path


def resolve_repo_path(value: Any) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def torch_dtype(name: str, torch: Any) -> Any:
    if name.lower() == "float16":
        return torch.float16
    if name.lower() == "bfloat16":
        return torch.bfloat16
    if name.lower() == "float32":
        return torch.float32
    raise ValueError(f"Unsupported VideoAlign torch_dtype={name!r}.")


def prepare_videoalign_compat() -> None:
    """Patch DanceGRPO VideoAlign for the shared FastVideo environment.

    DanceGRPO's VideoAlign code hardcodes FlashAttention-2 for Qwen2-VL. The
    current baseline sweep venv has a flash_attn namespace from flash-attn-4 but
    not the package metadata that transformers uses to validate FlashAttention-2,
    which makes evaluator construction fail before scoring. Keep this workaround
    in the wrapper rather than editing the vendored DanceGRPO repo.
    """
    patch_transformers_video_input_alias()

    import fastvideo.models.videoalign.inference as videoalign_inference

    if getattr(videoalign_inference, "_baseline_sweep_sdpa_compat", False):
        return

    import fastvideo.models.videoalign.trainer as videoalign_trainer
    import fastvideo.models.videoalign.train_reward as train_reward
    import fastvideo.models.videoalign.vision_process as vision_process

    patch_videoalign_base_model_path(train_reward)
    patch_videoalign_reward_model_layout(videoalign_trainer)
    patch_videoalign_peft_checkpoint_load()
    patch_videoalign_video_reader(vision_process)

    original_training_config = videoalign_inference.TrainingConfig

    def TrainingConfig(*args, **kwargs):
        # Force SDPA for evaluator inference. This avoids an environment-level
        # flash_attn compatibility problem without changing model weights,
        # prompts, video sampling, or reward normalization.
        kwargs["disable_flash_attn2"] = True
        return original_training_config(*args, **kwargs)

    videoalign_inference.TrainingConfig = TrainingConfig
    videoalign_inference._baseline_sweep_sdpa_compat = True


def patch_videoalign_video_reader(vision_process: Any) -> None:
    import torchvision.io as torchvision_io

    if hasattr(torchvision_io, "read_video"):
        return
    if getattr(torchvision_io, "_baseline_sweep_pyav_read_video", False):
        return

    import av
    import torch

    def read_video_with_pyav(
        filename,
        start_pts=0.0,
        end_pts=None,
        pts_unit="sec",
        output_format="THWC",
    ):
        # torchvision 0.26 removed read_video, but DanceGRPO's vendored
        # qwen-vl-utils still calls it. Provide the small subset VideoAlign uses
        # through PyAV, which is already present in the baseline sweep venv.
        if pts_unit != "sec":
            raise ValueError("VideoAlign PyAV read_video shim only supports pts_unit='sec'.")
        path = str(filename)
        if path.startswith("file://"):
            path = path[7:]

        container = av.open(path)
        try:
            stream = container.streams.video[0]
            video_fps = float(stream.average_rate) if stream.average_rate else 1.0
            frames = []
            for frame in container.decode(video=0):
                timestamp = frame.time
                if timestamp is None and frame.pts is not None and stream.time_base is not None:
                    timestamp = float(frame.pts * stream.time_base)
                if timestamp is not None and timestamp < start_pts:
                    continue
                if end_pts is not None and timestamp is not None and timestamp > end_pts:
                    break
                frames.append(torch.from_numpy(frame.to_rgb().to_ndarray()))
        finally:
            container.close()

        if not frames:
            raise RuntimeError(f"Could not decode any video frames: {filename}")
        video = torch.stack(frames)
        if output_format == "TCHW":
            video = video.permute(0, 3, 1, 2).contiguous()
        elif output_format != "THWC":
            raise ValueError(f"Unsupported output_format={output_format!r}.")
        return video, None, {"video_fps": video_fps}

    torchvision_io.read_video = read_video_with_pyav
    torchvision_io._baseline_sweep_pyav_read_video = True
    vision_process.get_video_reader_backend.cache_clear()


def patch_videoalign_reward_model_layout(videoalign_trainer: Any) -> None:
    reward_model_cls = videoalign_trainer.Qwen2VLRewardModelBT
    if getattr(reward_model_cls, "_baseline_sweep_layout_compat", False):
        return

    original_forward = reward_model_cls.forward

    def forward_with_videoalign_layout_aliases(self, *args, **kwargs):
        patch_qwen2vl_model_layout_aliases(self)
        return original_forward(self, *args, **kwargs)

    reward_model_cls.forward = forward_with_videoalign_layout_aliases
    reward_model_cls._baseline_sweep_layout_compat = True


def patch_qwen2vl_model_layout_aliases(model: Any) -> None:
    # DanceGRPO's copied VideoAlign forward still uses the older transformers
    # Qwen2-VL layout, where token embeddings lived at self.model.embed_tokens.
    # The shared FastVideo venv uses a newer transformers layout with the
    # language model nested under self.model.language_model.
    inner_model = getattr(model, "model", None)
    language_model = getattr(inner_model, "language_model", None)
    if language_model is not None and not hasattr(inner_model, "embed_tokens"):
        inner_model.embed_tokens = language_model.embed_tokens


def patch_videoalign_peft_checkpoint_load() -> None:
    from peft import PeftModelForCausalLM

    if getattr(PeftModelForCausalLM, "_baseline_sweep_videoalign_checkpoint_compat", False):
        return

    original_load_state_dict = PeftModelForCausalLM.load_state_dict

    def load_state_dict_with_videoalign_remap(self, state_dict, *args, **kwargs):
        state_dict = remap_videoalign_state_dict_for_new_transformers(state_dict)
        return original_load_state_dict(self, state_dict, *args, **kwargs)

    PeftModelForCausalLM.load_state_dict = load_state_dict_with_videoalign_remap
    PeftModelForCausalLM._baseline_sweep_videoalign_checkpoint_compat = True


def remap_videoalign_state_dict_for_new_transformers(state_dict: Any) -> Any:
    if not isinstance(state_dict, dict) or not needs_videoalign_key_remap(state_dict):
        return state_dict

    return {
        remap_videoalign_state_key_for_new_transformers(key): value
        for key, value in state_dict.items()
    }


def needs_videoalign_key_remap(state_dict: dict[str, Any]) -> bool:
    old_visual_prefix = "base_model.model.visual."
    new_visual_prefix = "base_model.model.model.visual."
    old_language_prefixes = (
        "base_model.model.model.embed_tokens.",
        "base_model.model.model.layers.",
        "base_model.model.model.norm.",
    )
    new_language_prefix = "base_model.model.model.language_model."
    has_old_visual_keys = any(key.startswith(old_visual_prefix) for key in state_dict)
    has_new_visual_keys = any(key.startswith(new_visual_prefix) for key in state_dict)
    has_old_language_keys = any(
        key.startswith(prefix)
        for key in state_dict
        for prefix in old_language_prefixes
    )
    has_new_language_keys = any(key.startswith(new_language_prefix) for key in state_dict)
    return (
        has_old_visual_keys
        and not has_new_visual_keys
    ) or (
        has_old_language_keys
        and not has_new_language_keys
    )


def remap_videoalign_state_key_for_new_transformers(key: str) -> str:
    # DanceGRPO's released VideoAlign model.pth was saved with visual tower
    # and language-model keys from an older PEFT/Qwen2-VL layout:
    #   base_model.model.visual.* -> base_model.model.model.visual.*
    #   base_model.model.model.layers.* -> base_model.model.model.language_model.layers.*
    # The newer transformers in this environment nests the language model under
    # language_model, matching the HPSv3 compatibility issue.
    old_visual_prefix = "base_model.model.visual."
    if key.startswith(old_visual_prefix):
        return f"base_model.model.model.visual.{key.removeprefix(old_visual_prefix)}"
    old_language_prefix = "base_model.model.model."
    if key.startswith(old_language_prefix):
        language_key = key.removeprefix(old_language_prefix)
        if (
            language_key.startswith("embed_tokens.")
            or language_key.startswith("layers.")
            or language_key.startswith("norm.")
        ):
            return f"base_model.model.model.language_model.{language_key}"
    return key


def patch_videoalign_base_model_path(train_reward: Any) -> None:
    if getattr(train_reward, "_baseline_sweep_base_model_compat", False):
        return

    original_auto_processor_from_pretrained = train_reward.AutoProcessor.from_pretrained
    original_reward_model_from_pretrained = train_reward.Qwen2VLRewardModelBT.from_pretrained

    def auto_processor_from_pretrained(model_name_or_path, *args, **kwargs):
        return original_auto_processor_from_pretrained(
            resolve_videoalign_base_model(model_name_or_path),
            *args,
            **kwargs,
        )

    def reward_model_from_pretrained(model_name_or_path, *args, **kwargs):
        return original_reward_model_from_pretrained(
            resolve_videoalign_base_model(model_name_or_path),
            *args,
            **kwargs,
        )

    train_reward.AutoProcessor.from_pretrained = auto_processor_from_pretrained
    train_reward.Qwen2VLRewardModelBT.from_pretrained = reward_model_from_pretrained
    train_reward._baseline_sweep_base_model_compat = True


def resolve_videoalign_base_model(model_name_or_path: Any) -> Any:
    # DanceGRPO's released inference code hardcodes "./Qwen2-VL-2B-Instruct",
    # while the downloaded checkpoint config names "Qwen/Qwen2-VL-2B-Instruct".
    # Prefer a real local checkout when present; otherwise use the HF model id.
    if str(model_name_or_path) == "./Qwen2-VL-2B-Instruct" and not Path(model_name_or_path).exists():
        return "Qwen/Qwen2-VL-2B-Instruct"
    return model_name_or_path


def patch_transformers_video_input_alias() -> None:
    from transformers import image_utils

    # Newer transformers no longer exposes VideoInput from image_utils. Some
    # copied Qwen2-VL processors still import it for typing, so restore the alias
    # locally for evaluator imports.
    if not hasattr(image_utils, "VideoInput") and hasattr(image_utils, "ImageInput"):
        image_utils.VideoInput = image_utils.ImageInput


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
