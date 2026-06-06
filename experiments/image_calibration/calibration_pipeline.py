#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METADATA = REPO_ROOT / "evaluations/geneval/prompts/evaluation_metadata.jsonl"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/image_calibration"

DEFAULT_REWRITE_SYSTEM_PROMPT = (
    "You are a prompt rewriting assistant for text-to-image generation. "
    "Rewrite the user's prompt into one semantically equivalent prompt. "
    "Preserve every object, count, color, spatial relation, visible text string, "
    "and other required attribute exactly. You may add concise visual detail, "
    "composition, lighting, camera, or style cues only when they do not change "
    "the requested content. Do not add new objects, remove objects, change counts, "
    "change colors, or change text. Return only the rewritten prompt."
)

DEFAULT_REWRITE_USER_TEMPLATE = (
    "Original prompt:\n{prompt}\n\n"
    "Return a single enhanced prompt that preserves the same semantic requirements."
)

REWRITER_SPECS: dict[str, dict[str, str]] = {
    "none": {
        "kind": "none",
        "model_id": "none",
        "description": "Use the original prompt without rewriting.",
    },
    "qwen25_vl_3b": {
        "kind": "qwen_vl",
        "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "description": "PromptRL-style Qwen2.5-VL-3B-Instruct prompt enhancement.",
    },
    "qwen25_3b": {
        "kind": "qwen_text",
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "description": "Text-only Qwen2.5-3B-Instruct prompt rewrite comparison.",
    },
    "qwen25_1_5b": {
        "kind": "qwen_text",
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "description": "Smaller text-only Qwen2.5 instruct prompt rewrite comparison.",
    },
}


def now_id() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def sanitize_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return text.strip("_") or "unnamed"


def clean_rewrite(text: str) -> str:
    text = text.strip()
    answer_matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.I | re.S)
    if answer_matches:
        text = answer_matches[-1].strip()
    text = re.sub(r"</?answer>", "", text, flags=re.I).strip()
    text = re.sub(r"^```(?:text)?", "", text, flags=re.I).strip()
    text = re.sub(r"```$", "", text).strip()
    text = re.sub(r"^(rewritten|enhanced)\s+prompt\s*:\s*", "", text, flags=re.I).strip()
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()
    return " ".join(text.split())


def parse_unifiedreward_score(output_text: str) -> float | None:
    match = re.search(r"Final Score\s*:\s*([1-5](?:\.[0-9]+)?)", output_text, flags=re.I)
    if match:
        return float(match.group(1))
    numeric = re.fullmatch(r"(?:score\s*[:=]\s*)?([1-5](?:\.[0-9]+)?)\.?", output_text.strip(), flags=re.I)
    return float(numeric.group(1)) if numeric else None


def selected_metadata(path: Path, limit: int | None, offset: int, tags: set[str] | None) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < offset:
                continue
            metadata = json.loads(line)
            if tags and metadata.get("tag") not in tags:
                continue
            metadata["_source_index"] = idx
            rows.append(metadata)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def cmd_select_metadata(args: argparse.Namespace) -> None:
    source = Path(args.metadata)
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} already exists; pass --overwrite to replace it.")

    tags = set(args.tags or []) or None
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    with source.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            metadata = json.loads(line)
            tag = metadata["tag"]
            if tags and tag not in tags:
                continue
            if len(grouped[tag]) < args.per_tag:
                grouped[tag].append((idx, metadata))

    selected: list[dict[str, Any]] = []
    source_indices: list[int] = []
    for tag in sorted(grouped):
        for source_idx, metadata in grouped[tag]:
            selected.append(metadata)
            source_indices.append(source_idx)

    write_jsonl(output, selected)
    write_json(
        output.with_suffix(output.suffix + ".config.json"),
        {
            "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
            "source": str(source.resolve()),
            "output": str(output.resolve()),
            "per_tag": args.per_tag,
            "tags": sorted(grouped),
            "count": len(selected),
            "source_indices": source_indices,
        },
    )
    print(f"[select-metadata] wrote {len(selected)} rows to {output}")


def cmd_import_prompts(args: argparse.Namespace) -> None:
    source = Path(args.input)
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} already exists; pass --overwrite to replace it.")

    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            prompt = line.strip()
            if not prompt:
                continue
            if args.skip_comments and prompt.startswith("#"):
                continue
            row: dict[str, Any] = {
                "prompt": prompt,
                "benchmark": args.benchmark_name,
                "source_file": str(source.resolve()),
                "source_line": idx + 1,
            }
            if args.tag:
                row["tag"] = args.tag
            rows.append(row)

    write_jsonl(output, rows)
    print(f"[import-prompts] wrote {len(rows)} rows to {output}")


def load_text_file_or_default(path: str | None, default: str) -> str:
    if not path:
        return default
    return Path(path).read_text(encoding="utf-8")


class PromptRewriter:
    def __init__(
        self,
        condition: str,
        system_prompt: str,
        user_template: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        use_system_prompt: bool,
    ) -> None:
        if condition not in REWRITER_SPECS:
            raise ValueError(f"Unknown rewrite condition {condition!r}; choices: {sorted(REWRITER_SPECS)}")
        self.condition = condition
        self.spec = REWRITER_SPECS[condition]
        self.system_prompt = system_prompt
        self.user_template = user_template
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.use_system_prompt = use_system_prompt
        self.model = None
        self.tokenizer = None
        self.processor = None

    def load(self) -> None:
        if self.spec["kind"] == "none":
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

        model_id = self.spec["model_id"]
        if self.spec["kind"] == "qwen_vl":
            from transformers import Qwen2_5_VLForConditionalGeneration

            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            ).eval()
        elif self.spec["kind"] == "qwen_text":
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            ).eval()
        else:
            raise ValueError(f"Unsupported rewriter kind: {self.spec['kind']}")

    def rewrite(self, prompt: str) -> str:
        if self.spec["kind"] == "none":
            return prompt
        user_prompt = self.user_template.format(prompt=prompt)
        if self.spec["kind"] == "qwen_vl":
            return self._rewrite_qwen_vl(user_prompt)
        return self._rewrite_qwen_text(user_prompt)

    def _rewrite_qwen_text(self, user_prompt: str) -> str:
        import torch

        assert self.model is not None
        assert self.tokenizer is not None
        messages = []
        if self.use_system_prompt and self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        do_sample = self.temperature > 0
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=do_sample,
                temperature=self.temperature if do_sample else None,
                top_p=self.top_p if do_sample else None,
            )
        generated = generated[:, inputs.input_ids.shape[1]:]
        return clean_rewrite(self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0])

    def _rewrite_qwen_vl(self, user_prompt: str) -> str:
        import torch

        assert self.model is not None
        assert self.processor is not None
        messages = []
        if self.use_system_prompt and self.system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": self.system_prompt}]})
        messages.append({"role": "user", "content": [{"type": "text", "text": user_prompt}]})
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], padding=True, return_tensors="pt").to(self.model.device)
        do_sample = self.temperature > 0
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=do_sample,
                temperature=self.temperature if do_sample else None,
                top_p=self.top_p if do_sample else None,
            )
        generated = generated[:, inputs.input_ids.shape[1]:]
        return clean_rewrite(self.processor.batch_decode(generated, skip_special_tokens=True)[0])


def cmd_rewrite(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir) if args.run_dir else DEFAULT_OUTPUT_ROOT / now_id()
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata_rows = selected_metadata(Path(args.metadata), args.limit, args.offset, set(args.tags or []) or None)
    system_prompt = load_text_file_or_default(args.system_prompt_file, DEFAULT_REWRITE_SYSTEM_PROMPT)
    user_template = load_text_file_or_default(args.user_template_file, DEFAULT_REWRITE_USER_TEMPLATE)
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    use_system_prompt = not args.no_system_prompt

    config = {
        "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "metadata_path": str(Path(args.metadata).resolve()),
        "limit": args.limit,
        "offset": args.offset,
        "tags": args.tags,
        "conditions": conditions,
        "sample_count": args.sample_count,
        "seed_base": args.seed_base,
        "rewrite_system_prompt": system_prompt,
        "rewrite_use_system_prompt": use_system_prompt,
        "rewrite_user_template": user_template,
        "rewrite_max_new_tokens": args.max_new_tokens,
        "rewrite_temperature": args.temperature,
        "rewrite_top_p": args.top_p,
        "rewriter_specs": {c: REWRITER_SPECS[c] for c in conditions},
    }
    write_json(run_dir / "run_config_rewrite.json", config)

    ledger_path = run_dir / "prompt_ledger.jsonl"
    if args.overwrite and ledger_path.exists():
        ledger_path.unlink()

    for condition in conditions:
        rewriter = PromptRewriter(
            condition=condition,
            system_prompt=system_prompt,
            user_template=user_template,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            use_system_prompt=use_system_prompt,
        )
        rewriter.load()
        for metadata in metadata_rows:
            original_prompt = metadata["prompt"]
            start = time.perf_counter()
            try:
                final_prompt = rewriter.rewrite(original_prompt)
                error = None
            except Exception as exc:  # noqa: BLE001
                final_prompt = original_prompt
                error = repr(exc)
            rewrite_time = time.perf_counter() - start
            for sample_idx in range(args.sample_count):
                seed = args.seed_base + int(metadata["_source_index"]) * args.sample_count + sample_idx
                append_jsonl(
                    ledger_path,
                    {
                        "prompt_index": int(metadata["_source_index"]),
                        "sample_index": sample_idx,
                        "condition": condition,
                        "rewriter_kind": REWRITER_SPECS[condition]["kind"],
                        "rewriter_model_id": REWRITER_SPECS[condition]["model_id"],
                        "rewrite_system_prompt": system_prompt if condition != "none" and use_system_prompt else None,
                        "rewrite_use_system_prompt": use_system_prompt if condition != "none" else False,
                        "rewrite_user_prompt": user_template.format(prompt=original_prompt) if condition != "none" else None,
                        "original_prompt": original_prompt,
                        "rewritten_prompt": final_prompt if condition != "none" else None,
                        "final_prompt": final_prompt,
                        "seed": seed,
                        "metadata": {k: v for k, v in metadata.items() if k != "_source_index"},
                        "rewrite_error": error,
                        "rewrite_wall_time_sec": rewrite_time,
                    },
                )
                print(f"[rewrite] {condition} idx={metadata['_source_index']} sample={sample_idx} error={error}")
    print(f"[rewrite] ledger: {ledger_path}")


def image_output_paths(run_dir: Path, condition: str, prompt_index: int, sample_index: int) -> tuple[Path, Path]:
    prompt_dir = run_dir / "images" / condition / f"{prompt_index:05d}"
    sample_dir = prompt_dir / "samples"
    image_path = sample_dir / f"{sample_index:04d}.png"
    return prompt_dir, image_path


def resolve_generator_backend(model_path: str, requested_backend: str) -> str:
    if requested_backend != "auto":
        return requested_backend
    if "flux" in model_path.lower():
        return "diffusers_flux"
    return "fastvideo"


class DiffusersFluxImageGenerator:
    def __init__(self, model_path: str, cpu_offload: bool) -> None:
        import torch
        from diffusers import FluxPipeline

        self.torch = torch
        self.pipe = FluxPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        )
        if cpu_offload:
            self.pipe.enable_model_cpu_offload()
            self.device = "cuda"
        else:
            self.pipe.to("cuda")
            self.device = "cuda"

    def generate_image(
        self,
        prompt: str,
        output_path: Path,
        height: int,
        width: int,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int,
        negative_prompt: str,
    ) -> dict[str, str]:
        generator = self.torch.Generator(device=self.device).manual_seed(seed)
        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "max_sequence_length": 512,
        }
        if negative_prompt:
            kwargs["negative_prompt"] = negative_prompt
        image = self.pipe(**kwargs).images[0]
        image.save(output_path)
        return {"video_path": str(output_path)}

    def shutdown(self) -> None:
        del self.pipe
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def cmd_generate(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    ledger = read_jsonl(Path(args.ledger) if args.ledger else run_dir / "prompt_ledger.jsonl")
    conditions = set(args.conditions.split(",")) if args.conditions else None
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = args.attention_backend
    if args.disable_nvfp4:
        os.environ.pop("FASTVIDEO_NVFP4_FA4", None)

    generator_backend = resolve_generator_backend(args.model_path, args.generator_backend)

    init_kwargs = {
        "num_gpus": args.num_gpus,
        "workload_type": "t2i",
        "sp_size": 1,
        "tp_size": 1,
        "dit_cpu_offload": args.cpu_offload,
        "dit_layerwise_offload": False,
        "text_encoder_cpu_offload": args.cpu_offload,
        "vae_cpu_offload": args.cpu_offload,
        "image_encoder_cpu_offload": False,
        "pin_cpu_memory": False,
        "use_fsdp_inference": False,
    }
    write_json(
        run_dir / "run_config_generate.json",
        {
            "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
            "model_path": args.model_path,
            "generator_backend": generator_backend,
            "height": args.height,
            "width": args.width,
            "steps": args.steps,
            "guidance": args.guidance,
            "negative_prompt": args.negative_prompt,
            "attention_backend": args.attention_backend,
            "num_gpus": args.num_gpus,
            "cpu_offload": args.cpu_offload,
            "metric_target_prompt": "original_prompt",
            "init_kwargs": init_kwargs,
        },
    )
    artifact_path = run_dir / "artifacts.jsonl"
    if args.overwrite and artifact_path.exists():
        artifact_path.unlink()
    existing_artifacts = set()
    if artifact_path.exists():
        for row in read_jsonl(artifact_path):
            existing_artifacts.add((row["condition"], int(row["prompt_index"]), int(row["sample_index"])))

    if generator_backend == "fastvideo":
        from fastvideo import VideoGenerator

        generator = VideoGenerator.from_pretrained(model_path=args.model_path, **init_kwargs)

        def generate_one(row: dict[str, Any], image_path: Path) -> dict[str, Any]:
            return generator.generate_video(
                row["final_prompt"],
                output_path=str(image_path),
                height=args.height,
                width=args.width,
                num_frames=1,
                fps=1,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                seed=int(row["seed"]),
                negative_prompt=args.negative_prompt,
                save_video=True,
                return_frames=False,
            )

    elif generator_backend == "diffusers_flux":
        generator = DiffusersFluxImageGenerator(args.model_path, args.cpu_offload)

        def generate_one(row: dict[str, Any], image_path: Path) -> dict[str, Any]:
            return generator.generate_image(
                prompt=row["final_prompt"],
                output_path=image_path,
                height=args.height,
                width=args.width,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                seed=int(row["seed"]),
                negative_prompt=args.negative_prompt,
            )

    else:
        raise ValueError(f"Unsupported generator backend: {generator_backend}")

    try:
        for row in ledger:
            if conditions and row["condition"] not in conditions:
                continue
            artifact_key = (row["condition"], int(row["prompt_index"]), int(row["sample_index"]))
            if artifact_key in existing_artifacts and not args.overwrite:
                print(f"[generate] skipped_recorded {row['condition']} idx={row['prompt_index']} sample={row['sample_index']}")
                continue
            prompt_dir, image_path = image_output_paths(
                run_dir,
                row["condition"],
                int(row["prompt_index"]),
                int(row["sample_index"]),
            )
            (prompt_dir / "samples").mkdir(parents=True, exist_ok=True)
            metadata_path = prompt_dir / "metadata.jsonl"
            if not metadata_path.exists() or args.overwrite:
                metadata_path.write_text(json.dumps(row["metadata"], ensure_ascii=False) + "\n", encoding="utf-8")
            if image_path.exists() and not args.overwrite:
                status = "skipped_existing"
                generation_time = None
            else:
                start = time.perf_counter()
                result = generate_one(row, image_path)
                generation_time = time.perf_counter() - start
                status = "generated"
                if result.get("video_path") and Path(result["video_path"]) != image_path:
                    image_path = Path(result["video_path"])
            artifact = {
                **row,
                "generator_model_path": args.model_path,
                "height": args.height,
                "width": args.width,
                "num_inference_steps": args.steps,
                "guidance_scale": args.guidance,
                "negative_prompt": args.negative_prompt,
                "attention_backend": args.attention_backend,
                "image_path": str(image_path),
                "geneval_prompt_dir": str(prompt_dir),
                "generation_status": status,
                "generation_wall_time_sec": generation_time,
            }
            append_jsonl(artifact_path, artifact)
            existing_artifacts.add(artifact_key)
            print(f"[generate] {status} {row['condition']} idx={row['prompt_index']} sample={row['sample_index']} -> {image_path}")
    finally:
        generator.shutdown()
    print(f"[generate] artifacts: {artifact_path}")


class PickScoreScorer:
    def __init__(self, device: str) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        self.device = device
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").eval().to(device)

    def score(self, image_path: str, prompt: str) -> float:
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        image_inputs = self.processor(images=[image], padding=True, truncation=True, max_length=77, return_tensors="pt").to(self.device)
        text_inputs = self.processor(text=[prompt], padding=True, truncation=True, max_length=77, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            image_embs = self.model.get_image_features(**image_inputs)
            image_embs = image_embs / self.torch.norm(image_embs, dim=-1, keepdim=True)
            text_embs = self.model.get_text_features(**text_inputs)
            text_embs = text_embs / self.torch.norm(text_embs, dim=-1, keepdim=True)
            score = self.model.logit_scale.exp() * (text_embs @ image_embs.T)[0, 0]
        return float(score.detach().cpu())


class HPSScorer:
    def __init__(self, device: str, hps_version: str, checkpoint: str | None) -> None:
        import torch
        import huggingface_hub

        sys.path.insert(0, str(REPO_ROOT / "evaluations/HPSv2"))
        from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
        from hpsv2.utils import hps_version_map

        self.device = device
        self.torch = torch
        self.tokenizer = get_tokenizer("ViT-H-14")
        self.model, _, self.preprocess = create_model_and_transforms(
            "ViT-H-14",
            "laion2B-s32B-b79K",
            precision="amp",
            device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False,
        )
        cp = checkpoint or huggingface_hub.hf_hub_download("xswu/HPSv2", hps_version_map[hps_version])
        state = torch.load(cp, map_location=device)
        self.model.load_state_dict(state["state_dict"])
        self.model = self.model.to(device).eval()

    def score(self, image_path: str, prompt: str) -> float:
        from PIL import Image

        image = self.preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(self.device)
        text = self.tokenizer([prompt]).to(self.device)
        with self.torch.inference_mode(), self.torch.amp.autocast("cuda", enabled=self.device.startswith("cuda")):
            outputs = self.model(image, text)
            score = self.torch.diagonal(outputs["image_features"] @ outputs["text_features"].T)[0]
        # HPS benchmark reporting multiplies the normalized image-text dot product by 100.
        return float((score * 100.0).detach().cpu())


class UnifiedRewardScorer:
    def __init__(self, device: str, model_id: str, max_new_tokens: int) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.torch = torch
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        from qwen_vl_utils import process_vision_info

        self.process_vision_info = process_vision_info

    @staticmethod
    def problem(prompt: str) -> str:
        return (
            "You are given a text caption and a generated image based on that caption. "
            "Your task is to evaluate this image based on two key criteria:\n"
            "1. Alignment with the Caption: Assess how well this image aligns with the provided caption. "
            "Consider the accuracy of depicted objects, their relationships, and attributes as described in the caption.\n"
            "2. Overall Image Quality: Examine the visual quality of this image, including clarity, detail preservation, "
            "color accuracy, and overall aesthetic appeal.\n"
            "Based on the above criteria, assign a score from 1 to 5 after 'Final Score:'.\n"
            "Your task is provided as follows:\n"
            f"Text Caption: [{prompt}]"
        )

    def score(self, image_path: str, prompt: str) -> tuple[float | None, str]:
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": self.problem(prompt)},
            ],
        }]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        generated_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return parse_unifiedreward_score(output_text), output_text


class UnifiedRewardSGLangScorer:
    def __init__(self, api_base: str, api_key: str, model_name: str, timeout: float) -> None:
        import requests

        self.requests = requests
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = timeout

    @staticmethod
    def problem(prompt: str) -> str:
        return (
            "<image>\n"
            "You are given a text caption and a generated image based on that caption. "
            "Your task is to evaluate this image based on two key criteria:\n"
            "1. Alignment with the Caption: Assess how well this image aligns with the provided caption. "
            "Consider the accuracy of depicted objects, their relationships, and attributes as described in the caption.\n"
            "2. Overall Image Quality: Examine the visual quality of this image, including clarity, detail preservation, "
            "color accuracy, and overall aesthetic appeal.\n"
            "Based on the above criteria, assign a score from 1 to 5 after 'Final Score:'.\n"
            "Your task is provided as follows:\n"
            f"Text Caption: [{prompt}]"
        )

    @staticmethod
    def image_url(image_path: str) -> str:
        from PIL import Image

        image = Image.open(image_path).convert("RGB").resize((512, 512))
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image;base64,{encoded}"

    def score(self, image_path: str, prompt: str) -> tuple[float | None, str]:
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": self.image_url(image_path)},
                        },
                        {
                            "type": "text",
                            "text": self.problem(prompt),
                        },
                    ],
                },
            ],
            "temperature": 0,
        }
        response = self.requests.post(
            f"{self.api_base}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        output_text = data["choices"][0]["message"]["content"]
        return parse_unifiedreward_score(output_text), output_text


class OCRBenchmarkScorer:
    def __init__(self, use_gpu: bool) -> None:
        sys.path.insert(0, str(REPO_ROOT / "evaluations/flow_grpo"))
        from paddleocr import PaddleOCR

        cache_dir = REPO_ROOT / ".cache/paddleocr"
        self.ocr = PaddleOCR(
            use_angle_cls=False,
            lang="en",
            use_gpu=use_gpu,
            show_log=False,
            det_model_dir=str(cache_dir / "whl/det/en/en_PP-OCRv3_det_infer"),
            rec_model_dir=str(cache_dir / "whl/rec/en/en_PP-OCRv4_rec_infer"),
            cls_model_dir=str(cache_dir / "whl/cls/ch_ppocr_mobile_v2.0_cls_infer"),
        )

    def score(self, image_path: str, prompt: str) -> float:
        from Levenshtein import distance
        from PIL import Image
        import numpy as np

        target_text = prompt.split('"')[1].replace(" ", "").lower()
        image = np.array(Image.open(image_path).convert("RGB"))
        try:
            result = self.ocr.ocr(image, cls=False)
            recognized_text = "".join([res[1][0] if res[1][1] > 0 else "" for res in result[0]]) if result[0] else ""
            recognized_text = recognized_text.replace(" ", "").lower()
            if target_text in recognized_text:
                dist = 0
            else:
                dist = min(distance(recognized_text, target_text), len(target_text))
        except Exception as exc:  # noqa: BLE001
            print(f"OCR processing failed: {exc!r}")
            dist = len(target_text)
        return 1 - dist / len(target_text)


def cmd_eval_preference(args: argparse.Namespace) -> None:
    import torch

    run_dir = Path(args.run_dir)
    artifacts = read_jsonl(Path(args.artifacts) if args.artifacts else run_dir / "artifacts.jsonl")
    metrics = {m.strip() for m in args.metrics.split(",") if m.strip()}
    allowed_metrics = {"pickscore", "hps", "ur", "ocr"}
    unknown_metrics = metrics - allowed_metrics
    if unknown_metrics:
        raise ValueError(f"Unknown metrics {sorted(unknown_metrics)}; choices: {sorted(allowed_metrics)}")
    if args.cpu:
        device = "cpu"
    elif args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but torch.cuda.is_available() is false.")
    eval_config = {
        "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "metrics": sorted(metrics),
        "metric_prompt": args.metric_prompt,
        "device": device,
        "hps_version": args.hps_version,
        "hps_checkpoint": args.hps_checkpoint,
        "ur_backend": args.ur_backend,
        "ur_model_id": args.ur_model_id,
        "ur_max_new_tokens": args.ur_max_new_tokens,
        "ur_api_base": args.ur_api_base,
        "ur_api_key": args.ur_api_key,
        "ur_model_name": args.ur_model_name,
        "ur_request_timeout": args.ur_request_timeout,
        "ocr_use_gpu": args.ocr_use_gpu if args.ocr_use_gpu is not None else device == "cuda",
        "scorer_load_wall_time_sec": {},
    }
    scorers: dict[str, Any] = {}
    if "pickscore" in metrics:
        start = time.perf_counter()
        scorers["pickscore"] = PickScoreScorer(device)
        eval_config["scorer_load_wall_time_sec"]["pickscore"] = time.perf_counter() - start
    if "hps" in metrics:
        start = time.perf_counter()
        scorers["hps"] = HPSScorer(device, args.hps_version, args.hps_checkpoint)
        eval_config["scorer_load_wall_time_sec"]["hps"] = time.perf_counter() - start
    if "ur" in metrics:
        start = time.perf_counter()
        if args.ur_backend == "local":
            scorers["ur"] = UnifiedRewardScorer(device, args.ur_model_id, args.ur_max_new_tokens)
        elif args.ur_backend == "sglang":
            scorers["ur"] = UnifiedRewardSGLangScorer(
                args.ur_api_base,
                args.ur_api_key,
                args.ur_model_name,
                args.ur_request_timeout,
            )
        else:
            raise ValueError(f"Unsupported UR backend: {args.ur_backend}")
        eval_config["scorer_load_wall_time_sec"]["ur"] = time.perf_counter() - start
    if "ocr" in metrics:
        start = time.perf_counter()
        scorers["ocr"] = OCRBenchmarkScorer(eval_config["ocr_use_gpu"])
        eval_config["scorer_load_wall_time_sec"]["ocr"] = time.perf_counter() - start
    write_json(run_dir / "run_config_eval_preference.json", eval_config)

    out_path = run_dir / "metrics_preference.jsonl"
    if args.overwrite and out_path.exists():
        out_path.unlink()
    existing_metrics = set()
    if out_path.exists():
        for row in read_jsonl(out_path):
            recorded_metrics = tuple(sorted(row.get("requested_metrics", [])))
            recorded_prompt_mode = row.get("metric_prompt_mode", args.metric_prompt)
            existing_metrics.add(
                (
                    row["condition"],
                    int(row["prompt_index"]),
                    int(row["sample_index"]),
                    recorded_metrics,
                    recorded_prompt_mode,
                )
            )
    for artifact in artifacts:
        metric_key = (
            artifact["condition"],
            int(artifact["prompt_index"]),
            int(artifact["sample_index"]),
            tuple(sorted(metrics)),
            args.metric_prompt,
        )
        if metric_key in existing_metrics and not args.overwrite:
            print(f"[eval-preference] skipped_recorded {artifact['condition']} idx={artifact['prompt_index']} sample={artifact['sample_index']}")
            continue
        image_path = artifact["image_path"]
        eval_prompt = artifact["original_prompt"] if args.metric_prompt == "original" else artifact["final_prompt"]
        row_start = time.perf_counter()
        row: dict[str, Any] = {
            "condition": artifact["condition"],
            "prompt_index": artifact["prompt_index"],
            "sample_index": artifact["sample_index"],
            "image_path": image_path,
            "requested_metrics": sorted(metrics),
            "metric_prompt_mode": args.metric_prompt,
            "metric_prompt": eval_prompt,
        }
        for name, scorer in scorers.items():
            metric_start = time.perf_counter()
            try:
                if name == "ur":
                    score, raw = scorer.score(image_path, eval_prompt)
                    row["ur"] = score
                    row["ur_raw_output"] = raw
                else:
                    row[name] = scorer.score(image_path, eval_prompt)
            except Exception as exc:  # noqa: BLE001
                row[f"{name}_error"] = repr(exc)
            row[f"{name}_wall_time_sec"] = time.perf_counter() - metric_start
        row["eval_wall_time_sec"] = time.perf_counter() - row_start
        append_jsonl(out_path, row)
        existing_metrics.add(metric_key)
        print(f"[eval-preference] {artifact['condition']} idx={artifact['prompt_index']} sample={artifact['sample_index']} -> {row}")
    summarize_run(run_dir)


def cmd_eval_geneval(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    conditions = [p.name for p in (run_dir / "images").iterdir() if p.is_dir()]
    if args.conditions:
        wanted = set(args.conditions.split(","))
        conditions = [c for c in conditions if c in wanted]
    if args.server_url:
        cmd_eval_geneval_server(args, run_dir, conditions)
        return
    status: dict[str, Any] = {
        "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "conditions": conditions,
        "model_path": args.model_path,
        "results": {},
    }
    try:
        import mmdet  # noqa: F401
        import open_clip  # noqa: F401
        import clip_benchmark  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        status["status"] = "missing_dependencies"
        status["error"] = repr(exc)
        status["note"] = (
            "Official GenEval requires the legacy mmdet/mmcv/open_clip stack. "
            "The generated image folders already match GenEval's expected layout."
        )
        write_json(run_dir / "geneval_status.json", status)
        print(json.dumps(status, indent=2))
        return

    for condition in conditions:
        image_dir = run_dir / "images" / condition
        out_file = run_dir / "geneval" / f"{condition}_results.jsonl"
        cmd = [
            sys.executable,
            str(REPO_ROOT / "evaluations/geneval/evaluation/evaluate_images.py"),
            str(image_dir),
            "--outfile",
            str(out_file),
            "--model-path",
            args.model_path,
        ]
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
        summary_cmd = [
            sys.executable,
            str(REPO_ROOT / "evaluations/geneval/evaluation/summary_scores.py"),
            str(out_file),
        ]
        summary = subprocess.check_output(summary_cmd, cwd=str(REPO_ROOT), text=True)
        summary_path = run_dir / "geneval" / f"{condition}_summary.txt"
        summary_path.write_text(summary, encoding="utf-8")
        status["results"][condition] = {
            "results_jsonl": str(out_file),
            "summary_txt": str(summary_path),
            "summary_stdout": summary,
        }
    status["status"] = "ok"
    write_json(run_dir / "geneval_status.json", status)
    summarize_run(run_dir)


def mean(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def mean_non_placeholder(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None and v >= 0]
    return sum(vals) / len(vals) if vals else None


def timing_stats(values: list[float | None]) -> dict[str, float | int] | None:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return {
        "count": len(vals),
        "sum": sum(vals),
        "mean": sum(vals) / len(vals),
        "min": min(vals),
        "max": max(vals),
    }


def add_summary_metric(
    summary: dict[str, Any],
    condition: str,
    metric: str,
    value: float | None,
    count: int | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    if value is None:
        return
    stat: dict[str, Any] = {"mean": float(value)}
    if count is not None:
        stat["count"] = count
    if extra:
        stat.update(extra)
    summary["conditions"].setdefault(condition, {})[metric] = stat


def escape_md_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    text = " ".join(text.split())
    return text.replace("|", "\\|")


def rel_to_run(run_dir: Path, value: str | None) -> str:
    if not value:
        return ""
    path = Path(value)
    try:
        return str(path.resolve().relative_to(run_dir.resolve()))
    except (OSError, ValueError):
        return str(path)


def write_review_report(run_dir: Path, summary: dict[str, Any]) -> None:
    lines = ["# Image Calibration Run Review", ""]
    lines.append(f"Run directory: `{run_dir}`")
    lines.append("")

    config_paths = sorted(run_dir.glob("run_config_*.json"))
    if config_paths:
        lines.append("## Config Files")
        for path in config_paths:
            lines.append(f"- `{path.name}`")
        lines.append("")

    rewrite_config_path = run_dir / "run_config_rewrite.json"
    if rewrite_config_path.exists():
        rewrite_config = json.loads(rewrite_config_path.read_text(encoding="utf-8"))
        lines.append("## Rewrite Knobs")
        lines.append(f"- Conditions: `{', '.join(rewrite_config.get('conditions', []))}`")
        lines.append(f"- Sample count: `{rewrite_config.get('sample_count')}`")
        lines.append(f"- Seed base: `{rewrite_config.get('seed_base')}`")
        lines.append(f"- Max new tokens: `{rewrite_config.get('rewrite_max_new_tokens')}`")
        lines.append(f"- Temperature: `{rewrite_config.get('rewrite_temperature')}`")
        lines.append(f"- Top-p: `{rewrite_config.get('rewrite_top_p')}`")
        lines.append(f"- Use system prompt: `{rewrite_config.get('rewrite_use_system_prompt', True)}`")
        lines.append("")
        lines.append("### Rewrite System Prompt")
        lines.append("```text")
        if rewrite_config.get("rewrite_use_system_prompt", True):
            lines.append(rewrite_config.get("rewrite_system_prompt", ""))
        else:
            lines.append("(disabled)")
        lines.append("```")
        lines.append("")
        lines.append("### Rewrite User Template")
        lines.append("```text")
        lines.append(rewrite_config.get("rewrite_user_template", ""))
        lines.append("```")
        lines.append("")

    generate_config_path = run_dir / "run_config_generate.json"
    if generate_config_path.exists():
        generate_config = json.loads(generate_config_path.read_text(encoding="utf-8"))
        lines.append("## Generator Knobs")
        for key in (
            "model_path",
            "height",
            "width",
            "steps",
            "guidance",
            "negative_prompt",
            "attention_backend",
            "num_gpus",
            "cpu_offload",
        ):
            lines.append(f"- {key}: `{generate_config.get(key)}`")
        lines.append("")

    if summary.get("conditions"):
        lines.append("## Metric Summary")
        lines.append("| Condition | Metric | Mean | Count | Delta vs none |")
        lines.append("| --- | --- | ---: | ---: | ---: |")
        for condition, metrics in sorted(summary["conditions"].items()):
            for metric, stat in sorted(metrics.items()):
                mean_value = stat.get("mean")
                delta = stat.get("delta_vs_none")
                delta_cell = f"{delta:.6f}" if delta is not None else ""
                lines.append(
                    "| "
                    f"{escape_md_cell(condition)} | "
                    f"{escape_md_cell(metric)} | "
                    f"{mean_value:.6f} | "
                    f"{escape_md_cell(stat.get('count', ''))} | "
                    f"{delta_cell} |"
                )
        lines.append("")

    if summary.get("timing"):
        lines.append("## Timing Summary")
        lines.append("| Stage | Condition | Count | Total sec | Mean sec |")
        lines.append("| --- | --- | ---: | ---: | ---: |")
        for stage, by_condition in sorted(summary["timing"].items()):
            for condition, stat in sorted(by_condition.items()):
                lines.append(
                    "| "
                    f"{escape_md_cell(stage)} | "
                    f"{escape_md_cell(condition)} | "
                    f"{escape_md_cell(stat.get('count', ''))} | "
                    f"{float(stat.get('sum', 0.0)):.3f} | "
                    f"{float(stat.get('mean', 0.0)):.3f} |"
                )
        lines.append("")

    artifacts_by_key: dict[tuple[str, int, int], dict[str, Any]] = {}
    artifact_path = run_dir / "artifacts.jsonl"
    if artifact_path.exists():
        for row in read_jsonl(artifact_path):
            artifacts_by_key[(row["condition"], int(row["prompt_index"]), int(row["sample_index"]))] = row

    ledger_path = run_dir / "prompt_ledger.jsonl"
    if ledger_path.exists():
        lines.append("## Prompt And Artifact Ledger")
        lines.append("| Condition | Prompt | Sample | Seed | Input Prompt | Output Prompt | Generated Artifact |")
        lines.append("| --- | ---: | ---: | ---: | --- | --- | --- |")
        for row in read_jsonl(ledger_path):
            key = (row["condition"], int(row["prompt_index"]), int(row["sample_index"]))
            artifact = artifacts_by_key.get(key, {})
            lines.append(
                "| "
                f"{escape_md_cell(row['condition'])} | "
                f"{row['prompt_index']} | "
                f"{row['sample_index']} | "
                f"{row['seed']} | "
                f"{escape_md_cell(row['original_prompt'])} | "
                f"{escape_md_cell(row['final_prompt'])} | "
                f"`{escape_md_cell(rel_to_run(run_dir, artifact.get('image_path')) or 'pending')}` |"
            )
        lines.append("")

    raw_files = [
        "prompt_ledger.jsonl",
        "artifacts.jsonl",
        "metrics_preference.jsonl",
        "geneval_status.json",
        "geneval/server_results.jsonl",
        "summary.json",
    ]
    existing = [name for name in raw_files if (run_dir / name).exists()]
    if existing:
        lines.append("## Raw Review Files")
        for name in existing:
            lines.append(f"- `{name}`")
        lines.append("")

    (run_dir / "run_review.md").write_text("\n".join(lines), encoding="utf-8")


def cmd_eval_geneval_server(args: argparse.Namespace, run_dir: Path, conditions: list[str]) -> None:
    import pickle
    from io import BytesIO

    import requests
    from PIL import Image

    artifacts = read_jsonl(Path(args.artifacts) if args.artifacts else run_dir / "artifacts.jsonl")
    wanted = set(conditions)
    out_dir = run_dir / "geneval"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "server_results.jsonl"
    if args.overwrite and rows_path.exists():
        rows_path.unlink()

    grouped_artifacts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for artifact in artifacts:
        if artifact["condition"] in wanted:
            grouped_artifacts[artifact["condition"]].append(artifact)

    status: dict[str, Any] = {
        "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "status": "ok",
        "mode": "reward_server",
        "server_url": args.server_url,
        "server_batch_size": args.server_batch_size,
        "server_pad_multiple": args.server_pad_multiple,
        "conditions": conditions,
        "results": {},
        "note": (
            "Server mode posts images and GenEval metadata to reward-server. "
            "group_* means ignore the server's -10 placeholders for groups that do not apply to an image."
        ),
    }
    for condition, rows in sorted(grouped_artifacts.items()):
        start = time.perf_counter()
        all_scores: list[float] = []
        all_rewards: list[float] = []
        all_strict_rewards: list[float] = []
        all_group_rewards: dict[str, list[float]] = defaultdict(list)
        all_group_strict_rewards: dict[str, list[float]] = defaultdict(list)
        for chunk_start in range(0, len(rows), args.server_batch_size):
            chunk_rows = rows[chunk_start:chunk_start + args.server_batch_size]
            image_bytes = []
            metadatas = []
            for row in chunk_rows:
                image = Image.open(row["image_path"]).convert("RGB")
                buffer = BytesIO()
                image.save(buffer, format="JPEG")
                image_bytes.append(buffer.getvalue())
                metadatas.append(row["metadata"])
            real_count = len(chunk_rows)
            pad_count = 0
            if args.server_pad_multiple and real_count % args.server_pad_multiple:
                pad_count = args.server_pad_multiple - (real_count % args.server_pad_multiple)
                image_bytes.extend([image_bytes[-1]] * pad_count)
                metadatas.extend([metadatas[-1]] * pad_count)
            payload = {
                "images": image_bytes,
                "meta_datas": metadatas,
                "only_strict": args.only_strict,
            }
            print(
                f"[eval-geneval] {condition} chunk={chunk_start // args.server_batch_size} "
                f"real={real_count} padded={pad_count}"
            )
            response = requests.post(args.server_url, data=pickle.dumps(payload), timeout=args.timeout)
            response.raise_for_status()
            data = pickle.loads(response.content)
            scores = [float(v) for v in data["scores"][:real_count]]
            rewards = [float(v) for v in data["rewards"][:real_count]]
            strict_rewards = [float(v) for v in data["strict_rewards"][:real_count]]
            all_scores.extend(scores)
            all_rewards.extend(rewards)
            all_strict_rewards.extend(strict_rewards)
            for key, values in data["group_rewards"].items():
                all_group_rewards[key].extend(float(v) for v in values[:real_count])
            for key, values in data["group_strict_rewards"].items():
                all_group_strict_rewards[key].extend(float(v) for v in values[:real_count])
            for idx, row in enumerate(chunk_rows):
                append_jsonl(
                    rows_path,
                    {
                        "condition": condition,
                        "prompt_index": row["prompt_index"],
                        "sample_index": row["sample_index"],
                        "image_path": row["image_path"],
                        "score": scores[idx],
                        "reward": rewards[idx],
                        "strict_reward": strict_rewards[idx],
                        "metadata": row["metadata"],
                    },
                )
        group_rewards = {
            key: mean_non_placeholder(values)
            for key, values in all_group_rewards.items()
        }
        group_strict_rewards = {
            key: mean_non_placeholder(values)
            for key, values in all_group_strict_rewards.items()
        }
        group_reward_vals = [v for v in group_rewards.values() if v is not None]
        group_strict_vals = [v for v in group_strict_rewards.values() if v is not None]
        status["results"][condition] = {
            "count": len(rows),
            "wall_time_sec": time.perf_counter() - start,
            "score_mean": mean(all_scores),
            "reward_mean": mean(all_rewards),
            "strict_reward_mean": mean(all_strict_rewards),
            "group_rewards_mean": group_rewards,
            "group_strict_rewards_mean": group_strict_rewards,
            "overall_from_group_rewards": mean(group_reward_vals),
            "overall_from_group_strict_rewards": mean(group_strict_vals),
        }
    write_json(run_dir / "geneval_status.json", status)
    summarize_run(run_dir)


def summarize_run(run_dir: Path) -> None:
    summary: dict[str, Any] = {"conditions": {}}
    pref_path = run_dir / "metrics_preference.jsonl"
    if pref_path.exists():
        grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for row in read_jsonl(pref_path):
            for metric in ("pickscore", "hps", "ur", "ocr"):
                value = row.get(metric)
                if metric == "ur" and not isinstance(value, int | float) and row.get("ur_raw_output"):
                    value = parse_unifiedreward_score(str(row["ur_raw_output"]))
                if isinstance(value, int | float):
                    grouped[row["condition"]][metric].append(float(value))
        for condition, vals in grouped.items():
            for metric, metric_vals in vals.items():
                add_summary_metric(summary, condition, metric, mean(metric_vals), len(metric_vals))

    geneval_path = run_dir / "geneval_status.json"
    if geneval_path.exists():
        geneval_status = json.loads(geneval_path.read_text(encoding="utf-8"))
        summary["geneval_status"] = {
            "status": geneval_status.get("status"),
            "mode": geneval_status.get("mode", "official"),
            "server_url": geneval_status.get("server_url"),
            "note": geneval_status.get("note"),
            "error": geneval_status.get("error"),
        }
        if geneval_status.get("status") == "ok":
            for condition, result in geneval_status.get("results", {}).items():
                count = result.get("count")
                add_summary_metric(summary, condition, "geneval_score", result.get("score_mean"), count)
                add_summary_metric(summary, condition, "geneval_reward", result.get("reward_mean"), count)
                add_summary_metric(summary, condition, "geneval_strict_reward", result.get("strict_reward_mean"), count)
                add_summary_metric(
                    summary,
                    condition,
                    "geneval_group_reward",
                    result.get("overall_from_group_rewards"),
                    count,
                    {"group_means": result.get("group_rewards_mean")},
                )
                add_summary_metric(
                    summary,
                    condition,
                    "geneval_group_strict_reward",
                    result.get("overall_from_group_strict_rewards"),
                    count,
                    {"group_means": result.get("group_strict_rewards_mean")},
                )

    timing: dict[str, dict[str, dict[str, float | int]]] = {}
    ledger_path = run_dir / "prompt_ledger.jsonl"
    if ledger_path.exists():
        grouped_rewrite: dict[str, list[float | None]] = defaultdict(list)
        for row in read_jsonl(ledger_path):
            grouped_rewrite[row["condition"]].append(row.get("rewrite_wall_time_sec"))
        stage = {
            condition: stat
            for condition, values in grouped_rewrite.items()
            if (stat := timing_stats(values)) is not None
        }
        if stage:
            timing["rewrite"] = stage

    artifact_path = run_dir / "artifacts.jsonl"
    if artifact_path.exists():
        grouped_generate: dict[str, list[float | None]] = defaultdict(list)
        for row in read_jsonl(artifact_path):
            grouped_generate[row["condition"]].append(row.get("generation_wall_time_sec"))
        stage = {
            condition: stat
            for condition, values in grouped_generate.items()
            if (stat := timing_stats(values)) is not None
        }
        if stage:
            timing["generate"] = stage

    if pref_path.exists():
        grouped_pref: dict[str, list[float | None]] = defaultdict(list)
        for row in read_jsonl(pref_path):
            grouped_pref[row["condition"]].append(row.get("eval_wall_time_sec"))
        stage = {
            condition: stat
            for condition, values in grouped_pref.items()
            if (stat := timing_stats(values)) is not None
        }
        if stage:
            timing["eval_preference"] = stage

    if geneval_path.exists():
        geneval_status = json.loads(geneval_path.read_text(encoding="utf-8"))
        stage = {
            condition: stat
            for condition, result in geneval_status.get("results", {}).items()
            if (stat := timing_stats([result.get("wall_time_sec")])) is not None
        }
        if stage:
            timing["eval_geneval"] = stage
    if timing:
        summary["timing"] = timing

    baseline = summary["conditions"].get("none", {})
    for condition, vals in summary["conditions"].items():
        if condition == "none":
            continue
        for metric, stat in list(vals.items()):
            base_mean = baseline.get(metric, {}).get("mean") if isinstance(baseline.get(metric), dict) else None
            if base_mean is not None and stat.get("mean") is not None:
                stat["delta_vs_none"] = stat["mean"] - base_mean
    write_json(run_dir / "summary.json", summary)
    lines = ["# Image Calibration Summary", ""]
    for condition, vals in sorted(summary["conditions"].items()):
        lines.append(f"## {condition}")
        for metric, stat in sorted(vals.items()):
            delta = stat.get("delta_vs_none")
            delta_s = f", delta_vs_none={delta:.4f}" if delta is not None else ""
            count_s = stat.get("count", "")
            lines.append(f"- {metric}: mean={stat['mean']:.4f}, n={count_s}{delta_s}")
        lines.append("")
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    write_review_report(run_dir, summary)


def cmd_summarize(args: argparse.Namespace) -> None:
    summarize_run(Path(args.run_dir))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prompt rewrite vs no-rewrite image calibration pipeline.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("select-metadata")
    p.add_argument("--metadata", default=str(DEFAULT_METADATA))
    p.add_argument("--output", required=True)
    p.add_argument("--per-tag", type=int, default=2)
    p.add_argument("--tags", nargs="*", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_select_metadata)

    p = sub.add_parser("import-prompts")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--benchmark-name", required=True)
    p.add_argument("--tag", default=None)
    p.add_argument("--skip-comments", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_import_prompts)

    p = sub.add_parser("rewrite")
    p.add_argument("--run-dir", default=None)
    p.add_argument("--metadata", default=str(DEFAULT_METADATA))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--tags", nargs="*", default=None)
    p.add_argument("--conditions", default="none,qwen25_vl_3b")
    p.add_argument("--sample-count", type=int, default=1)
    p.add_argument("--seed-base", type=int, default=42)
    p.add_argument("--system-prompt-file", default=None)
    p.add_argument("--no-system-prompt", action="store_true")
    p.add_argument("--user-template-file", default=None)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_rewrite)

    p = sub.add_parser("generate")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--ledger", default=None)
    p.add_argument("--conditions", default=None)
    p.add_argument("--model-path", default="stabilityai/stable-diffusion-3.5-medium")
    p.add_argument("--generator-backend", choices=["auto", "fastvideo", "diffusers_flux"], default="auto")
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--guidance", type=float, default=6.0)
    p.add_argument("--negative-prompt", default="")
    p.add_argument("--attention-backend", default="FLASH_ATTN")
    p.add_argument("--disable-nvfp4", action="store_true", default=True)
    p.add_argument("--enable-nvfp4", dest="disable_nvfp4", action="store_false")
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument("--cpu-offload", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("eval-preference")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--artifacts", default=None)
    p.add_argument("--metrics", default="pickscore,hps,ur")
    p.add_argument("--metric-prompt", choices=["original", "final"], default="original")
    p.add_argument("--hps-version", default="v2.0", choices=["v2.0", "v2.1"])
    p.add_argument("--hps-checkpoint", default=None)
    p.add_argument("--ur-backend", default="local", choices=["local", "sglang"])
    p.add_argument("--ur-model-id", default="CodeGoat24/UnifiedReward-qwen-7b")
    p.add_argument("--ur-max-new-tokens", type=int, default=256)
    p.add_argument("--ur-api-base", default="http://127.0.0.1:17140/v1")
    p.add_argument("--ur-api-key", default="flowgrpo")
    p.add_argument("--ur-model-name", default="UnifiedReward-7b-v1.5")
    p.add_argument("--ur-request-timeout", type=float, default=120.0)
    p.add_argument("--ocr-use-gpu", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_eval_preference)

    p = sub.add_parser("eval-geneval")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--artifacts", default=None)
    p.add_argument("--conditions", default=None)
    p.add_argument("--model-path", default=str(REPO_ROOT / "models/geneval"))
    p.add_argument("--server-url", default=None)
    p.add_argument("--only-strict", action="store_true")
    p.add_argument("--server-batch-size", type=int, default=64)
    p.add_argument("--server-pad-multiple", type=int, default=64)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_eval_geneval)

    p = sub.add_parser("summarize")
    p.add_argument("--run-dir", required=True)
    p.set_defaults(func=cmd_summarize)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
