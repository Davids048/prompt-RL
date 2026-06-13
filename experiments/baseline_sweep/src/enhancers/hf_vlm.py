#!/usr/bin/env python3
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any


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


def enhance(
    prompt_rows: list[dict[str, Any]],
    enhancer: dict[str, Any],
    trial: dict[str, Any],
    run_dir: Any,
) -> list[dict[str, Any]]:
    del trial, run_dir
    import torch
    from transformers import AutoProcessor

    model_id = enhancer["name"]
    params = enhancer.get("params", {})
    template = Path(enhancer["template"]).read_text(encoding="utf-8")
    trust_remote_code = bool(params.get("trust_remote_code", False))
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model = load_model(model_id, params, torch).eval()

    rows: list[dict[str, Any]] = []
    try:
        for row in prompt_rows:
            start = time.perf_counter()
            item = dict(row)
            item.update({
                "enhancer_model": model_id,
                "enhancer_alias": enhancer["alias"],
                "enhancer_backend": enhancer["backend"],
                "enhancer_template": enhancer.get("template"),
                "enhancer_params": params,
                "eval_prompt": row["original_prompt"],
            })
            user_prompt = template.format(prompt=row["original_prompt"])
            try:
                messages = [{"role": "user", "content": [{"type": "text", "text": user_prompt}]}]
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text], padding=True, return_tensors="pt").to(model.device)
                temperature = float(params.get("temperature", 0.0))
                do_sample = temperature > 0
                generation_kwargs: dict[str, Any] = {
                    "max_new_tokens": int(params.get("max_new_tokens", 256)),
                    "do_sample": do_sample,
                }
                if do_sample:
                    generation_kwargs["temperature"] = temperature
                    generation_kwargs["top_p"] = float(params.get("top_p", 1.0))
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        **generation_kwargs,
                    )
                generated = generated[:, inputs.input_ids.shape[1]:]
                enhanced_prompt = clean_rewrite(processor.batch_decode(generated, skip_special_tokens=True)[0])
                item.update({
                    "enhanced_prompt": enhanced_prompt,
                    "generation_prompt": enhanced_prompt,
                    "enhancement_error": None,
                    "enhancement_wall_time_sec": time.perf_counter() - start,
                })
            except Exception as exc:  # noqa: BLE001
                item.update({
                    "enhanced_prompt": None,
                    "generation_prompt": None,
                    "enhancement_error": repr(exc),
                    "enhancement_wall_time_sec": time.perf_counter() - start,
                })
            rows.append(item)
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def load_model(model_id: str, params: dict[str, Any], torch: Any) -> Any:
    from transformers import AutoModelForImageTextToText

    kwargs = {
        "torch_dtype": torch_dtype(params, torch),
        "device_map": params.get("device_map", "auto"),
        "trust_remote_code": bool(params.get("trust_remote_code", False)),
    }
    try:
        return AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    except ValueError:
        pass

    fallback_names = [
        "Qwen3VLForConditionalGeneration",
        "Qwen2_5_VLForConditionalGeneration",
        "Qwen2VLForConditionalGeneration",
    ]
    import transformers

    for class_name in fallback_names:
        model_class = getattr(transformers, class_name, None)
        if model_class is None:
            continue
        try:
            return model_class.from_pretrained(model_id, **kwargs)
        except ValueError:
            continue
    raise ValueError(f"No supported HF VLM loader found for enhancer model {model_id!r}.")


def torch_dtype(params: dict[str, Any], torch: Any) -> Any:
    dtype_name = str(params.get("torch_dtype", "bfloat16")).lower()
    if dtype_name in {"auto", "none"}:
        return "auto"
    dtype = getattr(torch, dtype_name, None)
    if dtype is None:
        raise ValueError(f"Unsupported torch_dtype={dtype_name!r}.")
    return dtype
