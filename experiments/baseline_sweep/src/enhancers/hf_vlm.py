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
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model_id = enhancer["name"]
    params = enhancer.get("params", {})
    template = Path(enhancer["template"]).read_text(encoding="utf-8")
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()

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
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=int(params.get("max_new_tokens", 256)),
                        do_sample=do_sample,
                        temperature=temperature if do_sample else None,
                        top_p=float(params.get("top_p", 1.0)) if do_sample else None,
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
