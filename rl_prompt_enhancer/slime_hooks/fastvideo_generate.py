"""Slime custom_generate hook that turns FastVideo into a reward tool."""

from __future__ import annotations

import os
import re
import json
import hashlib
from typing import Any

from slime.utils.http_utils import post
from slime.utils.types import Sample

from rl_prompt_enhancer.fastvideo_bridge.schema import default_generator

DEFAULT_TEMPLATE = (
    "Please provide an enhanced prompt for the following image generation prompt "
    "to make the image more realistic, detailed, with clear separation and "
    "precise alignment of all entities.\n"
    "Original prompt: {prompt}. Directly provide the improved prompt in "
    "<answer> </answer> tags."
)

OUTPUT_RULES = (
    "Output only the final image prompt inside <answer>...</answer>. "
    "Do not include reasoning, analysis, markdown, or labels."
)


def _service_urls() -> list[str]:
    raw_urls = os.environ.get("RLPE_FASTVIDEO_SERVICE_URLS")
    if raw_urls:
        urls = [url.strip().rstrip("/") for url in raw_urls.split(",") if url.strip()]
        if urls:
            return urls
    base = os.environ.get("RLPE_FASTVIDEO_SERVICE_URL", "http://127.0.0.1:18080")
    return [base.rstrip()]


def _service_url(payload: dict[str, Any]) -> str:
    # Route independent image requests across FastVideo data-parallel workers.
    urls = _service_urls()
    if len(urls) == 1:
        return urls[0] + "/generate_and_score"
    key = str(payload.get("request_id") or payload.get("comparison_group_id") or "")
    digest = hashlib.blake2s(key.encode("utf-8"), digest_size=4).digest()
    worker_index = int.from_bytes(digest, "big") % len(urls)
    return urls[worker_index] + "/generate_and_score"


def _template() -> str:
    path = os.environ.get("RLPE_PROMPT_TEMPLATE_PATH")
    if path:
        with open(path, encoding="utf-8") as handle:
            return handle.read().strip()
    return DEFAULT_TEMPLATE


def _generator_config() -> dict[str, Any]:
    # YAML launches pass the rendered generator JSON so config.yaml stays authoritative.
    raw = os.environ.get("RLPE_GENERATOR_CONFIG_JSON")
    if raw:
        generator = json.loads(raw)
        if not isinstance(generator, dict):
            raise RuntimeError("RLPE_GENERATOR_CONFIG_JSON must be a JSON object")
        return generator

    # Backward-compatible fallback for older rendered runs.
    path = os.environ.get("RLPE_GENERATOR_CONFIG_PATH")
    if not path:
        return default_generator()
    with open(path, encoding="utf-8") as handle:
        generator = json.load(handle)
    if not isinstance(generator, dict):
        raise RuntimeError(f"generator config must be a JSON object: {path}")
    return generator


def _original_prompt(sample: Sample) -> str:
    # Prefer metadata because Slime may wrap sample.prompt with a chat template.
    metadata = sample.metadata or {}
    if metadata.get("original_prompt"):
        return str(metadata["original_prompt"])
    if metadata.get("prompt"):
        return str(metadata["prompt"])
    if isinstance(sample.prompt, str):
        return sample.prompt
    return str(sample.prompt)


def _strip_thinking(response: str) -> str:
    without_closed_blocks = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL | re.IGNORECASE)
    return re.sub(r"<think>.*", "", without_closed_blocks, flags=re.DOTALL | re.IGNORECASE).strip()


def _strip_tags(response: str) -> str:
    without_thinking = _strip_thinking(response)
    return re.sub(r"</?answer>", "", without_thinking, flags=re.IGNORECASE).strip()


def _clean_prompt_text(response: str) -> str:
    stripped = _strip_tags(response)
    return re.sub(r"^(answer|enhanced prompt|improved prompt)\s*:\s*", "", stripped, flags=re.IGNORECASE).strip()


def _parse_answer(response: str, original_prompt: str) -> tuple[str, str]:
    # Invalid prompt-enhancer formatting should be visible in metadata, not abort rollout.
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", response, flags=re.DOTALL)
    if match:
        answer = _clean_prompt_text(match.group(1))
        if answer:
            return answer, "answer_tags"
        recovered = _clean_prompt_text(response)
        if recovered:
            return recovered, "empty_answer_recovered_from_text"
        return original_prompt, "empty_answer_used_original_prompt"

    recovered = _clean_prompt_text(response)
    if recovered:
        return recovered, "missing_answer_tags_recovered_from_text"
    return original_prompt, "missing_answer_tags_used_original_prompt"


def _stable_index(value: Any, fallback: int) -> int:
    if value is None:
        return fallback
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _request_ids(sample: Sample, evaluation: bool = False) -> tuple[str, str, int]:
    """Build stable artifact IDs while keeping eval artifacts out of train paths."""
    group_index = _stable_index(sample.group_index, _stable_index(sample.index, 0))
    sample_index = _stable_index(sample.index, group_index)
    rollout_index = _stable_index(sample.rollout_id, sample_index)
    phase_prefix = "eval_" if evaluation else ""
    comparison_group_id = f"{phase_prefix}prompt{group_index:06d}"
    request_id = f"{comparison_group_id}_rollout{rollout_index:06d}_sample{sample_index:06d}"
    seed_base = int(os.environ.get("RLPE_SEED_BASE", "430000"))
    return request_id, comparison_group_id, seed_base + group_index


def _generation_payload(
    sample: Sample,
    original_prompt: str,
    enhanced_prompt: str,
    evaluation: bool = False,
) -> dict[str, Any]:
    """Create the FastVideo request payload from one Slime rollout sample."""
    request_id, comparison_group_id, seed = _request_ids(sample, evaluation=evaluation)
    return {
        "request_id": request_id,
        "original_prompt": original_prompt,
        "enhanced_prompt": enhanced_prompt,
        "artifact_kind": "image",
        "comparison_group_id": comparison_group_id,
        "seed": seed,
        "generator": _generator_config(),
    }


async def generate(
    args,
    sample: Sample,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Sample:
    """Generate an enhanced prompt, score its FastVideo image, and attach rewards."""
    from slime.rollout.sglang_rollout import GenerateState

    # Build the prompt-enhancer request from the original image prompt.
    state = GenerateState(args)
    original_prompt = _original_prompt(sample)
    sample.metadata = sample.metadata or {}
    sample.metadata["original_prompt"] = original_prompt
    sample.metadata["rlpe_phase"] = "eval" if evaluation else "train"

    prompt_text = f"{_template().format(prompt=original_prompt)}\n{OUTPUT_RULES}"
    prompt_ids = state.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    sglang_sampling_params = dict(sampling_params)
    stop_values = sglang_sampling_params.get("stop") or []
    if isinstance(stop_values, str):
        stop_values = [stop_values]
    sglang_sampling_params["stop"] = list(dict.fromkeys([*stop_values, "</answer>"]))
    sglang_sampling_params["no_stop_trim"] = True

    # Ask the current policy model for an enhanced prompt and keep token logprobs.
    payload = {
        "text": prompt_text,
        "sampling_params": sglang_sampling_params,
        "return_logprob": True,
    }
    sglang_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    output = await post(sglang_url, payload)

    meta_info = output["meta_info"]
    if meta_info["finish_reason"]["type"] == "abort":
        sample.status = Sample.Status.ABORTED
        return sample
    if "output_token_logprobs" not in meta_info:
        raise RuntimeError("SGLang output did not include output_token_logprobs")

    output_token_logprobs = meta_info["output_token_logprobs"]
    response_token_ids = [item[1] for item in output_token_logprobs]
    response_log_probs = [item[0] for item in output_token_logprobs]
    if len(response_token_ids) != len(response_log_probs):
        raise RuntimeError(
            "SGLang response token/logprob length mismatch: "
            f"{len(response_token_ids)} tokens vs {len(response_log_probs)} logprobs"
        )
    response_text = output["text"]
    enhanced_prompt, parse_status = _parse_answer(response_text, original_prompt)

    # Populate the Slime sample fields that downstream GRPO training expects.
    sample.prompt = prompt_text
    sample.tokens = prompt_ids + response_token_ids
    sample.response = response_text
    sample.response_length = len(response_token_ids)
    sample.loss_mask = [1] * len(response_token_ids)
    sample.rollout_log_probs = response_log_probs
    sample.metadata["prompt_enhancer"] = {
        "original_prompt": original_prompt,
        "raw_response": response_text,
        "enhanced_prompt": enhanced_prompt,
        "parse_status": parse_status,
    }
    sample.update_from_meta_info(args, meta_info)

    # FastVideo returns artifact metadata plus the reward dict consumed by Slime.
    service_payload = _generation_payload(sample, original_prompt, enhanced_prompt, evaluation=evaluation)
    sample.metadata["fastvideo_request"] = service_payload
    service_url = _service_url(service_payload)
    sample.metadata["fastvideo_service_url"] = service_url
    service_response = await post(service_url, service_payload)
    if service_response.get("status") != "completed":
        raise RuntimeError(f"FastVideo service did not complete: {service_response}")

    sample.metadata["fastvideo_generation"] = service_response
    sample.metadata["fastvideo_reward"] = service_response["rewards"]
    sample.reward = service_response["rewards"]
    return sample
