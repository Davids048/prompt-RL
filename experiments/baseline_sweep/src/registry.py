#!/usr/bin/env python3
from __future__ import annotations

from prompt_sources import geneval as geneval_prompt_source
from enhancers import hf_vlm, none
from generators import fastvideo
from eval import geneval as geneval_eval


PROMPT_SOURCES = {
    "geneval": geneval_prompt_source.load_prompts,
}

ENHANCERS = {
    "none": none.enhance,
    "hf_vlm": hf_vlm.enhance,
}

GENERATORS = {
    "fastvideo": fastvideo.generate,
    "fastvideo_image": fastvideo.generate,
}

EVALS = {
    "geneval": geneval_eval.run,
}
