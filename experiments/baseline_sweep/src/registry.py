#!/usr/bin/env python3
from __future__ import annotations

from prompt_sources import geneval as geneval_prompt_source
from prompt_sources import txt as txt_prompt_source
from prompt_sources import vbench as vbench_prompt_source
from enhancers import hf_vlm, none
from generators import fastvideo
from eval import geneval as geneval_eval
from eval import hpsv3 as hpsv3_eval
from eval import vbench as vbench_eval
from eval import videoalign as videoalign_eval


PROMPT_SOURCES = {
    "geneval": geneval_prompt_source.load_prompts,
    "txt": txt_prompt_source.load_prompts,
    "vbench": vbench_prompt_source.load_prompts,
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
    "hpsv3": hpsv3_eval.run,
    "vbench": vbench_eval.run,
    "videoalign": videoalign_eval.run,
}
