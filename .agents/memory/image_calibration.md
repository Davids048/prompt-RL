# Image Calibration Memory

Last updated: 2026-05-28

Goal: establish a PromptRL-style image calibration sanity check before moving to video. The pilot compares images generated from original prompts against images generated from rewritten prompts, using the same original prompt as the evaluator prompt.

Pipeline:

```text
+---------------------------------------------+
| GenEval metadata prompts                    |
+---------------------------------------------+
                      |
                      v
+---------------------------------------------+
| rewrite ledger: none / rewrite condition    |
+---------------------------------------------+
                      |
                      v
+---------------------------------------------+
| SD3.5 Medium generation through FastVideo   |
| integration                                 |
+---------------------------------------------+
                      |
                      v
+---------------------------------------------+
| preference metrics: PickScore, HPS,         |
| UnifiedReward                               |
+---------------------------------------------+
                      |
                      v
+---------------------------------------------+
| GenEval reward-server scoring               |
+---------------------------------------------+
                      |
                      v
+---------------------------------------------+
| summaries and HTML review reports           |
+---------------------------------------------+
```

Prompt rewrite system prompt used in the pilot:

```text
You are a prompt rewriting assistant for text-to-image generation. Rewrite the user's prompt into one semantically equivalent prompt. Preserve every object, count, color, spatial relation, visible text string, and other required attribute exactly. You may add concise visual detail, composition, lighting, camera, or style cues only when they do not change the requested content. Do not add new objects, remove objects, change counts, change colors, or change text. Return only the rewritten prompt.
```

User prompt template:

```text
Original prompt:
{prompt}

Return a single enhanced prompt that preserves the same semantic requirements.
```

Pilot data:

```text
Metadata: experiments/image_calibration/geneval_balanced_12.jsonl
Prompt count: 12
Sampling: 2 prompts from each GenEval group
Groups: single_object, two_object, counting, colors, position, color_attr
Samples per prompt: 1
Generator: stabilityai/stable-diffusion-3.5-medium
Resolution: 1024 x 1024
Steps: 20
Guidance: 6.0
Attention backend: FLASH_ATTN
```

Run 1: no rewrite versus Qwen2.5-VL-3B-Instruct.

```text
Run dir: outputs/image_calibration/pilot_sd35_balanced12
Summary: outputs/image_calibration/pilot_sd35_balanced12/summary.md
Review: outputs/image_calibration/pilot_sd35_balanced12/run_review.md
HTML: outputs/image_calibration/pilot_sd35_balanced12/qwen25_vl_3b_prompt_comparison.html
```

Aggregate results:

```text
+----------------+---------------+----------------+-----------+---------+---------------+
| Condition      | GenEval score | GenEval reward | PickScore | HPS     | UnifiedReward |
+----------------+---------------+----------------+-----------+---------+---------------+
| none           | 0.5069        | 0.5000         | 21.9514   | 27.7865 | 4.1133        |
+----------------+---------------+----------------+-----------+---------+---------------+
| qwen25_vl_3b   | 0.5556        | 0.5000         | 21.7025   | 27.5755 | 3.8350        |
+----------------+---------------+----------------+-----------+---------+---------------+
```

Run 2: no rewrite versus Qwen2.5-3B-Instruct.

```text
Run dir: outputs/image_calibration/pilot_sd35_balanced12_qwen25_3b
Summary: outputs/image_calibration/pilot_sd35_balanced12_qwen25_3b/summary.md
Review: outputs/image_calibration/pilot_sd35_balanced12_qwen25_3b/run_review.md
HTML: outputs/image_calibration/pilot_sd35_balanced12_qwen25_3b/qwen25_3b_prompt_comparison.html
```

Aggregate results:

```text
+----------------+---------------+----------------+-----------+---------+---------------+
| Condition      | GenEval score | GenEval reward | PickScore | HPS     | UnifiedReward |
+----------------+---------------+----------------+-----------+---------+---------------+
| none           | 0.5069        | 0.5000         | 21.9514   | 27.7865 | 4.1133        |
+----------------+---------------+----------------+-----------+---------+---------------+
| qwen25_3b      | 0.4861        | 0.3333         | 21.3306   | 27.0612 | 3.7233        |
+----------------+---------------+----------------+-----------+---------+---------------+
```

Caveats:

The 12-prompt set is only a pipeline sanity check. It is not a faithful PromptRL reproduction. A proper GenEval run should use `evaluations/geneval/prompts/evaluation_metadata.jsonl`, which has 553 prompts.

UnifiedReward was attempted on all generated images, but the local prompt caused many malformed outputs without a parseable `Final Score:`. The reported UnifiedReward means are based on the parseable subset only. Raw outputs are recorded in each run's `metrics_preference.jsonl`.

The rewrite did change every rewritten-condition prompt. Some changes were semantically risky, for example adding `sports`, `vintage`, `wooden`, `red`, `brown`, or extra scene context. The HTML reports are the fastest way to inspect row-level before/after behavior.

## 2026-05-28 PromptRL Prompt-Enhancement Finding

PromptRL's paper PDF does not expose a reusable natural-language system prompt. The released `G-U-N/UniRL` `prompt_rl` branch uses a GenEval prompt-refinement user template instead: ask Qwen2.5-VL to provide an enhanced image-generation prompt that makes the image more realistic and detailed, with clear separation and precise alignment of all entities, then return the improved prompt in `<answer> </answer>` tags. This is materially less conservative than the local pilot prompt, which asked for strict semantic equivalence and exact preservation of objects, counts, colors, spatial relations, and text.

For reruns intended to match PromptRL-style prompt enhancement, use a PromptRL-style user template and output parser, not the earlier conservative system prompt alone. Keep the exact rewrite prompt, before/after prompts, generation artifacts, metric scores, and per-stage timing in the run directory.

## 2026-05-28 Full PromptRL-Style SD3.5 Medium Run

Run root: `outputs/image_calibration/promptrl_full_sd35_20260528_0439`.

This run replaced the conservative pilot rewrite instruction with the PromptRL-style user-only template from the released code. "PromptRL-style rewrite" means each original benchmark prompt is substituted into the template below, sent to `Qwen/Qwen2.5-VL-3B-Instruct` with no system prompt, and the text inside `<answer>...</answer>` is used as the rewritten image-generation prompt. The full run did not include the text-only `Qwen/Qwen2.5-3B-Instruct` condition.

Exact rewrite prompt sent to the LM:

```text
Please provide an enhanced prompt for the following image generation prompt to make the image more realistic, detailed, with clear separation and precise alignment of all entities.
Original prompt: {prompt}. Directly provide the improved prompt in <answer> </answer> tags.
```

Where to find this configuration:

```text
Rewrite prompt file: experiments/image_calibration/prompts/promptrl_geneval_enhance_user.txt
Run manifest: outputs/image_calibration/promptrl_full_sd35_20260528_0439/run_manifest.json
Runner: experiments/image_calibration/run_promptrl_full_benchmarks.sh
Pipeline: experiments/image_calibration/calibration_pipeline.py
```

Rewrite configuration:

```text
Rewriter: Qwen/Qwen2.5-VL-3B-Instruct
System prompt: none; runner passes --no-system-prompt and manifest records prompt_enhancer_system_prompt: null
Temperature: 0.7
Top-p: 0.9
Max new tokens: 256
Output parser: text inside <answer>...</answer>, implemented in calibration_pipeline.py
```

NOTE: Generator configuration provenance:

The manifest values `height=1024`, `width=1024`, `steps=20`, and `guidance=6.0` are experiment-local settings for this run, not verified official SD3.5 Medium defaults. They were set directly in `experiments/image_calibration/run_promptrl_full_benchmarks.sh` and mirrored as CLI defaults in `experiments/image_calibration/calibration_pipeline.py generate`; each benchmark subrun also wrote the resolved values to its own `run_config_generate.json`.

```text
Manifest: outputs/image_calibration/promptrl_full_sd35_20260528_0439/run_manifest.json
Runner defaults: experiments/image_calibration/run_promptrl_full_benchmarks.sh
Pipeline generate defaults: experiments/image_calibration/calibration_pipeline.py
Resolved GenEval config: outputs/image_calibration/promptrl_full_sd35_20260528_0439/geneval_full/run_config_generate.json
Resolved OCR config: outputs/image_calibration/promptrl_full_sd35_20260528_0439/ocr1k/run_config_generate.json
Resolved PickScore-SFW config: outputs/image_calibration/promptrl_full_sd35_20260528_0439/pickscore_sfw/run_config_generate.json
```

The reason for these values was pragmatic: keep the full run at a square 1-megapixel-ish evaluation resolution while limiting runtime. The 20-step and 6.0-guidance settings were inherited from the earlier pilot/smoke runs. They should not be cited as upstream defaults. 
Local source comparison of defaults:
```text
+----------------------------------------------+------------------------------------------------------------------------------------------------------------------------------+--------------------------------+-------+----------------+--------------------------------------------------------------------+
| Source                                       | Location                                                                                                                     | Resolution                     | Steps | Guidance / CFG | Notes                                                              |
+----------------------------------------------+------------------------------------------------------------------------------------------------------------------------------+--------------------------------+-------+----------------+--------------------------------------------------------------------+
| This full PromptRL-style run                 | `outputs/image_calibration/promptrl_full_sd35_20260528_0439/run_manifest.json`                                               | 1024x1024                      | 20    | 6.0            | Actual run setting selected by our wrapper/pipeline.               |
|                                              | per-task `run_config_generate.json` files                                                                                    |                                |       |                |                                                                    |
+----------------------------------------------+------------------------------------------------------------------------------------------------------------------------------+--------------------------------+-------+----------------+--------------------------------------------------------------------+
| FastVideo `sd35_medium` preset               | `FastVideo/fastvideo/pipelines/basic/sd35/presets.py`                                                                        | 512x512                        | 28    | 6.0            | FastVideo's registered SD3.5 Medium preset. Guidance matches our   |
|                                              |                                                                                                                              |                                |       |                | run; resolution and steps do not.                                  |
+----------------------------------------------+------------------------------------------------------------------------------------------------------------------------------+--------------------------------+-------+----------------+--------------------------------------------------------------------+
| FastVideo generic sampling default           | `FastVideo/fastvideo/api/sampling_param.py`                                                                                  | 1280x720                       | 50    | 1.0            | Generic sampling default, not SD3.5-specific.                      |
+----------------------------------------------+------------------------------------------------------------------------------------------------------------------------------+--------------------------------+-------+----------------+--------------------------------------------------------------------+
| Installed Diffusers SD3 pipeline default     | `.venv/lib/python3.12/site-packages/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py`                   | pipeline/model-derived if      | 28    | 7.0            | Code default for `StableDiffusion3Pipeline.__call__`, not the      |
|                                              |                                                                                                                              | unset                          |       |                | FastVideo path.                                                    |
+----------------------------------------------+------------------------------------------------------------------------------------------------------------------------------+--------------------------------+-------+----------------+--------------------------------------------------------------------+
| Cached SD3.5 Medium README example           | `.cache/huggingface/hub/models--stabilityai--stable-diffusion-3.5-medium/snapshots/.../README.md`                            | not specified in example       | 40    | 4.5            | Official model-card example, not a default.                        |
+----------------------------------------------+------------------------------------------------------------------------------------------------------------------------------+--------------------------------+-------+----------------+--------------------------------------------------------------------+
| Cached SD3.5 Medium ComfyUI workflow         | `.cache/huggingface/hub/models--stabilityai--stable-diffusion-3.5-medium/snapshots/.../SD3.5M_example_workflow.json`         | 1280x768                       | 40    | 5.5            | Example ComfyUI workflow, not a default.                           |
+----------------------------------------------+------------------------------------------------------------------------------------------------------------------------------+--------------------------------+-------+----------------+--------------------------------------------------------------------+
```

--- 

Pipeline:
```text
                         +------------------+
                         | Original prompts |
                         +------------------+
                                  |
                 +----------------+----------------+
                 |                                 |
                 v                                 v
   +-----------------------------------+   +-----------------------------------------------+
   | none condition: prompt unchanged  |   | qwen25_vl_3b condition: PromptRL-style        |
   |                                   |   | rewrite                                       |
   +-----------------------------------+   +-----------------------------------------------+
                 |                                 |
                 +----------------+----------------+
                                  |
                                  v
                         +-----------------------+
                         | SD3.5 Medium          |
                         | generation            |
                         +-----------------------+
                                  |
                                  v
        +-------------------------------------------------------+
        | GenEval / OCR / PickScore / HPS / UnifiedReward       |
        | scoring                                               |
        +-------------------------------------------------------+
                                  |
                                  v
        +-------------------------------------------------------+
        | summaries and row-level HTML comparison               |
        +-------------------------------------------------------+
```

Benchmark coverage:

```text
GenEval full: 553 prompts, 1106 generated images across none and qwen25_vl_3b
OCR1k: 1018 prompts, 2036 generated images across none and qwen25_vl_3b
PickScore-SFW: 1024 prompts, 2048 generated images across none and qwen25_vl_3b
```

Experiment configuration by task:

```text
+------------------+----------------------------------------------------------------------------------------------------------+----------------------------------------------+----------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------+
| Task             | Prompts used and location                                                                                | Count                                        | Rewrite prompt used                                                              | Evaluation used and harness                                                                                      |
+------------------+----------------------------------------------------------------------------------------------------------+----------------------------------------------+----------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------+
| GenEval full     | Original/eval prompts from `evaluations/geneval/prompts/evaluation_metadata.jsonl`;                      | 553 prompts, 1106 images for none +          | PromptRL template in                                                             | GenEval reward-server metrics: score, reward, strict reward, group reward, strict group reward.                  |
|                  | run ledger at                                                                                            | qwen25_vl_3b                                 | `experiments/image_calibration/prompts/promptrl_geneval_enhance_user.txt`;       | Harness: `experiments/image_calibration/calibration_pipeline.py eval-geneval` calling                            |
|                  | `outputs/image_calibration/promptrl_full_sd35_20260528_0439/geneval_full/prompt_ledger.jsonl`            |                                              | no system prompt                                                                 | `evaluations/reward-server/app_geneval.py` via `.envs/reward_server/bin/gunicorn`.                               |
|                  |                                                                                                          |                                              |                                                                                  | Results: `geneval_full/geneval/server_results.jsonl` and `geneval_full/summary.md`.                              |
+------------------+----------------------------------------------------------------------------------------------------------+----------------------------------------------+----------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------+
| OCR1k            | Raw prompts from `evaluations/flow_grpo/dataset/ocr/test.txt`,                                           | 1018 imported prompts, 2036 images for none  | Same PromptRL template;                                                          | OCR score against the original prompt using                                                                      |
|                  | imported to `outputs/image_calibration/promptrl_full_sd35_20260528_0439/metadata/ocr1k.jsonl`;           | + qwen25_vl_3b                               | original OCR prompt substituted into `{prompt}`;                                 | `experiments/image_calibration/calibration_pipeline.py eval-preference --metrics ocr --metric-prompt original`.  |
|                  | run ledger at `ocr1k/prompt_ledger.jsonl`                                                                |                                              | no system prompt                                                                 | Harness class: `OCRBenchmarkScorer` in `calibration_pipeline.py`, backed by PaddleOCR.                           |
|                  |                                                                                                          |                                              |                                                                                  | Results: `ocr1k/metrics_preference.jsonl` and `ocr1k/summary.md`.                                                |
+------------------+----------------------------------------------------------------------------------------------------------+----------------------------------------------+----------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------+
| PickScore-SFW    | Raw prompts from `evaluations/flow_grpo/dataset/pickscore_sfw/test.txt`,                                 | 1024 prompts, 2048 images for none +         | Same PromptRL template;                                                          | Preference metrics scored against the original prompt: PickScore, HPS v2.1, UnifiedReward.                       |
|                  | imported to `outputs/image_calibration/promptrl_full_sd35_20260528_0439/metadata/pickscore_sfw.jsonl`;   | qwen25_vl_3b                                 | original PickScore-SFW prompt substituted into `{prompt}`;                       | Harness: `calibration_pipeline.py eval-preference`; `PickScoreScorer` loads `yuvalkirstain/PickScore_v1`,        |
|                  | run ledger at `pickscore_sfw/prompt_ledger.jsonl`                                                        |                                              | no system prompt                                                                 | `HPSScorer` uses `evaluations/HPSv2`, and `UnifiedRewardSGLangScorer` calls an sglang server launched from       |
|                  |                                                                                                          |                                              |                                                                                  | `.envs/sglang/bin/python -m sglang.launch_server` with `CodeGoat24/UnifiedReward-7b-v1.5`.                       |
|                  |                                                                                                          |                                              |                                                                                  | Results: `pickscore_sfw/metrics_preference.jsonl` and `pickscore_sfw/summary.md`.                                |
+------------------+----------------------------------------------------------------------------------------------------------+----------------------------------------------+----------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------+
```

Primary summaries:

```text
GenEval: outputs/image_calibration/promptrl_full_sd35_20260528_0439/geneval_full/summary.md
OCR1k: outputs/image_calibration/promptrl_full_sd35_20260528_0439/ocr1k/summary.md
PickScore-SFW: outputs/image_calibration/promptrl_full_sd35_20260528_0439/pickscore_sfw/summary.md
HTML comparison: outputs/image_calibration/promptrl_full_sd35_20260528_0439/comparison_table.html
Manifest: outputs/image_calibration/promptrl_full_sd35_20260528_0439/run_manifest.json
```

Aggregate results:

```text
+-------------------+---------------------------+----------+----------------+----------+
| Benchmark         | Metric                    | none     | qwen25_vl_3b   | Delta    |
+-------------------+---------------------------+----------+----------------+----------+
| GenEval full      | group reward              | 0.4614   | 0.6103         | +0.1489  |
+-------------------+---------------------------+----------+----------------+----------+
| GenEval full      | strict group reward       | 0.3484   | 0.4931         | +0.1447  |
+-------------------+---------------------------+----------+----------------+----------+
| GenEval full      | reward                    | 0.4467   | 0.6004         | +0.1537  |
+-------------------+---------------------------+----------+----------------+----------+
| GenEval full      | score                     | 0.4625   | 0.5980         | +0.1355  |
+-------------------+---------------------------+----------+----------------+----------+
| GenEval full      | strict reward             | 0.3327   | 0.4828         | +0.1501  |
+-------------------+---------------------------+----------+----------------+----------+
| OCR1k             | OCR                       | 0.4447   | 0.5153         | +0.0706  |
+-------------------+---------------------------+----------+----------------+----------+
| PickScore-SFW     | PickScore                 | 21.4588  | 21.5690        | +0.1101  |
+-------------------+---------------------------+----------+----------------+----------+
| PickScore-SFW     | HPS                       | 29.1591  | 29.7003        | +0.5412  |
+-------------------+---------------------------+----------+----------------+----------+
| PickScore-SFW     | UnifiedReward             | 3.4893   | 3.6408         | +0.1515  |
+-------------------+---------------------------+----------+----------------+----------+
```

Implementation fixes made during this run:

```text
experiments/image_calibration/calibration_pipeline.py
  Added <answer> parsing, import-prompts, --no-system-prompt, timing logs, OCR scoring, UnifiedReward sglang scoring, UR numeric-output parsing, and GenEval server chunk padding.

experiments/image_calibration/run_promptrl_full_benchmarks.sh
  Added the full benchmark runner for PromptRL-style rewrites with SD3.5 Medium.
```

A subagent audit found the PromptRL-style rewrites were materially changed rather than near-identical. There were zero exact matches in the rewritten condition; average prompt length expanded from 7.6 to 89.4 words on GenEval, 33.8 to 103.8 words on OCR1k, and 18.5 to 88.1 words on PickScore-SFW. The main caveat is semantic drift: many rewrites add realism, lighting, background, composition, and camera cues, and some OCR prompts change text content or format.


## 2026-05-29 Full PromptRL-Style FLUX.1-dev Run Started

Run root: `outputs/image_calibration/promptrl_full_flux1dev_20260529_000245`.

This run is the same full PromptRL-style comparison as `outputs/image_calibration/promptrl_full_sd35_20260528_0439`, but with the generator changed to `black-forest-labs/FLUX.1-dev`. To keep the prompt side fixed, the runner copies the SD3.5 run's `prompt_ledger.jsonl` and `run_config_rewrite.json` files for GenEval full, OCR1k, and PickScore-SFW instead of rerunning the stochastic Qwen rewrite.

Runner and code:

```text
Runner: experiments/image_calibration/run_promptrl_full_benchmarks_flux1dev.sh
Pipeline: experiments/image_calibration/calibration_pipeline.py
Source run: outputs/image_calibration/promptrl_full_sd35_20260528_0439
Log: outputs/image_calibration/promptrl_full_flux1dev_20260529_000245/run.log
```

Generator configuration:

```text
Model: black-forest-labs/FLUX.1-dev
Backend: Diffusers FluxPipeline via --generator-backend diffusers_flux
Resolution: 1024x1024
Steps: 20
Guidance: 6.0
Conditions: none,qwen25_vl_3b
```

Prompt-ledger invariant checked at launch: the FLUX run ledgers are byte-identical to the SD3.5 source ledgers. SHA256 pairs matched for `geneval_full/prompt_ledger.jsonl`, `ocr1k/prompt_ledger.jsonl`, and `pickscore_sfw/prompt_ledger.jsonl`. The run had started generating GenEval images; completion still requires all 5190 images, the same evaluator outputs, summaries, and `comparison_table.html`.
