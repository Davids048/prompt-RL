# Baseline Sweep Plan

Baseline Sweep is a clean restart of the prompt enhancement experiments. The
old `experiments/image_calibration/` code and outputs remain useful as
provenance, but the new suite should be easier to configure, inspect, resume,
and extend.

Goal: find prompt enhancement methods that improve image and video generation
while preserving the user's original intent.

## Core Terms

`suite`
: The Baseline Sweep experiment family and its code under
  `experiments/baseline_sweep/`.

`config`
: One YAML file that describes an experiment or sweep. A config declares which
  prompt sources, enhancers, generators, evals, seeds, and budgets to use.

`trial`
: One concrete expanded experiment unit from a config. A trial combines one
  enhancer setting, one generator setting, one task/eval setup, and the resolved
  seed/sample settings.

`prompt_source`
: The source of original prompts and task metadata. Examples: GenEval prompts,
  OCR1k prompts, DrawBench prompts, VBench prompt subsets, or a custom prompt
  file.

`enhancer`
: A prompt enhancement setting. It may be no enhancement, a text instruct model,
  a vision-language model, a thinking model, or another model/API. An enhancer
  maps `original_prompt` to `enhanced_prompt`.

`generator`
: The image or video generation setting. Examples: FLUX.1-dev, Stable Diffusion
  3.5 Medium, Wan 2.1 1.3B as image with one frame, or Wan 2.1 1.3B as video.

`eval`
: A benchmark, scoring harness, or external evaluation path. Examples: GenEval,
  OCR, PickScore, HPSv3, VideoAlign, and VBench. Different evals may come from
  different repos and produce different output shapes; do not force them into a
  single metric interface too early.

`records`
: JSONL provenance files written during a run. Records make the run auditable:
  what prompt was loaded, what enhancement was produced, what generation prompt
  was used, what artifact was generated, and what eval output was produced.

## Prompt Contract

The suite enforces one prompt rule for now:

```text
original_prompt
  The benchmark/user prompt from the prompt source.

enhanced_prompt
  The enhancer output, if an enhancer model is used.

generation_prompt
  enhanced_prompt if an enhancer model is used,
  otherwise original_prompt.

eval_prompt
  Always original_prompt.
```

This keeps the headline question stable:

```text
Given original user intent P, does generating from enhanced(P) produce a better
image or video for P than generating from P directly?
```

Scoring against `original_prompt` prevents the enhancer from moving the target
toward a prompt that the generator or eval happens to like.

## Records

`records.jsonl` should contain one row per artifact attempt. A row is written
only when generation and eval for that artifact have finished, or when the
attempt has failed and should be documented. Rows are final once written.

Each record should be enough to answer:

```text
What did the prompt source ask for?
Which enhancer was used?
What enhanced prompt was produced?
What prompt did the generator receive?
What prompt did eval use?
Which seed, generator config, artifact path, and eval output belong together?
```

Each record should include full prompt text and prompt metadata inline, plus the
key resolved fields needed to understand the artifact. The full resolved config
lives in `resolved_config.yaml` and the full trial config lives in
`run_plan.jsonl`, so artifact records do not need to duplicate entire nested
configs.

Example successful record:

```json
{
  "trial_id": "trial_000042",
  "task": "geneval",
  "prompt_index": 17,
  "sample_index": 0,
  "prompt_source": "geneval",
  "prompt_source_path": "evaluations/geneval/prompts/evaluation_metadata.jsonl",
  "prompt_source_index": 17,
  "original_prompt": "a red cube on top of a blue sphere",
  "prompt_metadata": {"tag": "position"},
  "enhancer_model": "Qwen/Qwen2.5-VL-3B-Instruct",
  "enhancer_backend": "hf_vlm",
  "enhancer_template": "prompt/templates/image/geneval/promptrl_style_v1.txt",
  "enhanced_prompt": "A realistic studio photograph...",
  "generation_prompt": "A realistic studio photograph...",
  "eval_prompt": "a red cube on top of a blue sphere",
  "generator_model": "stabilityai/stable-diffusion-3.5-medium",
  "generator_backend": "fastvideo_image",
  "seed": 420017,
  "artifact_path": "artifacts/trial_000042/000017_0000_geneval.png",
  "eval": {
    "geneval": {
      "score": 1.0,
      "source": "eval/trial_000042/geneval/server_results.jsonl"
    }
  },
  "error": null
}
```

For no enhancement:

```json
{
  "enhancer_model": null,
  "enhanced_prompt": null,
  "generation_prompt": "a red cube on top of a blue sphere",
  "eval_prompt": "a red cube on top of a blue sphere"
}
```

If generation or eval fails, still write a record for the attempted
prompt/sample. Failed records are not counted in metric means, but
`summary.json` should include failure counts.

Example failed record:

```json
{
  "trial_id": "trial_000042",
  "task": "geneval",
  "prompt_index": 17,
  "sample_index": 0,
  "original_prompt": "a red cube on top of a blue sphere",
  "generation_prompt": "A realistic studio photograph...",
  "eval_prompt": "a red cube on top of a blue sphere",
  "artifact_path": null,
  "eval": {},
  "error": {
    "stage": "generation",
    "message": "CUDA out of memory",
    "log_path": "logs/trial_000042_geneval_stable-diffusion-3.5-medium_Qwen2.5-VL-3B-Instruct.log"
  }
}
```

## Run Directory

Each experiment run writes a small set of files:

```text
outputs/baseline_sweep/<run_name>/
  config.yaml
  resolved_config.yaml
  run_plan.jsonl
  records.jsonl
  artifacts/
  eval/
  logs/
  summary.json
  summary.md
```

`config.yaml`
: Copy of the YAML the user wrote.

`resolved_config.yaml`
: The actual fully expanded config the code uses after defaults, paths, aliases,
  and environment-derived values are resolved.

`run_plan.jsonl`
: Planner output. It contains the expanded trials, usually the Cartesian product
  of the selected YAML axes. A single-trial config produces a one-row plan.
  Trial rows should be fully resolved so the runner can execute them without
  re-reading YAML definitions.

`records.jsonl`
: One file for what happened: prompt loading, enhancements, generations, eval
  results, and errors. Use one final row per artifact attempt.

`artifacts/`
: Final generated images and videos required for eval. Do not store thumbnails
  or intermediate files here unless an eval genuinely requires them.

`eval/`
: Raw eval outputs from external repos/tools, organized by trial first. Leave raw
  eval files as produced by the external tools; do not copy or rename them into
  standard filenames for v0.

```text
eval/
  trial_000001/
    geneval/
  trial_000002/
    geneval/
```

`logs/`
: Runtime logs, with one log per trial. The filename should include the trial id
  and high-level settings.

```text
logs/trial_000001_geneval_stable-diffusion-3.5-medium_none.log
logs/trial_000002_geneval_stable-diffusion-3.5-medium_Qwen2.5-VL-3B-Instruct.log
```

`summary.json` and `summary.md`
: Aggregated machine-readable and human-readable results. Metric aggregates
  should use successful eval records only; failed records should be counted
  separately.

Resume behavior should not trust `records.jsonl` alone. It should also check
the actual artifact and eval files before deciding to skip work.

## Execution Flow

Use one CLI command for v0:

```text
python src/cli.py run <config.yaml>
```

Top-level flow:

```text
1. Copy the user YAML to config.yaml.
2. Resolve config defaults, paths, and generated aliases into resolved_config.yaml.
3. Expand the config into fully resolved trial rows in run_plan.jsonl.
4. Loop over run_plan.jsonl.
5. Call runner.py once per trial row.
6. Write summary.json and summary.md.
```

`runner.py` executes one trial row. It should batch work by stage:

```text
1. Load all prompts for the trial.
2. Enhance all prompts, or reuse matching prior enhancements.
3. Generate all artifacts.
4. Run evals.
5. Write one final record per artifact attempt.
```

Enhancement reuse is allowed when all of these match:

```text
original prompt
enhancer model
enhancer backend/settings
prompt template
```

Failure behavior:

```text
If a prompt/sample fails, write an error record and continue.
Do not stop the whole trial for one failed prompt/sample.
Failed records do not count in metric means.
```

Sampling and seed behavior:

```text
samples_per_prompt belongs to the task/eval setup.
If an eval/task only requires one sample per prompt, default to 1.

For paired enhancer comparisons, the noise seed must be identical for the same
task, generator, prompt_index, and sample_index across enhancer settings. This
keeps the prompt conditioning as the intended experimental difference.
```

In practice, the planner should derive per-prompt/sample seeds independently of
the enhancer choice. For example, `none` and `Qwen/...` trials for the same
task/generator/prompt/sample should receive the same seed.

## Next Design Sections

The next planning decisions should be handled one at a time:

1. Code and folder structure.
2. YAML config structure.
3. Run output structure and record files.
4. Execution flow and resume semantics.
5. First minimal Baseline Sweep config.

## Code Structure

Baseline Sweep should be a thin orchestration suite. It should implement
experiment control, prompt handling, records, and adapters. It should reuse
generation and eval implementations from FastVideo and the repos under
`evaluations/` whenever possible.

Proposed structure:

```text
experiments/baseline_sweep/
  README.md

  configs/
    smoke/
    image/
    video/

  prompt/
    templates/
      image/
        general/
        geneval/
        ocr/
      video/
        general/
        vbench/

  src/
    cli.py
    config.py
    planner.py
    registry.py
    runner.py

    prompt_sources/
    enhancers/
    generators/
    eval/
    reports/

  scripts/
```

Generated outputs should live outside the suite code:

```text
outputs/baseline_sweep/
```

`configs/`
: YAML experiment configs. Subset choices belong in configs because they are
  experiment choices, not reusable data assets.

`prompt/templates/`
: Versioned `.txt` prompt templates for prompt enhancement LMs. Organize by
  broad use case or task family. Example files:

```text
prompt/templates/image/general/realism_detail_v1.txt
prompt/templates/image/general/strict_semantic_v1.txt
prompt/templates/image/geneval/promptrl_style_v1.txt
prompt/templates/image/ocr/preserve_text_v1.txt
prompt/templates/video/general/temporal_motion_v1.txt
```

`src/cli.py`
: Command-line entry point for one YAML run. It parses commands, creates the run
  directory, copies `config.yaml`, writes `resolved_config.yaml`, calls the
  planner, loops over `run_plan.jsonl`, calls `runner.py` for each trial, and
  invokes reporting.

`src/config.py`
: Load YAML, validate it, resolve relative paths, and write the resolved config.

`src/planner.py`
: Expand a config into concrete trials. It should not load models, generate
  images/videos, or run evals.

  Every config goes through the planner. A single-trial YAML is just the special
  case where the planner writes one trial. A sweep YAML writes many trials. The
  runner should consume the planned trials, not reinterpret the original YAML.

`src/registry.py`
: Lightweight dispatch tables from config names/backends to wrapper functions.
  Keep this dumb: no model loading, no experiment logic, and no abstract class
  hierarchy.

`src/runner.py`
: Execute one trial row from `run_plan.jsonl`. It owns trial-level orchestration:
  load prompts, run enhancer or no enhancement, run generator, run evals, and
  write records/errors for that trial.

`src/prompt_sources/`
: Thin adapters that load original prompts and metadata from external prompt
  sources. Examples: GenEval metadata, OCR1k prompts, DrawBench prompts, VBench
  prompt sets, or a direct JSONL/text file if needed later.

`src/enhancers/`
: Thin wrappers that take `original_prompt`, call the configured prompt
  enhancement model/template, and produce `enhanced_prompt` or `null`.

`src/generators/`
: Thin wrappers that load/invoke image or video generation models and write
  artifacts. These should call FastVideo when possible instead of
  reimplementing model logic.

`src/eval/`
: Thin wrappers that run selected eval/benchmark/scoring tools on artifacts.
  These call external repos/tools and record their outputs
  without forcing all evals into one metric-runner shape too early.

`src/reports/`
: Summaries, comparison tables, and HTML review reports.

`scripts/`
: Optional shell helpers for repeated launch commands, Slurm wrappers, or
  environment setup snippets. Keep scripts thin; experiment choices should stay
  in YAML configs.

Wrappers should be small plain-function modules around external tools. Do not
normalize everything into adapter classes or abstract base classes before the
eval and generation shapes are understood.

Most heavy imports and real work should stay outside Baseline Sweep:

```text
prompt_sources/
  read benchmark prompt files from evaluations/... and normalize rows.

enhancers/
  call Transformers, OpenAI-compatible clients, or other model APIs.

generators/
  call FastVideo generation code.

eval/
  call evaluations/... repos or external metric packages.

reports/
  read records/summary outputs and produce human-facing summaries or HTML.
```

Baseline Sweep owns orchestration, config, planning, prompt contract
enforcement, records, and reporting. External packages/repos own model loading,
generation internals, benchmark logic, and metric implementations.

Example wrapper shape:

```text
src/prompt_sources/geneval.py
  load_prompts(config) -> prompt records

src/enhancers/hf_text.py
  enhance(prompt records, config) -> enhancement records

src/generators/fastvideo_image.py
  generate(generation records, config) -> artifact records

src/eval/geneval.py
  run(generation/artifact records, config) -> eval records
```

Records are a run artifact, not a required central code module. Each related
module can write its own rows to `records.jsonl` directly. Do not add a
`records.py` wrapper just to hide simple `json.dumps` / file append operations.
If JSONL mechanics become duplicated enough to hurt, extract the smallest
possible helper later.

Example registry shape:

```text
PROMPT_SOURCES["geneval"] = prompt_sources.geneval.load_prompts
ENHANCERS["hf_text"] = enhancers.hf_text.enhance
GENERATORS["fastvideo_image"] = generators.fastvideo_image.generate
EVALS["geneval"] = eval.geneval.run
```

Implement in Baseline Sweep:

- YAML loading and validation.
- Trial planning.
- Prompt contract enforcement.
- Record writing.
- Artifact path layout.
- Resume/status checks.
- Thin adapters to external code.
- Summary/report generation.

Reuse from external code:

- FastVideo generation internals.
- GenEval prompt metadata and scoring logic.
- VBench benchmark/eval logic.
- OCR, PickScore, HPS, VideoAlign implementations.
- Model loading libraries and upstream model APIs where appropriate.

## YAML Config Structure

One YAML config can describe either one trial or a sweep. Scalars in `sweep`
mean one choice. Lists in `sweep` mean multiple choices. Every config still goes
through the planner and produces `run_plan.jsonl`.

Example:

```yaml
run: image_geneval_smoke_v1
output_root: outputs/baseline_sweep
seed: 420000

enhancers:
  none: null
  Qwen/Qwen2.5-VL-3B-Instruct:
    backend: hf_vlm
    template: prompt/templates/image/geneval/promptrl_style_v1.txt
    params:
      temperature: 0.7
      top_p: 0.9
      max_new_tokens: 256

generators:
  stabilityai/stable-diffusion-3.5-medium:
    backend: fastvideo_image
    params:
      width: 1024
      height: 1024
      steps: 20
      sampler: euler

tasks:
  geneval:
    prompt_source: geneval
    source_path: evaluations/geneval/prompts/evaluation_metadata.jsonl
    subset:
      mode: first_n
      n: 8
    samples_per_prompt: 1
    eval:
      - geneval

sweep:
  enhancer:
    - none
    - Qwen/Qwen2.5-VL-3B-Instruct
  generator: stabilityai/stable-diffusion-3.5-medium
  task: geneval
```

Config rules:

- `samples_per_prompt` belongs under `tasks`, not top-level, because sampling
  needs differ by eval/task. If an eval/task only requires one sample per
  prompt, default to `1`.
- `none` is the reserved enhancer option for no prompt enhancement.
- Enhancer and generator definitions use exact model names as YAML keys.
- `backend` says how to call a model, for example `hf_vlm`,
  `fastvideo_image`, or `fastvideo_video`.
- Task names should be task families like `geneval`, `ocr`, `drawbench`, or
  `vbench`; smoke/full behavior belongs in the task `subset`.
- The prompt contract is not configurable in YAML: `generation_prompt` uses the
  enhanced prompt when an enhancer is used, and `eval_prompt` is always the
  original prompt.
- Generator params such as height, width, denoising steps, sampler/scheduler,
  and CFG/guidance should come from the relevant paper/config when the run is
  meant to reproduce a paper setting. For PromptRL image calibration, prefer the
  PromptRL paper or released config when it specifies a value; otherwise use the
  generator/backend default. Record the resolved value and its source in
  `resolved_config.yaml`.
- Current local research notes say PromptRL Table 1 used SD3 at 1024 resolution
  and 20 Euler steps. They do not establish a CFG/guidance value, so
  CFG/guidance should use the generator/backend default unless a PromptRL
  released config or paper source specifies otherwise.
- Do not silently inherit generation params from old exploratory runs. If a
  value is chosen manually, mark it as manual/local in `resolved_config.yaml`.

## Current Experiment Axes

Prompt enhancement:

- Model family: no enhancement, text instruct model, vision-language model,
  thinking model.
- Model size: small, medium, large.
- Prompt template and generation settings.

Generation:

- Image: FLUX.1-dev, Stable Diffusion 3.5 Medium, Wan 2.1 1.3B with one frame.
- Video: Wan 2.1 1.3B.
- Generation settings: CFG, denoising steps, seed control, resolution, frame
  count for video. Use paper/released-config values when reproducing a paper
  setting; otherwise use model/backend defaults and record the provenance.

Eval:

- Image generation: GenEval, OCR/text rendering, human preference style prompts
  with preference scoring.
- Video: HPSv3, VideoAlign, VBench.

## Initial Sweep Strategy

Start small before expanding the grid.

1. Image anchor smoke.
   Use no enhancement, one text LM, one VLM, and one thinking model. Use tiny
   subsets of GenEval, OCR1k, and DrawBench. Keep `samples_per_prompt` at `1`
   unless a selected eval requires more.

2. Image controlled sweep.
   Expand enhancer coverage while keeping generator settings, seeds, and evals
   fixed.

3. Video probe.
   Use the best few image enhancers with Wan 2.1 1.3B video and a small set of
   video evals.

4. Video full sweep.
   Scale only after prompt handling, records, seed policy, and eval behavior are
   stable.

## Minimal Baseline V0

The first implementation target is a tiny image GenEval run that proves the new
suite works end to end.

```text
run:
  image_geneval_smoke_v1

enhancers:
  none
  Qwen/Qwen2.5-VL-3B-Instruct

generator:
  stabilityai/stable-diffusion-3.5-medium
  backend: FastVideo

task/eval:
  GenEval only
  small balanced GenEval subset
  samples_per_prompt: 1
```

Use Stable Diffusion 3.5 Medium for v0 because it exists locally and should be
supported through FastVideo. Do not include FLUX, Wan, OCR, DrawBench, HPS,
VideoAlign, or VBench in the first implementation slice.

Runtime assumptions:

```text
Python environment:
  /home/hal-jundas/codes/UniRL/.venv

Slurm jobs available for scripts:
  4882
  4884

Slurm launch mode:
  srun --overlap --jobid=<job_id> ...
```

Slurm details should live in `experiments/baseline_sweep/scripts/`, not inside
the Python orchestration code. Use overlap mode so the allocated nodes are not
released after eval.

## Risks And Assumptions

- The previous FLUX path used Diffusers. If Baseline Sweep should route all
  generation through FastVideo, first verify FastVideo support for FLUX.1-dev,
  Stable Diffusion 3.5 Medium, and Wan 2.1 1.3B.
- Prompt enhancement templates are major experimental variables and should be
  versioned as first-class config inputs.
- Primary eval always uses `original_prompt`; any later diagnostic score against
  an enhanced prompt should be clearly separated from headline results.
