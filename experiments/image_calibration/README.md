# Image Calibration Baseline

This folder holds the first calibration path for prompt enhancement experiments. The current target is an image-generation sanity check against the PromptRL-style setup before expanding to video generators.

The pipeline is:

```text
GenEval metadata prompts
  -> prompt ledger with rewrite/no-rewrite conditions
  -> FastVideo SD3.5 Medium image generation
  -> preference metrics and GenEval-compatible artifact layout
  -> run summary
```

Use the workspace virtual environment:

```bash
export VIRTUAL_ENV=/home/hal-jundas/codes/UniRL/.venv
export PATH="$VIRTUAL_ENV/bin:$PATH"
export HF_TOKEN="$(cat /home/hal-jundas/.cache/huggingface/token)"
export HF_HOME=/home/hal-jundas/codes/UniRL/.cache/huggingface
export TRANSFORMERS_CACHE=/home/hal-jundas/codes/UniRL/.cache/huggingface/transformers
```

All GPU commands should use the existing Slurm allocation with overlap:

```bash
srun --overlap --jobid=4745 bash -lc '<command>'
```

Do not start or cancel the holder step for job `4745`.

## Smoke Commands

Create prompt rewrites:

```bash
python experiments/image_calibration/calibration_pipeline.py rewrite \
  --run-dir outputs/image_calibration/smoke_sd35 \
  --limit 1 \
  --conditions none,qwen25_vl_3b \
  --overwrite
```

Generate images with FastVideo SD3.5 Medium:

```bash
srun --overlap --jobid=4745 bash -lc 'export VIRTUAL_ENV=/home/hal-jundas/codes/UniRL/.venv; export PATH="$VIRTUAL_ENV/bin:$PATH"; export HF_TOKEN="$(cat /home/hal-jundas/.cache/huggingface/token)"; export HF_HOME=/home/hal-jundas/codes/UniRL/.cache/huggingface; export TRANSFORMERS_CACHE=/home/hal-jundas/codes/UniRL/.cache/huggingface/transformers; python experiments/image_calibration/calibration_pipeline.py generate --run-dir outputs/image_calibration/smoke_sd35 --height 512 --width 512 --steps 2 --attention-backend TORCH_SDPA --overwrite'
```

Evaluate preference metrics:

```bash
srun --overlap --jobid=4745 bash -lc 'export VIRTUAL_ENV=/home/hal-jundas/codes/UniRL/.venv; export PATH="$VIRTUAL_ENV/bin:$PATH"; export HF_TOKEN="$(cat /home/hal-jundas/.cache/huggingface/token)"; export HF_HOME=/home/hal-jundas/codes/UniRL/.cache/huggingface; export TRANSFORMERS_CACHE=/home/hal-jundas/codes/UniRL/.cache/huggingface/transformers; python experiments/image_calibration/calibration_pipeline.py eval-preference --run-dir outputs/image_calibration/smoke_sd35 --metrics pickscore,hps --device cuda'
```

HPS is emitted on the benchmark/table scale, `normalized image-text dot product * 100`, so it is comparable to PromptRL-style HPS values.
Every `summarize` call writes both `summary.json` / `summary.md` and `run_review.md`. The review file is the human-readable audit surface: it lists the config files, rewrite knobs, full rewrite instruction prompt, user template, generator knobs, metric deltas, input prompts, rewritten prompts, seeds, and generated artifact paths.

Record official GenEval status:

```bash
python experiments/image_calibration/calibration_pipeline.py eval-geneval \
  --run-dir outputs/image_calibration/smoke_sd35
```

If a GenEval reward-server is running on the same node, score through the server instead of importing `mmdet` in this `.venv`:

```bash
python experiments/image_calibration/calibration_pipeline.py eval-geneval \
  --run-dir outputs/image_calibration/smoke_sd35 \
  --server-url http://127.0.0.1:18085 \
  --overwrite
```

## Current Notes

The generated image folder layout matches official GenEval, but the official evaluator currently should not run inside the shared `.venv` because it requires the `mmdet`/`mmcv`/`open_clip` stack. Flow-GRPO uses a separate GenEval reward server for the same reason, and that repo is cloned at `evaluations/reward-server`. The reward-server env is prepared at `.envs/reward_server` with `torch==2.11.0+cu129`, locally built `mmcv-full==1.7.2` CUDA ops for `sm_100`, editable `mmdet==2.28.2`, and headless OpenCV. The reusable MMCV wheel is stored at `.cache/wheels/mmcv_full-1.7.2-cp310-cp310-linux_aarch64.whl`; plain `mmcv==2.x` should not be installed in this env. The remaining runtime blocker is that `srun --overlap --jobid=4745` currently cannot contact the Slurm controller. The pipeline writes `geneval_status.json` instead of silently failing when the local imports are missing.

To start the GenEval reward server when overlap access is available:

```bash
srun --overlap --jobid=4745 bash -lc 'cd /home/hal-jundas/codes/UniRL/evaluations/reward-server; export MPLCONFIGDIR=/tmp/matplotlib-reward-server; export GENEVAL_DEVICE=cuda; export GENEVAL_NUM_DEVICES=1; /home/hal-jundas/codes/UniRL/.envs/reward_server/bin/gunicorn "app_geneval:create_app()"'
```

The first Qwen2.5-VL-3B smoke rewrite changed `a photo of a bench` into `a photograph of an empty wooden bench in a park setting`. That is useful for surfacing the prompt-enhancement knob: even a conservative system prompt can add attributes or scene context, so the rewrite system prompt should be treated as an explicit experimental condition.

## Pilot Subset

The file `experiments/image_calibration/geneval_balanced_12.jsonl` contains two prompts from each GenEval task group. It is intended for the first non-smoke calibration pass before running the full 553-prompt GenEval set.

Create rewrite ledgers for that subset with:

```bash
python experiments/image_calibration/calibration_pipeline.py rewrite \
  --run-dir outputs/image_calibration/pilot_sd35_balanced12 \
  --metadata experiments/image_calibration/geneval_balanced_12.jsonl \
  --conditions none,qwen25_vl_3b \
  --overwrite
```

For a stricter semantic-preservation condition, add:

```bash
--system-prompt-file experiments/image_calibration/prompts/strict_semantic_rewrite.txt
```

The reproducible pilot launcher is:

```bash
experiments/image_calibration/run_pilot_sd35_balanced12.sh
```

It checks job `4745` with `srun --overlap`, then runs rewrite, SD3.5 Medium generation, PickScore/HPS/UnifiedReward, and GenEval reward-server scoring. It exits before doing any GPU work if the overlap probe fails.
