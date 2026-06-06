# Current State

Last updated: 2026-06-06

This workspace is now a top-level experiment hub at `/home/hal-jundas/codes/UniRL`. It is not being used as the original UniRL git repository.

The main directories are:

```text
papers/
  PDFs and related-work notes.

FastVideo/
  Local FastVideo checkout used for image/video generation integration.

evaluations/
  Evaluator repositories and reward-server support.

experiments/image_calibration/
  Prompt rewrite versus no-rewrite calibration pipeline.

outputs/image_calibration/
  Generated images, metrics, summaries, and HTML reports.

.agents/memory/
  Agent-facing project memory.
```

Root versioning policy is intentionally narrow. Track code, prompts, memory, research Markdown/text notes, `externals.lock.md`, and `patches/reward-server/*.patch`. Do not track `plan.md`, `reports/`, generated outputs, local environments, caches, raw artifacts, or nested third-party repo checkouts.

Completed image-calibration work now includes both the original 12-prompt SD3.5 Medium pilot and a full PromptRL-style rerun. The full rerun used all GenEval prompts plus OCR1k and PickScore-SFW, comparing no rewrite against PromptRL-style rewrites from `Qwen/Qwen2.5-VL-3B-Instruct`; it did not include the text-only `Qwen/Qwen2.5-3B-Instruct` condition. The run root is `outputs/image_calibration/promptrl_full_sd35_20260528_0439`.

Important caution: if GPU work uses Slurm job `4745`, use overlap mode only:

```text
srun --overlap --jobid=4745 ...
```

Do not cancel or release the holder allocation unless the user explicitly asks. Earlier work killed only the active `torchrun` step, not the allocation holder.
