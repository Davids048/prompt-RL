# Agent Memory Index

Last updated: 2026-06-05

This directory records persistent project context so future agents can quickly understand what has happened without rereading the whole conversation. It is intentionally shallow and topic-oriented.

```text
current_state.md
  What this workspace is, what exists now, and what assumptions are safe.

experiment_handoff.md
  Fast-start current state of completed, partial, and aborted image-calibration experiments.

image_calibration.md
  The SD3.5 Medium prompt-rewrite pilot, results, artifacts, caveats, and HTML reports.

infrastructure.md
  Environments, FastVideo, evaluator repos, GenEval reward server, Slurm/GPU notes.

evaluations.md
  Inventory of `evaluations/` subdirectories, git remotes, workspace use, and local changes.

open_questions.md
  Known uncertainties and next decisions before scaling experiments.

change_log.md
  Dated progress log.
```

Mental model:

```text
papers/
  research memory and related-work notes

FastVideo/
  generation framework

evaluations/
  cloned evaluator repos and reward-server support

experiments/
  runnable experiment scripts and pipeline code

outputs/
  generated artifacts, metrics, summaries, and HTML review reports

.agents/memory/
  compact agent-facing project memory
```

Read order for future agents:

1. `memory/current_state.md`
2. `memory/experiment_handoff.md` for the current image-calibration experiment state
3. The task-specific memory file, usually `memory/image_calibration.md`
4. `memory/evaluations.md` when touching evaluator code, prompt pools, or reward-server setup
5. `memory/open_questions.md` before proposing the next run

Update policy: after a meaningful run, environment change, evaluator fix, or result interpretation, update the relevant memory file and add one line to `memory/change_log.md`.
