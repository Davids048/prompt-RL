# RL Prompt Enhancer

This directory is the single source surface for the RL prompt-enhancer pipeline.
Keep reusable code and source launch profiles here.

Reusable data inputs live outside experiment run records, under
`/home/hal-jundas/codes/UniRL/data/rl_prompt_enhancer/`.

The current human-facing launch profile is:

```text
/home/hal-jundas/codes/UniRL/rl_prompt_enhancer/scripts/launch_exp/image_grpo/
```

It contains four source-default command files:

```text
experiment.sh
01_start_ray_head.sh
02_start_fastvideo_workers.sh
03_submit_slime_grpo.sh
```

The generator and validation config defaults live under:

```text
/home/hal-jundas/codes/UniRL/rl_prompt_enhancer/configs/
```

Normal launch flow:

```text
rl_prompt_enhancer/scripts/launch_exp/image_grpo/*.sh
  -> copy into experiments/<run_id>/commands/
  -> copy rl_prompt_enhancer/configs/fastvideo_image_grpo_*.{json,yaml} into experiments/<run_id>/configs/
  -> edit the run-local command files directly
  -> snapshot default and used command/config files under experiments/<run_id>/snapshot/
  -> launch commands/01_start_ray_head.sh
  -> launch commands/02_start_fastvideo_workers.sh
  -> launch commands/03_submit_slime_grpo.sh
  -> logs, checkpoints, ledgers, artifacts, wandb
```

`experiments/` is for generated run records and run-local launch copies.
Reusable prompt data stays in `data/`. Do not launch from the source profile
directory; launch from a run's `commands/` directory.

The legacy YAML launcher remains in this package only as historical machinery
while the shell-first workflow is reviewed. Do not add new launch behavior to
the YAML path.

Do not add maintained launcher scripts or canonical configs under
`/home/hal-jundas/codes/UniRL/experiments/rl_prompt_enhancer`; use this package
instead. That experiment-side source directory has been removed and should not
be recreated.
