# RL Prompt Enhancer

This directory is the single source surface for the RL prompt-enhancer pipeline.
Keep reusable code, launch logic, and central configs here.

Reusable data inputs live outside experiment run records, under
`/home/hal-jundas/codes/UniRL/data/rl_prompt_enhancer/`.

Normal launch flow:

```text
data/rl_prompt_enhancer/... + rl_prompt_enhancer/configs/*.yaml
  -> python -m rl_prompt_enhancer.yaml_launcher --config <yaml>
  -> experiments/<run_id>/snapshot/
  -> experiments/<run_id>/commands/
  -> experiments/<run_id>/logs, checkpoints, ledgers, artifacts, wandb
```

`experiments/` is for generated run records. Reusable prompt data stays in
`data/`; the launcher copies the YAML to each run's `snapshot/config.yaml` and
renders exact shell commands under that run's `commands/` directory before
launching Ray, FastVideo, and Slime.

Do not add maintained launcher scripts or canonical configs under
`experiments/rl_prompt_enhancer`; use this package instead.
