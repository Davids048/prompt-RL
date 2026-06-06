# Baseline Sweep

Baseline Sweep is the clean prompt-enhancement experiment suite described in
`/home/hal-jundas/codes/UniRL/plan.md`.

V0 is intentionally small:

```text
none vs Qwen/Qwen2.5-VL-3B-Instruct
stabilityai/stable-diffusion-3.5-medium through FastVideo
balanced tiny GenEval subset
GenEval reward-server eval
```

Run locally inside the workspace environment:

```bash
/home/hal-jundas/codes/UniRL/.venv/bin/python \
  /home/hal-jundas/codes/UniRL/experiments/baseline_sweep/src/cli.py \
  run \
  /home/hal-jundas/codes/UniRL/experiments/baseline_sweep/configs/smoke/image_geneval_smoke_v1.yaml
```

For Slurm, use the helper in `scripts/`. It uses overlap mode and keeps Slurm
details out of the Python orchestration code.
