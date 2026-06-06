# Infrastructure Memory

Last updated: 2026-05-28

Primary Python environment:

```text
.venv
Python: 3.12
Torch: CUDA 12.9 stack observed during setup
FastVideo: installed editable in this workspace
Flash Attention 4 / CuTe path: installed for Blackwell/B200-oriented work
```

Evaluator environment:

```text
.envs/reward_server
Python: 3.10
Purpose: GenEval reward-server dependency isolation
```

GenEval reward-server notes:

```text
Repo: evaluations/reward-server
Server command used by pilot: gunicorn app_geneval:create_app()
Local URL used by pilot: http://127.0.0.1:18085
MMCV: locally built mmcv-full 1.7.2 CUDA ops for sm_100
Do not install plain mmcv 2.x in this env because it shadows legacy exports expected by MMDetection 2.x.
```

For the full evaluator repo inventory, including all `evaluations/` subdirectories, git remotes, current workspace usage, and local modifications, read `memory/evaluations.md`.

FastVideo and generation:

```text
FastVideo lives at FastVideo/
Pilot generator: stabilityai/stable-diffusion-3.5-medium
Pilot outputs are PNG images, not videos.
Future video work should use FastVideo for Wan2.1 / HunyuanVideo-style T2V experiments.
```

Slurm/GPU caution:

If using the active GPU allocation mentioned in prior work, use:

```text
srun --overlap --jobid=4745 ...
```

Never cancel the allocation or kill the holder process unless explicitly requested. Inspect before killing any process.
