# Infrastructure Memory

Last updated: 2026-06-06

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

FastVideo VBench eval inventory, 2026-06-06:

```text
FastVideo's `fastvideo.eval` source registers all 16 original VBench T2V
metrics under `vbench.<dimension>`. The canonical 16 from the vendored
VBench `build_full_dimension_list()` are:

subject_consistency, background_consistency, aesthetic_quality,
imaging_quality, object_class, multiple_objects, color,
spatial_relationship, scene, temporal_style, overall_consistency,
human_action, temporal_flickering, motion_smoothness, dynamic_degree,
appearance_style.

No standard VBench metric is missing from source. Runtime/default coverage
is more nuanced: the arbitrary-video LTX2 example defaults to 8 metrics;
`color`, `object_class`, `multiple_objects`, and `spatial_relationship`
need GRiT/detectron2; `scene` is implemented through an AVoCaDO/Qwen
adapter and needs the fuller eval dependency path; `temporal_style` reuses
the `overall_consistency` ViCLIP logic with different VBench prompt
semantics.
```

External model checkouts:

```text
Wan2.1/
  GitHub repo: https://github.com/Wan-Video/Wan2.1
  HEAD: 9737cba9c1c3c4d04b33fcad41c111989865d315
  Cloned on 2026-06-06 as a shallow source checkout for official generation-default inspection.

Wan2.1-T2V-1.3B/
  Hugging Face repo: https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B
  HEAD: 37ec512624d61f7aa208f7ea8140a131f93afc9a
  Cloned on 2026-06-06 with GIT_LFS_SKIP_SMUDGE=1 for config/README inspection.
  Large checkpoint, tokenizer, and image files are LFS pointer files until LFS blobs are pulled.
```

Wan2.1 T2V-1.3B generation-default finding, 2026-06-06:

Scope: T2V-1.3B only. The official Wan source default, official 1.3B README/Gradio recommendation, FastVideo registered Wan preset, FastVideo shipped inference YAML, and the specific Baseline Sweep run config do not all agree.

| Setting             | Official Wan code default                  | Official 1.3B README/Gradio recommendation | FastVideo registered `wan_t2v_1_3b` preset | FastVideo `scripts/inference/inference_wan.yaml` | Specific Baseline Sweep run YAML            |
| ------------------- | ------------------------------------------ | ------------------------------------------ | ------------------------------------------ | ------------------------------------------------ | ------------------------------------------- |
| Model/task          | `t2v-1.3B` if explicitly chosen            | `t2v-1.3B`                                 | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`         | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`               | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`         |
| Backend             | official Wan                               | official Wan                               | FastVideo                                  | FastVideo                                        | FastVideo                                   |
| Resolution          | must pass `832*480` or `480*832`           | `832*480` CLI; `480*832` Gradio default    | `832x480`                                  | `832x480`                                        | `480x480`                                   |
| Frames              | `81`                                       | `81` via underlying call                   | `81`                                       | `77`                                             | `53`                                        |
| FPS                 | `16`                                       | `16`                                       | `16`                                       | `16`                                             | `8`                                         |
| Sampling steps      | `50`                                       | `50`                                       | `50`                                       | `50`                                             | `16`                                        |
| Shift / flow shift  | `5.0`                                      | `8.0` recommended/UI default               | pipeline config default `3.0`              | override `8.0`                                   | not specified in YAML                       |
| Guidance / CFG      | `5.0`                                      | `6.0` recommended/UI default               | `3.0`                                      | `6.0`                                            | `6.0`                                       |
| Solver              | `unipc`                                    | `unipc` unless overridden                  | not exposed in preset defaults             | not specified                                    | not specified                               |
| Seed                | randomized from `-1`                       | `-1`, randomized                           | `1024` generic sampling default behavior   | `1024`                                           | run seed `430000`                           |
| Negative prompt     | Chinese shared Wan negative prompt         | empty in Gradio unless user enters one     | English Wan negative prompt                | English Wan negative prompt                      | English Wan negative prompt                 |

Specific run caveat:

```text
/home/hal-jundas/codes/UniRL/outputs/baseline_sweep/wan21_13b_dancegrpo_vbench_rewards_v1/config.yaml
```

This specific run config does not match either the official Wan 1.3B recommendation or the main FastVideo Wan 1.3B defaults: it uses `480x480`, `53` frames, `8` fps, and `16` sampling steps. Treat results from only this run as potentially less informative than desired for judging standard Wan2.1 T2V-1.3B behavior or comparing prompt-enhancer quality under typical 480p generation settings. The caveat is scoped only to the run path above.

Sources:

- `/home/hal-jundas/codes/UniRL/Wan2.1/generate.py`: official parser defaults and task-conditioned defaults.
- `/home/hal-jundas/codes/UniRL/Wan2.1/wan/configs/__init__.py`: supported T2V-1.3B sizes are `480*832` and `832*480`.
- `/home/hal-jundas/codes/UniRL/Wan2.1/wan/configs/shared_config.py`: official shared `sample_fps=16` and Chinese negative prompt.
- `/home/hal-jundas/codes/UniRL/Wan2.1/gradio/t2v_1.3B_singleGPU.py`: official 1.3B Gradio UI defaults: 50 steps, guide 6.0, shift 8.0.
- `/home/hal-jundas/codes/UniRL/Wan2.1/README.md`: official README recommends `--sample_shift 8` and `--sample_guide_scale 6` for T2V-1.3B.
- `/home/hal-jundas/codes/UniRL/FastVideo/fastvideo/pipelines/basic/wan/presets.py`: FastVideo registered `wan_t2v_1_3b` preset defaults.
- `/home/hal-jundas/codes/UniRL/FastVideo/fastvideo/configs/pipelines/wan.py`: FastVideo Wan T2V 480P pipeline `flow_shift=3.0`.
- `/home/hal-jundas/codes/UniRL/FastVideo/scripts/inference/inference_wan.yaml`: FastVideo shipped Wan inference example values.
- `/home/hal-jundas/codes/UniRL/outputs/baseline_sweep/wan21_13b_dancegrpo_vbench_rewards_v1/config.yaml`: specific Baseline Sweep run values and caveat.

Slurm/GPU caution:

For active Wan2.1-1.3B Baseline Sweep video runs, the relevant holder jobs are:

```text
4882 -> hpc-rack-2-3, 4 GPUs
4884 -> hpc-rack-2-9, 4 GPUs
```

Use overlap mode only, for example:

```text
srun --overlap --jobid=4882 ...
srun --overlap --jobid=4884 ...
```

Never cancel the allocation or kill the holder process unless explicitly requested. Inspect before killing any process.

Restricted VBench Baseline Sweep launch, 2026-06-07:

```text
Config: experiments/baseline_sweep/configs/video/wan21_13b_vbench6_prompt_enhancer_v2.yaml
Run dir: outputs/baseline_sweep/wan21_13b_vbench6_prompt_enhancer_v2
tmux session: baseline_sweep_vbench6_v2
Jobs: 4882 handles trials 1-3; 4884 handles trials 4-5, all via srun --overlap
Prompt count: 251 unique official VBench prompts * 5 samples = 1255 videos/trial
Generator settings: 832x480, 81 frames, 16 fps, 50 steps, CFG 6.0, flow_shift 8.0
VBench dimensions: subject_consistency, background_consistency, motion_smoothness, dynamic_degree, aesthetic_quality, imaging_quality
```

Companion HPSv3/VideoAlign Baseline Sweep launch, 2026-06-07:

```text
Config: experiments/baseline_sweep/configs/video/wan21_13b_dancegrpo200_hpsv3_videoalign_v2.yaml
Run dir: outputs/baseline_sweep/wan21_13b_dancegrpo200_hpsv3_videoalign_v2
tmux session: baseline_sweep_dancegrpo200_v2
Jobs: 4882 handles trials 1-3; 4884 handles trials 4-5, all via srun --overlap
Prompt count: 200 DanceGRPO/VidProM prompts * 1 sample = 200 videos/trial
Generator settings: 832x480, 81 frames, 16 fps, 50 steps, CFG 6.0, flow_shift 8.0
Eval: HPSv3 and VideoAlign
```
