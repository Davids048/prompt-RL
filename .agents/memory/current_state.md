# Current State

Last updated: 2026-06-14

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

rl_prompt_enhancer/
  Tracked reusable code for the RL prompt-enhancer experiments, including
  the Slime custom-generate hook, FastVideo generate-and-score bridge service,
  shared request/response schema, and prompt-data conversion helpers.

slime/
  Local checkout of https://github.com/THUDM/slime for RL-framework comparison and RL prompt-enhancer porting.
  Cloned 2026-06-13 at 5d7296a77a83bb249b257ee1f082d83db16a8079.
  Current preference: use Slime because it is more lightweight; revisit Miles later only if needed.
  User-requested exception: `slime/` is no longer ignored by the root `.gitignore` as of 2026-06-13.

miles/
  Ignored checkout of https://github.com/radixark/miles for RL-framework comparison.
  Cloned 2026-06-13 at 969c2a70b8816465e923d2d9ba3e0ee76507a1e3.
  Jun Li said the code is actually all the same.
```

RL prompt-enhancer execution status as of 2026-06-14:

- Phase 1 Slime sanity with `Qwen/Qwen3.5-9B` succeeded at `/home/hal-jundas/codes/UniRL/experiments/rl_prompt_enhancer/phase1_slime_sanity/runs/phase1_qwen35_9b_gsm8k_short_20260613`.
- Phase 2 image-only FastVideo+Slime GRPO succeeded at `/home/hal-jundas/codes/UniRL/experiments/rl_prompt_enhancer/phase2_fastvideo_image_grpo/runs/phase2_fastvideo_image_grpo_20260613_181035`.
- Scaled Phase 2 image-only FastVideo+Slime GRPO launched at `/home/hal-jundas/codes/UniRL/experiments/rl_prompt_enhancer/phase2_fastvideo_image_grpo/runs/phase2_fastvideo_image_grpo_full_prompt_clean_20260614_004449` using Slime job `4972` on `hpc-rack-2-8` and FastVideo job `4973` on `hpc-rack-2-12`; local logs show GRPO training through at least `train/step: 7`, FastVideo request/reward/artifact ledgers, and W&B run `https://wandb.ai/js202/prompt%20RL/runs/g9rr98vc`.
- The FastVideo image-GRPO prompt-enhancer launch has a YAML-source-of-truth path: use `/home/hal-jundas/codes/UniRL/experiments/rl_prompt_enhancer/launch/launch_phase2.sh --config /home/hal-jundas/codes/UniRL/rl_prompt_enhancer/configs/fastvideo_image_grpo.yaml`. The two-update e2e test succeeded at `/home/hal-jundas/codes/UniRL/experiments/rlprompt_phase2_yml_config_e2e_test` using Slurm job `4882` on live-verified node `hpc-rack-2-3`, with Slime/Ray on GPUs `0,1,2` and FastVideo on GPU `3`; it produced 6 image artifacts, 6 reward rows, checkpoint `checkpoints/iter_0000001`, and W&B run `https://wandb.ai/js202/prompt%20RL/runs/rnumvv2w`.
- Full-epoch Phase 2 image-only FastVideo+Slime GRPO launched at `/home/hal-jundas/codes/UniRL/experiments/rlprompt_phase2_full_epoch_group8`: full DiffusionNFT PickScore prompt JSONL with 25,432 rows, `num_frames = 1`, `num_rollout = 25432`, `rollout_batch_size = 1`, `n_samples_per_prompt = 8`, `global_batch_size = 8`, DMD2 validation baseline enabled before training by setting `skip_eval_before_train: false`, Slime on Slurm job `4884` / `hpc-rack-2-9`, and FastVideo on Slurm job `4882` / `hpc-rack-2-3`. FastVideo request-level data parallel validation passed with four one-GPU workers on ports `18080,18081,18082,18083`; evidence lives at `/home/hal-jundas/codes/UniRL/experiments/rlprompt_phase2_full_epoch_group8/validation/fastvideo_dp/20260614_102225`. The first launch attempt failed before training because the Slime node lacked node-local `/dev/shm/Qwen3.5-9B_torch_dist`; setup recreated it on job `4884` with evidence at `/home/hal-jundas/codes/UniRL/experiments/rlprompt_phase2_full_epoch_group8/env_setup/phase2_slime_node_torch_dist_20260614_102846`. The relaunched Ray job `raysubmit_x7V8kgvzERbQvxXi` was `RUNNING` when monitoring stopped after W&B run `https://wandb.ai/js202/prompt%20RL/runs/twmirro5` became visible.
- The previous scaled Phase 2 run used the full DiffusionNFT PickScore JSONL prompt snapshot with 25,432 rows, `num_frames = 1`, FastVideo on 4 GPUs with `RLPE_FASTVIDEO_HSDP_SHARD_DIM=4`, Slime `N_SAMPLES_PER_PROMPT=4`, `GLOBAL_BATCH_SIZE=4`, and `NUM_ROLLOUT=32`.
- The reusable Slime/FastVideo bridge for Phase 2 now lives in tracked root code under `/home/hal-jundas/codes/UniRL/rl_prompt_enhancer/`; experiment scripts reference `rl_prompt_enhancer.slime_hooks.fastvideo_generate.generate` and `rl_prompt_enhancer.fastvideo_bridge.generate_and_score_server:app`.
- The Phase 2 run snapshot is deduplicated under `/home/hal-jundas/codes/UniRL/experiments/rl_prompt_enhancer/phase2_fastvideo_image_grpo/runs/phase2_fastvideo_image_grpo_20260613_181035/snapshot/`; use `snapshot/shared/` for shared config/source evidence, `snapshot/slime/` for Slime/Ray settings and prompt inputs, and `snapshot/fastvideo_service/` for service-side settings.
- Phase 2 used Slime/Ray on GPUs `0,1,2`, FastVideo service on GPU `3`, `num_frames=1`, PickScore+CLIPScore `avg` reward, and W&B run `https://wandb.ai/js202/prompt%20RL/runs/rygtxv7u`.
- The first Phase 2 submit failed because Slime needed chat/list-shaped prompts when a processor was present; the successful run added `--apply-chat-template` and preserved the raw image prompt as `metadata.original_prompt`.
- RL prompt-enhancer workflow boundary: the public interface is now direct `setup -> launch -> result`. Use `/home/hal-jundas/codes/UniRL/experiments/rl_prompt_enhancer/env_setup/prepare_phase1_env.sh`, `/home/hal-jundas/codes/UniRL/experiments/rl_prompt_enhancer/launch/launch_phase1.sh`, and YAML-backed `/home/hal-jundas/codes/UniRL/experiments/rl_prompt_enhancer/launch/launch_phase2.sh --config <phase2-yaml>`. Agent validation, command-generation scaffolding, preflight history, and pycache were moved under `/home/hal-jundas/codes/UniRL/experiments/rl_prompt_enhancer/_internal/`; old generated/debug run directories remain under `/home/hal-jundas/codes/UniRL/experiments/rl_prompt_enhancer/_archive/`.

Root versioning policy is intentionally narrow. Track code, prompts, memory, research Markdown/text notes, `externals.lock.md`, and `patches/reward-server/*.patch`. Do not track `plan.md`, `reports/`, generated outputs, local environments, caches, raw artifacts, or nested third-party repo checkouts except the current user-requested `slime/` visibility exception.

Completed image-calibration work now includes both the original 12-prompt SD3.5 Medium pilot and a full PromptRL-style rerun. The full rerun used all GenEval prompts plus OCR1k and PickScore-SFW, comparing no rewrite against PromptRL-style rewrites from `Qwen/Qwen2.5-VL-3B-Instruct`; it did not include the text-only `Qwen/Qwen2.5-3B-Instruct` condition. The run root is `outputs/image_calibration/promptrl_full_sd35_20260528_0439`.

Important caution: if GPU work uses Slurm job `4745`, use overlap mode only:

```text
srun --overlap --jobid=4745 ...
```

Do not cancel or release the holder allocation unless the user explicitly asks. Earlier work killed only the active `torchrun` step, not the allocation holder.
