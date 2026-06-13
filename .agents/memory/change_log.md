# Change Log

## 2026-06-08

- Corrected the prompt-video report's green-row best-enhancer rule from raw metric averaging to equal-weight normalized averaging across measured metrics, preventing high-scale metrics such as `imaging_quality` from dominating low-scale metrics such as `aesthetic_quality`.
- Updated the paged prompt-video HTML report to color the best enhancer per original prompt light green using average measured metrics only, and to bold/black/underline per-metric maximum score cells.
- Updated the paged prompt-video HTML report so currently loaded prompt-page videos loop and rows alternate white/gray backgrounds by enhancer setting.
- Updated the paged prompt-video HTML report so each currently loaded prompt page uses muted autoplay inline video players.
- Converted `outputs/baseline_sweep/wan21_13b_vbench6_prompt_enhancer_v2/prompt_video_report.html` into a paged one-original-prompt-at-a-time viewer backed by `prompt_video_report_manifest.json` and per-prompt JSON files, reducing initial page load and keeping the table horizontally scrollable/readable.
- Added Baseline Sweep HTML prompt-video report generation and rendered `outputs/baseline_sweep/wan21_13b_vbench6_prompt_enhancer_v2/prompt_video_report.html`; it groups the completed restricted VBench run by original prompt with enhancer prompts, VBench scores, and embedded videos.

## 2026-06-07

- Created and launched companion reward run `wan21_13b_dancegrpo200_hpsv3_videoalign_v2` from `experiments/baseline_sweep/configs/video/wan21_13b_dancegrpo200_hpsv3_videoalign_v2.yaml`; it uses the same five enhancer settings and Wan2.1-T2V-1.3B generation config as the restricted VBench run, but evaluates HPSv3 and VideoAlign on the first 200 DanceGRPO/VidProM prompts.
- Created and launched restricted VBench Baseline Sweep `wan21_13b_vbench6_prompt_enhancer_v2` from `experiments/baseline_sweep/configs/video/wan21_13b_vbench6_prompt_enhancer_v2.yaml`; it uses `none` plus the four prior Qwen VLM enhancers, Wan2.1-T2V-1.3B at `832x480`, 81 frames, 16 fps, 50 steps, CFG 6.0, flow shift 8.0, seed base `430000`, and only six VBench dimensions that avoid detectron2/GRiT.

## 2026-06-06

- Confirmed FastVideo's `fastvideo.eval` source registers all 16 original VBench T2V metrics; none are missing, though default/example coverage and runtime dependencies split the metrics into 8 generic defaults, 4 detectron2/GRiT structured metrics, `scene` via the fuller Qwen/AVoCaDO path, and remaining prompt/fps-aware metrics.
- Retried and promoted staged HPSv3 plus VideoAlign evals for `outputs/baseline_sweep/wan21_13b_dancegrpo_vbench_rewards_v1`: active records now have 397 successful rows, zero failed eval rows, and trial 4 remains at 97 existing videos with no regeneration.
- Recorded a source-backed Wan2.1 T2V-1.3B generation-default comparison in infrastructure memory, including the caveat that only `outputs/baseline_sweep/wan21_13b_dancegrpo_vbench_rewards_v1/config.yaml` uses nonstandard `480x480`, `53` frame, `8` fps, `16` step settings.
- Cloned official source repo `Wan-Video/Wan2.1` into `Wan2.1/` at commit `9737cba9c1c3c4d04b33fcad41c111989865d315`; inspected `generate.py`, `wan/configs/`, and bundled Gradio scripts for generation defaults.
- Cloned `Wan-AI/Wan2.1-T2V-1.3B` into `Wan2.1-T2V-1.3B/` as a shallow LFS-pointer checkout at commit `37ec512624d61f7aa208f7ea8140a131f93afc9a`; inspected the model repo generation/default config surface.
- Patched the Wan2.1-1.3B Baseline Sweep video run before artifacts were written: VBench official prompts now dedupe duplicate prompt strings and merge dimensions, HPSv3 frame rewards are batched, and the sweep relaunched in tmux `baseline_sweep_wan21` across Slurm jobs `4882` and `4884`.
- Added Baseline Sweep wrappers for HPSv3, VideoAlign, and VBench plus generic HF image-text-to-text enhancer loading for Qwen2.5/Qwen3-VL variants; lightweight checks passed for syntax, config expansion, prompt counts, and diff whitespace.
- Cloned DanceGRPO at `evaluations/DanceGRPO` and prepared the Baseline Sweep Wan2.1-1.3B video reward config to load the top 100 VidProM-derived prompts from the DanceGRPO source file.
- Updated Baseline Sweep FastVideo generation so `workload_type`, `num_frames`, and `fps` come from YAML; artifact extensions now follow image/video workload type, and GenEval trials reject video workloads before generation.
- Added and launched the new Baseline Sweep suite under `experiments/baseline_sweep/`; the smoke run at `outputs/baseline_sweep/image_geneval_smoke_v1` completed with 12/12 successful GenEval records, paired seeds across `none` and `Qwen/Qwen2.5-VL-3B-Instruct`, and no failed records.
- Added root versioning policy for the experiment hub: track lightweight code, prompts, memory, research notes, external repository pins, and reward-server patches while ignoring `plan.md`, `reports/`, generated outputs, caches, environments, and nested third-party checkouts.

## 2026-06-05

- Added `/home/hal-jundas/codes/UniRL/papers/image_video_model_eval_report.md`, a polished synthesis of the image/video model evaluation notes from `papers/evals.md`, plus a high-level Notion ingestion plan for reading-list and eval-wiki updates.
- Added `memory/experiment_handoff.md` as the fast-start current-state handoff for completed, partial, and aborted image-calibration experiments; linked it from `.agents/README.md`.

## 2026-05-29

- Started the full PromptRL-style FLUX.1-dev run at `outputs/image_calibration/promptrl_full_flux1dev_20260529_000245`; it reuses byte-identical prompt ledgers from the SD3.5 source run and changes the generator to `black-forest-labs/FLUX.1-dev` via Diffusers `FluxPipeline`.

## 2026-05-28

- Added reward-server provenance to `memory/evaluations.md`: Flow-GRPO README section `3. Reward Preparation` -> `GenEval` points to `https://github.com/yifan123/reward-server`, which is why this workspace uses `evaluations/reward-server` for dependency-isolated GenEval scoring.
- Added `memory/evaluations.md` with an aligned inventory of every `evaluations/` subdirectory, git remote, workspace use, and local changes; linked it from the memory index and infrastructure notes.
- Replaced the SD3.5 generator-config provenance prose with a source/config comparison table that includes the actual run, FastVideo `sd35_medium`, FastVideo generic sampling, Diffusers, README, and ComfyUI settings.
- Added generator-config provenance for the full PromptRL-style run: `1024x1024`, 20 steps, and guidance 6.0 were experiment-local settings inherited from the pilot, not official SD3.5 defaults.
- Expanded the full PromptRL-style run memory with the exact rewrite prompt, no-system-prompt setting, prompt locations, benchmark prompt counts, and evaluator harness locations.
- Logged the full PromptRL-style SD3.5 Medium run at `outputs/image_calibration/promptrl_full_sd35_20260528_0439`: no rewrite versus `Qwen/Qwen2.5-VL-3B-Instruct`, all GenEval prompts plus OCR1k and PickScore-SFW, with summaries and `comparison_table.html`.
- Recorded PromptRL prompt-enhancement finding: the paper did not expose a system prompt, while the released code uses a user-template requesting realistic/detailed enhancement, clear entity separation/alignment, and `<answer>` output tags.
- Created `.agents` memory architecture and root `AGENTS.md` pointer.
- Added HTML prompt comparison report for the Qwen2.5-3B non-VL run at `outputs/image_calibration/pilot_sd35_balanced12_qwen25_3b/qwen25_3b_prompt_comparison.html`.
- Added HTML prompt comparison report for the Qwen2.5-VL-3B run at `outputs/image_calibration/pilot_sd35_balanced12/qwen25_vl_3b_prompt_comparison.html`.

## 2026-05-27

- Completed SD3.5 Medium image calibration pilot for no rewrite versus Qwen2.5-VL-3B and no rewrite versus Qwen2.5-3B.
- Recorded prompt ledgers, generated images, preference metrics, GenEval server results, summaries, and run review files under `outputs/image_calibration/`.
- Confirmed the 12-prompt pilot was only a balanced GenEval subset and not a full PromptRL reproduction.
