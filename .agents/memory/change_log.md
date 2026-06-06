# Change Log

## 2026-06-06

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
