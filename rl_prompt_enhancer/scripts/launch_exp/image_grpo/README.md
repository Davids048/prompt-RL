# FastVideo Image-GRPO Launch Profile

This directory is a copyable source default profile. Do not launch these files
in place.

The run-local command files should live under:

```text
/home/hal-jundas/codes/UniRL/experiments/<run_id>/commands/
```

## Runbook

State the experiment contract before creating a real run:

```text
Goal: <research question/action>
Setup: <setup state and evidence location>
Launch: bash commands/01_start_ray_head.sh, bash commands/02_start_fastvideo_workers.sh, bash commands/03_submit_slime_grpo.sh
Result: /home/hal-jundas/codes/UniRL/experiments/<run_id>
Class: real experiment
```

Create the run directory and copy the source defaults:

```bash
RUN_DIR=/home/hal-jundas/codes/UniRL/experiments/<run_id>
PROFILE=/home/hal-jundas/codes/UniRL/rl_prompt_enhancer/scripts/launch_exp/image_grpo
CONFIGS=/home/hal-jundas/codes/UniRL/rl_prompt_enhancer/configs

mkdir -p "${RUN_DIR}/commands" "${RUN_DIR}/configs" "${RUN_DIR}/logs"
mkdir -p "${RUN_DIR}/snapshot/default_profile/commands" "${RUN_DIR}/snapshot/default_profile/configs"
cp "${PROFILE}/experiment.sh" "${RUN_DIR}/commands/"
cp "${PROFILE}/01_start_ray_head.sh" "${RUN_DIR}/commands/"
cp "${PROFILE}/02_start_fastvideo_workers.sh" "${RUN_DIR}/commands/"
cp "${PROFILE}/03_submit_slime_grpo.sh" "${RUN_DIR}/commands/"
cp "${CONFIGS}/fastvideo_image_grpo_generator.json" "${RUN_DIR}/configs/"
cp "${CONFIGS}/fastvideo_image_grpo_validation.yaml" "${RUN_DIR}/configs/"
cp "${PROFILE}/"*.sh "${RUN_DIR}/snapshot/default_profile/commands/"
cp "${CONFIGS}/fastvideo_image_grpo_generator.json" "${RUN_DIR}/snapshot/default_profile/configs/"
cp "${CONFIGS}/fastvideo_image_grpo_validation.yaml" "${RUN_DIR}/snapshot/default_profile/configs/"
```

Edit the run-local copies directly:

```text
commands/experiment.sh
commands/01_start_ray_head.sh
commands/02_start_fastvideo_workers.sh
commands/03_submit_slime_grpo.sh
configs/fastvideo_image_grpo_generator.json
configs/fastvideo_image_grpo_validation.yaml
```

Before launching, snapshot the exact scripts that will be used:

```bash
mkdir -p "${RUN_DIR}/snapshot/used_commands/attempt_001/commands"
mkdir -p "${RUN_DIR}/snapshot/used_commands/attempt_001/configs"
cp "${RUN_DIR}/commands/"*.sh "${RUN_DIR}/snapshot/used_commands/attempt_001/commands/"
cp "${RUN_DIR}/configs/"* "${RUN_DIR}/snapshot/used_commands/attempt_001/configs/"
bash -n "${RUN_DIR}/commands/"*.sh
```

Launch from the run directory in separate terminal or tmux panes:

```bash
cd "${RUN_DIR}"
bash commands/01_start_ray_head.sh
bash commands/02_start_fastvideo_workers.sh
bash commands/03_submit_slime_grpo.sh
```

For relaunches, create a new used-command snapshot such as
`snapshot/used_commands/attempt_002/` before running again.

## Placement Rules

- `experiment.sh`: shared run identity, Slurm topology, Ray address, FastVideo
  host/ports, W&B target, and common repository paths.
- `01_start_ray_head.sh`: Ray head command and Slime-node launch mechanics.
- `02_start_fastvideo_workers.sh`: FastVideo service command, generation model,
  worker GPUs, reward models, and reward weights.
- `03_submit_slime_grpo.sh`: prompt data, prompt-enhancer checkpoint, validation
  config paths, generator config path, and Slime GRPO training flags.
- `configs/fastvideo_image_grpo_generator.json`: FastVideo request settings used
  by the Slime custom-generate hook.
- `configs/fastvideo_image_grpo_validation.yaml`: Slime eval config for the
  validation baseline.

If a value only matters to one launch action, keep it in that action script.
Do not add a materializer script or a separate override registry.
