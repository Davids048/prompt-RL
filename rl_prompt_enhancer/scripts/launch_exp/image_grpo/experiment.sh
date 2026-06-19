#!/usr/bin/env bash
# Shared run manifest for FastVideo image-GRPO prompt-enhancer launches.
#
# Copy this file and the numbered launch scripts into:
#   experiments/<run_id>/commands/
#
# Edit the run-local copies before launching. Do not launch from this source
# profile directory.

COMMANDS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${COMMANDS_DIR}/.." && pwd)"
RUN_NAME="$(basename "${RUN_DIR}")"

REPO_ROOT="/home/hal-jundas/codes/UniRL"
PYTHON="${REPO_ROOT}/.venv/bin/python"
SLIME_ROOT="${REPO_ROOT}/slime"
FASTVIDEO_ROOT="${REPO_ROOT}/FastVideo"
MEGATRON_ROOT="${REPO_ROOT}/Megatron-LM"

EXPERIMENT_GOAL="Run image-only GRPO for one epoch over the full DiffusionNFT PickScore prompt set with group size 8."
EXPERIMENT_SETUP="Uses Qwen/Qwen3.5-9B, torch-dist checkpoint, full DiffusionNFT PickScore prompts, FastVideo, PickScore, CLIPScore, and a validation baseline before training."
EXPERIMENT_CLASS="real experiment"

SLIME_JOB_ID="4884"
SLIME_NODELIST="hpc-rack-2-9"
SLIME_CUDA_VISIBLE_DEVICES="0,1,2,3"
SLIME_NUM_GPUS="4"

FASTVIDEO_JOB_ID="4882"
FASTVIDEO_NODELIST="hpc-rack-2-3"
FASTVIDEO_PUBLIC_HOST="hpc-rack-2-3"
FASTVIDEO_SERVICE_HOST="0.0.0.0"
FASTVIDEO_SERVICE_PORTS=(18080 18081 18082 18083)

RAY_MASTER_ADDR="127.0.0.1"
RAY_ADDRESS="http://127.0.0.1:8265"
RAY_PORT="6385"
RAY_DASHBOARD_PORT="8265"
RAY_CLIENT_SERVER_PORT="10001"
RAY_TEMP_DIR="/tmp/rlpe2_ray"

WANDB_PROJECT="prompt RL"
WANDB_GROUP="${RUN_NAME}"
WANDB_DISABLE_RANDOM_SUFFIX="1"
WANDB_DIR="${RUN_DIR}/wandb"

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  cat <<CONTRACT
Goal: ${EXPERIMENT_GOAL}
Setup: ${EXPERIMENT_SETUP}
Launch:
  bash ${COMMANDS_DIR}/01_start_ray_head.sh
  bash ${COMMANDS_DIR}/02_start_fastvideo_workers.sh
  bash ${COMMANDS_DIR}/03_submit_slime_grpo.sh
Result: ${RUN_DIR}
Class: ${EXPERIMENT_CLASS}
CONTRACT
fi
