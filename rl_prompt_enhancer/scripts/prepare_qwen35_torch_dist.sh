#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/hal-jundas/codes/UniRL}"
SLIME_ROOT="${SLIME_ROOT:-${REPO_ROOT}/slime}"
MEGATRON_ROOT="${MEGATRON_ROOT:-${REPO_ROOT}/Megatron-LM}"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-9B}"
MODEL_NAME="${MODEL_NAME:-Qwen3.5-9B}"
MODEL_TYPE="${MODEL_TYPE:-qwen3.5-9B}"
MODEL_DIR="${MODEL_DIR:-${REPO_ROOT}/.cache/rl_prompt_enhancer/phase1/models/${MODEL_NAME}}"
TORCH_DIST_CKPT="${TORCH_DIST_CKPT:-/dev/shm/${MODEL_NAME}_torch_dist}"
NUM_GPUS="${NUM_GPUS:-4}"
RUN_ID="${RUN_ID:-qwen35_torch_dist_$(date -u +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/experiments/env_setup/rl_prompt_enhancer/${RUN_ID}}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/summaries"
{
  echo "Goal: Prepare Qwen3.5-9B torch-dist checkpoint for RL prompt-enhancer launches."
  echo "Setup: ${RUN_DIR}"
  echo "Launch: ${REPO_ROOT}/rl_prompt_enhancer/scripts/prepare_qwen35_torch_dist.sh"
  echo "Result: ${TORCH_DIST_CKPT}"
  echo "Class: setup"
} | tee "${RUN_DIR}/summaries/experiment_contract.txt"

export VIRTUAL_ENV="${REPO_ROOT}/.venv"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}:${SLIME_ROOT}:${REPO_ROOT}/FastVideo:${PYTHONPATH:-}"

{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "repo_root=${REPO_ROOT}"
  echo "slime_root=${SLIME_ROOT}"
  echo "megatron_root=${MEGATRON_ROOT}"
  echo "model_id=${MODEL_ID}"
  echo "model_dir=${MODEL_DIR}"
  echo "torch_dist_ckpt=${TORCH_DIST_CKPT}"
  echo "num_gpus=${NUM_GPUS}"
} > "${RUN_DIR}/logs/settings.env"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "dry-run: would prepare ${TORCH_DIST_CKPT}"
  exit 0
fi

for path in "${SLIME_ROOT}" "${MEGATRON_ROOT}" "${PYTHON}"; do
  if [[ ! -e "${path}" ]]; then
    echo "missing required path: ${path}" >&2
    exit 1
  fi
done

if [[ ! -e "${MODEL_DIR}/config.json" ]]; then
  if ! command -v hf >/dev/null 2>&1; then
    echo "missing model at ${MODEL_DIR} and no hf command is available" >&2
    exit 1
  fi
  mkdir -p "${MODEL_DIR}"
  hf download "${MODEL_ID}" --local-dir "${MODEL_DIR}" 2>&1 | tee "${RUN_DIR}/logs/download_model.log"
fi

if [[ -d "${TORCH_DIST_CKPT}" ]] && find "${TORCH_DIST_CKPT}" -mindepth 1 -print -quit | grep -q .; then
  echo "torch-dist checkpoint already present: ${TORCH_DIST_CKPT}"
else
  mkdir -p "$(dirname "${TORCH_DIST_CKPT}")"
  (
    cd "${SLIME_ROOT}"
    source "${SLIME_ROOT}/scripts/models/${MODEL_TYPE}.sh"
    PYTHONPATH="${MEGATRON_ROOT}:${PYTHONPATH:-}" \
      torchrun \
        --nproc-per-node "${NUM_GPUS}" \
        tools/convert_hf_to_torch_dist.py \
        "${MODEL_ARGS[@]}" \
        --hf-checkpoint "${MODEL_DIR}" \
        --save "${TORCH_DIST_CKPT}"
  ) 2>&1 | tee "${RUN_DIR}/logs/convert_hf_to_torch_dist.log"
fi

{
  echo "# Qwen3.5 Torch-Dist Setup"
  echo
  echo "- Model: \`${MODEL_DIR}\`"
  echo "- Torch-dist checkpoint: \`${TORCH_DIST_CKPT}\`"
  echo "- Megatron root: \`${MEGATRON_ROOT}\`"
  echo "- GPUs: \`${NUM_GPUS}\`"
} > "${RUN_DIR}/summaries/summary.md"

echo "setup complete: ${RUN_DIR}"
