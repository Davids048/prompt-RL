#!/usr/bin/env bash
set -euo pipefail

# Launch-only GenEval evaluation entry point. This script assumes the uv
# environment, model caches, reward ckpts, and dataset already exist.

# Repository and environment locations.
REPO_ROOT="/mnt/weka/home/junda.su/codes/prompt-RL/DiffusionNFT"
MAIN_VENV="${MAIN_VENV:-${REPO_ROOT}/.venv}"

# Run artifact locations.
RUN_ID="${RUN_ID:-geneval_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/evaluation_output/${RUN_ID}}"
OUT_LOG="${OUTPUT_DIR}/launch.out"
ERR_LOG="${OUTPUT_DIR}/launch.err"

# Distributed evaluation topology.
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29518}"
EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"

# GenEval inference configuration for evaluating the same MultiReward LoRA.
LORA_HF_PATH="${LORA_HF_PATH:-worstcoder/SD3.5M-DiffusionNFT-MultiReward}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-1.0}"
MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-40}"
RESOLUTION="${RESOLUTION:-512}"

# Enter the repo and load user-provided secrets or launch overrides.
cd "${REPO_ROOT}"
if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env"
  set +a
fi

# Keep GenEval color crop loading in-process; worker subprocesses can crash
# after CUDA models have already been initialized in each torchrun rank.
readonly GENEVAL_COLOR_NUM_WORKERS=0

# Create the run directory before tee so launch logs live beside results.
mkdir -p "${OUTPUT_DIR}"
exec > >(tee "${OUT_LOG}") 2> >(tee "${ERR_LOG}" >&2)

echo "===== PRE-LAUNCH VERIFICATION: assumes setup is already complete ====="
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set. Add it to ${REPO_ROOT}/.env or export it before launching." >&2
  exit 1
fi

if [[ ! -f "${REPO_ROOT}/dataset/geneval/test_metadata.jsonl" ]]; then
  echo "Missing GenEval metadata file: ${REPO_ROOT}/dataset/geneval/test_metadata.jsonl" >&2
  exit 1
fi

GENEVAL_PROMPTS="$(awk 'NF {count += 1} END {print count + 0}' "${REPO_ROOT}/dataset/geneval/test_metadata.jsonl")"
echo "geneval_prompts=${GENEVAL_PROMPTS}"

# Runtime environment shared by torchrun and the reward/model loaders.
export PATH="${MAIN_VENV}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-${REPO_ROOT}/.cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${REPO_ROOT}/.cache}"
export CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES}"
export GENEVAL_COLOR_NUM_WORKERS

echo "Pre-launch verification passed."
echo

echo "===== GENEVAL EVALUATION LAUNCH ====="
echo "Launch time: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Repo: ${REPO_ROOT}"
echo "Output: ${OUTPUT_DIR}"
echo "stdout log: ${OUT_LOG}"
echo "stderr log: ${ERR_LOG}"
echo "CUDA_VISIBLE_DEVICES=${EVAL_CUDA_VISIBLE_DEVICES}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "LORA_HF_PATH=${LORA_HF_PATH}"
echo "GUIDANCE_SCALE=${GUIDANCE_SCALE}"
echo "MIXED_PRECISION=${MIXED_PRECISION}"
echo "NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS}"
echo "RESOLUTION=${RESOLUTION}"
echo "GENEVAL_COLOR_NUM_WORKERS=${GENEVAL_COLOR_NUM_WORKERS}"
echo

"${MAIN_VENV}/bin/torchrun" --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  scripts/evaluation.py \
  --lora_hf_path "${LORA_HF_PATH}" \
  --model_type sd3 \
  --dataset geneval \
  --guidance_scale "${GUIDANCE_SCALE}" \
  --mixed_precision "${MIXED_PRECISION}" \
  --num_inference_steps "${NUM_INFERENCE_STEPS}" \
  --resolution "${RESOLUTION}" \
  --output_dir "${OUTPUT_DIR}" \
  --save_images

echo
echo "GenEval eval complete."
echo "Results: ${OUTPUT_DIR}/evaluation_results.jsonl"
echo "Averages: ${OUTPUT_DIR}/average_scores.json"
