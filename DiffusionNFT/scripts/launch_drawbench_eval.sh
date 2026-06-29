#!/usr/bin/env bash
set -euo pipefail

# Launch-only DrawBench evaluation entry point. This script assumes the uv
# environments, model caches, reward ckpts, and dataset already exist.

# Repository and environment locations.
REPO_ROOT="/mnt/weka/home/junda.su/codes/prompt-RL/DiffusionNFT"
MAIN_VENV="${MAIN_VENV:-${REPO_ROOT}/.venv}"
SGLANG_VENV="${SGLANG_VENV:-${REPO_ROOT}/.venv-sglang}"

# Run artifact locations.
RUN_ID="${RUN_ID:-drawbench_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/evaluation_output/${RUN_ID}}"
OUT_LOG="${OUTPUT_DIR}/launch.out"
ERR_LOG="${OUTPUT_DIR}/launch.err"
UNIFIEDREWARD_LOG="${OUTPUT_DIR}/unifiedreward_server.log"

# Distributed evaluation topology.
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29517}"
EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
UNIFIEDREWARD_CUDA_VISIBLE_DEVICES="${UNIFIEDREWARD_CUDA_VISIBLE_DEVICES:-7}"

# DrawBench inference configuration matching the first reproduction run.
LORA_HF_PATH="${LORA_HF_PATH:-worstcoder/SD3.5M-DiffusionNFT-MultiReward}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-1.0}"
MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-40}"
RESOLUTION="${RESOLUTION:-512}"

# UnifiedReward service settings. The scorer code hard-codes this local
# endpoint with api_key=flowgrpo, so the launcher always starts that service.
UNIFIEDREWARD_ENDPOINT="http://127.0.0.1:17140/v1/models"
UNIFIEDREWARD_MODEL_PATH="${UNIFIEDREWARD_MODEL_PATH:-CodeGoat24/UnifiedReward-7b-v1.5}"
UNIFIEDREWARD_MEM_FRACTION_STATIC="${UNIFIEDREWARD_MEM_FRACTION_STATIC:-0.25}"
UNIFIEDREWARD_MAX_RUNNING_REQUESTS="${UNIFIEDREWARD_MAX_RUNNING_REQUESTS:-8}"
UNIFIEDREWARD_STARTUP_TIMEOUT="${UNIFIEDREWARD_STARTUP_TIMEOUT:-900}"
UNIFIEDREWARD_PID=""

# Enter the repo and load user-provided secrets or launch overrides.
cd "${REPO_ROOT}"
if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env"
  set +a
fi

# Create the run directory before tee so launch logs live beside results.
mkdir -p "${OUTPUT_DIR}"
exec > >(tee "${OUT_LOG}") 2> >(tee "${ERR_LOG}" >&2)

# Helper functions for the UnifiedReward service started by this launcher.
is_unifiedreward_ready() {
  : "Check the fixed local UnifiedReward HTTP endpoint."
  "${MAIN_VENV}/bin/python" - <<'PY'
import sys
import urllib.request

try:
    request = urllib.request.Request(
        "http://127.0.0.1:17140/v1/models",
        headers={"Authorization": "Bearer flowgrpo"},
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        response.read()
except Exception:
    raise SystemExit(1)

raise SystemExit(0)
PY
}

cleanup_unifiedreward() {
  : "Stop only the UnifiedReward server started by this launcher."
  if [[ -n "${UNIFIEDREWARD_PID}" ]] && kill -0 "${UNIFIEDREWARD_PID}" 2>/dev/null; then
    echo "Stopping UnifiedReward server pid ${UNIFIEDREWARD_PID}"
    kill "${UNIFIEDREWARD_PID}" 2>/dev/null || true
    wait "${UNIFIEDREWARD_PID}" 2>/dev/null || true
  fi
}
trap cleanup_unifiedreward EXIT

wait_for_unifiedreward() {
  : "Poll the fixed UnifiedReward endpoint and fail with server logs on startup errors."
  local deadline=$((SECONDS + UNIFIEDREWARD_STARTUP_TIMEOUT))

  while (( SECONDS < deadline )); do
    if is_unifiedreward_ready; then
      echo "UnifiedReward endpoint is ready: ${UNIFIEDREWARD_ENDPOINT}"
      return 0
    fi

    if [[ -n "${UNIFIEDREWARD_PID}" ]] && ! kill -0 "${UNIFIEDREWARD_PID}" 2>/dev/null; then
      echo "UnifiedReward server exited before becoming ready. Last log lines:" >&2
      tail -n 80 "${UNIFIEDREWARD_LOG}" >&2 || true
      exit 1
    fi

    sleep 5
  done

  echo "Timed out waiting for UnifiedReward after ${UNIFIEDREWARD_STARTUP_TIMEOUT}s. Last log lines:" >&2
  tail -n 80 "${UNIFIEDREWARD_LOG}" >&2 || true
  exit 1
}

echo "===== PRE-LAUNCH VERIFICATION: assumes setup is already complete ====="
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set. Add it to ${REPO_ROOT}/.env or export it before launching." >&2
  exit 1
fi

if [[ ! -f "${REPO_ROOT}/dataset/drawbench/test.txt" ]]; then
  echo "Missing DrawBench prompt file: ${REPO_ROOT}/dataset/drawbench/test.txt" >&2
  exit 1
fi

DRAWBENCH_PROMPTS="$(awk 'NF {count += 1} END {print count + 0}' "${REPO_ROOT}/dataset/drawbench/test.txt")"
echo "drawbench_prompts=${DRAWBENCH_PROMPTS}"

# Runtime environment shared by torchrun and the reward/model loaders.
export PATH="${MAIN_VENV}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-${REPO_ROOT}/.cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${REPO_ROOT}/.cache}"
export CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES}"

echo "Pre-launch verification passed."
echo

echo "===== LAUNCH-TIME UNIFIEDREWARD SERVICE ====="
echo "Starting UnifiedReward on CUDA_VISIBLE_DEVICES=${UNIFIEDREWARD_CUDA_VISIBLE_DEVICES}"
echo "UnifiedReward log: ${UNIFIEDREWARD_LOG}"
# Mirror the server's combined stdout/stderr to the terminal, launch.out, and
# the dedicated UnifiedReward log while keeping the Python process as $!.
(
  export PATH="${SGLANG_VENV}/bin:${PATH}"
  export CUDA_VISIBLE_DEVICES="${UNIFIEDREWARD_CUDA_VISIBLE_DEVICES}"
  exec "${SGLANG_VENV}/bin/python" -m sglang.launch_server \
    --model-path "${UNIFIEDREWARD_MODEL_PATH}" \
    --api-key flowgrpo \
    --port 17140 \
    --chat-template chatml-llava \
    --enable-p2p-check \
    --mem-fraction-static "${UNIFIEDREWARD_MEM_FRACTION_STATIC}" \
    --max-running-requests "${UNIFIEDREWARD_MAX_RUNNING_REQUESTS}" \
    2>&1
) > >(tee "${UNIFIEDREWARD_LOG}") &
UNIFIEDREWARD_PID=$!
wait_for_unifiedreward
echo

echo "===== DRAWBENCH EVALUATION LAUNCH ====="
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
echo

export CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES}"

"${MAIN_VENV}/bin/torchrun" --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  scripts/evaluation.py \
  --lora_hf_path "${LORA_HF_PATH}" \
  --model_type sd3 \
  --dataset drawbench \
  --guidance_scale "${GUIDANCE_SCALE}" \
  --mixed_precision "${MIXED_PRECISION}" \
  --num_inference_steps "${NUM_INFERENCE_STEPS}" \
  --resolution "${RESOLUTION}" \
  --output_dir "${OUTPUT_DIR}" \
  --save_images

echo
echo "DrawBench eval complete."
echo "Results: ${OUTPUT_DIR}/evaluation_results.jsonl"
echo "Averages: ${OUTPUT_DIR}/average_scores.json"
