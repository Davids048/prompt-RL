#!/usr/bin/env bash
set -euo pipefail

# Launch-only OCR evaluation entry point. The script creates the run directory
# before torchrun so stdout/stderr and evaluation artifacts share one location.

REPO_ROOT="/mnt/weka/home/junda.su/codes/prompt-RL/DiffusionNFT"
RUN_ID="${RUN_ID:-ocr_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/evaluation_output/${RUN_ID}}"
OUT_LOG="${OUTPUT_DIR}/launch.out"
ERR_LOG="${OUTPUT_DIR}/launch.err"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
LORA_HF_PATH="${LORA_HF_PATH:-worstcoder/SD3.5M-DiffusionNFT-MultiReward}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-1.0}"
MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-40}"
RESOLUTION="${RESOLUTION:-512}"

cd "${REPO_ROOT}"

if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env"
  set +a
fi

if [[ ! -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  echo "Missing Python environment: ${REPO_ROOT}/.venv" >&2
  exit 1
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set. Add it to ${REPO_ROOT}/.env or export it before launching." >&2
  exit 1
fi

"${REPO_ROOT}/.venv/bin/python" - <<'PY'
import importlib.util
import sys

missing = [
    name
    for name in ("paddle", "paddleocr", "Levenshtein")
    if importlib.util.find_spec(name) is None
]

if missing:
    print(
        "Missing OCR dependencies: "
        + ", ".join(missing)
        + ". Install paddlepaddle-gpu==2.6.2 paddleocr==2.9.1 python-Levenshtein first.",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY

mkdir -p "${OUTPUT_DIR}"

export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-${REPO_ROOT}/.cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${REPO_ROOT}/.cache}"
export PADDLEOCR_HOME="${PADDLEOCR_HOME:-${REPO_ROOT}/.cache/paddleocr}"
export CUDA_VISIBLE_DEVICES

echo "OCR eval output directory: ${OUTPUT_DIR}"
echo "stdout log: ${OUT_LOG}"
echo "stderr log: ${ERR_LOG}"

{
  echo "Launch time: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "Repo: ${REPO_ROOT}"
  echo "Output: ${OUTPUT_DIR}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
  echo "LORA_HF_PATH=${LORA_HF_PATH}"
  echo "GUIDANCE_SCALE=${GUIDANCE_SCALE}"
  echo "MIXED_PRECISION=${MIXED_PRECISION}"
  echo "NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS}"
  echo "RESOLUTION=${RESOLUTION}"
  echo

  "${REPO_ROOT}/.venv/bin/torchrun" --nproc_per_node="${NPROC_PER_NODE}" \
    scripts/evaluation.py \
    --lora_hf_path "${LORA_HF_PATH}" \
    --model_type sd3 \
    --dataset ocr \
    --guidance_scale "${GUIDANCE_SCALE}" \
    --mixed_precision "${MIXED_PRECISION}" \
    --num_inference_steps "${NUM_INFERENCE_STEPS}" \
    --resolution "${RESOLUTION}" \
    --output_dir "${OUTPUT_DIR}" \
    --save_images
} > >(tee "${OUT_LOG}") 2> >(tee "${ERR_LOG}" >&2)
