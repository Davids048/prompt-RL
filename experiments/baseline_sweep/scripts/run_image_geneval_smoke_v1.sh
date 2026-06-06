#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/hal-jundas/codes/UniRL}"
JOB_ID="${JOB_ID:-4882}"
CONFIG="${CONFIG:-${REPO_ROOT}/experiments/baseline_sweep/configs/smoke/image_geneval_smoke_v1.yaml}"
GENEVAL_URL="${GENEVAL_URL:-http://127.0.0.1:18085}"
GENEVAL_NUM_DEVICES="${GENEVAL_NUM_DEVICES:-1}"
GENEVAL_TIMEOUT="${GENEVAL_TIMEOUT:-900}"

srun --overlap --jobid="${JOB_ID}" bash -lc "
set -euo pipefail
cd '${REPO_ROOT}'
export VIRTUAL_ENV='${REPO_ROOT}/.venv'
export PATH=\"\${VIRTUAL_ENV}/bin:\${PATH}\"
export HF_HOME='${REPO_ROOT}/.cache/huggingface'
export TRANSFORMERS_CACHE='${REPO_ROOT}/.cache/huggingface/transformers'
export GENEVAL_URL='${GENEVAL_URL}'
export GENEVAL_TIMEOUT='${GENEVAL_TIMEOUT}'
export GENEVAL_DEVICE=cuda
export GENEVAL_NUM_DEVICES='${GENEVAL_NUM_DEVICES}'
export GENEVAL_GUNICORN='${REPO_ROOT}/.envs/reward_server/bin/gunicorn'
export GENEVAL_SERVER_CWD='${REPO_ROOT}/evaluations/reward-server'
export GENEVAL_SERVER_LOG='${REPO_ROOT}/outputs/baseline_sweep/geneval_reward_server_${JOB_ID}.log'
export MPLCONFIGDIR=/tmp/matplotlib-reward-server
if [[ -f /home/hal-jundas/.cache/huggingface/token ]]; then
  export HF_TOKEN=\"\$(cat /home/hal-jundas/.cache/huggingface/token)\"
fi

mkdir -p outputs/baseline_sweep
cd '${REPO_ROOT}'
python experiments/baseline_sweep/src/cli.py run '${CONFIG}'
"
