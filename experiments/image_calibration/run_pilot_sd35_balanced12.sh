#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/hal-jundas/codes/UniRL}"
JOB_ID="${JOB_ID:-4745}"
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/outputs/image_calibration/pilot_sd35_balanced12}"
METADATA="${METADATA:-${REPO_ROOT}/experiments/image_calibration/geneval_balanced_12.jsonl}"
CONDITIONS="${CONDITIONS:-none,qwen25_vl_3b}"
HEIGHT="${HEIGHT:-1024}"
WIDTH="${WIDTH:-1024}"
STEPS="${STEPS:-20}"
GUIDANCE="${GUIDANCE:-6.0}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-FLASH_ATTN}"
METRICS="${METRICS:-pickscore,hps,ur}"
GENEVAL_URL="${GENEVAL_URL:-http://127.0.0.1:18085}"
GENEVAL_TIMEOUT="${GENEVAL_TIMEOUT:-900}"
GENEVAL_NUM_DEVICES="${GENEVAL_NUM_DEVICES:-1}"

case "${RUN_DIR}" in
  /*) ;;
  *) RUN_DIR="${REPO_ROOT}/${RUN_DIR}" ;;
esac

case "${METADATA}" in
  /*) ;;
  *) METADATA="${REPO_ROOT}/${METADATA}" ;;
esac

export VIRTUAL_ENV="${REPO_ROOT}/.venv"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
export HF_HOME="${REPO_ROOT}/.cache/huggingface"
export TRANSFORMERS_CACHE="${REPO_ROOT}/.cache/huggingface/transformers"
if [[ -f /home/hal-jundas/.cache/huggingface/token ]]; then
  export HF_TOKEN="$(cat /home/hal-jundas/.cache/huggingface/token)"
fi

echo "[probe] checking Slurm overlap job ${JOB_ID}"
srun --overlap --jobid="${JOB_ID}" nvidia-smi -L

echo "[stage] rewrite, generate, preference metrics"
srun --overlap --jobid="${JOB_ID}" bash -lc "
set -euo pipefail
cd '${REPO_ROOT}'
export VIRTUAL_ENV='${REPO_ROOT}/.venv'
export PATH=\"\${VIRTUAL_ENV}/bin:\${PATH}\"
export HF_HOME='${HF_HOME}'
export TRANSFORMERS_CACHE='${TRANSFORMERS_CACHE}'
export FASTVIDEO_ATTENTION_BACKEND='${ATTENTION_BACKEND}'
if [[ -f /home/hal-jundas/.cache/huggingface/token ]]; then
  export HF_TOKEN=\"\$(cat /home/hal-jundas/.cache/huggingface/token)\"
fi
python experiments/image_calibration/calibration_pipeline.py rewrite \
  --run-dir '${RUN_DIR}' \
  --metadata '${METADATA}' \
  --conditions '${CONDITIONS}' \
  --sample-count 1 \
  --seed-base 42 \
  --temperature 0.0 \
  --max-new-tokens 128 \
  --overwrite
python experiments/image_calibration/calibration_pipeline.py generate \
  --run-dir '${RUN_DIR}' \
  --height '${HEIGHT}' \
  --width '${WIDTH}' \
  --steps '${STEPS}' \
  --guidance '${GUIDANCE}' \
  --attention-backend '${ATTENTION_BACKEND}' \
  --overwrite
python experiments/image_calibration/calibration_pipeline.py eval-preference \
  --run-dir '${RUN_DIR}' \
  --metrics '${METRICS}' \
  --metric-prompt original \
  --device cuda \
  --overwrite
python experiments/image_calibration/calibration_pipeline.py summarize \
  --run-dir '${RUN_DIR}'
"

echo "[stage] GenEval reward server and scoring"
srun --overlap --jobid="${JOB_ID}" bash -lc "
set -euo pipefail
mkdir -p '${RUN_DIR}'
cd '${REPO_ROOT}/evaluations/reward-server'
export MPLCONFIGDIR=/tmp/matplotlib-reward-server
export GENEVAL_DEVICE=cuda
export GENEVAL_NUM_DEVICES='${GENEVAL_NUM_DEVICES}'
export HF_HOME='${HF_HOME}'
export TRANSFORMERS_CACHE='${TRANSFORMERS_CACHE}'
if [[ -f /home/hal-jundas/.cache/huggingface/token ]]; then
  export HF_TOKEN=\"\$(cat /home/hal-jundas/.cache/huggingface/token)\"
fi
'${REPO_ROOT}/.envs/reward_server/bin/gunicorn' 'app_geneval:create_app()' > '${RUN_DIR}/geneval_server.log' 2>&1 &
server_pid=\$!
trap 'kill \${server_pid} >/dev/null 2>&1 || true' EXIT
python - <<'PY'
import socket
import time
from urllib.parse import urlparse

parsed = urlparse('${GENEVAL_URL}')
host = parsed.hostname or '127.0.0.1'
port = parsed.port or 80
deadline = time.time() + ${GENEVAL_TIMEOUT}
while time.time() < deadline:
    try:
        with socket.create_connection((host, port), timeout=2):
            break
    except OSError:
        time.sleep(5)
else:
    raise SystemExit('GenEval server did not become reachable before timeout')
PY
cd '${REPO_ROOT}'
export VIRTUAL_ENV='${REPO_ROOT}/.venv'
export PATH=\"\${VIRTUAL_ENV}/bin:\${PATH}\"
python experiments/image_calibration/calibration_pipeline.py eval-geneval \
  --run-dir '${RUN_DIR}' \
  --server-url '${GENEVAL_URL}' \
  --timeout '${GENEVAL_TIMEOUT}' \
  --overwrite
python experiments/image_calibration/calibration_pipeline.py summarize \
  --run-dir '${RUN_DIR}'
"

echo "[done] ${RUN_DIR}"
