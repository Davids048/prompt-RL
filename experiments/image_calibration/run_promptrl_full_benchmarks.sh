#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/hal-jundas/codes/UniRL}"
JOB_ID="${JOB_ID:-4745}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/outputs/image_calibration/promptrl_full_sd35_$(date -u +%Y%m%d_%H%M%S)}"
CONDITIONS="${CONDITIONS:-none,qwen25_vl_3b}"
PROMPTRL_TEMPLATE="${PROMPTRL_TEMPLATE:-${REPO_ROOT}/experiments/image_calibration/prompts/promptrl_geneval_enhance_user.txt}"
HEIGHT="${HEIGHT:-1024}"
WIDTH="${WIDTH:-1024}"
STEPS="${STEPS:-20}"
GUIDANCE="${GUIDANCE:-6.0}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-FLASH_ATTN}"
GENEVAL_URL="${GENEVAL_URL:-http://127.0.0.1:18085}"
GENEVAL_TIMEOUT="${GENEVAL_TIMEOUT:-900}"
GENEVAL_NUM_DEVICES="${GENEVAL_NUM_DEVICES:-1}"
UR_URL="${UR_URL:-http://127.0.0.1:17140}"
UR_TIMEOUT="${UR_TIMEOUT:-900}"
SGLANG_PYTHON="${SGLANG_PYTHON:-${REPO_ROOT}/.envs/sglang/bin/python}"

GENEVAL_METADATA="${GENEVAL_METADATA:-${REPO_ROOT}/evaluations/geneval/prompts/evaluation_metadata.jsonl}"
OCR_PROMPTS="${OCR_PROMPTS:-${REPO_ROOT}/evaluations/flow_grpo/dataset/ocr/test.txt}"
PICKSCORE_PROMPTS="${PICKSCORE_PROMPTS:-${REPO_ROOT}/evaluations/flow_grpo/dataset/pickscore_sfw/test.txt}"

case "${RUN_ROOT}" in
  /*) ;;
  *) RUN_ROOT="${REPO_ROOT}/${RUN_ROOT}" ;;
esac

export VIRTUAL_ENV="${REPO_ROOT}/.venv"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
export HF_HOME="${REPO_ROOT}/.cache/huggingface"
export TRANSFORMERS_CACHE="${REPO_ROOT}/.cache/huggingface/transformers"
export CUDAHOSTCXX="${CUDAHOSTCXX:-/home/shared-bin/local/usr/bin/g++}"
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:---compiler-bindir /home/shared-bin/local/usr/bin/g++}"
if [[ -f /home/hal-jundas/.cache/huggingface/token ]]; then
  export HF_TOKEN="$(cat /home/hal-jundas/.cache/huggingface/token)"
fi

GENEVAL_RUN="${RUN_ROOT}/geneval_full"
OCR_RUN="${RUN_ROOT}/ocr1k"
PICKSCORE_RUN="${RUN_ROOT}/pickscore_sfw"
OCR_METADATA="${RUN_ROOT}/metadata/ocr1k.jsonl"
PICKSCORE_METADATA="${RUN_ROOT}/metadata/pickscore_sfw.jsonl"
MANIFEST="${RUN_ROOT}/run_manifest.json"

mkdir -p "${RUN_ROOT}/metadata"

python experiments/image_calibration/calibration_pipeline.py import-prompts \
  --input "${OCR_PROMPTS}" \
  --output "${OCR_METADATA}" \
  --benchmark-name ocr1k \
  --overwrite

python experiments/image_calibration/calibration_pipeline.py import-prompts \
  --input "${PICKSCORE_PROMPTS}" \
  --output "${PICKSCORE_METADATA}" \
  --benchmark-name pickscore_sfw \
  --overwrite

cat > "${MANIFEST}" <<JSON
{
  "run_root": "${RUN_ROOT}",
  "conditions": "${CONDITIONS}",
  "prompt_enhancer_template": "${PROMPTRL_TEMPLATE}",
  "prompt_enhancer_system_prompt": null,
  "rewrite_temperature": 0.7,
  "rewrite_top_p": 0.9,
  "rewrite_max_new_tokens": 256,
  "generator": {
    "model_path": "stabilityai/stable-diffusion-3.5-medium",
    "height": ${HEIGHT},
    "width": ${WIDTH},
    "steps": ${STEPS},
    "guidance": ${GUIDANCE}
  },
  "benchmarks": {
    "geneval_full": "${GENEVAL_METADATA}",
    "ocr1k": "${OCR_METADATA}",
    "pickscore_sfw": "${PICKSCORE_METADATA}"
  }
}
JSON

echo "[probe] checking Slurm overlap job ${JOB_ID}"
srun --overlap --jobid="${JOB_ID}" nvidia-smi -L

run_rewrite_generate() {
  local run_dir="$1"
  local metadata="$2"
  local seed_base="$3"
  echo "[stage] rewrite+generate ${run_dir}"
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
  --run-dir '${run_dir}' \
  --metadata '${metadata}' \
  --conditions '${CONDITIONS}' \
  --sample-count 1 \
  --seed-base '${seed_base}' \
  --user-template-file '${PROMPTRL_TEMPLATE}' \
  --no-system-prompt \
  --temperature 0.7 \
  --top-p 0.9 \
  --max-new-tokens 256 \
  --overwrite
python experiments/image_calibration/calibration_pipeline.py generate \
  --run-dir '${run_dir}' \
  --height '${HEIGHT}' \
  --width '${WIDTH}' \
  --steps '${STEPS}' \
  --guidance '${GUIDANCE}' \
  --attention-backend '${ATTENTION_BACKEND}' \
  --overwrite
python experiments/image_calibration/calibration_pipeline.py summarize \
  --run-dir '${run_dir}'
"
}

run_rewrite_generate "${GENEVAL_RUN}" "${GENEVAL_METADATA}" 420000
run_rewrite_generate "${OCR_RUN}" "${OCR_METADATA}" 520000
run_rewrite_generate "${PICKSCORE_RUN}" "${PICKSCORE_METADATA}" 620000

echo "[stage] OCR scoring"
srun --overlap --jobid="${JOB_ID}" bash -lc "
set -euo pipefail
cd '${REPO_ROOT}'
export VIRTUAL_ENV='${REPO_ROOT}/.venv'
export PATH=\"\${VIRTUAL_ENV}/bin:\${PATH}\"
export HF_HOME='${HF_HOME}'
export TRANSFORMERS_CACHE='${TRANSFORMERS_CACHE}'
python experiments/image_calibration/calibration_pipeline.py eval-preference \
  --run-dir '${OCR_RUN}' \
  --metrics ocr \
  --metric-prompt original \
  --device cuda \
  --no-ocr-use-gpu \
  --overwrite
"

echo "[stage] PickScore/HPS/UnifiedReward scoring"
srun --overlap --jobid="${JOB_ID}" bash -lc "
set -euo pipefail
cd '${REPO_ROOT}'
export VIRTUAL_ENV='${REPO_ROOT}/.venv'
export PATH=\"\${VIRTUAL_ENV}/bin:\${PATH}\"
export HF_HOME='${HF_HOME}'
export TRANSFORMERS_CACHE='${TRANSFORMERS_CACHE}'
python experiments/image_calibration/calibration_pipeline.py eval-preference \
  --run-dir '${PICKSCORE_RUN}' \
  --metrics pickscore,hps \
  --metric-prompt original \
  --hps-version v2.1 \
  --device cuda \
  --overwrite
"

echo "[stage] UnifiedReward sglang scoring"
srun --overlap --jobid="${JOB_ID}" bash -lc "
set -euo pipefail
cd '${REPO_ROOT}'
export HF_HOME='${HF_HOME}'
export TRANSFORMERS_CACHE='${TRANSFORMERS_CACHE}'
export CUDAHOSTCXX='${CUDAHOSTCXX}'
export NVCC_PREPEND_FLAGS='${NVCC_PREPEND_FLAGS}'
if [[ -f /home/hal-jundas/.cache/huggingface/token ]]; then
  export HF_TOKEN=\"\$(cat /home/hal-jundas/.cache/huggingface/token)\"
fi
if [[ ! -x '${SGLANG_PYTHON}' ]]; then
  echo 'Missing sglang python at ${SGLANG_PYTHON}' >&2
  exit 2
fi
'${SGLANG_PYTHON}' -m sglang.launch_server \
  --model-path CodeGoat24/UnifiedReward-7b-v1.5 \
  --api-key flowgrpo \
  --port 17140 \
  --chat-template chatml-llava \
  --enable-p2p-check \
  --mem-fraction-static 0.85 \
  --disable-cuda-graph \
  --disable-overlap-schedule > '${PICKSCORE_RUN}/unifiedreward_sglang.log' 2>&1 &
server_pid=\$!
trap 'kill \${server_pid} >/dev/null 2>&1 || true' EXIT
python - <<'PY'
import socket
import time
from urllib.parse import urlparse

parsed = urlparse('${UR_URL}')
host = parsed.hostname or '127.0.0.1'
port = parsed.port or 17140
deadline = time.time() + ${UR_TIMEOUT}
while time.time() < deadline:
    try:
        with socket.create_connection((host, port), timeout=2):
            break
    except OSError:
        time.sleep(5)
else:
    raise SystemExit('UnifiedReward sglang server did not become reachable before timeout')
PY
export VIRTUAL_ENV='${REPO_ROOT}/.venv'
export PATH=\"\${VIRTUAL_ENV}/bin:\${PATH}\"
python experiments/image_calibration/calibration_pipeline.py eval-preference \
  --run-dir '${PICKSCORE_RUN}' \
  --metrics ur \
  --metric-prompt original \
  --ur-backend sglang \
  --ur-api-base '${UR_URL}/v1' \
  --ur-api-key flowgrpo \
  --ur-model-name UnifiedReward-7b-v1.5 \
  --device cpu
"

echo "[stage] GenEval reward server and scoring"
srun --overlap --jobid="${JOB_ID}" bash -lc "
set -euo pipefail
mkdir -p '${GENEVAL_RUN}'
cd '${REPO_ROOT}/evaluations/reward-server'
export MPLCONFIGDIR=/tmp/matplotlib-reward-server
export GENEVAL_DEVICE=cuda
export GENEVAL_NUM_DEVICES='${GENEVAL_NUM_DEVICES}'
export HF_HOME='${HF_HOME}'
export TRANSFORMERS_CACHE='${TRANSFORMERS_CACHE}'
if [[ -f /home/hal-jundas/.cache/huggingface/token ]]; then
  export HF_TOKEN=\"\$(cat /home/hal-jundas/.cache/huggingface/token)\"
fi
'${REPO_ROOT}/.envs/reward_server/bin/gunicorn' 'app_geneval:create_app()' > '${GENEVAL_RUN}/geneval_server.log' 2>&1 &
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
  --run-dir '${GENEVAL_RUN}' \
  --server-url '${GENEVAL_URL}' \
  --timeout '${GENEVAL_TIMEOUT}' \
  --overwrite
"

python experiments/image_calibration/calibration_pipeline.py summarize --run-dir "${GENEVAL_RUN}"
python experiments/image_calibration/calibration_pipeline.py summarize --run-dir "${OCR_RUN}"
python experiments/image_calibration/calibration_pipeline.py summarize --run-dir "${PICKSCORE_RUN}"

echo "[done] ${RUN_ROOT}"
