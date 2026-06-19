#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${SCRIPT_DIR}" == *"/rl_prompt_enhancer/scripts/launch_exp/"* ]]; then
  echo "Copy this launch profile into experiments/<run_id>/commands/ before launching." >&2
  exit 1
fi

# shellcheck source=experiment.sh
source "${SCRIPT_DIR}/experiment.sh"

UVICORN_BIN="${PYTHON%/python}/uvicorn"
SERVICE_PYTHONPATH="${REPO_ROOT}:${SLIME_ROOT}:${FASTVIDEO_ROOT}"

FASTVIDEO_MODEL="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
FASTVIDEO_CUDA_VISIBLE_DEVICES=(0 1 2 3)
FASTVIDEO_ARTIFACT_ROOT="${RUN_DIR}/fastvideo_service/artifacts"
FASTVIDEO_LEDGER_ROOT="${RUN_DIR}/fastvideo_service/ledgers"
FASTVIDEO_EXECUTION_BACKEND="mp"
FASTVIDEO_WORKLOAD_TYPE="t2v"

PICKSCORE_PROCESSOR_MODEL="laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
PICKSCORE_MODEL="yuvalkirstain/PickScore_v1"
CLIPSCORE_MODEL="openai/clip-vit-large-patch14"
REWARD_DEVICE="cuda"
REWARD_WEIGHTS_JSON='{"pickscore":1.0,"clipscore":1.0}'
SCALAR_REWARD_KEY="avg"

mkdir -p "${RUN_DIR}/logs" "${FASTVIDEO_ARTIFACT_ROOT}" "${FASTVIDEO_LEDGER_ROOT}"
exec > >(tee -a "${RUN_DIR}/logs/02_start_fastvideo_workers.log") 2>&1

if [[ "${#FASTVIDEO_SERVICE_PORTS[@]}" -ne "${#FASTVIDEO_CUDA_VISIBLE_DEVICES[@]}" ]]; then
  echo "FASTVIDEO_SERVICE_PORTS and FASTVIDEO_CUDA_VISIBLE_DEVICES must have the same length." >&2
  exit 1
fi

echo "Starting ${#FASTVIDEO_SERVICE_PORTS[@]} FastVideo workers on ${FASTVIDEO_NODELIST}"
echo "Worker ports: ${FASTVIDEO_SERVICE_PORTS[*]}"
echo "Worker GPUs: ${FASTVIDEO_CUDA_VISIBLE_DEVICES[*]}"

srun \
  --overlap \
  --jobid "${FASTVIDEO_JOB_ID}" \
  --nodes=1 \
  --ntasks=1 \
  --export=ALL \
  --nodelist "${FASTVIDEO_NODELIST}" \
  bash <<FASTVIDEO_WORKERS
set -euo pipefail

worker_ports=(${FASTVIDEO_SERVICE_PORTS[*]})
worker_devices=(${FASTVIDEO_CUDA_VISIBLE_DEVICES[*]})
pids=()

cleanup() {
  for pid in "\${pids[@]:-}"; do
    kill "\${pid}" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for index in "\${!worker_ports[@]}"; do
  port="\${worker_ports[\${index}]}"
  device="\${worker_devices[\${index}]}"
  artifact_root="${FASTVIDEO_ARTIFACT_ROOT}/worker_\${index}"
  mkdir -p "\${artifact_root}"

  echo "Worker \${index}: GPU \${device}, port \${port}"
  env \\
    PYTHONPATH="${SERVICE_PYTHONPATH}" \\
    PYTHONNOUSERSITE=1 \\
    CUDA_VISIBLE_DEVICES="\${device}" \\
    RLPE_LEDGER_ROOT="${FASTVIDEO_LEDGER_ROOT}" \\
    RLPE_FASTVIDEO_MODEL="${FASTVIDEO_MODEL}" \\
    RLPE_FASTVIDEO_EXECUTION_BACKEND="${FASTVIDEO_EXECUTION_BACKEND}" \\
    RLPE_FASTVIDEO_WORKLOAD_TYPE="${FASTVIDEO_WORKLOAD_TYPE}" \\
    RLPE_FASTVIDEO_WORKER_ID="\${index}" \\
    RLPE_FASTVIDEO_SERVICE_PORT="\${port}" \\
    RLPE_FASTVIDEO_OUTPUT_ROOT="\${artifact_root}" \\
    RLPE_FASTVIDEO_NUM_GPUS=1 \\
    RLPE_FASTVIDEO_TP_SIZE=1 \\
    RLPE_FASTVIDEO_SP_SIZE=1 \\
    RLPE_FASTVIDEO_HSDP_REPLICATE_DIM=1 \\
    RLPE_FASTVIDEO_HSDP_SHARD_DIM=1 \\
    PICKSCORE_PROCESSOR_MODEL="${PICKSCORE_PROCESSOR_MODEL}" \\
    PICKSCORE_MODEL="${PICKSCORE_MODEL}" \\
    CLIPSCORE_MODEL="${CLIPSCORE_MODEL}" \\
    RLPE_REWARD_DEVICE="${REWARD_DEVICE}" \\
    RLPE_REWARD_WEIGHTS_JSON='${REWARD_WEIGHTS_JSON}' \\
    RLPE_SCALAR_REWARD_KEY="${SCALAR_REWARD_KEY}" \\
    "${UVICORN_BIN}" rl_prompt_enhancer.fastvideo_bridge.generate_and_score_server:app \\
      --host "${FASTVIDEO_SERVICE_HOST}" \\
      --port "\${port}" \\
      --log-level info &
  pids+=("\$!")
done

echo "FastVideo workers are running. Keep this command alive for the experiment."
wait -n
FASTVIDEO_WORKERS
