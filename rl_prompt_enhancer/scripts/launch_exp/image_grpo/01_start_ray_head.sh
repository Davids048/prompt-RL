#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${SCRIPT_DIR}" == *"/rl_prompt_enhancer/scripts/launch_exp/"* ]]; then
  echo "Copy this launch profile into experiments/<run_id>/commands/ before launching." >&2
  exit 1
fi

# shellcheck source=experiment.sh
source "${SCRIPT_DIR}/experiment.sh"

RAY_BIN="${PYTHON%/python}/ray"
mkdir -p "${RUN_DIR}/logs"
exec > >(tee -a "${RUN_DIR}/logs/01_start_ray_head.log") 2>&1

echo "Starting Ray head on ${SLIME_NODELIST}"
echo "Slime GPUs: ${SLIME_CUDA_VISIBLE_DEVICES}"
echo "Ray address: ${RAY_ADDRESS}"

srun \
  --overlap \
  --jobid "${SLIME_JOB_ID}" \
  --nodes=1 \
  --ntasks=1 \
  --export=ALL \
  --nodelist "${SLIME_NODELIST}" \
  bash <<RAY_HEAD
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${SLIME_CUDA_VISIBLE_DEVICES}"

"${RAY_BIN}" start \\
  --head \\
  --node-ip-address "${RAY_MASTER_ADDR}" \\
  --num-gpus "${SLIME_NUM_GPUS}" \\
  --port "${RAY_PORT}" \\
  --dashboard-port "${RAY_DASHBOARD_PORT}" \\
  --ray-client-server-port "${RAY_CLIENT_SERVER_PORT}" \\
  --temp-dir "${RAY_TEMP_DIR}" \\
  --disable-usage-stats

echo "Ray head is running. Keep this command alive for the experiment."
while true; do
  sleep 3600
done
RAY_HEAD
