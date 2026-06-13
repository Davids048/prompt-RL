#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/hal-jundas/codes/UniRL}"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
CONFIG="${CONFIG:-${REPO_ROOT}/experiments/baseline_sweep/configs/video/wan21_13b_dancegrpo200_hpsv3_videoalign_v2.yaml}"
JOB_A="${JOB_A:-4882}"
JOB_B="${JOB_B:-4884}"

cd "${REPO_ROOT}"

prepare_output="$("${PYTHON}" experiments/baseline_sweep/src/cli.py prepare "${CONFIG}")"
printf '%s\n' "${prepare_output}"
RUN_DIR="$(printf '%s\n' "${prepare_output}" | awk -F= '/run_dir=/{print $2}' | tail -n 1)"
if [[ -z "${RUN_DIR}" ]]; then
  echo "Could not parse run_dir from prepare output." >&2
  exit 1
fi

mapfile -t TRIAL_IDS < <("${PYTHON}" experiments/baseline_sweep/src/cli.py list-trials "${RUN_DIR}")
if [[ "${#TRIAL_IDS[@]}" -ne 5 ]]; then
  echo "Expected 5 trials, found ${#TRIAL_IDS[@]}." >&2
  exit 1
fi

LAUNCH_LOG_DIR="${RUN_DIR}/launch_logs"
mkdir -p "${LAUNCH_LOG_DIR}"

launch_job() {
  local job_id="$1"
  shift
  local launch_log="${LAUNCH_LOG_DIR}/job_${job_id}.log"
  local trials=("$@")
  local trials_payload=""
  local trial
  for trial in "${trials[@]}"; do
    trials_payload+="${trial}"$'\n'
  done

  TRIALS_PAYLOAD="${trials_payload}" srun --overlap --jobid="${job_id}" bash -lc "
set -euo pipefail
cd '${REPO_ROOT}'
export VIRTUAL_ENV='${REPO_ROOT}/.venv'
export PATH=\"\${VIRTUAL_ENV}/bin:\${PATH}\"
export HF_HOME='${REPO_ROOT}/.cache/huggingface'
export TRANSFORMERS_CACHE='${REPO_ROOT}/.cache/huggingface/transformers'
export TORCH_HOME='${REPO_ROOT}/.cache/torch'
export MPLCONFIGDIR='/tmp/matplotlib-baseline-sweep-${job_id}'
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
if [[ -f /home/hal-jundas/.cache/huggingface/token ]]; then
  export HF_TOKEN=\"\$(cat /home/hal-jundas/.cache/huggingface/token)\"
fi
mkdir -p '${LAUNCH_LOG_DIR}'
mapfile -t assigned_trials <<< \"\${TRIALS_PAYLOAD}\"

pids=()
for local_gpu in \"\${!assigned_trials[@]}\"; do
  trial_id=\"\${assigned_trials[\${local_gpu}]}\"
  if [[ -z \"\${trial_id}\" ]]; then
    continue
  fi
  trial_log='${LAUNCH_LOG_DIR}/'\${trial_id}'_job${job_id}_gpu'\${local_gpu}'.log'
  (
    export CUDA_VISIBLE_DEVICES=\"\${local_gpu}\"
    echo \"[launch] job=${job_id} gpu=\${local_gpu} trial=\${trial_id} start=\$(date -Is)\"
    '${PYTHON}' experiments/baseline_sweep/src/cli.py run-trial '${RUN_DIR}' \"\${trial_id}\"
    echo \"[launch] job=${job_id} gpu=\${local_gpu} trial=\${trial_id} done=\$(date -Is)\"
  ) > \"\${trial_log}\" 2>&1 &
  pids+=(\"\$!\")
done

failed=0
for pid in \"\${pids[@]}\"; do
  if ! wait \"\${pid}\"; then
    failed=1
  fi
done
exit \"\${failed}\"
" > "${launch_log}" 2>&1 &
}

launch_job "${JOB_A}" "${TRIAL_IDS[@]:0:3}"
pid_a="$!"
launch_job "${JOB_B}" "${TRIAL_IDS[@]:3:2}"
pid_b="$!"

failed=0
if ! wait "${pid_a}"; then
  failed=1
fi
if ! wait "${pid_b}"; then
  failed=1
fi

"${PYTHON}" experiments/baseline_sweep/src/cli.py summary "${RUN_DIR}"
exit "${failed}"
