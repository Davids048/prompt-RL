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
SLIME_PYTHONPATH="${REPO_ROOT}:${SLIME_ROOT}:${FASTVIDEO_ROOT}:${MEGATRON_ROOT}"
SLIME_RUNTIME_ENV_PATH="${RUN_DIR}/snapshot/slime_runtime_env.json"
GENERATOR_CONFIG_PATH="${RUN_DIR}/configs/fastvideo_image_grpo_generator.json"
VALIDATION_JSONL_PATH="${REPO_ROOT}/data/rl_prompt_enhancer/validation/dmd2_text_only_validation.jsonl"
VALIDATION_EVAL_CONFIG_PATH="${RUN_DIR}/configs/fastvideo_image_grpo_validation.yaml"

mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/snapshot" "${RUN_DIR}/checkpoints"

verify_prompt_data() {
  local prompt_jsonl="${REPO_ROOT}/data/rl_prompt_enhancer/prompts/diffusion_nft_pickscore_train_full.jsonl"
  if [[ -s "${prompt_jsonl}" ]]; then
    return
  fi

  echo "Prompt JSONL not found: ${prompt_jsonl}" >&2
  exit 1
}

write_slime_runtime_env() {
  "${PYTHON}" - "${SLIME_RUNTIME_ENV_PATH}" "${SLIME_PYTHONPATH}" "${RUN_DIR}" "${SLIME_CUDA_VISIBLE_DEVICES}" "${FASTVIDEO_PUBLIC_HOST}" "${FASTVIDEO_SERVICE_PORTS[*]}" "${GENERATOR_CONFIG_PATH}" <<'PY'
import json
import sys

output, pythonpath, run_dir, cuda_devices, host, ports, generator = sys.argv[1:]
urls = ",".join(f"http://{host}:{port}" for port in ports.split())
runtime_env = {
    "env_vars": {
        "PYTHONPATH": pythonpath,
        "RAY_USE_UVLOOP": "0",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "CUDA_VISIBLE_DEVICES": cuda_devices,
        "WANDB_DIR": f"{run_dir}/wandb",
        "RLPE_FASTVIDEO_SERVICE_URL": urls.split(",")[0],
        "RLPE_FASTVIDEO_SERVICE_URLS": urls,
        "RLPE_PROMPT_TEMPLATE_PATH": "/home/hal-jundas/codes/UniRL/experiments/baseline_sweep/prompt/templates/image/geneval/promptrl_style_v1.txt",
        "RLPE_SEED_BASE": "430000",
        "RLPE_GENERATOR_CONFIG_PATH": generator,
        "RLPE_SCALAR_REWARD_KEY": "avg",
        "CC": "/home/shared-bin/local/usr/bin/gcc-13",
        "CXX": "/home/shared-bin/local/usr/bin/g++-13",
        "NVCC_PREPEND_FLAGS": "-ccbin=/home/shared-bin/local/usr/bin/g++-13",
    }
}
with open(output, "w", encoding="utf-8") as handle:
    handle.write(json.dumps(runtime_env, separators=(",", ":")) + "\n")
PY
}

require_paths() {
  local required=(
    "${PYTHON}"
    "${RAY_BIN}"
    "${SLIME_ROOT}/train.py"
    "${SLIME_ROOT}/scripts/models/qwen3.5-9B.sh"
    "${GENERATOR_CONFIG_PATH}"
    "${VALIDATION_JSONL_PATH}"
    "${VALIDATION_EVAL_CONFIG_PATH}"
    "${REPO_ROOT}/experiments/baseline_sweep/prompt/templates/image/geneval/promptrl_style_v1.txt"
    "${REPO_ROOT}/.cache/rl_prompt_enhancer/phase1/models/Qwen3.5-9B"
  )

  for path in "${required[@]}"; do
    [[ -e "${path}" ]] || {
      echo "required path missing: ${path}" >&2
      exit 1
    }
  done

  srun \
    --overlap \
    --jobid "${SLIME_JOB_ID}" \
    --nodes=1 \
    --ntasks=1 \
    --export=ALL \
    --nodelist "${SLIME_NODELIST}" \
    test -e /dev/shm/Qwen3.5-9B_torch_dist
}

wait_for_fastvideo_health() {
  local attempt port url
  for ((attempt = 1; attempt <= 30; attempt++)); do
    local failed=0
    for port in "${FASTVIDEO_SERVICE_PORTS[@]}"; do
      url="http://${FASTVIDEO_PUBLIC_HOST}:${port}/health"
      if ! srun --overlap --jobid "${SLIME_JOB_ID}" --nodes=1 --ntasks=1 --export=ALL --nodelist "${SLIME_NODELIST}" curl -fsS "${url}"; then
        failed=1
      fi
    done

    if [[ "${failed}" == "0" ]]; then
      return
    fi
    sleep 10
  done

  echo "FastVideo workers are not healthy." >&2
  exit 1
}

submit_slime_grpo() {
  # Slime owns model architecture defaults; the rest of this command is the
  # experiment launch surface.
  # shellcheck source=/dev/null
  source "${SLIME_ROOT}/scripts/models/qwen3.5-9B.sh"

  local runtime_env_json
  runtime_env_json="$(<"${SLIME_RUNTIME_ENV_PATH}")"

  (
    cd "${SLIME_ROOT}"
    srun \
      --overlap \
      --jobid "${SLIME_JOB_ID}" \
      --nodes=1 \
      --ntasks=1 \
      --export=ALL \
      --nodelist "${SLIME_NODELIST}" \
      "${RAY_BIN}" job submit \
        --address "${RAY_ADDRESS}" \
        --runtime-env-json "${runtime_env_json}" \
        -- \
        "${PYTHON}" "${SLIME_ROOT}/train.py" \
        "${MODEL_ARGS[@]}" \
        --hf-checkpoint "${REPO_ROOT}/.cache/rl_prompt_enhancer/phase1/models/Qwen3.5-9B" \
        --ref-load /dev/shm/Qwen3.5-9B_torch_dist \
        --save "${RUN_DIR}/checkpoints" \
        --save-interval 9999 \
        --prompt-data "${REPO_ROOT}/data/rl_prompt_enhancer/prompts/diffusion_nft_pickscore_train_full.jsonl" \
        --input-key original_prompt \
        --metadata-key metadata \
        --apply-chat-template \
        --custom-generate-function-path rl_prompt_enhancer.slime_hooks.fastvideo_generate.generate \
        --custom-eval-rollout-log-function-path rl_prompt_enhancer.slime_hooks.eval_wandb_images.log_eval_images \
        --reward-key avg \
        --num-rollout 25432 \
        --rollout-batch-size 1 \
        --n-samples-per-prompt 8 \
        --rollout-max-response-len 256 \
        --rollout-temperature 0.8 \
        --global-batch-size 8 \
        --eval-interval 20 \
        --eval-config "${VALIDATION_EVAL_CONFIG_PATH}" \
        --eval-reward-key avg \
        --optimizer adam \
        --lr 1.0e-6 \
        --lr-decay-style constant \
        --weight-decay 0.1 \
        --adam-beta1 0.9 \
        --adam-beta2 0.98 \
        --advantage-estimator grpo \
        --use-kl-loss \
        --kl-loss-coef 0.00 \
        --kl-loss-type low_var_kl \
        --entropy-coef 0.00 \
        --eps-clip 0.2 \
        --eps-clip-high 0.28 \
        --tensor-model-parallel-size 1 \
        --sequence-parallel \
        --pipeline-model-parallel-size 1 \
        --context-parallel-size 1 \
        --expert-model-parallel-size 1 \
        --expert-tensor-parallel-size 1 \
        --qkv-format bshd \
        --micro-batch-size 1 \
        --rollout-num-gpus-per-engine 1 \
        --sglang-mem-fraction-static 0.7 \
        --sglang-cuda-graph-max-bs 32 \
        --sglang-enable-metrics \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --accumulate-allreduce-grads-in-fp32 \
        --attention-softmax-in-fp32 \
        --attention-backend flash \
        --loss-mask-type qwen3_5 \
        --actor-num-nodes 1 \
        --actor-num-gpus-per-node "${SLIME_NUM_GPUS}" \
        --colocate \
        --use-wandb \
        --wandb-project "${WANDB_PROJECT}" \
        --wandb-group "${WANDB_GROUP}" \
        --wandb-dir "${WANDB_DIR}" \
        --disable-wandb-random-suffix
  ) 2>&1 | tee "${RUN_DIR}/logs/03_submit_slime_grpo.log"
}

main() {
  exec > >(tee -a "${RUN_DIR}/logs/03_submit_slime_grpo_prepare.log") 2>&1

  echo "Submitting training for ${RUN_NAME}"
  echo "FastVideo service: http://${FASTVIDEO_PUBLIC_HOST}:${FASTVIDEO_SERVICE_PORTS[0]}"

  require_paths
  verify_prompt_data
  write_slime_runtime_env
  wait_for_fastvideo_health
  submit_slime_grpo
}

main "$@"
