#!/usr/bin/env bash
# Multi-devpod grid sweep launcher.
# Distributes grid points across multiple devpods, each running 8 GPUs.
#
# Usage:
#   DEVPODS="fractal-llm-1 fractal-llm-2 fractal-llm-3" \
#   RUN_PREFIX=multi-sweep \
#   RES=16 \
#   ./scripts/multi_devpod_sweep.sh

set -euo pipefail
[[ ${DEBUG_MULTI:-0} -eq 1 ]] && set -x

DEVPODS_STR=${DEVPODS:-"fractal-llm-1 fractal-llm-2 fractal-llm-3"}
read -ra DEVPODS <<< "${DEVPODS_STR}"
NUM_DEVPODS=${#DEVPODS[@]}

if [[ ${NUM_DEVPODS} -eq 0 ]]; then
  echo "[multi] No devpods specified. Set DEVPODS=\"fractal-llm-1 fractal-llm-2 ...\"" >&2
  exit 1
fi

RUN_PREFIX=${RUN_PREFIX:-multi-$(date +%Y%m%d_%H%M%S)}
GRID_SWEEP_ID=${GRID_SWEEP_ID:-${RUN_PREFIX}}
RES=${RES:-16}
LR_MIN=${LR_MIN:-1e-5}
LR_MAX=${LR_MAX:-1e-3}
TOK_MIN=${TOK_MIN:-5e3}
TOK_MAX=${TOK_MAX:-5e5}
TOKENS_PER_RUN=${TOKENS_PER_RUN:-}
LR_FIXED=${LR_FIXED:-}
SEED=${SEED:-999}
MODEL_ID=${MODEL_ID:-}
DATASET_ID=${DATASET_ID:-}
DATASET_REVISION=${DATASET_REVISION:-}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-1024}
SWEEP_AXES=${SWEEP_AXES:-lr_tokens}
WANDB_PROJECT=${WANDB_PROJECT:-fractal-llm}
WANDB_ENTITY=${WANDB_ENTITY:-morgy}
FINETUNE_WANDB_TAGS=${FINETUNE_WANDB_TAGS:-fractal-grid,multi-devpod}
FRACTAL_STORAGE_DIR=${FRACTAL_STORAGE_DIR:-/var/tmp/fractal-llm}
HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-0}

TOTAL_POINTS=$((RES * RES))
POINTS_PER_DEVPOD=$(( (TOTAL_POINTS + NUM_DEVPODS - 1) / NUM_DEVPODS ))

echo "[multi] =============================================="
echo "[multi] Multi-DevPod Grid Sweep"
echo "[multi] =============================================="
echo "[multi] devpods: ${DEVPODS[*]}"
echo "[multi] grid: ${RES}x${RES} = ${TOTAL_POINTS} points"
echo "[multi] points/devpod: ~${POINTS_PER_DEVPOD}"
echo "[multi] sweep_id: ${GRID_SWEEP_ID}"
echo "[multi] run_prefix: ${RUN_PREFIX}"
echo "[multi] =============================================="

echo "[multi] Checking devpod connectivity..."
for pod in "${DEVPODS[@]}"; do
  if ! ssh -o ConnectTimeout=5 "${pod}.devpod" "echo 'connected'" &>/dev/null; then
    echo "[multi] ERROR: Cannot connect to ${pod}.devpod" >&2
    echo "[multi] Make sure the devpod is running: devpod list" >&2
    exit 1
  fi
  echo "[multi] + ${pod}"
done

ENV_VARS=(
  "DEVPOD_NAME"
  "RUN_PREFIX=${RUN_PREFIX}"
  "GRID_SWEEP_ID=${GRID_SWEEP_ID}"
  "RES=${RES}"
  "LR_MIN=${LR_MIN}"
  "LR_MAX=${LR_MAX}"
  "TOK_MIN=${TOK_MIN}"
  "TOK_MAX=${TOK_MAX}"
  "SEED=${SEED}"
  "MAX_SEQ_LEN=${MAX_SEQ_LEN}"
  "SWEEP_AXES=${SWEEP_AXES}"
  "WANDB_PROJECT=${WANDB_PROJECT}"
  "WANDB_ENTITY=${WANDB_ENTITY}"
  "FINETUNE_WANDB_TAGS=${FINETUNE_WANDB_TAGS}"
  "FRACTAL_STORAGE_DIR=${FRACTAL_STORAGE_DIR}"
  "HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE}"
)
[[ -n "${TOKENS_PER_RUN}" ]] && ENV_VARS+=("TOKENS_PER_RUN=${TOKENS_PER_RUN}")
[[ -n "${LR_FIXED}" ]] && ENV_VARS+=("LR_FIXED=${LR_FIXED}")
[[ -n "${MODEL_ID}" ]] && ENV_VARS+=("MODEL_ID=${MODEL_ID}")
[[ -n "${DATASET_ID}" ]] && ENV_VARS+=("DATASET_ID=${DATASET_ID}")
[[ -n "${DATASET_REVISION}" ]] && ENV_VARS+=("DATASET_REVISION=${DATASET_REVISION}")

echo "[multi] Launching sweeps..."
pids=()
for i in "${!DEVPODS[@]}"; do
  pod="${DEVPODS[$i]}"
  env_str=""
  for var in "${ENV_VARS[@]}"; do
    if [[ "${var}" == "DEVPOD_NAME" ]]; then
      env_str+="DEVPOD_NAME=${pod} "
    else
      env_str+="${var} "
    fi
  done

  echo "[multi] Starting ${pod} (devpod $((i+1))/${NUM_DEVPODS})..."
  ssh "${pod}.devpod" "bash -lc '
    cd /workspaces/fractal-llm && \
    source .env 2>/dev/null || true && \
    ${env_str} \
    tmux new-session -d -s grid_sweep \"./scripts/grid_sweep.sh 2>&1 | tee /var/tmp/fractal-llm/grid_sweep_\$(date +%Y%m%d_%H%M%S).log\" && \
    echo \"[${pod}] Started grid_sweep in tmux session\"
  '" &
  pids+=($!)
done

for pid in "${pids[@]}"; do
  wait "${pid}" || true
done

echo "[multi] =============================================="
echo "[multi] All devpods launched!"
echo "[multi] =============================================="
echo "[multi] Monitor with:"
for pod in "${DEVPODS[@]}"; do
  echo "  ssh -t ${pod}.devpod 'tmux attach -t grid_sweep'"
done
echo ""
echo "[multi] W&B: https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}"
echo "[multi] Filter: config.grid_sweep_id = \"${GRID_SWEEP_ID}\""
