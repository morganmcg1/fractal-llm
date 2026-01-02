#!/usr/bin/env bash
# Probe the maximum `device_batch_size` that fits on a single GPU for src/finetune.py.
# Uses a short run (fixed steps) and binary-searches the max batch size.

set -euo pipefail
[[ ${DEBUG_PROBE:-0} -eq 1 ]] && set -x

GPU=${GPU:-0}
RUN_PREFIX=${RUN_PREFIX:-batch-probe-$(date +%Y%m%d_%H%M%S)}
GRID_SWEEP_ID=${GRID_SWEEP_ID:-${RUN_PREFIX}}
FRACTAL_STORAGE_DIR=${FRACTAL_STORAGE_DIR:-/var/tmp/fractal-llm}
LOG_DIR=${LOG_DIR:-${FRACTAL_STORAGE_DIR}/results/batch_probe_logs/${RUN_PREFIX}}

LR=${LR:-3e-4}
SEED=${SEED:-999}
NUM_ITERATIONS=${NUM_ITERATIONS:-2}
NUM_TOKENS=${NUM_TOKENS:-5000}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-1024}

BS_START=${BS_START:-8}
BS_MAX=${BS_MAX:-256}

MODEL_ID=${MODEL_ID:-${MODEL_ARTIFACT:-}}
DATASET_ID=${DATASET_ID:-}
DATASET_REVISION=${DATASET_REVISION:-}

WANDB_PROJECT=${WANDB_PROJECT:-fractal-llm}
WANDB_ENTITY=${WANDB_ENTITY:-morgy}
FINETUNE_WANDB_TAGS=${FINETUNE_WANDB_TAGS:-batch-probe}

mkdir -p "${LOG_DIR}"
echo "[probe] logging to ${LOG_DIR}"
echo "[probe] GPU=${GPU} LR=${LR} seed=${SEED} steps=${NUM_ITERATIONS} tokens_goal=${NUM_TOKENS} max_seq_len=${MAX_SEQ_LEN}"
echo "[probe] batch_size_start=${BS_START} batch_size_max=${BS_MAX}"
echo "[probe] W&B project=${WANDB_PROJECT} entity=${WANDB_ENTITY} tags=${FINETUNE_WANDB_TAGS} sweep_id=${GRID_SWEEP_ID}"

extra_args=()
[[ -n "${MODEL_ID}" ]] && extra_args+=(--model_id "${MODEL_ID}")
[[ -n "${DATASET_ID}" ]] && extra_args+=(--dataset_id "${DATASET_ID}")
[[ -n "${DATASET_REVISION}" ]] && extra_args+=(--dataset_revision "${DATASET_REVISION}")

run_trial() {
  local bs=$1
  local log="${LOG_DIR}/bs_${bs}.log"
  echo "[probe] TRY device_batch_size=${bs} -> ${log}"
  if ! (
    CUDA_VISIBLE_DEVICES=${GPU} HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-0} \
    FRACTAL_STORAGE_DIR=${FRACTAL_STORAGE_DIR} \
    WANDB_PROJECT=${WANDB_PROJECT} WANDB_ENTITY=${WANDB_ENTITY} \
    PYTHONUNBUFFERED=1 \
    uv run python -m src.finetune \
      --run "${RUN_PREFIX}-bs${bs}" \
      --wandb_tags "${FINETUNE_WANDB_TAGS}" \
      --grid_sweep_id "${GRID_SWEEP_ID}" \
      --learning_rate "${LR}" \
      --seed "${SEED}" \
      --deterministic True \
      --device_batch_size "${bs}" \
      --num_iterations "${NUM_ITERATIONS}" \
      --num_tokens "${NUM_TOKENS}" \
      --eval_every 0 \
      --log_every 1 \
      --save_artifacts False \
      --max_seq_len "${MAX_SEQ_LEN}" \
      "${extra_args[@]}" \
      2>&1 | tee "${log}"
  ); then
    return 1
  fi

  # Ensure the run produced a finite numeric final loss (reject inf/nan).
  if ! rg -n "Final: loss=[0-9]" "${log}" >/dev/null; then
    echo "[probe] FAILED (non-finite final loss) device_batch_size=${bs} log=${log}" >&2
    return 1
  fi
}

is_power_of_two() {
  local n=$1
  (( n > 0 && (n & (n - 1)) == 0 ))
}

last_ok=0
first_bad=0

bs=${BS_START}
while (( bs <= BS_MAX )); do
  if run_trial "${bs}"; then
    last_ok=${bs}
    if (( bs == BS_MAX )); then
      break
    fi
    if is_power_of_two "${bs}"; then
      bs=$((bs * 2))
    else
      bs=$((bs + 1))
    fi
  else
    first_bad=${bs}
    break
  fi
done

if (( last_ok == 0 )); then
  echo "[probe] No working batch size found (starting at ${BS_START})." >&2
  exit 2
fi

if (( first_bad == 0 )); then
  echo "[probe] Max batch size reached without failure: ${last_ok}"
  exit 0
fi

lo=${last_ok}
hi=$((first_bad - 1))
echo "[probe] coarse bracket: ok=${last_ok} bad=${first_bad} -> searching [${lo}, ${hi}]"

while (( lo < hi )); do
  mid=$(((lo + hi + 1) / 2))
  if run_trial "${mid}"; then
    lo=${mid}
  else
    hi=$((mid - 1))
  fi
done

echo "[probe] Max device_batch_size that fits: ${lo}"
