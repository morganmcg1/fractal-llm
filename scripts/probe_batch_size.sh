#!/usr/bin/env bash
# Probe the maximum `device_batch_size` that fits on a single GPU for src/finetune.py.
# Runs a multi-step training + periodic evals to catch allocator/sequence-length spikes.
# By default, searches batch sizes in steps of 8 (preferred divisibility).

set -euo pipefail
[[ ${DEBUG_PROBE:-0} -eq 1 ]] && set -x

GPU=${GPU:-0}
RUN_PREFIX=${RUN_PREFIX:-batch-probe-$(date +%Y%m%d_%H%M%S)}
GRID_SWEEP_ID=${GRID_SWEEP_ID:-${RUN_PREFIX}}
FRACTAL_STORAGE_DIR=${FRACTAL_STORAGE_DIR:-/var/tmp/fractal-llm}
LOG_DIR=${LOG_DIR:-${FRACTAL_STORAGE_DIR}/results/batch_probe_logs/${RUN_PREFIX}}

LR=${LR:-3e-4}
SEED=${SEED:-999}
NUM_ITERATIONS=${NUM_ITERATIONS:-100}
NUM_TOKENS=${NUM_TOKENS:-5000}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-1024}
EVAL_EVERY=${EVAL_EVERY:-33}         # 100 steps -> eval at 33, 66, 99 (3 eval rounds)
EVAL_BATCHES=${EVAL_BATCHES:-4}
LOG_EVERY=${LOG_EVERY:-10}

BS_START=${BS_START:-32}
BS_MAX=${BS_MAX:-256}
BS_STRIDE=${BS_STRIDE:-8}

MODEL_ID=${MODEL_ID:-${MODEL_ARTIFACT:-}}
DATASET_ID=${DATASET_ID:-}
DATASET_REVISION=${DATASET_REVISION:-}

WANDB_PROJECT=${WANDB_PROJECT:-fractal-llm}
WANDB_ENTITY=${WANDB_ENTITY:-morgy}
FINETUNE_WANDB_TAGS=${FINETUNE_WANDB_TAGS:-batch-probe}

mkdir -p "${LOG_DIR}"
echo "[probe] logging to ${LOG_DIR}"
echo "[probe] GPU=${GPU} LR=${LR} seed=${SEED} steps=${NUM_ITERATIONS} tokens_goal=${NUM_TOKENS} max_seq_len=${MAX_SEQ_LEN}"
echo "[probe] eval_every=${EVAL_EVERY} eval_batches=${EVAL_BATCHES} log_every=${LOG_EVERY}"
echo "[probe] batch_size_start=${BS_START} batch_size_max=${BS_MAX} stride=${BS_STRIDE}"
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
      --eval_every "${EVAL_EVERY}" \
      --eval_batches "${EVAL_BATCHES}" \
      --log_every "${LOG_EVERY}" \
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

last_ok=0
first_bad=0

bs=${BS_START}
if (( bs % BS_STRIDE != 0 )); then
  bs=$(( (bs / BS_STRIDE + 1) * BS_STRIDE ))
fi
while (( bs <= BS_MAX )); do
  if run_trial "${bs}"; then
    last_ok=${bs}
    bs=$((bs + BS_STRIDE))
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

echo "[probe] Max device_batch_size that fits (stride=${BS_STRIDE}): ${last_ok}"
