#!/usr/bin/env bash
# Run 8 single-GPU deterministic smoke trainings and verify matching losses.
set -euo pipefail

RUN_PREFIX=${RUN_PREFIX:-repro-$(date +%Y%m%d_%H%M%S)}
LR=${LR:-3e-4}
TOKENS=${TOKENS:-2000}
SEED=${SEED:-999}
NGPUS=${NGPUS:-8}   # number of single-GPU jobs (assumes contiguous IDs starting at 0)
FRACTAL_STORAGE_DIR=${FRACTAL_STORAGE_DIR:-/var/tmp/fractal-llm}
LOG_DIR=${LOG_DIR:-${FRACTAL_STORAGE_DIR}/results/repro_logs/${RUN_PREFIX}}
MODEL_ID=${MODEL_ID:-${MODEL_ARTIFACT:-}}
DATASET_ID=${DATASET_ID:-}
WANDB_PROJECT=${WANDB_PROJECT:-fractal-llm}
WANDB_ENTITY=${WANDB_ENTITY:-morgy}

mkdir -p "${LOG_DIR}"
echo "[repro] logging to ${LOG_DIR}"
echo "[repro] W&B project=${WANDB_PROJECT} entity=${WANDB_ENTITY}"

extra_args=()
if [[ -n "${MODEL_ID}" ]]; then
  extra_args+=(--model_id "${MODEL_ID}")
fi
if [[ -n "${DATASET_ID}" ]]; then
  extra_args+=(--dataset_id "${DATASET_ID}")
fi

pids=()
for gpu in $(seq 0 $((NGPUS - 1))); do
  log="${LOG_DIR}/run_${gpu}.log"
  echo "[repro] GPU ${gpu} -> ${log}"
  CUDA_VISIBLE_DEVICES=${gpu} HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-0} \
    WANDB_PROJECT=${WANDB_PROJECT} WANDB_ENTITY=${WANDB_ENTITY} \
    uv run python -m src.finetune \
      --run "${RUN_PREFIX}-g${gpu}" \
      --learning_rate "${LR}" \
      --num_tokens "${TOKENS}" \
      --eval_every 0 \
      --log_every 5 \
      --save_artifacts False \
      --debug False \
      --deterministic True \
      --seed "${SEED}" \
      "${extra_args[@]}" \
      >"${log}" 2>&1 &
  pids+=($!)
done

# Wait for all jobs
for pid in "${pids[@]}"; do
  wait "${pid}"
done

# Verify losses match across GPUs
uv run python - <<PY
import json, pathlib, re, sys
log_dir = pathlib.Path("${LOG_DIR}")
pattern = re.compile(r"Final: loss=([0-9.]+)")
losses = {}
for log in sorted(log_dir.glob("run_*.log")):
    txt = log.read_text()
    match = pattern.findall(txt)
    if not match:
        print(f"[repro] missing final loss in {log}", file=sys.stderr)
        sys.exit(1)
    losses[log.name] = float(match[-1])

print(json.dumps(losses, indent=2))
rounded = {round(v, 6) for v in losses.values()}
if len(rounded) != 1:
    print("[repro] Repro test FAILED: losses differ", file=sys.stderr)
    sys.exit(2)
print("[repro] Repro test PASSED")
PY
