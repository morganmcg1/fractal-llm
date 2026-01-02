#!/usr/bin/env bash
# Parallel single-GPU grid sweep launcher for fractal LR×tokens grids.
# One worker per GPU runs its assigned points sequentially (no GPU oversubscription).
# Logs per-run output and summarizes final losses.

set -euo pipefail
[[ ${DEBUG_GRID:-0} -eq 1 ]] && set -x

RUN_PREFIX=${RUN_PREFIX:-grid-$(date +%Y%m%d_%H%M%S)}
WANDB_RUN_PREFIX=${WANDB_RUN_PREFIX:-}
if [[ -n "${WANDB_RUN_PREFIX}" ]]; then
  RUN_PREFIX="${WANDB_RUN_PREFIX}"
fi
GRID_SWEEP_ID=${GRID_SWEEP_ID:-${RUN_PREFIX}}  # constant tag across all points in this sweep
RES=${RES:-4}                 # grid resolution per axis (RES x RES points)
LR_MIN=${LR_MIN:-1e-5}
LR_MAX=${LR_MAX:-1e-3}
TOK_MIN=${TOK_MIN:-5e3}
TOK_MAX=${TOK_MAX:-5e5}
GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
FRACTAL_STORAGE_DIR=${FRACTAL_STORAGE_DIR:-/var/tmp/fractal-llm}
LOG_DIR=${LOG_DIR:-${FRACTAL_STORAGE_DIR}/results/grid_logs/${RUN_PREFIX}}
MODEL_ID=${MODEL_ID:-${MODEL_ARTIFACT:-}}
DATASET_ID=${DATASET_ID:-}
DATASET_REVISION=${DATASET_REVISION:-}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-1024}
TOKENS_PER_RUN=${TOKENS_PER_RUN:-}  # optional override: fixed tokens instead of TOK_MIN..MAX grid
LR_FIXED=${LR_FIXED:-}              # optional override: fixed LR instead of LR_MIN..MAX grid
SEED=${SEED:-999}
WANDB_PROJECT=${WANDB_PROJECT:-fractal-llm}
WANDB_ENTITY=${WANDB_ENTITY:-morgy}
FINETUNE_WANDB_TAGS=${FINETUNE_WANDB_TAGS:-fractal-grid}

mkdir -p "${LOG_DIR}"
echo "[grid] logging to ${LOG_DIR}"
echo "[grid] W&B project=${WANDB_PROJECT} entity=${WANDB_ENTITY} tags=${FINETUNE_WANDB_TAGS} sweep_id=${GRID_SWEEP_ID}"

# Build grid points
mapfile -t GRID_POINTS < <(uv run python - <<PY
import numpy as np, os
res = int(os.environ.get("RES", "4"))
lr_fixed = os.environ.get("LR_FIXED")
tok_override = os.environ.get("TOKENS_PER_RUN")
if lr_fixed:
    lrs = np.array([float(lr_fixed)])
else:
    lr_min = float(os.environ.get("LR_MIN", "1e-5"))
    lr_max = float(os.environ.get("LR_MAX", "1e-3"))
    lrs = np.logspace(np.log10(lr_min), np.log10(lr_max), res)
if tok_override:
    toks = np.array([int(float(tok_override))])
else:
    tok_min = float(os.environ.get("TOK_MIN", "5e3"))
    tok_max = float(os.environ.get("TOK_MAX", "5e5"))
    toks = np.logspace(np.log10(tok_min), np.log10(tok_max), res).astype(int)
for i, lr in enumerate(lrs):
    for j, tok in enumerate(toks):
        print(f"{i},{j},{lr:.6g},{int(tok)}")
PY
)

if [[ ${#GRID_POINTS[@]} -eq 0 ]]; then
  echo "[grid] No grid points generated; check inputs" >&2
  exit 1
fi
echo "[grid] grid points: ${#GRID_POINTS[@]}"

extra_args=()
[[ -n "${MODEL_ID}" ]] && extra_args+=(--model_id "${MODEL_ID}")
[[ -n "${DATASET_ID}" ]] && extra_args+=(--dataset_id "${DATASET_ID}")
[[ -n "${DATASET_REVISION}" ]] && extra_args+=(--dataset_revision "${DATASET_REVISION}")

num_gpus=${#GPUS[@]}
if [[ ${num_gpus} -eq 0 ]]; then
  echo "[grid] No GPUs specified; set GPUS=\"0 1 2 ...\"" >&2
  exit 2
fi

echo "[grid] dispatching ${#GRID_POINTS[@]} points across ${num_gpus} GPU workers"
pids=()
for gpu_idx in "${!GPUS[@]}"; do
  gpu=${GPUS[$gpu_idx]}
  (
    set +e
    worker_rc=0
    for ((point_idx=gpu_idx; point_idx<${#GRID_POINTS[@]}; point_idx+=num_gpus)); do
      point=${GRID_POINTS[$point_idx]}
      IFS=',' read -r gi gj lr tok <<<"${point}"
      log="${LOG_DIR}/run_${gi}_${gj}.log"
      echo "[grid] GPU ${gpu} -> (${gi},${gj}) lr=${lr} tok=${tok} :: ${log}"

      if ! (
        CUDA_VISIBLE_DEVICES=${gpu} HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-0} \
          FRACTAL_STORAGE_DIR=${FRACTAL_STORAGE_DIR} \
          WANDB_PROJECT=${WANDB_PROJECT} WANDB_ENTITY=${WANDB_ENTITY} \
          PYTHONUNBUFFERED=1 \
          uv run python -m src.finetune \
            --run "${RUN_PREFIX}" \
            --grid_sweep_id "${GRID_SWEEP_ID}" \
            --grid_i "${gi}" \
            --grid_j "${gj}" \
            --learning_rate "${lr}" \
            --num_tokens "${tok}" \
            --eval_every 0 \
            --log_every 5 \
            --save_artifacts False \
            --deterministic True \
            --seed "${SEED}" \
            --max_seq_len "${MAX_SEQ_LEN}" \
            --wandb_tags "${FINETUNE_WANDB_TAGS}" \
            "${extra_args[@]}" \
            2>&1 | tee "${log}"
      ); then
        echo "[grid] FAILED gpu=${gpu} point=(${gi},${gj}) log=${log}" >&2
        worker_rc=1
      fi
    done
    exit "${worker_rc}"
  ) &
  pids+=($!)
done

rc=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    rc=1
  fi
done
echo "[grid] workers complete (rc=${rc})"

# Summarize results
uv run python - <<PY
import json, pathlib, re
log_dir = pathlib.Path("${LOG_DIR}")
pattern = re.compile(r"Final: loss=([0-9.]+)")
rows = []
for log in sorted(log_dir.glob("run_*.log")):
    txt = log.read_text()
    match = pattern.findall(txt)
    rows.append({"log": log.name, "final_loss": float(match[-1]) if match else None})
print(json.dumps(rows, indent=2))
unique = {r["final_loss"] for r in rows if r["final_loss"] is not None}
if len(unique) == 1:
    print(f"[grid] All runs matched final_loss={unique.pop():.4f}")
else:
    print(f"[grid] Final losses varied across runs: {sorted(unique)}")
PY
exit ${rc}
