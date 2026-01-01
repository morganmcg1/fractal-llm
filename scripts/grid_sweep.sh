#!/usr/bin/env bash
# Parallel single-GPU grid sweep launcher for fractal LR×tokens grids.
# Distributes grid points across available GPUs (round-robin, bounded concurrency),
# logs per-run output, and summarizes final losses.

set -euo pipefail
[[ ${DEBUG_GRID:-0} -eq 1 ]] && set -x

RUN_PREFIX=${RUN_PREFIX:-grid}
RES=${RES:-4}                 # grid resolution per axis (RES x RES points)
LR_MIN=${LR_MIN:-1e-5}
LR_MAX=${LR_MAX:-1e-3}
TOK_MIN=${TOK_MIN:-5e3}
TOK_MAX=${TOK_MAX:-5e5}
GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
LOG_DIR=${LOG_DIR:-results/grid_logs/${RUN_PREFIX}}
MODEL_ID=${MODEL_ID:-${MODEL_ARTIFACT:-}}
DATASET_ID=${DATASET_ID:-}
DATASET_REVISION=${DATASET_REVISION:-}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-256}
TOKENS_PER_RUN=${TOKENS_PER_RUN:-}  # optional override: fixed tokens instead of TOK_MIN..MAX grid
LR_FIXED=${LR_FIXED:-}              # optional override: fixed LR instead of LR_MIN..MAX grid

mkdir -p "${LOG_DIR}"
echo "[grid] logging to ${LOG_DIR}"

# Build grid points
mapfile -t GRID_POINTS < <(python - <<PY
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

max_jobs=${#GPUS[@]}
job_idx=0
cmd_file=$(mktemp)
trap 'st=$?; [[ ${DEBUG_GRID:-0} -eq 1 ]] && echo "[grid] exit status $st, cleaning ${cmd_file}"; rm -f "$cmd_file"' EXIT

for point in "${GRID_POINTS[@]}"; do
  [[ ${DEBUG_GRID:-0} -eq 1 ]] && echo "[grid] point ${job_idx}: ${point}"
  IFS=',' read -r gi gj lr tok <<<"${point}"
  gpu=${GPUS[$((job_idx % max_jobs))]}
  log="${LOG_DIR}/run_${gi}_${gj}.log"
  echo "[grid] GPU ${gpu} -> (${gi},${gj}) lr=${lr} tok=${tok} :: ${log}"

  cmd="CUDA_VISIBLE_DEVICES=${gpu} WANDB_MODE=disabled HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-0} \
python -m src.finetune \
  --run ${RUN_PREFIX}-g${gi}-${gj} \
  --grid_i ${gi} \
  --grid_j ${gj} \
  --learning_rate ${lr} \
  --num_tokens ${tok} \
  --eval_every 0 \
  --log_every 5 \
  --save_artifacts False \
  --deterministic True \
  --max_seq_len ${MAX_SEQ_LEN}"

  for arg in "${extra_args[@]}"; do
    cmd+=" ${arg}"
  done
  cmd+=" >${log} 2>&1"
  echo "${cmd}" >> "${cmd_file}"
  ((job_idx += 1))
done
echo "[grid] built ${job_idx} commands at ${cmd_file}"

# Run commands with bounded parallelism (one per GPU)
echo "[grid] dispatching ${#GRID_POINTS[@]} commands with ${max_jobs} workers"
set +e
xargs -P "${max_jobs}" -I CMD bash -lc "CMD" < "${cmd_file}"
rc=$?
set -e
echo "[grid] xargs exit code: ${rc}"

# Summarize results
python - <<PY
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
exit ${rc:-0}
