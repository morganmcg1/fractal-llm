#!/usr/bin/env bash
# Parallel single-GPU grid sweep launcher for fractal grids.
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
SWEEP_AXES=${SWEEP_AXES:-lr_tokens}  # lr_tokens | matrix_unembedding
RES=${RES:-4}                        # grid resolution per axis (RES x RES points)
LR_MIN=${LR_MIN:-1e-5}
LR_MAX=${LR_MAX:-1e-3}
TOK_MIN=${TOK_MIN:-5e3}
TOK_MAX=${TOK_MAX:-5e5}
MATRIX_LR_MIN=${MATRIX_LR_MIN:-1e-4}
MATRIX_LR_MAX=${MATRIX_LR_MAX:-3e-4}
UNEMBEDDING_LR_MIN=${UNEMBEDDING_LR_MIN:-2e-5}
UNEMBEDDING_LR_MAX=${UNEMBEDDING_LR_MAX:-1e-4}
GPU_IDS_STR=${GPUS-}
FRACTAL_STORAGE_DIR=${FRACTAL_STORAGE_DIR:-/var/tmp/fractal-llm}
LOG_DIR=${LOG_DIR:-${FRACTAL_STORAGE_DIR}/results/grid_logs/${RUN_PREFIX}}
MODEL_ID=${MODEL_ID:-${MODEL_ARTIFACT:-}}
DATASET_ID=${DATASET_ID:-}
DATASET_REVISION=${DATASET_REVISION:-}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-1024}
TOKENS_PER_RUN=${TOKENS_PER_RUN:-}  # optional override: fixed num_tokens instead of TOK_MIN..MAX grid
LR_FIXED=${LR_FIXED:-}              # optional override: fixed learning_rate instead of LR_MIN..MAX grid
SEED=${SEED:-999}
WANDB_PROJECT=${WANDB_PROJECT:-fractal-llm}
WANDB_ENTITY=${WANDB_ENTITY:-morgy}
FINETUNE_WANDB_TAGS=${FINETUNE_WANDB_TAGS:-fractal-grid}
MAX_RETRIES=${MAX_RETRIES:-3}       # per-point retries for transient failures (total attempts = 1 + MAX_RETRIES)
RETRY_BACKOFF_S=${RETRY_BACKOFF_S:-5}  # base backoff seconds (exponential-ish via multiplier)
RETRY_PATTERNS=${RETRY_PATTERNS:-"ReadTimeout|Read timed out|ConnectionPool|ConnectionError|Temporary failure in name resolution|502|503|504"}
SKIP_COMPLETED=${SKIP_COMPLETED:-1} # if 1, skip points whose log already contains a Final: line
LOG_SUMMARY=${LOG_SUMMARY:-1}       # if 1, log a W&B grid-summary run at the end (single-pod/local only by default)

# Multi-devpod orchestration (run this script locally with DEVPODS set).
# Example:
#   DEVPODS="fractal-llm-1 fractal-llm-2 fractal-llm-3" RES=16 TOKENS_PER_RUN=250000 \
#   SWEEP_AXES=matrix_unembedding MATRIX_LR_MIN=1e-4 MATRIX_LR_MAX=3e-2 \
#   UNEMBEDDING_LR_MIN=2e-5 UNEMBEDDING_LR_MAX=6e-3 \
#   ./scripts/grid_sweep.sh
DEVPODS_STR=${DEVPODS:-}            # space/comma-separated devpod workspace names (e.g., "fractal-llm-1 fractal-llm-2")
GRID_SWEEP_ROLE=${GRID_SWEEP_ROLE:-}  # orchestrator|worker (internal)
POD_INDEX=${POD_INDEX:-0}           # worker pod shard index in [0, NUM_PODS)
NUM_PODS=${NUM_PODS:-1}             # number of pods participating in the sweep
DEVPOD_NAME=${DEVPOD_NAME:-}        # optional label for logs/tags
DEVPOD_WORKDIR=${DEVPOD_WORKDIR:-}  # if unset in orchestrator mode, defaults per pod to /workspaces/<devpod-name>
DEVPOD_TMUX_SESSION=${DEVPOD_TMUX_SESSION:-grid_${RUN_PREFIX}}
AUTO_PULL=${AUTO_PULL:-1}           # if 1, run git pull --ff-only on each devpod before starting
AUTO_RESET=${AUTO_RESET:-1}         # if 1, run git reset --hard HEAD after pulling (restores missing tracked files)
AUTO_UV_SYNC=${AUTO_UV_SYNC:-1}     # if 1, run uv sync --frozen on each devpod before starting
WAIT_FOR_COMPLETION=${WAIT_FOR_COMPLETION:-0}  # if 1, wait for all devpod tmux sessions to exit
POLL_INTERVAL_S=${POLL_INTERVAL_S:-30}
COLLECT_LOGS=${COLLECT_LOGS:-0}     # if 1, stream logs back to local LOG_DIR after completion
SUMMARY_AFTER_COLLECT=${SUMMARY_AFTER_COLLECT:-0}  # if 1, run grid_sweep_summary locally after collecting logs

_quote() { printf "%q" "$1"; }
_assign() {
  local key=$1
  local val=${2-}
  if [[ -z "${val}" ]]; then
    echo "${key}="
  else
    echo "${key}=$(_quote "${val}")"
  fi
}

if [[ -n "${DEVPODS_STR}" ]] && [[ "${GRID_SWEEP_ROLE}" != "worker" ]]; then
  if ! command -v devpod >/dev/null 2>&1; then
    echo "[grid] DEVPODS is set but devpod CLI not found in PATH" >&2
    exit 2
  fi

  # Normalize separators (commas -> spaces), then split.
  DEVPODS_STR="${DEVPODS_STR//,/ }"
  # shellcheck disable=SC2206
  DEVPODS=(${DEVPODS_STR})
  NUM_PODS=${#DEVPODS[@]}
  if [[ "${NUM_PODS}" -le 0 ]]; then
    echo "[grid] No devpods specified. Set DEVPODS=\"fractal-llm-1 fractal-llm-2 ...\"" >&2
    exit 2
  fi

  echo "[grid] multi-devpod orchestrator mode"
  echo "[grid] devpods: ${DEVPODS[*]}"
  echo "[grid] sweep_axes=${SWEEP_AXES} res=${RES} tokens_per_run=${TOKENS_PER_RUN:-<grid>}"
  echo "[grid] run_prefix=${RUN_PREFIX} sweep_id=${GRID_SWEEP_ID} tmux_session=${DEVPOD_TMUX_SESSION}"

  # Launch a worker tmux session on each devpod.
  pids=()
  for pod_idx in "${!DEVPODS[@]}"; do
    pod="${DEVPODS[$pod_idx]}"
    echo "[grid] launching worker ${pod_idx}/${NUM_PODS} on devpod=${pod}"
    pod_workdir="${DEVPOD_WORKDIR:-/workspaces/${pod}}"

    # Build env assignments for the tmux command (shell-escaped).
    env_assign=(
      "GRID_SWEEP_ROLE=worker"
      "DEVPODS="
      "$(_assign POD_INDEX "${pod_idx}")"
      "$(_assign NUM_PODS "${NUM_PODS}")"
      "$(_assign DEVPOD_NAME "${pod}")"
      "$(_assign DEVPOD_WORKDIR "${pod_workdir}")"
      "$(_assign DEVPOD_TMUX_SESSION "${DEVPOD_TMUX_SESSION}")"
      "$(_assign RUN_PREFIX "${RUN_PREFIX}")"
      "$(_assign GRID_SWEEP_ID "${GRID_SWEEP_ID}")"
      "$(_assign SWEEP_AXES "${SWEEP_AXES}")"
      "$(_assign RES "${RES}")"
      "$(_assign LR_MIN "${LR_MIN}")"
      "$(_assign LR_MAX "${LR_MAX}")"
      "$(_assign TOK_MIN "${TOK_MIN}")"
      "$(_assign TOK_MAX "${TOK_MAX}")"
      "$(_assign MATRIX_LR_MIN "${MATRIX_LR_MIN}")"
      "$(_assign MATRIX_LR_MAX "${MATRIX_LR_MAX}")"
      "$(_assign UNEMBEDDING_LR_MIN "${UNEMBEDDING_LR_MIN}")"
      "$(_assign UNEMBEDDING_LR_MAX "${UNEMBEDDING_LR_MAX}")"
      "$(_assign FRACTAL_STORAGE_DIR "${FRACTAL_STORAGE_DIR}")"
      "$(_assign LOG_DIR "${LOG_DIR}")"
      "$(_assign MODEL_ID "${MODEL_ID}")"
      "$(_assign DATASET_ID "${DATASET_ID}")"
      "$(_assign DATASET_REVISION "${DATASET_REVISION}")"
      "$(_assign MAX_SEQ_LEN "${MAX_SEQ_LEN}")"
      "$(_assign TOKENS_PER_RUN "${TOKENS_PER_RUN}")"
      "$(_assign LR_FIXED "${LR_FIXED}")"
      "$(_assign SEED "${SEED}")"
      "$(_assign WANDB_PROJECT "${WANDB_PROJECT}")"
      "$(_assign WANDB_ENTITY "${WANDB_ENTITY}")"
      "$(_assign FINETUNE_WANDB_TAGS "${FINETUNE_WANDB_TAGS}")"
      "$(_assign MAX_RETRIES "${MAX_RETRIES}")"
      "$(_assign RETRY_BACKOFF_S "${RETRY_BACKOFF_S}")"
      "$(_assign RETRY_PATTERNS "${RETRY_PATTERNS}")"
      "$(_assign SKIP_COMPLETED "${SKIP_COMPLETED}")"
      "LOG_SUMMARY=0"
    )
    tmux_cmd="${env_assign[*]} ./scripts/grid_sweep.sh"

    devpod --silent ssh "${pod}" --command "bash -lc '
      set -euo pipefail
      cd ${pod_workdir}
      if [[ -f .env ]]; then source .env; fi
      if [[ ${AUTO_PULL} -eq 1 ]]; then
        git pull --ff-only || echo \"[grid] WARNING: git pull failed on ${pod}\"
      fi
      if [[ ${AUTO_RESET} -eq 1 ]]; then
        git reset --hard HEAD || echo \"[grid] WARNING: git reset failed on ${pod}\"
      fi
      if [[ ${AUTO_UV_SYNC} -eq 1 ]]; then
        uv sync --frozen || echo \"[grid] WARNING: uv sync failed on ${pod}\"
      fi
      if tmux has-session -t ${DEVPOD_TMUX_SESSION} 2>/dev/null; then
        echo \"[grid] ERROR: tmux session already exists on ${pod}: ${DEVPOD_TMUX_SESSION}\" >&2
        exit 3
      fi
      tmux new-session -d -s ${DEVPOD_TMUX_SESSION} -c ${pod_workdir} \"${tmux_cmd}\"
      echo \"[grid] started ${pod}: tmux attach -t ${DEVPOD_TMUX_SESSION}\"
    '" &
    pids+=($!)
  done

  launch_rc=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      launch_rc=1
    fi
  done
  if [[ "${launch_rc}" -ne 0 ]]; then
    echo "[grid] ERROR: one or more devpods failed to launch" >&2
    exit 3
  fi

  echo "[grid] all devpods launched"
  echo "[grid] monitor:"
  for pod in "${DEVPODS[@]}"; do
    echo "  devpod ssh ${pod}   # then: tmux attach -t ${DEVPOD_TMUX_SESSION}"
  done

  if [[ "${WAIT_FOR_COMPLETION}" == "1" ]]; then
    echo "[grid] waiting for completion (poll=${POLL_INTERVAL_S}s)"
    while true; do
      alive=0
      for pod in "${DEVPODS[@]}"; do
        if devpod --silent ssh "${pod}" --command "bash -lc 'tmux has-session -t ${DEVPOD_TMUX_SESSION} 2>/dev/null'"; then
          alive=$((alive + 1))
        fi
      done
      if [[ "${alive}" -eq 0 ]]; then
        break
      fi
      echo "[grid] ${alive}/${NUM_PODS} devpods still running..."
      sleep "${POLL_INTERVAL_S}"
    done
    echo "[grid] all devpods complete"
  fi

  if [[ "${COLLECT_LOGS}" == "1" ]]; then
    mkdir -p "${LOG_DIR}"
    echo "[grid] collecting logs into ${LOG_DIR}"
    for pod in "${DEVPODS[@]}"; do
      echo "[grid] collecting from ${pod}"
      devpod --silent ssh "${pod}" --command "bash -lc '
        set -euo pipefail
        if [[ -d ${LOG_DIR} ]]; then
          cd ${LOG_DIR}
          tar -cf - run_*.log run_*.attempt*.log 2>/dev/null || tar -cf - --files-from /dev/null
        else
          tar -cf - --files-from /dev/null
        fi
      '" | tar -C "${LOG_DIR}" -xf -
    done
  fi

  if [[ "${SUMMARY_AFTER_COLLECT}" == "1" ]]; then
    if [[ "${COLLECT_LOGS}" != "1" ]]; then
      echo "[grid] SUMMARY_AFTER_COLLECT=1 requires COLLECT_LOGS=1" >&2
      exit 2
    fi
    echo "[grid] logging combined grid summary from ${LOG_DIR}"
    uv run python -m src.grid_sweep_summary \
      --log_dir "${LOG_DIR}" \
      --run_prefix "${RUN_PREFIX}" \
      --grid_sweep_id "${GRID_SWEEP_ID}" \
      --sweep_axes "${SWEEP_AXES}" \
      --resolution "${RES}" \
      --wandb_project "${WANDB_PROJECT}" \
      --wandb_entity "${WANDB_ENTITY}" \
      --wandb_tags "${FINETUNE_WANDB_TAGS}" \
      --storage_dir "${FRACTAL_STORAGE_DIR}"
  fi

  exit 0
fi

mkdir -p "${LOG_DIR}"
echo "[grid] logging to ${LOG_DIR}"
echo "[grid] W&B project=${WANDB_PROJECT} entity=${WANDB_ENTITY} tags=${FINETUNE_WANDB_TAGS} sweep_id=${GRID_SWEEP_ID}"
echo "[grid] sweep_axes=${SWEEP_AXES} res=${RES} tokens_per_run=${TOKENS_PER_RUN:-<grid>}"

# Auto-detect GPUs when GPUS isn't set (safe default for devpod 1×GPU workspaces).
GPU_IDS=()
if [[ -n "${GPU_IDS_STR}" ]]; then
  # shellcheck disable=SC2206
  GPU_IDS=(${GPU_IDS_STR})
else
  gpu_count=0
  if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_count=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
  fi
  if [[ "${gpu_count}" -le 0 ]]; then
    gpu_count=$(uv run python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)
  fi
  if [[ "${gpu_count}" -le 0 ]]; then
    echo "[grid] No GPUs detected. Set GPUS=\"0\" (CPU runs are not supported here)." >&2
    exit 2
  fi
  for ((i=0; i<gpu_count; i++)); do
    GPU_IDS+=("${i}")
  done
fi

# Build grid points
mapfile -t GRID_POINTS < <(uv run python - <<PY
import numpy as np, os
mode = os.environ.get("SWEEP_AXES", "lr_tokens")
res = int(os.environ.get("RES", "4"))
tokens_per_run = os.environ.get("TOKENS_PER_RUN")

if mode == "lr_tokens":
    lr_fixed = os.environ.get("LR_FIXED")
    if lr_fixed:
        lrs = np.array([float(lr_fixed)])
    else:
        lr_min = float(os.environ.get("LR_MIN", "1e-5"))
        lr_max = float(os.environ.get("LR_MAX", "1e-3"))
        lrs = np.logspace(np.log10(lr_min), np.log10(lr_max), res)

    if tokens_per_run:
        toks = np.array([int(float(tokens_per_run))])
    else:
        tok_min = float(os.environ.get("TOK_MIN", "5e3"))
        tok_max = float(os.environ.get("TOK_MAX", "5e5"))
        toks = np.logspace(np.log10(tok_min), np.log10(tok_max), res).astype(int)

    for i, lr in enumerate(lrs):
        for j, tok in enumerate(toks):
            print(f"{i},{j},{lr:.6g},{int(tok)}")

elif mode == "matrix_unembedding":
    if not tokens_per_run:
        raise SystemExit("TOKENS_PER_RUN is required for SWEEP_AXES=matrix_unembedding")
    tok = int(float(tokens_per_run))

    mmin = float(os.environ.get("MATRIX_LR_MIN", "1e-4"))
    mmax = float(os.environ.get("MATRIX_LR_MAX", "3e-4"))
    umin = float(os.environ.get("UNEMBEDDING_LR_MIN", "2e-5"))
    umax = float(os.environ.get("UNEMBEDDING_LR_MAX", "1e-4"))
    matrix_lrs = np.logspace(np.log10(mmin), np.log10(mmax), res)
    unembedding_lrs = np.logspace(np.log10(umin), np.log10(umax), res)

    for i, mlr in enumerate(matrix_lrs):
        for j, ulr in enumerate(unembedding_lrs):
            print(f"{i},{j},{mlr:.6g},{ulr:.6g},{tok}")

else:
    raise SystemExit(f"Unknown SWEEP_AXES={mode!r} (expected lr_tokens or matrix_unembedding)")
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

num_gpus=${#GPU_IDS[@]}
if [[ ${num_gpus} -eq 0 ]]; then
  echo "[grid] No GPUs specified; set GPUS=\"0 1 2 ...\"" >&2
  exit 2
fi

if [[ "${NUM_PODS}" -le 0 ]]; then
  echo "[grid] NUM_PODS must be >= 1; got ${NUM_PODS}" >&2
  exit 2
fi
if [[ "${POD_INDEX}" -lt 0 ]] || [[ "${POD_INDEX}" -ge "${NUM_PODS}" ]]; then
  echo "[grid] POD_INDEX must satisfy 0 <= POD_INDEX < NUM_PODS; got POD_INDEX=${POD_INDEX} NUM_PODS=${NUM_PODS}" >&2
  exit 2
fi

echo "[grid] dispatching ${#GRID_POINTS[@]} points across ${num_gpus} GPU workers (pod ${POD_INDEX}/${NUM_PODS} ${DEVPOD_NAME})"
pids=()
for gpu_idx in "${!GPU_IDS[@]}"; do
  gpu=${GPU_IDS[$gpu_idx]}
  (
    set +e
    worker_rc=0
    start_idx=$((POD_INDEX + NUM_PODS * gpu_idx))
    stride=$((NUM_PODS * num_gpus))
    for ((point_idx=start_idx; point_idx<${#GRID_POINTS[@]}; point_idx+=stride)); do
      point=${GRID_POINTS[$point_idx]}
      lr_args=()
      if [[ "${SWEEP_AXES}" == "matrix_unembedding" ]]; then
        IFS=',' read -r gi gj mlr ulr tok <<<"${point}"
        lr_desc="matrix_lr=${mlr} unembedding_lr=${ulr}"
        lr_args+=(--matrix_lr "${mlr}" --unembedding_lr "${ulr}")
      else
        IFS=',' read -r gi gj lr tok <<<"${point}"
        lr_desc="learning_rate=${lr}"
        lr_args+=(--learning_rate "${lr}")
      fi
      log="${LOG_DIR}/run_${gi}_${gj}.log"
      echo "[grid] GPU ${gpu} -> (${gi},${gj}) ${lr_desc} tok=${tok} :: ${log}"

      if [[ "${SKIP_COMPLETED}" == "1" ]] && [[ -f "${log}" ]] && grep -q "^Final: loss=" "${log}"; then
        echo "[grid] SKIP (${gi},${gj}) already has Final: in ${log}"
        continue
      fi

      attempt=0
      point_ok=0
      while [[ "${attempt}" -le "${MAX_RETRIES}" ]]; do
        attempt=$((attempt + 1))
        attempt_log="${LOG_DIR}/run_${gi}_${gj}.attempt${attempt}.log"
        echo "[grid] GPU ${gpu} -> (${gi},${gj}) attempt ${attempt}/$((MAX_RETRIES + 1)) :: ${attempt_log}"

        CUDA_VISIBLE_DEVICES=${gpu} HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-0} \
          FRACTAL_STORAGE_DIR=${FRACTAL_STORAGE_DIR} \
          WANDB_PROJECT=${WANDB_PROJECT} WANDB_ENTITY=${WANDB_ENTITY} \
          PYTHONUNBUFFERED=1 \
          uv run python -m src.finetune \
            --run "${RUN_PREFIX}" \
            --grid_sweep_id "${GRID_SWEEP_ID}" \
            --grid_i "${gi}" \
            --grid_j "${gj}" \
            "${lr_args[@]}" \
            --num_tokens "${tok}" \
            --eval_every 0 \
            --eval_batches 0 \
            --log_every 5 \
            --save_artifacts False \
            --deterministic True \
            --seed "${SEED}" \
            --max_seq_len "${MAX_SEQ_LEN}" \
            --wandb_tags "${FINETUNE_WANDB_TAGS}" \
            "${extra_args[@]}" \
            2>&1 | tee "${attempt_log}"
        cmd_rc=${PIPESTATUS[0]}

        # Canonical log always points at the last attempt (success or final failure).
        cp -f "${attempt_log}" "${log}"

        if [[ "${cmd_rc}" -eq 0 ]]; then
          point_ok=1
          break
        fi

        if grep -Eq "${RETRY_PATTERNS}" "${attempt_log}"; then
          if [[ "${attempt}" -le "${MAX_RETRIES}" ]]; then
            sleep_s=$((RETRY_BACKOFF_S * attempt))
            echo "[grid] RETRY (${gi},${gj}) in ${sleep_s}s (rc=${cmd_rc}; matched RETRY_PATTERNS)"
            sleep "${sleep_s}"
            continue
          fi
        fi

        echo "[grid] FAILED gpu=${gpu} point=(${gi},${gj}) rc=${cmd_rc} log=${log}" >&2
        worker_rc=1
        break
      done
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

# Summarize results + log a single W&B "grid-summary" run with the image/table.
summary_rc=0
if [[ "${LOG_SUMMARY}" == "1" ]]; then
  if ! uv run python -m src.grid_sweep_summary \
    --log_dir "${LOG_DIR}" \
    --run_prefix "${RUN_PREFIX}" \
    --grid_sweep_id "${GRID_SWEEP_ID}" \
    --sweep_axes "${SWEEP_AXES}" \
    --resolution "${RES}" \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_entity "${WANDB_ENTITY}" \
    --wandb_tags "${FINETUNE_WANDB_TAGS}" \
    --storage_dir "${FRACTAL_STORAGE_DIR}"; then
    summary_rc=1
    echo "[grid] WARNING: failed to log grid summary to W&B (log_dir=${LOG_DIR})" >&2
  fi
fi

if [[ ${summary_rc} -ne 0 ]]; then
  rc=1
fi
exit ${rc}
