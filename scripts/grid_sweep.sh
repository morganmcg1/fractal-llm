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

echo "[grid] dispatching ${#GRID_POINTS[@]} points across ${num_gpus} GPU workers"
pids=()
for gpu_idx in "${!GPU_IDS[@]}"; do
  gpu=${GPU_IDS[$gpu_idx]}
  (
    set +e
    worker_rc=0
    for ((point_idx=gpu_idx; point_idx<${#GRID_POINTS[@]}; point_idx+=num_gpus)); do
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

# Summarize results + log a single W&B "grid-summary" run with the image/table.
summary_rc=0
if ! uv run python - <<'PY'
import json
import math
import os
import pathlib
import re
from dataclasses import dataclass, asdict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage

import wandb


@dataclass
class PointResult:
    grid_i: int
    grid_j: int
    num_tokens: int
    tokens_seen: int | None
    final_loss: float | None
    converged: bool | None
    error: str | None
    log: str
    learning_rate: float | None = None
    matrix_lr: float | None = None
    unembedding_lr: float | None = None


def _parse_overrides(text: str) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for m in re.finditer(r"^Overriding: ([a-zA-Z0-9_]+) = (.+)$", text, flags=re.MULTILINE):
        overrides[m.group(1)] = m.group(2).strip()
    return overrides


def _parse_final_line(text: str) -> tuple[float | None, int | None, bool | None]:
    m = re.search(
        r"Final: loss=([0-9.]+) tokens_seen=([0-9,]+) converged=(True|False)",
        text,
    )
    if not m:
        return None, None, None
    loss = float(m.group(1))
    tokens_seen = int(m.group(2).replace(",", ""))
    converged = m.group(3) == "True"
    return loss, tokens_seen, converged


def _parse_error(text: str) -> str | None:
    # Keep the last error if multiple occur.
    errs = re.findall(r"^\[ERROR\] training failed: (.+)$", text, flags=re.MULTILINE)
    return errs[-1].strip() if errs else None


def _fractal_cmap():
    colors_diverged = ["#8B0000", "#CD5C5C", "#FA8072"]  # Dark red to light red
    colors_converged = ["#ADD8E6", "#4169E1", "#00008B"]  # Light blue to dark blue
    colors = colors_diverged + ["#FFFFFF"] + colors_converged
    positions = [0.0, 0.15, 0.35, 0.5, 0.65, 0.85, 1.0]
    return LinearSegmentedColormap.from_list("fractal", list(zip(positions, colors)))


def _build_grids(points: list[PointResult], resolution: int):
    fractal_grid = np.zeros((resolution, resolution))
    loss_grid = np.full((resolution, resolution), np.nan)
    convergence_grid = np.zeros((resolution, resolution))

    converged_losses: list[float] = []
    for p in points:
        if p.converged and p.final_loss is not None and math.isfinite(p.final_loss):
            loss_grid[p.grid_i, p.grid_j] = p.final_loss
            converged_losses.append(p.final_loss)
            convergence_grid[p.grid_i, p.grid_j] = 1
        else:
            convergence_grid[p.grid_i, p.grid_j] = 0

    if converged_losses:
        loss_min, loss_max = min(converged_losses), max(converged_losses)
        loss_range = loss_max - loss_min if loss_max > loss_min else 1.0
    else:
        loss_min, loss_max, loss_range = 0.0, 1.0, 1.0

    for p in points:
        if p.converged and p.final_loss is not None and math.isfinite(p.final_loss):
            intensity = (p.final_loss - loss_min) / loss_range
            fractal_grid[p.grid_i, p.grid_j] = 0.3 + 0.7 * intensity
        else:
            fractal_grid[p.grid_i, p.grid_j] = -1.0

    return fractal_grid, loss_grid, convergence_grid


def _compute_fractal(convergence_grid: np.ndarray):
    boundary = ndimage.binary_dilation(convergence_grid) ^ ndimage.binary_erosion(convergence_grid)

    def box_count(binary_image, box_size):
        h, w = binary_image.shape
        count = 0
        for i in range(0, h, box_size):
            for j in range(0, w, box_size):
                if binary_image[i : i + box_size, j : j + box_size].any():
                    count += 1
        return count

    sizes = [s for s in [2, 4, 8, 16, 32, 64] if s < boundary.shape[0]]
    if sizes:
        counts = [box_count(boundary, s) for s in sizes]
        log_sizes = np.log(sizes)
        log_counts = np.log(np.array(counts) + 1)
        coeffs = np.polyfit(log_sizes, log_counts, 1)
        fractal_dimension = -coeffs[0]
    else:
        counts = []
        fractal_dimension = float("nan")
    return {
        "fractal_dimension": float(fractal_dimension),
        "box_sizes": sizes,
        "box_counts": [int(c) for c in counts],
        "boundary_pixels": int(boundary.sum()),
        "converged_ratio": float(convergence_grid.sum() / convergence_grid.size),
    }


def _safe_extent(vmin: float, vmax: float) -> tuple[float, float]:
    if vmin <= 0 or vmax <= 0:
        raise ValueError(f"Extent must be positive; got vmin={vmin} vmax={vmax}")
    if vmin == vmax:
        # Expand slightly to avoid zero-width extent.
        vmin = vmin * 0.9
        vmax = vmax * 1.1
    return math.log10(vmin), math.log10(vmax)


log_dir = pathlib.Path(os.environ["LOG_DIR"])
run_prefix = os.environ["RUN_PREFIX"]
grid_sweep_id = os.environ.get("GRID_SWEEP_ID", run_prefix)
mode = os.environ.get("SWEEP_AXES", "lr_tokens")
resolution = int(os.environ.get("RES", "4"))
wandb_project = os.environ.get("WANDB_PROJECT", "fractal-llm")
wandb_entity = os.environ.get("WANDB_ENTITY", "morgy")
wandb_tags_raw = os.environ.get("FINETUNE_WANDB_TAGS", "fractal-grid")
fractal_storage_dir = os.environ.get("FRACTAL_STORAGE_DIR", "/var/tmp/fractal-llm")

# Keep W&B files out of the workspace (devpod has a tiny /workspaces volume).
os.environ.setdefault("WANDB_DIR", str(pathlib.Path(fractal_storage_dir) / "wandb"))
os.environ.setdefault("WANDB_CONFIG_DIR", str(pathlib.Path(fractal_storage_dir) / "wandb" / "config"))
os.environ.setdefault("WANDB_CACHE_DIR", str(pathlib.Path(fractal_storage_dir) / "wandb" / "cache"))

log_files = sorted(log_dir.glob("run_*.log"))
if not log_files:
    raise SystemExit(f"No logs found in {log_dir}")

points: list[PointResult] = []
name_re = re.compile(r"run_(\d+)_(\d+)\.log$")
for log_path in log_files:
    m = name_re.search(log_path.name)
    if not m:
        continue
    gi = int(m.group(1))
    gj = int(m.group(2))
    txt = log_path.read_text(errors="replace")
    overrides = _parse_overrides(txt)
    final_loss, tokens_seen, converged = _parse_final_line(txt)
    err = _parse_error(txt)

    num_tokens = int(str(overrides.get("num_tokens", "0")).replace("_", ""))
    learning_rate = float(overrides["learning_rate"]) if "learning_rate" in overrides else None
    matrix_lr = float(overrides["matrix_lr"]) if "matrix_lr" in overrides else None
    unembedding_lr = float(overrides["unembedding_lr"]) if "unembedding_lr" in overrides else None

    points.append(
        PointResult(
            grid_i=gi,
            grid_j=gj,
            num_tokens=num_tokens,
            tokens_seen=tokens_seen,
            final_loss=final_loss,
            converged=converged,
            error=err,
            log=log_path.name,
            learning_rate=learning_rate,
            matrix_lr=matrix_lr,
            unembedding_lr=unembedding_lr,
        )
    )

if len(points) != resolution * resolution:
    print(
        f"[grid-summary] WARNING: expected {resolution*resolution} points, found {len(points)} logs",
        flush=True,
    )

fractal_grid, loss_grid, convergence_grid = _build_grids(points, resolution)
fractal = _compute_fractal(convergence_grid)

if mode == "matrix_unembedding":
    xs = [p.unembedding_lr for p in points if p.unembedding_lr is not None]
    ys = [p.matrix_lr for p in points if p.matrix_lr is not None]
    x_label = "log₁₀(unembedding lr)"
    y_label = "log₁₀(matrix lr)"
    caption = "Matrix LR × Unembedding LR Grid"
elif mode == "lr_tokens":
    xs = [float(p.num_tokens) for p in points if p.num_tokens is not None and p.num_tokens > 0]
    ys = [p.learning_rate for p in points if p.learning_rate is not None]
    x_label = "log₁₀(tokens)"
    y_label = "log₁₀(learning rate)"
    caption = "LR × Tokens Grid"
else:
    raise SystemExit(f"Unknown SWEEP_AXES={mode!r}")

xmin, xmax = min(xs), max(xs)
ymin, ymax = min(ys), max(ys)
ex0, ex1 = _safe_extent(xmin, xmax)
ey0, ey1 = _safe_extent(ymin, ymax)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
extent = [ex0, ex1, ey0, ey1]

im1 = axes[0].imshow(
    fractal_grid,
    origin="lower",
    aspect="auto",
    cmap=_fractal_cmap(),
    vmin=-1.0,
    vmax=1.0,
    extent=extent,
)
axes[0].set_xlabel(x_label)
axes[0].set_ylabel(y_label)
axes[0].set_title("Trainability Boundary\n(Blue=Converged, Red=Diverged)")
plt.colorbar(im1, ax=axes[0]).set_label("Convergence")

im2 = axes[1].imshow(
    loss_grid,
    origin="lower",
    aspect="auto",
    cmap="viridis_r",
    extent=extent,
)
axes[1].set_xlabel(x_label)
axes[1].set_ylabel(y_label)
axes[1].set_title("Final Loss (converged)")
plt.colorbar(im2, ax=axes[1]).set_label("Loss")

im3 = axes[2].imshow(
    convergence_grid,
    origin="lower",
    aspect="auto",
    cmap="RdBu",
    vmin=0,
    vmax=1,
    extent=extent,
)
axes[2].set_xlabel(x_label)
axes[2].set_ylabel(y_label)
axes[2].set_title("Binary Convergence")
cbar3 = plt.colorbar(im3, ax=axes[2])
cbar3.set_ticks([0, 1])
cbar3.set_ticklabels(["Diverged", "Converged"])

plt.tight_layout()
out_prefix = log_dir / f"grid_summary_{run_prefix}"
img_path = out_prefix.with_suffix(".png")
json_path = out_prefix.with_suffix(".json")
fig.savefig(img_path, dpi=150, facecolor="white")
plt.close(fig)

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(
        {
            "config": {
                "run_prefix": run_prefix,
                "grid_sweep_id": grid_sweep_id,
                "sweep_axes": mode,
                "resolution": resolution,
                "wandb_project": wandb_project,
                "wandb_entity": wandb_entity,
            },
            "fractal": fractal,
            "points": [asdict(p) for p in points],
        },
        f,
        indent=2,
    )

tags = [t.strip() for t in wandb_tags_raw.split(",") if t.strip()]
if not any(t.lower() == "finetune" for t in tags):
    tags.append("finetune")
if grid_sweep_id:
    tags.append(grid_sweep_id)
tags.append("grid-summary")
tags = list(dict.fromkeys(tags))

run_base = run_prefix if run_prefix.endswith("-ft") else f"{run_prefix}-ft"
summary_name = f"{run_base}-grid-summary"

summary_run = wandb.init(
    project=wandb_project,
    entity=wandb_entity,
    name=summary_name,
    config={
        "run_prefix": run_prefix,
        "grid_sweep_id": grid_sweep_id,
        "sweep_axes": mode,
        "resolution": resolution,
        "fractal_dimension": fractal["fractal_dimension"],
        "boundary_pixels": fractal["boundary_pixels"],
        "converged_ratio": fractal["converged_ratio"],
    },
    tags=tags,
    settings=wandb.Settings(init_timeout=300, _service_wait=300),
)
summary_run.log(
    {
        "fractal/image": wandb.Image(str(img_path), caption=caption),
        "fractal/dimension": fractal["fractal_dimension"],
        "fractal/boundary_pixels": fractal["boundary_pixels"],
        "fractal/converged_ratio": fractal["converged_ratio"],
    }
)

if mode == "matrix_unembedding":
    columns = ["grid_i", "grid_j", "matrix_lr", "unembedding_lr", "num_tokens", "tokens_seen", "final_loss", "converged", "error", "log"]
    data = [
        [p.grid_i, p.grid_j, p.matrix_lr, p.unembedding_lr, p.num_tokens, p.tokens_seen, p.final_loss, p.converged, p.error, p.log]
        for p in points
    ]
else:
    columns = ["grid_i", "grid_j", "learning_rate", "num_tokens", "tokens_seen", "final_loss", "converged", "error", "log"]
    data = [
        [p.grid_i, p.grid_j, p.learning_rate, p.num_tokens, p.tokens_seen, p.final_loss, p.converged, p.error, p.log]
        for p in points
    ]
summary_run.log({"results_table": wandb.Table(columns=columns, data=data)})
summary_run.finish()

print(f"[grid-summary] wrote {img_path}")
print(f"[grid-summary] wrote {json_path}")
print(f"[grid-summary] logged W&B run: {wandb_entity}/{wandb_project} :: {summary_name}")
PY
then
  summary_rc=1
  echo "[grid] WARNING: failed to log grid summary to W&B (LOG_DIR=${LOG_DIR})" >&2
fi

if [[ ${summary_rc} -ne 0 ]]; then
  rc=1
fi
exit ${rc}
