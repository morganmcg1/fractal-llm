"""
Summarize a parallel grid sweep launched by scripts/grid_sweep.sh.

This reads the per-point log files (run_<i>_<j>.log), reconstructs the 2D grid,
renders the same 3-panel grid visualization as src/finetune.py, and logs a single
W&B summary run with the image + a results table.

Example:
  uv run python -m src.grid_sweep_summary \
    --log_dir /var/tmp/fractal-llm/results/grid_logs/my-sweep \
    --run_prefix my-sweep \
    --grid_sweep_id my-sweep \
    --sweep_axes matrix_unembedding \
    --resolution 2
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import simple_parsing as sp
import wandb
from matplotlib.colors import LinearSegmentedColormap
from rich.console import Console
from rich.panel import Panel
from scipy import ndimage

console = Console()


@dataclass
class Args:
    """Summarize a grid sweep and log a W&B summary run."""

    log_dir: Path  # Directory containing run_<i>_<j>.log files
    run_prefix: str  # Same prefix passed to finetune as --run
    resolution: int = 4  # Grid resolution per axis (RES)
    sweep_axes: str = "matrix_unembedding"  # matrix_unembedding | lr_tokens
    grid_sweep_id: str = ""  # Tag shared across all points (defaults to run_prefix)
    wandb_project: str = os.environ.get("WANDB_PROJECT", "fractal-llm")
    wandb_entity: str = os.environ.get("WANDB_ENTITY", "morgy")
    wandb_tags: str = os.environ.get("FINETUNE_WANDB_TAGS", "fractal-grid")
    storage_dir: Path = Path(os.environ.get("FRACTAL_STORAGE_DIR", "/var/tmp/fractal-llm"))


@dataclass
class PointResult:
    grid_i: int
    grid_j: int
    num_tokens: int
    tokens_seen: int | None
    final_loss: float | None
    trainable_ratio: float | None
    stable: bool | None
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


def _parse_final_line(text: str) -> tuple[float | None, int | None, bool | None, bool | None, float | None]:
    # New (2026-01): includes stable + trainable_ratio.
    m = re.search(
        r"Final: loss=([0-9.]+) tokens_seen=([0-9,]+) stable=(True|False) converged=(True|False) trainable_ratio=([0-9.]+)",
        text,
    )
    if m:
        loss = float(m.group(1))
        tokens_seen = int(m.group(2).replace(",", ""))
        stable = m.group(3) == "True"
        converged = m.group(4) == "True"
        trainable_ratio = float(m.group(5))
        return loss, tokens_seen, stable, converged, trainable_ratio

    # Old format (backwards compatible)
    m = re.search(
        r"Final: loss=([0-9.]+) tokens_seen=([0-9,]+) converged=(True|False)",
        text,
    )
    if not m:
        return None, None, None, None, None
    loss = float(m.group(1))
    tokens_seen = int(m.group(2).replace(",", ""))
    converged = m.group(3) == "True"
    return loss, tokens_seen, None, converged, None


def _parse_error(text: str) -> str | None:
    errs = re.findall(r"^\[ERROR\] training failed: (.+)$", text, flags=re.MULTILINE)
    return errs[-1].strip() if errs else None


def _fractal_cmap():
    """Create a diverging red-white-blue colormap for trainability visualization.

    Color mapping (vmin=-1, vmax=1):
      -1.0: Dark red (#8B0000) - Not trainable runs (includes both unstable AND stable-but-not-improving)
      -0.5: Salmon (#FA8072) - (unused in practice, all not trainable → -1.0)
       0.0: White (#FFFFFF) - Boundary (unused, trainable starts at 0.3)
       0.3: Light blue (#ADD8E6) - Trainable, highest loss among trainable
       0.65: Royal blue (#4169E1) - Trainable, medium loss
       1.0: Dark blue (#00008B) - Trainable, lowest loss (best)
    """
    # More granular color stops for smoother transitions
    colors = [
        "#8B0000",  # Dark red (not trainable)
        "#B22222",  # Firebrick
        "#CD5C5C",  # Indian red
        "#FA8072",  # Salmon
        "#FFC0CB",  # Pink (light not trainable)
        "#FFFFFF",  # White (boundary)
        "#E0FFFF",  # Light cyan
        "#ADD8E6",  # Light blue (worst trainable)
        "#87CEEB",  # Sky blue
        "#4169E1",  # Royal blue
        "#0000CD",  # Medium blue
        "#00008B",  # Dark blue (best trainable)
    ]
    positions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.65, 0.75, 0.85, 0.92, 1.0]
    return LinearSegmentedColormap.from_list("fractal", list(zip(positions, colors)))


def _build_grids(points: list[PointResult], resolution: int):
    """Build visualization grids from sweep results.

    Returns:
        fractal_grid: Trainability with loss-based intensity (for colormap visualization)
        loss_grid: Final loss values (NaN for non-trainable)
        convergence_grid: Binary trainability (1=trainable, 0=not trainable)
        stability_grid: Binary stability (1=stable/finite loss, 0=unstable/NaN)

    Note: Missing grid points (no data) are filled with NaN and will appear as
    white/transparent in the visualization. Use interpolation='nearest' in imshow
    to avoid anti-aliasing artifacts at boundaries.
    """
    # Initialize with NaN to distinguish missing data from actual 0 values
    fractal_grid = np.full((resolution, resolution), np.nan)
    loss_grid = np.full((resolution, resolution), np.nan)
    convergence_grid = np.full((resolution, resolution), np.nan)
    stability_grid = np.full((resolution, resolution), np.nan)

    converged_losses: list[float] = []
    for p in points:
        # Stability: did training complete with finite loss?
        stability_grid[p.grid_i, p.grid_j] = 1.0 if p.stable else 0.0

        # Trainability: did loss improve?
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
            # Invert intensity: lower loss → darker blue (value closer to 1.0)
            intensity = (p.final_loss - loss_min) / loss_range
            fractal_grid[p.grid_i, p.grid_j] = 1.0 - 0.7 * intensity  # Range: 0.3 (worst) to 1.0 (best)
        else:
            fractal_grid[p.grid_i, p.grid_j] = -1.0

    return fractal_grid, loss_grid, convergence_grid, stability_grid


def _fill_nan_nearest(grid: np.ndarray) -> np.ndarray:
    """Fill NaN values using nearest-neighbor interpolation.

    This is useful for partial sweeps where not all grid points have data.
    Uses scipy.ndimage.distance_transform_edt to find nearest valid values.
    """
    if not np.any(np.isnan(grid)):
        return grid
    mask = np.isnan(grid)
    if mask.all():
        return grid  # All NaN, nothing to interpolate from
    # Find indices of nearest non-NaN values
    from scipy.ndimage import distance_transform_edt
    _, indices = distance_transform_edt(mask, return_distances=True, return_indices=True)
    filled = grid[tuple(indices)]
    return filled


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
        vmin = vmin * 0.9
        vmax = vmax * 1.1
    return math.log10(vmin), math.log10(vmax)


def summarize_and_log(args: Args) -> tuple[Path, Path, str]:
    log_dir = args.log_dir.expanduser().resolve()
    if not log_dir.exists():
        raise FileNotFoundError(str(log_dir))

    sweep_id = args.grid_sweep_id or args.run_prefix

    # Keep W&B files off the workspace.
    os.environ.setdefault("WANDB_DIR", str(args.storage_dir / "wandb"))
    os.environ.setdefault("WANDB_CONFIG_DIR", str(args.storage_dir / "wandb" / "config"))
    os.environ.setdefault("WANDB_CACHE_DIR", str(args.storage_dir / "wandb" / "cache"))

    log_files = sorted(log_dir.glob("run_*.log"))
    if not log_files:
        raise RuntimeError(f"No logs found in {log_dir}")

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
        err = _parse_error(txt)
        final_loss, tokens_seen, stable, converged, trainable_ratio = _parse_final_line(txt)
        if stable is None:
            stable = err is None and final_loss is not None and math.isfinite(final_loss)

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
                trainable_ratio=trainable_ratio,
                stable=stable,
                converged=converged,
                error=err,
                log=log_path.name,
                learning_rate=learning_rate,
                matrix_lr=matrix_lr,
                unembedding_lr=unembedding_lr,
            )
        )

    expected = args.resolution * args.resolution
    if len(points) != expected:
        console.print(
            Panel(
                f"Expected {expected} points (resolution={args.resolution}), found {len(points)} logs in {log_dir}",
                title="grid-summary warning",
            )
        )

    fractal_grid, loss_grid, convergence_grid, stability_grid = _build_grids(points, args.resolution)

    # Fill gaps from partial sweeps using nearest-neighbor interpolation
    # This makes visualizations smoother when not all grid points have data
    fractal_grid_filled = _fill_nan_nearest(fractal_grid)
    stability_grid_filled = _fill_nan_nearest(stability_grid)
    convergence_grid_filled = _fill_nan_nearest(convergence_grid)
    # loss_grid keeps NaN for non-trainable (intentional gaps)

    fractal = _compute_fractal(convergence_grid_filled)

    if args.sweep_axes == "matrix_unembedding":
        xs = [p.unembedding_lr for p in points if p.unembedding_lr is not None]
        ys = [p.matrix_lr for p in points if p.matrix_lr is not None]
        x_label = "log₁₀(unembedding lr)"
        y_label = "log₁₀(matrix lr)"
        caption = "Matrix LR × Unembedding LR Grid"
    elif args.sweep_axes == "lr_tokens":
        xs = [float(p.num_tokens) for p in points if p.num_tokens and p.num_tokens > 0]
        ys = [p.learning_rate for p in points if p.learning_rate is not None]
        x_label = "log₁₀(tokens)"
        y_label = "log₁₀(learning rate)"
        caption = "LR × Tokens Grid"
    else:
        raise ValueError(f"Unknown sweep_axes={args.sweep_axes!r} (expected lr_tokens or matrix_unembedding)")

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    ex0, ex1 = _safe_extent(xmin, xmax)
    ey0, ey1 = _safe_extent(ymin, ymax)
    extent = [ex0, ex1, ey0, ey1]

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    # Panel 1: Trainability Boundary (with loss-intensity coloring)
    im1 = axes[0].imshow(
        fractal_grid_filled,
        origin="lower",
        aspect="auto",
        cmap=_fractal_cmap(),
        vmin=-1.0,
        vmax=1.0,
        extent=extent,
        interpolation="nearest",
    )
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel(y_label)
    axes[0].set_title("Trainability Boundary\n(Blue=Trainable, Red=Not Trainable)")
    plt.colorbar(im1, ax=axes[0]).set_label("Trainability")

    # Panel 2: Stability Boundary (the TRUE fractal boundary from the papers)
    im2 = axes[1].imshow(
        stability_grid_filled,
        origin="lower",
        aspect="auto",
        cmap="RdYlGn",  # Red=Unstable, Yellow=boundary, Green=Stable
        vmin=0,
        vmax=1,
        extent=extent,
        interpolation="nearest",
    )
    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel(y_label)
    axes[1].set_title("Stability Boundary\n(Green=Stable, Red=Unstable)")
    cbar2 = plt.colorbar(im2, ax=axes[1])
    cbar2.set_ticks([0, 1])
    cbar2.set_ticklabels(["Unstable", "Stable"])

    # Panel 3: Final Loss (trainable runs only)
    im3 = axes[2].imshow(
        loss_grid,  # Keep NaN for non-trainable (intentional gaps)
        origin="lower",
        aspect="auto",
        cmap="viridis_r",
        extent=extent,
        interpolation="nearest",
    )
    axes[2].set_xlabel(x_label)
    axes[2].set_ylabel(y_label)
    axes[2].set_title("Final Loss (trainable only)")
    plt.colorbar(im3, ax=axes[2]).set_label("Loss")

    # Panel 4: Binary Trainability
    im4 = axes[3].imshow(
        convergence_grid_filled,
        origin="lower",
        aspect="auto",
        cmap="RdBu",
        vmin=0,
        vmax=1,
        extent=extent,
        interpolation="nearest",
    )
    axes[3].set_xlabel(x_label)
    axes[3].set_ylabel(y_label)
    axes[3].set_title("Binary Trainability")
    cbar4 = plt.colorbar(im4, ax=axes[3])
    cbar4.set_ticks([0, 1])
    cbar4.set_ticklabels(["Not Trainable", "Trainable"])

    plt.tight_layout()
    out_prefix = log_dir / f"grid_summary_{args.run_prefix}"
    img_path = out_prefix.with_suffix(".png")
    json_path = out_prefix.with_suffix(".json")
    fig.savefig(img_path, dpi=150, facecolor="white")
    plt.close(fig)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "run_prefix": args.run_prefix,
                    "grid_sweep_id": sweep_id,
                    "sweep_axes": args.sweep_axes,
                    "resolution": args.resolution,
                    "wandb_project": args.wandb_project,
                    "wandb_entity": args.wandb_entity,
                },
                "fractal": fractal,
                "points": [asdict(p) for p in points],
            },
            f,
            indent=2,
        )

    tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
    if not any(t.lower() == "finetune" for t in tags):
        tags.append("finetune")
    tags += [sweep_id, "grid-summary"]
    tags = list(dict.fromkeys([t for t in tags if t]))

    run_base = args.run_prefix if args.run_prefix.endswith("-ft") else f"{args.run_prefix}-ft"
    summary_name = f"{run_base}-grid-summary"

    summary_run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=summary_name,
        config={
            "run_prefix": args.run_prefix,
            "grid_sweep_id": sweep_id,
            "sweep_axes": args.sweep_axes,
            "resolution": args.resolution,
            "fractal_dimension": fractal["fractal_dimension"],
            "boundary_pixels": fractal["boundary_pixels"],
            "converged_ratio": fractal["converged_ratio"],
        },
        tags=tags,
        settings=wandb.Settings(init_timeout=300, _service_wait=300),
    )
    # Compute stability ratio (separate from trainability/converged ratio)
    stable_ratio = float(stability_grid.sum() / stability_grid.size)

    summary_run.log(
        {
            "fractal/image": wandb.Image(str(img_path), caption=caption),
            "fractal/dimension": fractal["fractal_dimension"],
            "fractal/boundary_pixels": fractal["boundary_pixels"],
            "fractal/converged_ratio": fractal["converged_ratio"],
            "fractal/stable_ratio": stable_ratio,  # NEW: true stability (no NaN)
        }
    )

    if args.sweep_axes == "matrix_unembedding":
        columns = [
            "grid_i",
            "grid_j",
            "matrix_lr",
            "unembedding_lr",
            "num_tokens",
            "tokens_seen",
            "final_loss",
            "trainable_ratio",
            "stable",
            "converged",
            "error",
            "log",
        ]
        data = [
            [
                p.grid_i,
                p.grid_j,
                p.matrix_lr,
                p.unembedding_lr,
                p.num_tokens,
                p.tokens_seen,
                p.final_loss,
                p.trainable_ratio,
                p.stable,
                p.converged,
                p.error,
                p.log,
            ]
            for p in points
        ]
    else:
        columns = [
            "grid_i",
            "grid_j",
            "learning_rate",
            "num_tokens",
            "tokens_seen",
            "final_loss",
            "trainable_ratio",
            "stable",
            "converged",
            "error",
            "log",
        ]
        data = [
            [
                p.grid_i,
                p.grid_j,
                p.learning_rate,
                p.num_tokens,
                p.tokens_seen,
                p.final_loss,
                p.trainable_ratio,
                p.stable,
                p.converged,
                p.error,
                p.log,
            ]
            for p in points
        ]
    summary_run.log({"results_table": wandb.Table(columns=columns, data=data)})
    summary_run.finish()

    return img_path, json_path, summary_name


def main():
    args = sp.parse(Args)
    console.rule("[bold]grid sweep summary[/bold]")
    img_path, json_path, summary_name = summarize_and_log(args)
    console.print(
        Panel(
            f"[bold]Logged:[/bold] {args.wandb_entity}/{args.wandb_project} :: {summary_name}\n"
            f"[bold]Image:[/bold] {img_path}\n"
            f"[bold]JSON:[/bold]  {json_path}",
            title="grid-summary",
        )
    )


if __name__ == "__main__":
    main()
