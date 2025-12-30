"""
Visualization tools for fractal LLM training experiments.

Creates heatmaps of convergence/divergence boundaries and fractal analysis plots.
"""

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from scipy import ndimage
from rich.console import Console
import simple_parsing as sp

console = Console()


def load_results(results_path: str | Path) -> tuple[dict, pd.DataFrame]:
    """Load grid search results from JSON file."""
    with open(results_path, "r") as f:
        data = json.load(f)

    config = data["config"]
    df = pd.DataFrame(data["results"])
    return config, df


def build_grid(df: pd.DataFrame, resolution: int, metric: str = "converged") -> np.ndarray:
    """Build a 2D grid from results DataFrame."""
    grid = np.full((resolution, resolution), np.nan)

    for _, row in df.iterrows():
        i, j = int(row["grid_i"]), int(row["grid_j"])
        if 0 <= i < resolution and 0 <= j < resolution:
            if metric == "converged":
                grid[i, j] = 1.0 if row["converged"] else 0.0
            elif metric == "loss":
                loss = row["final_loss"]
                grid[i, j] = min(loss, 20.0)  # Cap for visualization
            else:
                grid[i, j] = row.get(metric, np.nan)

    return grid


def plot_convergence_heatmap(
    config: dict,
    df: pd.DataFrame,
    output_path: str | Path | None = None,
    title: str = "Convergence/Divergence Boundary",
) -> plt.Figure:
    """
    Plot heatmap showing converged (green) vs diverged (red) regions.
    """
    resolution = config["resolution"]
    grid = build_grid(df, resolution, metric="converged")

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Custom colormap: red (diverged) -> yellow (boundary) -> green (converged)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "convergence", ["#d62728", "#ffff00", "#2ca02c"]
    )

    # Create axis labels
    lr_min, lr_max = config["lr_min"], config["lr_max"]
    tokens_min, tokens_max = config["tokens_min"], config["tokens_max"]

    # Plot heatmap
    im = ax.imshow(
        grid,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=0,
        vmax=1,
        extent=[
            np.log10(tokens_min),
            np.log10(tokens_max),
            np.log10(lr_min),
            np.log10(lr_max),
        ],
    )

    ax.set_xlabel("log₁₀(Training Tokens)", fontsize=12)
    ax.set_ylabel("log₁₀(Learning Rate)", fontsize=12)
    ax.set_title(title, fontsize=14)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label="Converged")
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["Diverged", "Boundary", "Converged"])

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        console.print(f"[green]Saved heatmap to {output_path}")

    return fig


def plot_loss_heatmap(
    config: dict,
    df: pd.DataFrame,
    output_path: str | Path | None = None,
    title: str = "Final Training Loss",
) -> plt.Figure:
    """Plot heatmap of final loss values."""
    resolution = config["resolution"]
    grid = build_grid(df, resolution, metric="loss")

    fig, ax = plt.subplots(figsize=(10, 8))

    lr_min, lr_max = config["lr_min"], config["lr_max"]
    tokens_min, tokens_max = config["tokens_min"], config["tokens_max"]

    # Use log scale for loss with viridis colormap
    im = ax.imshow(
        np.log10(grid + 0.1),  # +0.1 to avoid log(0)
        origin="lower",
        aspect="auto",
        cmap="viridis_r",
        extent=[
            np.log10(tokens_min),
            np.log10(tokens_max),
            np.log10(lr_min),
            np.log10(lr_max),
        ],
    )

    ax.set_xlabel("log₁₀(Training Tokens)", fontsize=12)
    ax.set_ylabel("log₁₀(Learning Rate)", fontsize=12)
    ax.set_title(title, fontsize=14)

    cbar = plt.colorbar(im, ax=ax, label="log₁₀(Loss)")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        console.print(f"[green]Saved loss heatmap to {output_path}")

    return fig


def plot_boundary(
    config: dict,
    df: pd.DataFrame,
    output_path: str | Path | None = None,
    title: str = "Trainability Boundary",
) -> plt.Figure:
    """Plot just the boundary between converged and diverged regions."""
    resolution = config["resolution"]
    grid = build_grid(df, resolution, metric="converged")

    # Find boundary using morphological operations
    grid_bool = grid > 0.5
    boundary = ndimage.binary_dilation(grid_bool) ^ ndimage.binary_erosion(grid_bool)

    fig, ax = plt.subplots(figsize=(10, 8))

    lr_min, lr_max = config["lr_min"], config["lr_max"]
    tokens_min, tokens_max = config["tokens_min"], config["tokens_max"]

    ax.imshow(
        boundary.astype(float),
        origin="lower",
        aspect="auto",
        cmap="binary",
        extent=[
            np.log10(tokens_min),
            np.log10(tokens_max),
            np.log10(lr_min),
            np.log10(lr_max),
        ],
    )

    ax.set_xlabel("log₁₀(Training Tokens)", fontsize=12)
    ax.set_ylabel("log₁₀(Learning Rate)", fontsize=12)
    ax.set_title(title, fontsize=14)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        console.print(f"[green]Saved boundary plot to {output_path}")

    return fig


def plot_fractal_analysis(
    config: dict,
    df: pd.DataFrame,
    output_path: str | Path | None = None,
) -> tuple[plt.Figure, float]:
    """
    Plot box-counting fractal dimension analysis.
    Returns the figure and computed fractal dimension.
    """
    resolution = config["resolution"]
    grid = build_grid(df, resolution, metric="converged")

    # Find boundary
    grid_bool = grid > 0.5
    boundary = ndimage.binary_dilation(grid_bool) ^ ndimage.binary_erosion(grid_bool)

    # Box counting
    def box_count(binary_image: np.ndarray, box_size: int) -> int:
        h, w = binary_image.shape
        count = 0
        for i in range(0, h, box_size):
            for j in range(0, w, box_size):
                box = binary_image[i : i + box_size, j : j + box_size]
                if box.any():
                    count += 1
        return count

    # Compute for multiple box sizes (include 64 for higher resolution grids)
    sizes = [2, 4, 8, 16, 32, 64]
    sizes = [s for s in sizes if s < resolution]
    counts = [box_count(boundary, s) for s in sizes]

    # Filter out zero counts
    valid_idx = [i for i, c in enumerate(counts) if c > 0]
    sizes = [sizes[i] for i in valid_idx]
    counts = [counts[i] for i in valid_idx]

    if len(sizes) < 2:
        console.print("[yellow]Not enough data points for fractal analysis")
        return None, float("nan")

    # Fit line on log-log plot
    log_sizes = np.log(sizes)
    log_counts = np.log(counts)
    coeffs = np.polyfit(log_sizes, log_counts, 1)
    fractal_dim = -coeffs[0]

    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: boundary visualization
    lr_min, lr_max = config["lr_min"], config["lr_max"]
    tokens_min, tokens_max = config["tokens_min"], config["tokens_max"]

    ax1.imshow(
        boundary.astype(float),
        origin="lower",
        aspect="auto",
        cmap="hot",
        extent=[
            np.log10(tokens_min),
            np.log10(tokens_max),
            np.log10(lr_min),
            np.log10(lr_max),
        ],
    )
    ax1.set_xlabel("log₁₀(Training Tokens)", fontsize=12)
    ax1.set_ylabel("log₁₀(Learning Rate)", fontsize=12)
    ax1.set_title("Trainability Boundary", fontsize=14)

    # Right: log-log plot with fit
    ax2.scatter(log_sizes, log_counts, s=100, c="blue", zorder=5)
    fit_x = np.linspace(min(log_sizes), max(log_sizes), 100)
    fit_y = coeffs[0] * fit_x + coeffs[1]
    ax2.plot(fit_x, fit_y, "r--", linewidth=2, label=f"Fit (D = {fractal_dim:.3f})")

    ax2.set_xlabel("log(Box Size)", fontsize=12)
    ax2.set_ylabel("log(Box Count)", fontsize=12)
    ax2.set_title(f"Box Counting: Fractal Dimension = {fractal_dim:.3f}", fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        console.print(f"[green]Saved fractal analysis to {output_path}")

    return fig, fractal_dim


def create_all_visualizations(results_path: str | Path, output_dir: str | Path):
    """Create all visualization outputs from a results file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    console.rule("[bold blue]Fractal LLM Visualization")

    config, df = load_results(results_path)
    console.print(f"Loaded {len(df)} results from {results_path}")
    console.print(f"Resolution: {config['resolution']}x{config['resolution']}")

    # Convergence stats
    converged = df["converged"].sum()
    total = len(df)
    console.print(f"Converged: {converged}/{total} ({100*converged/total:.1f}%)")

    # Generate plots
    console.rule("Generating Plots")

    plot_convergence_heatmap(
        config, df, output_dir / "convergence_heatmap.png"
    )

    plot_loss_heatmap(
        config, df, output_dir / "loss_heatmap.png"
    )

    plot_boundary(
        config, df, output_dir / "boundary.png"
    )

    fig, fractal_dim = plot_fractal_analysis(
        config, df, output_dir / "fractal_analysis.png"
    )

    if not np.isnan(fractal_dim):
        console.print(f"\n[bold green]Fractal Dimension: {fractal_dim:.3f}")

        # Interpretation
        if fractal_dim < 1.2:
            console.print("[yellow]Boundary appears relatively smooth (low fractal dimension)")
        elif fractal_dim < 1.6:
            console.print("[green]Moderate fractal structure detected")
        else:
            console.print("[bold magenta]Strong fractal structure! Boundary is highly irregular")

    console.rule("[bold green]Complete")
    console.print(f"All outputs saved to {output_dir}/")


@dataclass
class Args:
    """Visualize fractal LLM training results."""

    results_path: str  # Path to grid search results JSON
    output_dir: str = "results/figures"  # Output directory for plots


if __name__ == "__main__":
    args = sp.parse(Args)
    create_all_visualizations(args.results_path, args.output_dir)
