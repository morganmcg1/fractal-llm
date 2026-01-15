"""
Create a zoom animation between three Trainability Boundary charts.

Downloads the W&B run images, extracts the Trainability panel, crops to the
heatmap area, and generates a smooth zoom + crossfade animation:
parent -> box1 -> box1-2.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import simple_parsing as sp
import wandb
from PIL import Image
from rich.console import Console
from rich.panel import Panel

console = Console()


@dataclass
class Args:
    """Create a zooming animation into nested trainability boxes."""

    parent_run: str  # W&B run id for parent grid summary
    box1_run: str  # W&B run id for first zoom grid summary
    box1_2_run: str  # W&B run id for second zoom grid summary
    out_dir: Path = Path("results/figures/zoom_animation_box_1-2")
    resolution: int = 64  # grid resolution per axis
    # Parent grid box (i, j) indices (inclusive)
    box1_i0: int = 48
    box1_i1: int = 63
    box1_j0: int = 0
    box1_j1: int = 15
    # Box1 grid sub-box for Box1-2 (i, j) indices (inclusive)
    box1_2_i0: int = 56
    box1_2_i1: int = 60
    box1_2_j0: int = 24
    box1_2_j1: int = 29
    hold_frames: int = 12
    zoom_frames: int = 24
    crossfade_frames: int = 8
    fps: int = 24
    pad_frac: float = 0.02


def _download_fractal_image(api: wandb.Api, run_id: str, out_dir: Path) -> Path:
    run = api.run(f"morgy/fractal-llm/{run_id}")
    pngs = [f.name for f in run.files() if f.name.endswith(".png")]
    if not pngs:
        raise RuntimeError(f"No png files found in run {run_id}")
    # Pick the trainability boundary image logged under fractal/image.
    pngs = [p for p in pngs if "media/images/fractal/image_" in p] or pngs
    file_name = pngs[0]
    console.print(f"[bold]download[/bold] {run_id} -> {file_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    run.file(file_name).download(root=out_dir, replace=True)
    return out_dir / file_name


def _split_first_panel(img: Image.Image) -> Image.Image:
    w, h = img.size
    panel_w = w // 4
    return img.crop((0, 0, panel_w, h))


def _find_heatmap_bbox(panel: Image.Image) -> tuple[int, int, int, int]:
    arr = np.asarray(panel.convert("RGB"), dtype=np.float32) / 255.0
    sat = arr.max(axis=2) - arr.min(axis=2)
    mask = sat > 0.12

    col_frac = mask.mean(axis=0)
    col_mask = col_frac > 0.08
    if not col_mask.any():
        return (0, 0, panel.size[0], panel.size[1])

    # Find contiguous column segments and keep the widest (heatmap > colorbar).
    cols = np.where(col_mask)[0]
    segments: list[tuple[int, int]] = []
    start = cols[0]
    prev = cols[0]
    for c in cols[1:]:
        if c != prev + 1:
            segments.append((start, prev))
            start = c
        prev = c
    segments.append((start, prev))
    seg = max(segments, key=lambda s: s[1] - s[0])

    x0, x1 = seg[0], seg[1] + 1
    sub = mask[:, x0:x1]
    row_frac = sub.mean(axis=1)
    row_mask = row_frac > 0.08
    if not row_mask.any():
        return (x0, 0, x1, panel.size[1])
    rows = np.where(row_mask)[0]
    y0, y1 = rows[0], rows[-1] + 1
    return (int(x0), int(y0), int(x1), int(y1))


def _expand_to_aspect(
    box: tuple[float, float, float, float],
    aspect: float,
    bounds: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = box
    bx0, by0, bx1, by1 = bounds
    w = x1 - x0
    h = y1 - y0
    if h <= 0 or w <= 0:
        return (bx0, by0, bx1, by1)
    cur = w / h
    if cur < aspect:
        new_w = h * aspect
        delta = (new_w - w) / 2
        x0 -= delta
        x1 += delta
    else:
        new_h = w / aspect
        delta = (new_h - h) / 2
        y0 -= delta
        y1 += delta
    # Clamp to bounds by shifting if needed.
    if x0 < bx0:
        x1 += (bx0 - x0)
        x0 = bx0
    if x1 > bx1:
        x0 -= (x1 - bx1)
        x1 = bx1
    if y0 < by0:
        y1 += (by0 - y0)
        y0 = by0
    if y1 > by1:
        y0 -= (y1 - by1)
        y1 = by1
    return (x0, y0, x1, y1)


def _box_from_indices(
    width: int,
    height: int,
    res: int,
    i0: int,
    i1: int,
    j0: int,
    j1: int,
    pad_frac: float,
    aspect: float,
) -> tuple[float, float, float, float]:
    cell_w = width / res
    cell_h = height / res
    x0 = j0 * cell_w
    x1 = (j1 + 1) * cell_w
    y0 = height - (i1 + 1) * cell_h
    y1 = height - i0 * cell_h
    pad_x = width * pad_frac
    pad_y = height * pad_frac
    x0 -= pad_x
    x1 += pad_x
    y0 -= pad_y
    y1 += pad_y
    bounds = (0.0, 0.0, float(width), float(height))
    return _expand_to_aspect((x0, y0, x1, y1), aspect, bounds)


def _interp_box(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
    t: float,
) -> tuple[float, float, float, float]:
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(4))  # type: ignore[return-value]


def _crop_resize(img: Image.Image, box: tuple[float, float, float, float], out_size: tuple[int, int]) -> Image.Image:
    crop = img.crop(tuple(map(int, box)))
    return crop.resize(out_size, resample=Image.Resampling.LANCZOS)


def _make_zoom(
    base_img: Image.Image,
    start_box: tuple[float, float, float, float],
    end_box: tuple[float, float, float, float],
    frames: int,
    out_size: tuple[int, int],
) -> list[Image.Image]:
    return [
        _crop_resize(base_img, _interp_box(start_box, end_box, t / (frames - 1)), out_size)
        for t in range(frames)
    ]


def _crossfade(a: Image.Image, b: Image.Image, frames: int) -> list[Image.Image]:
    return [Image.blend(a, b, t / (frames - 1)) for t in range(frames)]


def _save_gif(frames: Iterable[Image.Image], path: Path, fps: int) -> None:
    frames = list(frames)
    duration = int(1000 / fps)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        optimize=False,
    )


def main() -> None:
    args = sp.parse(Args)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api(timeout=120)

    parent_path = _download_fractal_image(api, args.parent_run, args.out_dir)
    box1_path = _download_fractal_image(api, args.box1_run, args.out_dir)
    box12_path = _download_fractal_image(api, args.box1_2_run, args.out_dir)

    parent_img = Image.open(parent_path)
    box1_img = Image.open(box1_path)
    box12_img = Image.open(box12_path)

    parent_panel = _split_first_panel(parent_img)
    box1_panel = _split_first_panel(box1_img)
    box12_panel = _split_first_panel(box12_img)

    parent_box = _find_heatmap_bbox(parent_panel)
    box1_box = _find_heatmap_bbox(box1_panel)
    box12_box = _find_heatmap_bbox(box12_panel)

    parent_heat = parent_panel.crop(parent_box)
    box1_heat = box1_panel.crop(box1_box)
    box12_heat = box12_panel.crop(box12_box)

    parent_heat_path = args.out_dir / "trainability_boundary_parent.png"
    box1_heat_path = args.out_dir / "trainability_boundary_box1.png"
    box12_heat_path = args.out_dir / "trainability_boundary_box1-2.png"
    parent_heat.save(parent_heat_path)
    box1_heat.save(box1_heat_path)
    box12_heat.save(box12_heat_path)

    out_w = 900
    out_h = int(out_w * parent_heat.size[1] / parent_heat.size[0])
    out_size = (out_w, out_h)
    aspect = out_w / out_h

    full_box = (0.0, 0.0, float(parent_heat.size[0]), float(parent_heat.size[1]))
    box1_zoom = _box_from_indices(
        width=parent_heat.size[0],
        height=parent_heat.size[1],
        res=args.resolution,
        i0=args.box1_i0,
        i1=args.box1_i1,
        j0=args.box1_j0,
        j1=args.box1_j1,
        pad_frac=args.pad_frac,
        aspect=aspect,
    )

    box12_zoom = _box_from_indices(
        width=box1_heat.size[0],
        height=box1_heat.size[1],
        res=args.resolution,
        i0=args.box1_2_i0,
        i1=args.box1_2_i1,
        j0=args.box1_2_j0,
        j1=args.box1_2_j1,
        pad_frac=args.pad_frac,
        aspect=aspect,
    )

    frames: list[Image.Image] = []
    parent_full = parent_heat.resize(out_size, resample=Image.Resampling.LANCZOS)
    box1_full = box1_heat.resize(out_size, resample=Image.Resampling.LANCZOS)
    box12_full = box12_heat.resize(out_size, resample=Image.Resampling.LANCZOS)

    frames += [parent_full] * args.hold_frames
    frames += _make_zoom(parent_heat, full_box, box1_zoom, args.zoom_frames, out_size)
    frames += _crossfade(frames[-1], box1_full, args.crossfade_frames)
    frames += [box1_full] * args.hold_frames

    box1_full_box = (0.0, 0.0, float(box1_heat.size[0]), float(box1_heat.size[1]))
    frames += _make_zoom(box1_heat, box1_full_box, box12_zoom, args.zoom_frames, out_size)
    frames += _crossfade(frames[-1], box12_full, args.crossfade_frames)
    frames += [box12_full] * args.hold_frames

    gif_path = args.out_dir / "trainability_zoom_box1_box1-2.gif"
    _save_gif(frames, gif_path, args.fps)

    console.print(
        Panel(
            f"Saved:\n- {gif_path}\n- {parent_heat_path}\n- {box1_heat_path}\n- {box12_heat_path}",
            title="zoom animation",
        )
    )


if __name__ == "__main__":
    main()
