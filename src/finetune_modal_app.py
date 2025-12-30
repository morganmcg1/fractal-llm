"""
Finetune nanochat-style on a single 8×GPU node (no Modal) and optionally sweep LR×token grids.

Usage:
  # single run on 8 GPUs
  torchrun --standalone --nproc_per_node=8 -m src.finetune_modal_app -- --run myrun --learning_rate=3e-4 --num_tokens=200000

  # grid search (runs sequentially, all ranks participate in every point)
  torchrun --standalone --nproc_per_node=8 -m src.finetune_modal_app -- --grid --resolution=16 --lr_min=1e-5 --lr_max=1e-3 --tokens_min=5e3 --tokens_max=5e5

Style notes:
- Mirrors nanochat/scripts/chat_sft.py: flat config globals + configurator overrides, compute_init/cleanup, DummyWandb, print0.
- Keeps DocVQA chat formatting from the previous Modal finetune script.
- Runs entirely locally; no Modal objects or volumes.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import itertools
from dotenv import load_dotenv
from contextlib import nullcontext
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Tuple
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage

import torch
import torch.distributed as dist
import numpy as np
import wandb
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Wire nanochat helpers (style compatibility)
REPO_ROOT = Path(__file__).resolve().parent.parent
NANOCHAT_DIR = REPO_ROOT / "third_party" / "nanochat"
DOTENV_CANDIDATES = [
    REPO_ROOT / ".env",
    Path(".env"),
    Path("/workspace/.env"),
    Path("/root/.env"),
]
for _cand in DOTENV_CANDIDATES:
    if _cand.exists():
        load_dotenv(_cand, override=False)

if NANOCHAT_DIR.exists() and str(NANOCHAT_DIR) not in sys.path:
    sys.path.insert(0, str(NANOCHAT_DIR))

# nanochat imports (after sys.path wiring)
from nanochat.checkpoint_manager import build_model, find_last_step
from nanochat.tokenizer import get_tokenizer
from nanochat.common import get_base_dir

try:
    from nanochat.common import (
        compute_init,
        compute_cleanup,
        print0,
        DummyWandb,
        autodetect_device_type,
    )
except Exception:
    # Fallback if nanochat isn't importable (e.g., trimmed checkout). Minimal shims.
    def print0(s: str = "", **kwargs):
        if int(os.environ.get("RANK", 0)) == 0:
            print(s, **kwargs)

    def autodetect_device_type():
        return "cuda" if torch.cuda.is_available() else "cpu"

    def compute_init(device_type="cuda"):
        is_ddp = all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world = int(os.environ.get("WORLD_SIZE", 1))
        device = torch.device(device_type if device_type != "cuda" else f"cuda:{local_rank}")
        if is_ddp and device_type == "cuda":
            torch.cuda.set_device(device)
            dist.init_process_group("nccl", device_id=device)
            dist.barrier()
        return is_ddp, rank, local_rank, world, device

    def compute_cleanup():
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

    class DummyWandb:
        def log(self, *args, **kwargs): ...
        def finish(self): ...

# ---------------------------------------------------------------------------
# Config (nanochat-style flat globals + configurator)
run = "dummy"  # wandb run name; "dummy" disables logging
model_id = os.environ.get("MODEL_ARTIFACT", "wandb:morgan/fractal-llm/nanochat-d20-speedrun:latest")
dataset_id = os.environ.get("DATASET_ID", "morgan/docvqa-nanochat")
max_seq_len = int(os.environ.get("MAX_SEQ_LEN", "128"))
dtype = "bfloat16"  # float32 | bfloat16
device_batch_size = 2
target_examples_per_step = 32  # like chat_sft
learning_rate = 3e-4
weight_decay = 0.1
init_lr_frac = 1.0
warmup_frac = 0.06
num_tokens = 200_000  # approximate global tokens to see
num_iterations = -1  # derived from num_tokens if -1
eval_every = 200
log_every = 20
seed = 42
gradient_checkpointing = False
source_stage = "sft"  # nanochat checkpoint family: base|mid|sft|rl
tokenizer_artifact = os.environ.get("TOKENIZER_ARTIFACT", None)

# Grid search knobs
grid = False
resolution = 8
lr_min = 1e-5
lr_max = 1e-3
tokens_min = 5_000
tokens_max = 500_000
checkpoint_every = 32

# Derived/configurable keys list
config_keys = [
    k
    for k, v in list(globals().items())
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]

# Allow CLI overrides using nanochat/configurator.py semantics if available.
configurator = NANOCHAT_DIR / "nanochat" / "configurator.py"
if configurator.exists():
    # Allow both "--key=value" and "--key value" forms; normalize before running configurator.
    if len(sys.argv) > 1:
        new_args = []
        args = sys.argv[1:]
        i = 0
        while i < len(args):
            arg = args[i]
            if arg.startswith("--") and "=" not in arg and i + 1 < len(args) and not args[i + 1].startswith("--"):
                new_args.append(f"{arg}={args[i+1]}")
                i += 2
            else:
                new_args.append(arg)
                i += 1
        sys.argv = [sys.argv[0]] + new_args
    exec(configurator.read_text())
user_config = {k: globals()[k] for k in config_keys}

# Respect WANDB_RUN override and attach suffix similar to chat_sft
env_run = os.environ.get("WANDB_RUN")
if env_run:
    run = env_run
if run != "dummy" and not run.endswith("-sft"):
    run = f"{run}-sft"

WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "fractal-llm")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "morgan")


# ---------------------------------------------------------------------------
# Data utilities

def format_docvqa(sample: dict) -> str:
    """Convert DocVQA style messages to a single chat text block."""
    msgs = sample.get("messages", [])
    if not msgs:
        return ""
    parts = []
    for msg in msgs:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"<|{role}|>\n{content}")
    parts.append("<|end|>")
    return "\n".join(parts)


class DocVQADataset(IterableDataset):
    """Streaming iterable dataset sharded across ranks."""

    def __init__(self, tokenizer, split: str, seed: int, world_size: int, rank: int):
        self.tokenizer = tokenizer
        self.split = split
        self.seed = seed
        self.world_size = world_size
        self.rank = rank

    def __iter__(self):
        ds = load_dataset(dataset_id, split=self.split, streaming=True)
        ds = ds.shuffle(seed=self.seed, buffer_size=2048)
        if self.world_size > 1:
            ds = ds.shard(num_shards=self.world_size, index=self.rank)
        for sample in ds:
            text = format_docvqa(sample)
            if not text:
                continue
            # nanochat tokenizer expects conversation structure; wrap into messages
            conversation = {"messages": [{"role": "user", "content": text}]}
            ids, mask = self.tokenizer.render_conversation(conversation)
            # Clamp / truncate to max_seq_len
            ids = ids[:max_seq_len]
            mask = mask[:max_seq_len]
            input_ids = torch.tensor(ids, dtype=torch.long)
            attn = torch.tensor([1 if m >= 0 else 0 for m in mask], dtype=torch.long)
            labels = input_ids.clone()
            # mask==0 -> positions we don't train on
            labels = torch.where(torch.tensor(mask) == 1, labels, torch.tensor(-100))
            yield input_ids, attn, labels


def build_dataloader(tokenizer, seed, world_size, rank) -> DataLoader:
    dataset = DocVQADataset(tokenizer, split="train", seed=seed, world_size=world_size, rank=rank)
    return DataLoader(dataset, batch_size=device_batch_size, pin_memory=True)


# ---------------------------------------------------------------------------
# Model helpers

def ensure_tokenizer(tokenizer_id: str | None, artifact_root: Path | None) -> Path:
    """
    Ensure nanochat tokenizer files are available under ~/.cache/nanochat/tokenizer.
    Strategy:
      1) If tokenizer already exists there, use it.
      2) Else, if artifact_root contains tokenizer files, copy them in.
      3) Else, if tokenizer_id is provided (wandb:...), download and copy.
      4) Else, raise.
    """
    base_dir = Path(get_base_dir())
    tok_dir = base_dir / "tokenizer"
    tok_dir.mkdir(parents=True, exist_ok=True)
    required = ["tokenizer.pkl", "tokenizer.json", "vocab.json", "merges.txt", "token_bytes.pt", "tokenizer_config.json"]

    def has_required(path: Path) -> bool:
        return all((path / r).exists() for r in ["tokenizer.pkl"])

    if has_required(tok_dir):
        return tok_dir

    def copy_from(src_dir: Path):
        for r in required:
            for cand in src_dir.rglob(r):
                target = tok_dir / cand.name
                target.write_bytes(cand.read_bytes())

    if artifact_root:
        # Look for tokenizer assets inside the already-downloaded model artifact
        for sub in [artifact_root, artifact_root / "tokenizer"]:
            if sub.exists():
                copy_from(sub)
        if has_required(tok_dir):
            return tok_dir

    if tokenizer_id:
        import wandb as _wandb

        art_path = tokenizer_id[len("wandb:") :] if tokenizer_id.startswith("wandb:") else tokenizer_id
        api = _wandb.Api()
        art = api.artifact(art_path, type="model")
        dl_root = tok_dir / "tmp_download"
        if dl_root.exists():
            shutil.rmtree(dl_root)
        dl_root.mkdir(parents=True, exist_ok=True)
        art_local = Path(art.download(root=str(dl_root)))
        copy_from(art_local)
        shutil.rmtree(dl_root, ignore_errors=True)
        if has_required(tok_dir):
            return tok_dir

    raise FileNotFoundError(
        "Tokenizer not found. Provide TOKENIZER_ARTIFACT (wandb:<entity>/<project>/<name>:<ver>) "
        "or place tokenizer files under ~/.cache/nanochat/tokenizer."
    )


def resolve_checkpoint_dir(model_ref: str) -> Path:
    """
    Download W&B artifact if needed, otherwise treat as local path.
    Expects nanochat checkpoint layout: checkpoints/model_*.pt + meta_*.json (or those files directly in root).
    """
    if model_ref.startswith("wandb:"):
        import wandb as _wandb
        art_path = model_ref[len("wandb:") :]
        api = _wandb.Api()
        art = api.artifact(art_path, type="model")
        cache_root = REPO_ROOT / "results" / "model_cache"
        cache_root.mkdir(parents=True, exist_ok=True)
        target = cache_root / art.name.replace(":", "_")
        if not target.exists():
            art.download(root=str(target))
        if (target / "checkpoints").exists():
            return target / "checkpoints"
        return target
    return Path(model_ref)


def load_model_and_tok(device_type: str):
    ckpt_dir = resolve_checkpoint_dir(model_id)
    tok_dir = ensure_tokenizer(tokenizer_artifact, ckpt_dir.parent if ckpt_dir.name == "checkpoints" else ckpt_dir)
    step = find_last_step(ckpt_dir)
    model, tokenizer, meta = build_model(ckpt_dir, step=step, device=torch.device(device_type), phase="train")
    model.to(device_type)
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Training core

@dataclass
class RunResult:
    run_name: str
    learning_rate: float
    num_tokens_target: int
    tokens_seen: int
    avg_loss: float
    final_loss: float
    converged: bool
    steps: int
    runtime_s: float
    error: str | None = None
    grid_i: int | None = None
    grid_j: int | None = None


# ---------------------------------------------------------------------------
# Visualization and analysis helpers

def build_grids(results: List[RunResult], resolution: int):
    """Construct convergence, loss, and intensity grids."""
    fractal_grid = np.zeros((resolution, resolution))
    loss_grid = np.full((resolution, resolution), np.nan)
    convergence_grid = np.zeros((resolution, resolution))

    converged_losses: List[float] = []

    for r in results:
        i = r.grid_i or 0
        j = r.grid_j or 0
        if r.converged and math.isfinite(r.final_loss):
            loss_grid[i, j] = r.final_loss
            converged_losses.append(r.final_loss)
            convergence_grid[i, j] = 1
        else:
            convergence_grid[i, j] = 0

    if converged_losses:
        loss_min, loss_max = min(converged_losses), max(converged_losses)
        loss_range = loss_max - loss_min if loss_max > loss_min else 1.0
    else:
        loss_min, loss_max, loss_range = 0.0, 1.0, 1.0

    for r in results:
        i = r.grid_i or 0
        j = r.grid_j or 0
        if r.converged and math.isfinite(r.final_loss):
            intensity = (r.final_loss - loss_min) / loss_range
            fractal_grid[i, j] = 0.3 + 0.7 * intensity  # Blue range
        else:
            fractal_grid[i, j] = -1.0  # Red for diverged / failed

    return fractal_grid, loss_grid, convergence_grid


def save_visualizations(
    results: List[RunResult],
    resolution: int,
    lr_min: float,
    lr_max: float,
    tokens_min: float,
    tokens_max: float,
    out_prefix: Path,
):
    """Create and save three-panel visualization. Returns image path."""
    fractal_grid, loss_grid, convergence_grid = build_grids(results, resolution)

    colors_diverged = ["#8B0000", "#CD5C5C", "#FA8072"]  # Dark red to light red
    colors_converged = ["#ADD8E6", "#4169E1", "#00008B"]  # Light blue to dark blue
    colors = colors_diverged + ["#FFFFFF"] + colors_converged
    positions = [0.0, 0.15, 0.35, 0.5, 0.65, 0.85, 1.0]
    fractal_cmap = LinearSegmentedColormap.from_list("fractal", list(zip(positions, colors)))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    im1 = axes[0].imshow(
        fractal_grid,
        origin="lower",
        aspect="auto",
        cmap=fractal_cmap,
        vmin=-1.0,
        vmax=1.0,
        extent=[np.log10(tokens_min), np.log10(tokens_max), np.log10(lr_min), np.log10(lr_max)],
    )
    axes[0].set_xlabel("log₁₀(tokens)")
    axes[0].set_ylabel("log₁₀(learning rate)")
    axes[0].set_title("Trainability Boundary\n(Blue=Converged, Red=Diverged)")
    cbar1 = plt.colorbar(im1, ax=axes[0])
    cbar1.set_label("Convergence")

    im2 = axes[1].imshow(
        loss_grid,
        origin="lower",
        aspect="auto",
        cmap="viridis_r",
        extent=[np.log10(tokens_min), np.log10(tokens_max), np.log10(lr_min), np.log10(lr_max)],
    )
    axes[1].set_xlabel("log₁₀(tokens)")
    axes[1].set_ylabel("log₁₀(learning rate)")
    axes[1].set_title("Final Loss (converged)")
    cbar2 = plt.colorbar(im2, ax=axes[1])
    cbar2.set_label("Loss")

    im3 = axes[2].imshow(
        convergence_grid,
        origin="lower",
        aspect="auto",
        cmap="RdBu",
        vmin=0,
        vmax=1,
        extent=[np.log10(tokens_min), np.log10(tokens_max), np.log10(lr_min), np.log10(lr_max)],
    )
    axes[2].set_xlabel("log₁₀(tokens)")
    axes[2].set_ylabel("log₁₀(learning rate)")
    axes[2].set_title("Binary Convergence")
    cbar3 = plt.colorbar(im3, ax=axes[2])
    cbar3.set_ticks([0, 1])
    cbar3.set_ticklabels(["Diverged", "Converged"])

    plt.tight_layout()
    out_path = out_prefix.with_suffix(".png")
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    return out_path, convergence_grid


def compute_fractal(convergence_grid: np.ndarray):
    """Compute box-counting fractal dimension of boundary."""
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

def derive_num_iterations(tokens_per_batch: int, grad_accum: int, tokens_goal: int, world_size: int) -> int:
    tokens_per_step = max(1, tokens_per_batch * grad_accum * max(1, world_size))
    return max(1, math.ceil(tokens_goal / tokens_per_step))


def train_once(
    learning_rate_override: float | None = None,
    num_tokens_override: int | None = None,
    grid_i: int | None = None,
    grid_j: int | None = None,
    runtime=None,
) -> RunResult:
    import time

    lr = float(learning_rate_override or learning_rate)
    tokens_goal = int(num_tokens_override or num_tokens)

    # Init compute/DDP once; reuse runtime to avoid re-init errors during grid sweeps
    if runtime is None:
        device_type = autodetect_device_type()
        ddp, rank, local_rank, world_size, device = compute_init(device_type)
        runtime = (ddp, rank, local_rank, world_size, device, device_type)
    else:
        ddp, rank, local_rank, world_size, device, device_type = runtime
    master = rank == 0

    # Seeds
    torch.manual_seed(seed + (grid_i or 0) + (grid_j or 0))
    np.random.seed(seed + (grid_i or 0) + (grid_j or 0))
    random.seed(seed + (grid_i or 0) + (grid_j or 0))

    # wandb
    use_dummy = run == "dummy" or not master
    run_name = run
    if grid and grid_i is not None and grid_j is not None:
        run_name = f"{run}-g{grid_i}-{grid_j}"
    wb = (
        DummyWandb()
        if use_dummy
        else wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=run_name,
            config=user_config | {"learning_rate": lr, "num_tokens": tokens_goal, "grid_i": grid_i, "grid_j": grid_j},
            save_code=True,
            settings=wandb.Settings(init_timeout=300, _service_wait=300),
        )
    )

    model, tokenizer = load_model_and_tok(device_type)
    model.to(device)
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    dataloader = build_dataloader(tokenizer, seed=seed, world_size=world_size, rank=rank)
    data_iter = iter(dataloader)

    # Peek first batch to estimate tokens/batch (then reuse it)
    first_batch = next(data_iter)
    input_ids, attn, labels = [x.to(device) for x in first_batch]
    tokens_per_batch = int(attn.sum().item())
    import itertools
    data_iter = itertools.chain([first_batch], data_iter)

    examples_per_step = device_batch_size * world_size
    grad_accum = max(1, target_examples_per_step // examples_per_step)
    total_steps = num_iterations if num_iterations > 0 else derive_num_iterations(tokens_per_batch, grad_accum, tokens_goal, world_size)
    warmup_steps = max(1, int(total_steps * warmup_frac))

    optim = torch.optim.AdamW(model.parameters(), lr=lr * init_lr_frac, weight_decay=weight_decay)

    def lr_mult(step_idx: int) -> float:
        if step_idx < warmup_steps:
            return (step_idx + 1) / warmup_steps
        return max(0.0, (total_steps - step_idx) / max(1, total_steps - warmup_steps))

    # metrics
    losses: List[float] = []
    tokens_seen = 0
    start_time = time.time()
    error = None
    final_loss = float("inf")

    autocast_dtype = torch.bfloat16 if dtype == "bfloat16" and device_type == "cuda" else torch.float32
    if device_type == "cuda":
        def cast_ctx():
            return torch.amp.autocast(device_type=device_type, dtype=autocast_dtype)
    else:
        def cast_ctx():
            return nullcontext()

    try:
        for step_idx in range(total_steps):
            optim.zero_grad(set_to_none=True)
            num_tokens_step = 0
            running_loss = 0.0
            for micro in range(grad_accum):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    batch = next(data_iter)
                batch = [b.to(device) for b in batch]
                b_input, b_attn, b_labels = batch
                with cast_ctx():
                    outputs = model(input_ids=b_input, attention_mask=b_attn, labels=b_labels)
                    loss = outputs.loss
                (loss / grad_accum).backward()
                running_loss += loss.detach()
                num_tokens_step += int((b_labels >= 0).sum().item())

            # lr schedule
            mult = lr_mult(step_idx)
            for pg in optim.param_groups:
                pg["lr"] = lr * mult
            optim.step()
            tokens_seen += num_tokens_step * world_size  # tokens counted on all ranks

            # reduce loss for logging
            loss_item = running_loss / grad_accum
            if ddp:
                dist.all_reduce(loss_item, op=dist.ReduceOp.AVG)
            loss_scalar = loss_item.item()
            final_loss = loss_scalar
            losses.append(loss_scalar)

            if master and (step_idx % log_every == 0 or step_idx + 1 == total_steps):
                print0(f"step {step_idx+1:05d}/{total_steps:05d} loss={loss_scalar:.4f} lr={lr*mult:.2e} tokens={tokens_seen:,}")
                wb.log(
                    {
                        "step": step_idx,
                        "train/loss": loss_scalar,
                        "lr": lr * mult,
                        "tokens_seen": tokens_seen,
                    }
                )

            if master and eval_every > 0 and (step_idx + 1) % eval_every == 0:
                wb.log({"eval/placeholder": loss_scalar})  # simple placeholder to mirror chat_sft cadence

    except Exception as exc:  # pylint: disable=broad-except
        error = str(exc)
        print0(f"[ERROR] training failed: {error}")

    runtime_s = time.time() - start_time
    converged = error is None and math.isfinite(final_loss) and final_loss < 10.0
    avg_loss = float(np.mean(losses)) if losses else float("inf")

    if master and not use_dummy:
        wb.log(
            {
                "final/loss": final_loss,
                "final/converged": converged,
                "final/tokens_seen": tokens_seen,
                "runtime_s": runtime_s,
            }
        )
        wb.finish()

    # cleanup
    del model
    torch.cuda.empty_cache()

    return RunResult(
        run_name=run,
        learning_rate=lr,
        num_tokens_target=tokens_goal,
        tokens_seen=tokens_seen,
        avg_loss=avg_loss,
        final_loss=final_loss,
        converged=converged,
        steps=total_steps,
        runtime_s=runtime_s,
        error=error,
        grid_i=grid_i,
        grid_j=grid_j,
    )


# ---------------------------------------------------------------------------
# Grid search orchestrator

def logspace(min_v: float, max_v: float, n: int):
    return np.logspace(np.log10(min_v), np.log10(max_v), n)


def run_grid_search():
    learning_rates = logspace(lr_min, lr_max, resolution)
    token_counts = logspace(tokens_min, tokens_max, resolution).astype(int)

    # initialize once and reuse across grid points
    device_type = autodetect_device_type()
    runtime = compute_init(device_type)
    runtime = runtime + (device_type,)

    if int(os.environ.get("RANK", 0)) == 0:
        print0(f"Grid search {resolution}x{resolution} ({resolution*resolution} runs)")
    results: List[RunResult] = []

    for i, lr_val in enumerate(learning_rates):
        for j, tok_val in enumerate(token_counts):
            if int(os.environ.get("RANK", 0)) == 0:
                print0(f"\n=== Grid ({i},{j}) lr={lr_val:.2e} tokens={tok_val:,} ===")
            res = train_once(
                learning_rate_override=float(lr_val),
                num_tokens_override=int(tok_val),
                grid_i=i,
                grid_j=j,
                runtime=runtime,
            )
            if int(os.environ.get("RANK", 0)) == 0:
                results.append(res)
                # checkpoint to disk
                ckpt_dir = REPO_ROOT / "results"
                ckpt_dir.mkdir(exist_ok=True, parents=True)
                ckpt_path = ckpt_dir / f"finetune_grid_{resolution}x{resolution}.json"
                with open(ckpt_path, "w", encoding="utf-8") as f:
                    json.dump([asdict(r) for r in results], f, indent=2)
                print0(f"Checkpointed {len(results)} results to {ckpt_path}")

    # Rank 0: finalize artifacts
    if int(os.environ.get("RANK", 0)) == 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = REPO_ROOT / "results"
        results_dir.mkdir(exist_ok=True, parents=True)
        prefix = results_dir / f"finetune_grid_{resolution}x{resolution}_{timestamp}"
        results_path = prefix.with_suffix(".json")

        config_dump = {
            "resolution": resolution,
            "lr_min": lr_min,
            "lr_max": lr_max,
            "tokens_min": tokens_min,
            "tokens_max": tokens_max,
            "model_id": model_id,
            "dataset_id": dataset_id,
            "max_seq_len": max_seq_len,
        }

        with open(results_path, "w", encoding="utf-8") as f:
            json.dump({"config": config_dump, "results": [asdict(r) for r in results]}, f, indent=2)

        img_path, convergence_grid = save_visualizations(
            results,
            resolution,
            lr_min,
            lr_max,
            tokens_min,
            tokens_max,
            prefix,
        )
        fractal = compute_fractal(convergence_grid)
        fractal_path = prefix.with_suffix(".fractal.json")
        with open(fractal_path, "w", encoding="utf-8") as f:
            json.dump(fractal, f, indent=2)

        print0(f"Saved results: {results_path}")
        print0(f"Saved visualization: {img_path}")
        print0(f"Fractal dimension: {fractal['fractal_dimension']:.3f}")

        if run != "dummy":
            summary_run = wandb.init(
                project=WANDB_PROJECT,
                entity=WANDB_ENTITY,
                name=f"{run}-grid-summary",
                config=config_dump | {"fractal_dimension": fractal["fractal_dimension"]},
                tags=["fractal-grid"],
            )

            summary_run.log(
                {
                    "fractal/image": wandb.Image(str(img_path), caption="LR × Tokens Grid"),
                    "fractal/dimension": fractal["fractal_dimension"],
                    "fractal/boundary_pixels": fractal["boundary_pixels"],
                    "fractal/converged_ratio": fractal["converged_ratio"],
                }
            )

            results_table = wandb.Table(
                columns=[
                    "grid_i",
                    "grid_j",
                    "learning_rate",
                    "num_tokens",
                    "tokens_seen",
                    "final_loss",
                    "converged",
                    "runtime_s",
                    "error",
                ],
                data=[
                    [
                        r.grid_i,
                        r.grid_j,
                        r.learning_rate,
                        r.num_tokens_target,
                        r.tokens_seen,
                        r.final_loss,
                        r.converged,
                        r.runtime_s,
                        r.error,
                    ]
                    for r in results
                ],
            )
            summary_run.log({"results_table": results_table})
            summary_run.finish()

    return results


# ---------------------------------------------------------------------------
# Entry

def main():
    if grid:
        run_grid_search()
    else:
        device_type = autodetect_device_type()
        runtime = compute_init(device_type)
        runtime = runtime + (device_type,)
        res = train_once(runtime=runtime)
        if int(os.environ.get("RANK", 0)) == 0:
            print0(f"\nFinal: loss={res.final_loss:.4f} tokens_seen={res.tokens_seen:,} converged={res.converged}")
    compute_cleanup()


if __name__ == "__main__":
    main()
