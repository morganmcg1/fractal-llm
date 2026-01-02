"""
Finetune nanochat-style on a single 8×GPU node (no Modal) and optionally sweep LR×token grids.

Usage:
  # single run on 8 GPUs (DDP, data parallel)
  torchrun --standalone --nproc_per_node=8 -m src.finetune -- --run myrun --learning_rate=3e-4 --num_tokens=200000

  # debug/smoke test (minimal data, fast iteration, saves artifacts)
  torchrun --standalone --nproc_per_node=1 -m src.finetune -- --debug

  # grid search (runs sequentially, all 8 GPUs work on each point together)
  torchrun --standalone --nproc_per_node=8 -m src.finetune -- --grid --resolution=16 --lr_min=1e-5 --lr_max=1e-3 --tokens_min=5e3 --tokens_max=5e5

  # PARALLEL grid search: run 8 independent single-GPU experiments (RECOMMENDED for fractal grids)
  # This gives 8x throughput for grid search by running different (lr, tokens) combinations in parallel
  for gpu in {0..7}; do
    CUDA_VISIBLE_DEVICES=$gpu python -m src.finetune --run grid-$gpu --learning_rate=<lr_$gpu> --num_tokens=<tokens_$gpu> &
  done
  wait

Scaling Strategy for Fractal Grid Search:
  - For maximum throughput: Run 8 independent single-GPU experiments in parallel
  - Each GPU runs a different (learning_rate, num_tokens) combination
  - Use a launcher script to distribute grid points across GPUs
  - This is 8x faster than sequential grid search with DDP

Reproducibility:
  - deterministic=True (default) enables full reproducibility
  - Same seed + same GPU = identical results
  - CUBLAS_WORKSPACE_CONFIG and torch.use_deterministic_algorithms are set
  - Grid points get unique seeds: seed + grid_i * 1000 + grid_j

Style notes:
- Mirrors nanochat/scripts/chat_sft.py: flat config globals + configurator overrides, compute_init/cleanup, DummyWandb, print0.
- Uses DocVQA dataset with proper conversation masking (assistant tokens only).
- Runs entirely locally; no Modal objects or volumes.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import subprocess
from dotenv import load_dotenv
from contextlib import nullcontext
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List
import shutil
import itertools

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
from torch.amp import GradScaler

# ---------------------------------------------------------------------------
# Reproducibility settings (critical for fractal grid experiments)
# Must be set before any CUDA operations
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # Deterministic cuBLAS
os.environ["PYTHONHASHSEED"] = "0"  # Deterministic Python hashing
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("NCCL_ALGO", "Ring")  # Stable reduction order (deterministic for single-node)
os.environ.setdefault("NCCL_PROTO", "Simple")
os.environ.setdefault("NCCL_MIN_NRINGS", "1")
os.environ.setdefault("NCCL_DEBUG", os.environ.get("NCCL_DEBUG", "WARN"))
os.environ.setdefault("HF_DATASETS_OFFLINE", os.environ.get("HF_DATASETS_OFFLINE", "0"))


def set_seed(seed: int, rank: int = 0):
    """Set all random seeds for full reproducibility.

    Model initialization uses the base seed for all ranks so weights match
    exactly; rank offset is applied only to RNG streams that affect data order.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Optional per-rank jitter for data-only randomness can be layered on top
    # by callers if needed without changing model initialization.


def enable_deterministic_mode():
    """Enable PyTorch deterministic algorithms for reproducibility.

    Note: This may have a small performance impact (~5-10%).
    """
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Disable auto-tuning for reproducibility
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))


def repro_context():
    """Collect reproducibility-relevant metadata for logging."""
    def _git_rev():
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                cwd=REPO_ROOT,
            )
            return result.stdout.strip()
        except Exception:
            return None

    ctx = {
        "seed": seed,
        "deterministic": deterministic,
        "model_id": model_id,
        "dataset_id": dataset_id,
        "dataset_revision": dataset_revision,
        "tokenizer_artifact": tokenizer_artifact,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "nccl": ".".join(map(str, torch.cuda.nccl.version())) if torch.cuda.is_available() else None,
        "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "nccl_algo": os.environ.get("NCCL_ALGO"),
        "nccl_proto": os.environ.get("NCCL_PROTO"),
        "nccl_min_nrings": os.environ.get("NCCL_MIN_NRINGS"),
        "hf_datasets_offline": os.environ.get("HF_DATASETS_OFFLINE"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "tf32": {
            "matmul": torch.backends.cuda.matmul.allow_tf32,
            "cudnn": torch.backends.cudnn.allow_tf32,
        },
        "torch_num_threads": torch.get_num_threads(),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "git_commit": _git_rev(),
    }
    return ctx

# ---------------------------------------------------------------------------
# Wire nanochat helpers (style compatibility)
REPO_ROOT = Path(__file__).resolve().parent.parent

# Default storage: push results, checkpoints, and HF caches to /var/tmp to avoid small workspace quotas.
STORAGE_ROOT = Path(os.environ.get("FRACTAL_STORAGE_DIR", "/var/tmp/fractal-llm"))
RESULTS_ROOT = STORAGE_ROOT / "results"
MODEL_CACHE_ROOT = RESULTS_ROOT / "model_cache"
CHECKPOINTS_ROOT = RESULTS_ROOT / "checkpoints"
for _path in [STORAGE_ROOT, RESULTS_ROOT, MODEL_CACHE_ROOT, CHECKPOINTS_ROOT]:
    _path.mkdir(parents=True, exist_ok=True)

# HuggingFace caches (datasets + hub) and nanochat base dir => /var/tmp by default.
_storage_env_defaults = {
    # Hugging Face caches
    "HF_HOME": "/var/tmp/huggingface",
    "HF_DATASETS_CACHE": "/var/tmp/huggingface/datasets",
    "HF_HUB_CACHE": "/var/tmp/huggingface/hub",
    # nanochat base dir
    "NANOCHAT_BASE_DIR": "/var/tmp/nanochat",
    # WandB local files/config/cache
    "WANDB_DIR": str(STORAGE_ROOT / "wandb"),
    "WANDB_CONFIG_DIR": str(STORAGE_ROOT / "wandb" / "config"),
    "WANDB_CACHE_DIR": str(STORAGE_ROOT / "wandb" / "cache"),
}
for _k, _v in _storage_env_defaults.items():
    os.environ.setdefault(_k, _v)
for _p in [
    Path(os.environ["HF_HOME"]),
    Path(os.environ["HF_DATASETS_CACHE"]),
    Path(os.environ["HF_HUB_CACHE"]),
    Path(os.environ["NANOCHAT_BASE_DIR"]),
    Path(os.environ["WANDB_DIR"]),
    Path(os.environ["WANDB_CONFIG_DIR"]),
    Path(os.environ["WANDB_CACHE_DIR"]),
]:
    _p.mkdir(parents=True, exist_ok=True)

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
from nanochat.checkpoint_manager import build_model, find_last_step, save_checkpoint
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
model_id = os.environ.get("MODEL_ARTIFACT", "morgy/fractal-llm/nanochat-fin-rl-artifact:v7")
dataset_id = os.environ.get("DATASET_ID", "morgan/docvqa-nanochat")
dataset_revision = os.environ.get("DATASET_REVISION", None)
max_seq_len = int(os.environ.get("MAX_SEQ_LEN", "1024"))  # nanochat default=2048, DocVQA avg ~400 tokens
dtype = "bfloat16"  # float32 | bfloat16 | float16 (bfloat16 matches model weights)
device_batch_size = 8  # aim for the largest batch that fits; override via CLI/ENV if needed
# Set 0 to force grad_accum_steps=1; otherwise acts as a target effective batch size.
target_examples_per_step = 0
learning_rate = 3e-4
weight_decay = 0.1
init_lr_frac = 1.0
warmup_frac = 0.06
final_lr_frac = 0.0  # cosine anneals to lr * final_lr_frac (0.0 = full decay to zero)
num_tokens = 200_000  # approximate global tokens to see
num_iterations = -1  # derived from num_tokens if -1
eval_every = 200
eval_batches = 4  # number of validation batches per eval pass
log_every = 20
seed = 999
deterministic = True  # if True, enable full reproducibility (slight perf hit ~5-10%)
debug = False  # if True, run a quick smoke test with minimal data
save_artifacts = False  # if True, save checkpoint and upload to W&B (auto-enabled in debug mode)
visualize = False  # if True, show model predictions before/after training
gradient_checkpointing = False
source_stage = "sft"  # nanochat checkpoint family: base|mid|sft|rl
tokenizer_artifact = os.environ.get("TOKENIZER_ARTIFACT", None)
convergence_loss_threshold = 2.0  # consider run converged if final CE below this and finite
wandb_tags = "finetune"  # comma-separated wandb tags (e.g., "finetune,debug")
grid_sweep_id = ""  # sweep identifier tag; auto-generated as YYYY-MM-DD_HH-MM if empty

# Grid search knobs
grid = False
resolution = 8
lr_min = 1e-5
lr_max = 1e-3
tokens_min = 5_000
tokens_max = 500_000
checkpoint_every = 32
grid_i = 0
grid_j = 0

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
if run != "dummy" and not run.endswith("-ft"):
    run = f"{run}-ft"

# Debug mode: override to minimal settings for a quick smoke test
if debug:
    num_tokens = 2000
    log_every = 1
    save_artifacts = True
    if run == "dummy":
        run = "debug-finetune"
    print0(f"[DEBUG MODE] num_tokens={num_tokens}, log_every={log_every}, save_artifacts={save_artifacts}")

WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "fractal-llm")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "morgy")


# ---------------------------------------------------------------------------
# Data utilities


class DocVQADataset(IterableDataset):
    """Streaming iterable dataset sharded across ranks."""

    def __init__(self, tokenizer, split: str, seed: int, world_size: int, rank: int):
        self.tokenizer = tokenizer
        self.split = split
        self.seed = seed
        self.world_size = world_size
        self.rank = rank

    def __iter__(self):
        load_kwargs = {"split": self.split, "streaming": True}
        if dataset_revision:
            load_kwargs["revision"] = dataset_revision
        ds = load_dataset(dataset_id, **load_kwargs)
        # Keep deterministic ordering to mirror fractal-boundary experiments.
        # Avoid shuffle buffers that introduce nondeterminism across runs.
        if not deterministic:
            ds = ds.shuffle(seed=self.seed, buffer_size=2048)
        if self.world_size > 1:
            ds = ds.shard(num_shards=self.world_size, index=self.rank)
        for sample in ds:
            # Pass messages directly - dataset already has proper {user, assistant} structure
            messages = sample.get("messages", [])
            if not messages:
                continue
            conversation = {"messages": messages}
            ids, mask = self.tokenizer.render_conversation(conversation)
            # Truncate to max_seq_len (keeping room for shift)
            orig_len = len(ids)
            ids = ids[:max_seq_len]
            mask = mask[:max_seq_len]
            if orig_len > max_seq_len and self.rank == 0:
                # Log truncation warning once per sample on rank 0
                print0(f"[WARN] Truncated sample from {orig_len} tokens to {max_seq_len} tokens")
            if len(ids) < 2:
                continue  # Need at least 2 tokens for next-token prediction

            # Shift for next-token prediction: input predicts next token
            # input_ids[i] should predict ids[i+1], so:
            #   input_ids = ids[:-1] (all but last)
            #   labels = ids[1:] (all but first, shifted by 1)
            input_ids = torch.tensor(ids[:-1], dtype=torch.long)
            labels = torch.tensor(ids[1:], dtype=torch.long)
            # Shift mask to align with labels (mask[i] corresponds to ids[i])
            mask_shifted = mask[1:]
            if sum(mask_shifted) == 0:
                # No assistant tokens to train on; skip to avoid NaN loss
                continue
            # Apply mask: positions where mask==0 should not contribute to loss
            # Use -1 as ignore_index (matches gpt.py F.cross_entropy ignore_index=-1)
            labels = torch.where(
                torch.tensor(mask_shifted, dtype=torch.long) == 1,
                labels,
                torch.tensor(-1, dtype=torch.long),
            )
            yield input_ids, labels


def collate_sft(batch, pad_token_id: int):
    """Collate variable-length sequences with dynamic padding to longest in batch."""
    input_ids_list, labels_list = zip(*batch)
    max_len = max(len(ids) for ids in input_ids_list)

    padded_input_ids = []
    padded_labels = []

    for ids, labels in zip(input_ids_list, labels_list):
        pad_len = max_len - len(ids)
        if pad_len > 0:
            padded_input_ids.append(torch.cat([ids, torch.full((pad_len,), pad_token_id, dtype=ids.dtype)]))
            padded_labels.append(torch.cat([labels, torch.full((pad_len,), -1, dtype=labels.dtype)]))
        else:
            padded_input_ids.append(ids)
            padded_labels.append(labels)

    return (
        torch.stack(padded_input_ids),
        torch.stack(padded_labels),
    )


def build_dataloader(tokenizer, seed, world_size, rank, split: str = "train") -> DataLoader:
    dataset = DocVQADataset(tokenizer, split=split, seed=seed, world_size=world_size, rank=rank)
    pad_token_id = tokenizer.encode_special("<|assistant_end|>")

    def collate_fn(batch):
        return collate_sft(batch, pad_token_id)

    return DataLoader(dataset, batch_size=device_batch_size, pin_memory=True, collate_fn=collate_fn)


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
    # RustBPETokenizer only needs tokenizer.pkl; get_token_bytes needs token_bytes.pt
    required = ["tokenizer.pkl", "token_bytes.pt"]

    def has_required(path: Path) -> bool:
        return all((path / r).exists() for r in required)

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
        cache_root = MODEL_CACHE_ROOT
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
    seed: int
    world_size: int
    device_type: str
    error: str | None = None
    grid_i: int | None = None
    grid_j: int | None = None


# ---------------------------------------------------------------------------
# Visualization and analysis helpers

def capture_baseline_samples(dataloader, device, num_samples: int = 5):
    """Capture fixed samples for before/after comparison."""
    samples = []
    for batch in dataloader:
        input_ids, labels = [b.to(device) for b in batch]
        for i in range(input_ids.shape[0]):
            if len(samples) >= num_samples:
                break
            samples.append({
                "input_ids": input_ids[i].clone(),
                "labels": labels[i].clone(),
            })
        if len(samples) >= num_samples:
            break
    return samples


def generate_sample_predictions(model, tokenizer, samples, device, cast_ctx, step: int, max_gen_tokens: int = 250):
    """Generate predictions for fixed samples and return table data.

    Args:
        model: The model to generate from
        tokenizer: Tokenizer for decoding
        samples: List of dicts with 'input_ids' and 'labels' tensors
        device: Device to run on
        cast_ctx: Autocast context function
        step: Training step (for logging)
        max_gen_tokens: Maximum tokens to generate per sample

    Returns:
        List of [step, sample_idx, prompt_text, generated_text, expected_text]
    """
    model.eval()
    table_data = []

    with torch.no_grad():
        for idx, sample in enumerate(samples):
            input_ids = sample["input_ids"].unsqueeze(0)
            labels = sample["labels"]

            # Find where assistant response starts (first non-masked token)
            non_masked = (labels >= 0).nonzero(as_tuple=True)[0]
            if len(non_masked) == 0:
                print0(f"[WARN] No non-masked tokens found in sample {idx} during sample generation.")
                continue

            prompt_end = int(non_masked[0].item())
            prompt_tokens = input_ids[0, :prompt_end]
            expected_tokens = labels[prompt_end:]
            expected_tokens = expected_tokens[expected_tokens >= 0]

            # Generate from model (greedy)
            gen_input = prompt_tokens.unsqueeze(0).to(device)
            generated = []
            max_gen_len = min(max_gen_tokens, len(expected_tokens) + 20)
            for _ in range(max_gen_len):
                with cast_ctx():
                    logits = model(gen_input)
                    if isinstance(logits, dict):
                        logits = logits.get("logits", logits)
                    if hasattr(logits, "shape") and len(logits.shape) == 3:
                        logits = logits[:, -1, :]
                next_token = logits.argmax(dim=-1)
                generated.append(next_token.item())
                gen_input = torch.cat([gen_input, next_token.unsqueeze(0)], dim=1)
                # Stop on EOS or assistant_end token
                if next_token.item() in [tokenizer.encode_special("<|assistant_end|>"), 0]:
                    break

            # Decode for table
            prompt_text = tokenizer.decode(prompt_tokens.tolist())
            expected_text = tokenizer.decode(expected_tokens.tolist()) if len(expected_tokens) > 0 else ""
            generated_text = tokenizer.decode(generated) if generated else ""

            table_data.append([
                step,
                idx,
                prompt_text,
                generated_text,
                expected_text,
            ])

    model.train()
    return table_data


def print_generation_table(
    table_data,
    stage: str,
    max_prompt: int = 200,
    max_expected: int = 100,
    max_generated: int = 100,
):
    """Print a generation table (as produced by generate_sample_predictions)."""
    if not table_data:
        print0(f"[WARN] No samples to display for {stage} predictions.")
        return 0

    def truncate(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[:limit] + "..."

    print0("=" * 60)
    print0(f"MODEL PREDICTIONS ({stage.upper()} training)")
    print0("=" * 60)

    for row in table_data:
        _, idx, prompt_text, generated_text, expected_text = row

        prompt_display = truncate(prompt_text, max_prompt)
        expected_display = truncate(expected_text or "", max_expected)
        generated_display = truncate(generated_text or "", max_generated)

        print0(f"\n--- Sample {idx + 1} ---")
        print0(f"Prompt: {prompt_display}")
        print0(f"Expected: {expected_display if expected_display else '(empty)'}")
        print0(f"Generated: {generated_display if generated_display else '(empty)'}")

    print0("=" * 60)
    return len(table_data)


def visualize_predictions(model, tokenizer, dataloader, device, cast_ctx, num_samples: int = 3, stage: str = "before"):
    """Generate and display model predictions on validation samples."""
    # Capture samples from dataloader
    samples = capture_baseline_samples(dataloader, device, num_samples=num_samples)

    # Generate predictions (shorter generation for display)
    table_data = generate_sample_predictions(
        model, tokenizer, samples, device, cast_ctx,
        step=0,  # step not used for display
        max_gen_tokens=50,
    )

    return print_generation_table(table_data, stage=stage)


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

def derive_num_iterations(tokens_per_batch: int, grad_accum_steps: int, tokens_goal: int, world_size: int) -> int:
    tokens_per_step = max(1, tokens_per_batch * grad_accum_steps * max(1, world_size))
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

    # Reproducibility: set all seeds and enable deterministic mode
    set_seed(seed, rank=rank)
    if deterministic:
        enable_deterministic_mode()
    run_repro = repro_context() | {
        "seed": seed,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "device": str(device),
    }
    if master:
        print0(f"[REPRO] {json.dumps(run_repro, indent=2)}")

    # wandb
    use_dummy = run == "dummy" or not master
    run_name = run
    if (grid or grid_i != 0 or grid_j != 0) and run:
        run_name = f"{run}-g{grid_i}-{grid_j}"

    # Build tags list from wandb_tags + grid_sweep_id; always include base "finetune" tag
    user_tags = [t.strip() for t in wandb_tags.split(",") if t.strip()]
    if not any(t.lower() == "finetune" for t in user_tags):
        user_tags.append("finetune")
    sweep_id = grid_sweep_id or datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_tags = list(dict.fromkeys(user_tags + [sweep_id]))  # preserve order, de-dup

    wb = (
        DummyWandb()
        if use_dummy
        else wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=run_name,
            config=user_config
            | {
                "learning_rate": lr,
                "num_tokens": tokens_goal,
                "grid_i": grid_i,
                "grid_j": grid_j,
                "seed": seed,
                "repro": run_repro,
                "grid_sweep_id": sweep_id,
            },
            tags=run_tags,
            save_code=True,
            settings=wandb.Settings(init_timeout=300, _service_wait=300),
        )
    )

    model, tokenizer = load_model_and_tok(device_type)
    # Cast model to float32 if dtype is float32, fp32 will have higher precision for granular sweeps
    if dtype == "float32":
        model = model.float()
    model.to(device)
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    if device_type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    dataloader = build_dataloader(tokenizer, seed=seed, world_size=world_size, rank=rank)
    pad_token_id = tokenizer.encode_special("<|assistant_end|>")
    data_iter = iter(dataloader)

    # Peek first batch to estimate tokens/batch (then reuse it)
    first_batch = next(data_iter)
    _, labels_peek = [x.to(device) for x in first_batch]
    tokens_per_batch = int((labels_peek >= 0).sum().item())
    data_iter = itertools.chain([first_batch], data_iter)

    examples_per_step = device_batch_size * world_size
    if target_examples_per_step <= 0:
        grad_accum_steps = 1
    else:
        assert (
            target_examples_per_step % examples_per_step == 0
        ), "Target examples per step must be divisible by examples per step"
        grad_accum_steps = target_examples_per_step // examples_per_step
    total_steps = num_iterations if num_iterations > 0 else derive_num_iterations(
        tokens_per_batch, grad_accum_steps, tokens_goal, world_size
    )
    warmup_steps = max(1, int(total_steps * warmup_frac))

    optim = torch.optim.AdamW(model.parameters(), lr=lr * init_lr_frac, weight_decay=weight_decay)

    def lr_mult(step_idx: int) -> float:
        # linear warmup
        if step_idx < warmup_steps:
            return (step_idx + 1) / warmup_steps
        # cosine decay from 1.0 to final_lr_frac
        progress = (step_idx - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return final_lr_frac + (1.0 - final_lr_frac) * cosine_decay

    # metrics
    losses: List[float] = []
    tokens_seen = 0
    start_time = time.time()
    error = None
    final_loss = float("inf")

    bf16_ok = device_type == "cuda" and torch.cuda.is_bf16_supported()
    use_bf16 = dtype == "bfloat16" and bf16_ok
    use_fp16 = dtype == "float16" and device_type == "cuda"
    use_autocast = use_bf16 or use_fp16
    autocast_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)
    scaler = GradScaler(enabled=use_fp16)  # disabled for fp32/bf16

    if use_autocast:
        def cast_ctx():
            return torch.amp.autocast(device_type=device_type, dtype=autocast_dtype)
    else:
        def cast_ctx():
            return nullcontext()

    # Log baseline samples for before/after comparison (use validation data)
    baseline_samples = []
    generations_table = None
    before_data = None
    if master:
        baseline_loader = build_dataloader(tokenizer, seed=seed + 999, world_size=1, rank=0, split="validation")
        baseline_samples = capture_baseline_samples(baseline_loader, device, num_samples=5)
        print0(f"Captured {len(baseline_samples)} validation samples for before/after comparison")

        # Generate predictions BEFORE training and log to wandb
        if baseline_samples and not use_dummy:
            generations_table = wandb.Table(
                columns=["step", "sample_idx", "prompt", "generated", "expected"],
                log_mode="INCREMENTAL",
            )
            before_data = generate_sample_predictions(model, tokenizer, baseline_samples, device, cast_ctx, step=0)
            if before_data:
                for row in before_data:
                    generations_table.add_data(*row)
                wb.log({"samples/generations_incr": generations_table})
                print0(f"Logged {len(before_data)} validation sample predictions (before training) to wandb table (incremental)")
                print_generation_table(before_data, stage="before")

    # Diagnostic: compute initial loss before any training
    if master:
        print0("=" * 60)
        print0("DIAGNOSTIC: Initial state before training")
        print0("=" * 60)
        model.eval()
        with torch.no_grad():
            diag_batch = next(iter(dataloader))
            diag_input, diag_labels = [b.to(device) for b in diag_batch]
            with cast_ctx():
                init_loss = model(diag_input, diag_labels).item()
            # Token statistics (exclude padding)
            nonpad_mask = (diag_input != pad_token_id) | (diag_labels >= 0)
            total_tokens = int(nonpad_mask.sum().item())
            trained_tokens = int((diag_labels >= 0).sum().item())
            masked_tokens = total_tokens - trained_tokens
            print0(f"Initial loss (before training): {init_loss:.4f}")
            print0(f"Batch shape: {diag_input.shape}")
            print0(f"Total non-pad tokens in batch: {total_tokens}")
            pct_trained = 0.0 if total_tokens == 0 else 100 * trained_tokens / total_tokens
            print0(f"Trained tokens (non-masked): {trained_tokens} ({pct_trained:.1f}%)")
            print0(f"Masked tokens (non-pad): {masked_tokens} ({100 - pct_trained:.1f}%)")
            wb.log({"diagnostic/initial_loss": init_loss, "diagnostic/trained_token_pct": pct_trained})
        model.train()
        print0("=" * 60)

    # Visualization: show predictions before training (fallback if no table data)
    if master and visualize and before_data is None:
        val_loader = build_dataloader(tokenizer, seed=seed + 999, world_size=1, rank=0, split="validation")
        visualize_predictions(model, tokenizer, val_loader, device, cast_ctx, num_samples=3, stage="before")

    # Training loop
    print0("=" * 60)
    print0("Training loop")
    print0("=" * 60)
    try:
        for step_idx in range(total_steps):
            optim.zero_grad(set_to_none=True)
            num_tokens_step = 0
            running_loss = 0.0
            for _ in range(grad_accum_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    batch = next(data_iter)
                batch = [b.to(device) for b in batch]
                b_input, b_labels = batch
                with cast_ctx():
                    loss = model(b_input, b_labels)
                scaled_loss = loss / grad_accum_steps
                if scaler.is_enabled():
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                running_loss += loss.detach()
                num_tokens_step += int((b_labels >= 0).sum().item())

            # lr schedule
            mult = lr_mult(step_idx)
            for pg in optim.param_groups:
                pg["lr"] = lr * mult
            if scaler.is_enabled():
                scaler.step(optim)
                scaler.update()
            else:
                optim.step()
            tokens_seen += num_tokens_step * world_size  # tokens counted on all ranks

            # reduce loss for logging
            loss_item = running_loss / grad_accum_steps
            if ddp:
                dist.all_reduce(loss_item, op=dist.ReduceOp.AVG)
            loss_scalar = loss_item.item()
            final_loss = loss_scalar
            if not math.isfinite(loss_scalar):
                error = "non-finite loss detected"
                break
            losses.append(loss_scalar)

            if master and (step_idx % log_every == 0 or step_idx + 1 == total_steps):
                print0(f"step {step_idx+1:05d}/{total_steps:05d} loss={loss_scalar:.4f} lr={lr*mult:.2e} tokens={tokens_seen:,}")
                wb.log(
                    {
                        "step": step_idx,
                        "train/loss": loss_scalar,
                        "train/lr": lr * mult,
                        "train/tokens_seen": tokens_seen,
                    }
                )

            if master and eval_every > 0 and (step_idx + 1) % eval_every == 0:
                # Lightweight validation on a few batches (rank 0 only)
                model.eval()
                val_loader = build_dataloader(tokenizer, seed=seed + 999, world_size=1, rank=0, split="validation")
                val_losses = []
                with torch.no_grad():
                    for b_idx, (v_input, v_labels) in enumerate(val_loader):
                        if b_idx >= eval_batches:
                            break
                        v_input, v_labels = v_input.to(device), v_labels.to(device)
                        with cast_ctx():
                            v_loss = model(v_input, v_labels)
                        val_losses.append(v_loss.item())
                if val_losses:
                    val_loss_mean = float(np.mean(val_losses))
                    wb.log({"val/loss": val_loss_mean, "val/batches": len(val_losses)})
                    print0(f"eval loss={val_loss_mean:.4f} over {len(val_losses)} batches")
                model.train()

    except Exception as exc:  # pylint: disable=broad-except
        error = str(exc)
        print0(f"[ERROR] training failed: {error}")

    runtime_s = time.time() - start_time
    converged = error is None and math.isfinite(final_loss) and final_loss < convergence_loss_threshold
    avg_loss = float(np.mean(losses)) if losses else float("inf")

    if device_type == "cuda" and master:
        max_alloc_gb = torch.cuda.max_memory_allocated() / (1024**3)
        max_reserved_gb = torch.cuda.max_memory_reserved() / (1024**3)
        print0(f"[MEM] max_allocated_gb={max_alloc_gb:.2f} max_reserved_gb={max_reserved_gb:.2f}")
        wb.log({"memory/max_allocated_gb": max_alloc_gb, "memory/max_reserved_gb": max_reserved_gb})

    # Generate predictions AFTER training on same baseline samples and log to wandb
    after_data = None
    if master and generations_table is not None and baseline_samples and not use_dummy:
        after_data = generate_sample_predictions(model, tokenizer, baseline_samples, device, cast_ctx, step=total_steps)
        if after_data:
            for row in after_data:
                generations_table.add_data(*row)
            wb.log({"samples/generations_incr": generations_table})
            print0(f"Logged {len(after_data)} validation sample predictions (after training, step {total_steps}) to wandb table (incremental)")
            print_generation_table(after_data, stage="after")

    # Log a final immutable table for stable viewing in the W&B UI
    if master and generations_table is not None and not use_dummy and generations_table.data:
        final_table = wandb.Table(
            columns=generations_table.columns,
            data=generations_table.data,
            log_mode="IMMUTABLE",
        )
        wb.log({"samples/generations": final_table})
        print0(f"Logged final generations table with {len(generations_table.data)} rows to wandb")

    # Display: show predictions after training (fallback if no table data)
    if master and visualize and after_data is None:
        val_loader = build_dataloader(tokenizer, seed=seed + 999, world_size=1, rank=0, split="validation")
        visualize_predictions(model, tokenizer, val_loader, device, cast_ctx, num_samples=3, stage="after")

    # Save checkpoint and upload artifact (single runs only, not grid)
    if master and save_artifacts and not grid:
        # Unwrap DDP if needed
        raw_model = model.module if ddp else model

        # Save checkpoint locally
        checkpoint_dir = CHECKPOINTS_ROOT / run_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model_config_kwargs = raw_model.config.__dict__ if hasattr(raw_model, "config") else {}
        save_checkpoint(
            str(checkpoint_dir),
            total_steps,
            raw_model.state_dict(),
            None,  # don't save optimizer state
            {
                "step": total_steps,
                "final_loss": final_loss,
                "tokens_seen": tokens_seen,
                "converged": converged,
                "learning_rate": lr,
                "model_config": model_config_kwargs,
            },
            rank=0,
        )
        print0(f"Saved checkpoint to {checkpoint_dir}")

        # Upload to W&B as artifact
        if not use_dummy:
            artifact_name = f"{run_name}-artifact" if not run_name.endswith("-artifact") else run_name
            art = wandb.Artifact(artifact_name, type="model")
            art.add_dir(str(checkpoint_dir), name="checkpoints")
            # Bundle tokenizer if available
            tok_dir = Path(get_base_dir()) / "tokenizer"
            if tok_dir.is_dir():
                art.add_dir(str(tok_dir), name="tokenizer")
            wb.log_artifact(art, aliases=["finetune", run_name, "latest"])
            print0(f"Uploaded artifact: {artifact_name}")

    if master and not use_dummy:
        wb.summary["final_train_loss"] = final_loss
        wb.summary["converged"] = converged
        wb.summary["tokens_seen"] = tokens_seen
        wb.summary["runtime_s"] = runtime_s
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
        seed=seed,
        world_size=world_size,
        device_type=device_type,
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
                ckpt_dir = RESULTS_ROOT
                ckpt_dir.mkdir(exist_ok=True, parents=True)
                ckpt_path = ckpt_dir / f"finetune_grid_{resolution}x{resolution}.json"
                with open(ckpt_path, "w", encoding="utf-8") as f:
                    json.dump([asdict(r) for r in results], f, indent=2)
                print0(f"Checkpointed {len(results)} results to {ckpt_path}")

    # Rank 0: finalize artifacts
    if int(os.environ.get("RANK", 0)) == 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = RESULTS_ROOT
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
            "repro": repro_context(),
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
            summary_tags = ["fractal-grid", "finetune"]
            summary_run = wandb.init(
                project=WANDB_PROJECT,
                entity=WANDB_ENTITY,
                name=f"{run}-grid-summary",
                config=config_dump | {"fractal_dimension": fractal["fractal_dimension"]},
                tags=summary_tags,
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
        ddp, rank, local_rank, world_size, device, _device_type = runtime
        res = train_once(runtime=runtime, grid_i=grid_i, grid_j=grid_j)
        if int(os.environ.get("RANK", 0)) == 0:
            print0(f"\nFinal: loss={res.final_loss:.4f} tokens_seen={res.tokens_seen:,} converged={res.converged}")

            # Save results JSON (single run artifact)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = RESULTS_ROOT
            results_dir.mkdir(exist_ok=True, parents=True)
            results_path = results_dir / f"finetune_single_{run}_{timestamp}.json"
            repro = repro_context() | {
                "seed": res.seed,
                "world_size": res.world_size,
                "device_type": res.device_type,
            }
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump({"config": user_config, "result": asdict(res), "repro": repro}, f, indent=2)
            print0(f"Saved results: {results_path}")

        exit_code = 1 if res.error is not None else 0
        if ddp and dist.is_available() and dist.is_initialized():
            # Sync failure across ranks so torchrun sees a consistent exit status.
            flag = torch.tensor(exit_code, device=device)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            exit_code = int(flag.item())

        compute_cleanup()
        if exit_code != 0:
            raise SystemExit(exit_code)
        return

    compute_cleanup()


if __name__ == "__main__":
    main()
