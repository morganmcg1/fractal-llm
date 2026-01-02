#!/usr/bin/env python3
"""
Chat wrapper that loads a model from a W&B artifact and serves the chat UI.

This is a thin wrapper around third_party/nanochat/scripts/chat_web.py that:
1. Downloads the model from W&B artifact (or uses a local checkpoint path)
2. Sets up the directory structure nanochat expects
3. Launches the FastAPI chat server

Usage:
    # Load from W&B artifact (default: morgy/fractal-llm/nanochat-fin-rl-artifact:v7)
    uv run src/chat.py

    # Specify a different W&B artifact
    uv run src/chat.py --model wandb:morgy/fractal-llm/my-finetuned-artifact:v1

    # Use a local checkpoint directory
    uv run src/chat.py --model /path/to/checkpoints

    # With custom settings
    uv run src/chat.py --port 8080 --temperature 0.6 --max-tokens 1024
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import simple_parsing as sp
from rich.console import Console
from rich.panel import Panel

console = Console()

# Project paths
REPO_ROOT = Path(__file__).resolve().parent.parent
NANOCHAT_DIR = REPO_ROOT / "third_party" / "nanochat"

# Default storage for downloaded artifacts
STORAGE_ROOT = Path(os.environ.get("FRACTAL_STORAGE_DIR", Path.home() / ".cache" / "fractal-llm"))
MODEL_CACHE_ROOT = STORAGE_ROOT / "model_cache"
NANOCHAT_CACHE = STORAGE_ROOT / "nanochat"


@dataclass
class Args:
    """Chat server with W&B artifact model loading"""

    model: str = os.environ.get("MODEL_ARTIFACT", "morgy/fractal-llm/nanochat-fin-rl-artifact:v7")
    """W&B artifact path (entity/project/name:version) or local checkpoint directory"""

    port: int = 8000
    """Port to run the server on"""

    host: str = "0.0.0.0"
    """Host to bind the server to"""

    temperature: float = 0.8
    """Default temperature for generation"""

    top_k: int = 50
    """Default top-k sampling parameter"""

    max_tokens: int = 512
    """Default max tokens for generation"""

    num_gpus: int = 1
    """Number of GPUs to use (CUDA only)"""

    dtype: str = "bfloat16"
    """Data type: float32 or bfloat16"""


def resolve_artifact(model_ref: str) -> Path:
    """Download W&B artifact if needed, otherwise treat as local path.

    Returns the path to the checkpoint directory containing model_*.pt files.
    """
    # Strip wandb: prefix if present
    if model_ref.startswith("wandb:"):
        model_ref = model_ref[len("wandb:"):]

    # Check if it's a local path
    local_path = Path(model_ref)
    if local_path.exists():
        # Find the actual checkpoint directory
        if (local_path / "checkpoints").exists():
            return local_path / "checkpoints"
        if list(local_path.glob("model_*.pt")):
            return local_path
        # Look one level deeper
        for subdir in local_path.iterdir():
            if subdir.is_dir() and list(subdir.glob("model_*.pt")):
                return subdir
        console.print(f"[yellow]Warning: No model_*.pt found in {local_path}, using as-is[/yellow]")
        return local_path

    # Download from W&B
    console.print(f"[blue]Downloading artifact:[/blue] {model_ref}")
    import wandb

    api = wandb.Api()
    art = api.artifact(model_ref, type="model")

    MODEL_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    cache_name = art.name.replace(":", "_").replace("/", "_")
    target = MODEL_CACHE_ROOT / cache_name

    if not target.exists():
        console.print(f"[dim]Downloading to {target}...[/dim]")
        art.download(root=str(target))
        console.print(f"[green]Downloaded![/green]")
    else:
        console.print(f"[dim]Using cached artifact at {target}[/dim]")

    # Find checkpoint directory
    if (target / "checkpoints").exists():
        return target / "checkpoints"
    if list(target.glob("model_*.pt")):
        return target
    # Look for subdirectory with checkpoints
    for subdir in target.iterdir():
        if subdir.is_dir() and list(subdir.glob("model_*.pt")):
            return subdir

    return target


def ensure_tokenizer(artifact_root: Path) -> Path:
    """Ensure tokenizer files are available in the nanochat cache.

    Looks for tokenizer files in the artifact and copies them to the expected location.
    """
    NANOCHAT_CACHE.mkdir(parents=True, exist_ok=True)
    tok_dir = NANOCHAT_CACHE / "tokenizer"
    tok_dir.mkdir(parents=True, exist_ok=True)

    required = ["tokenizer.pkl", "token_bytes.pt"]

    def has_required(path: Path) -> bool:
        return all((path / r).exists() for r in required)

    if has_required(tok_dir):
        return tok_dir

    # Search for tokenizer files in artifact
    search_dirs = [
        artifact_root,
        artifact_root / "tokenizer",
        artifact_root.parent,
        artifact_root.parent / "tokenizer",
    ]

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for req_file in required:
            for found in search_dir.rglob(req_file):
                target = tok_dir / found.name
                if not target.exists():
                    console.print(f"[dim]Copying {found.name} to tokenizer cache[/dim]")
                    shutil.copy2(found, target)

    if has_required(tok_dir):
        console.print(f"[green]Tokenizer ready at {tok_dir}[/green]")
        return tok_dir

    # If still missing, the model might be using the default nanochat tokenizer
    console.print(
        "[yellow]Warning: Tokenizer files not found in artifact. "
        "Nanochat will download the default tokenizer.[/yellow]"
    )
    return tok_dir


def setup_nanochat_structure(checkpoint_dir: Path, model_tag: str = "artifact") -> tuple[str, str]:
    """Set up the directory structure nanochat expects.

    nanochat's load_model expects:
        $NANOCHAT_BASE_DIR/{source}_checkpoints/{model_tag}/model_*.pt

    We create this structure by symlinking to the actual checkpoint directory.

    Returns (source, model_tag) to pass to chat_web.py
    """
    # Use a custom source name that we'll create
    source = "sft"  # Use sft as the source type
    source_dir_name = "chatsft_checkpoints"

    checkpoints_base = NANOCHAT_CACHE / source_dir_name
    checkpoints_base.mkdir(parents=True, exist_ok=True)

    model_tag_dir = checkpoints_base / model_tag

    # Remove existing symlink/dir if it exists
    if model_tag_dir.is_symlink():
        model_tag_dir.unlink()
    elif model_tag_dir.exists():
        shutil.rmtree(model_tag_dir)

    # Create symlink to the actual checkpoint directory
    model_tag_dir.symlink_to(checkpoint_dir.resolve())
    console.print(f"[dim]Linked {model_tag_dir} -> {checkpoint_dir}[/dim]")

    return source, model_tag


def main():
    args = sp.parse(Args)

    console.print(Panel.fit(
        "[bold blue]NanoChat Server[/bold blue]\n"
        f"Model: {args.model}\n"
        f"Port: {args.port}",
        title="fractal-llm chat"
    ))

    # Resolve the model artifact/path
    checkpoint_dir = resolve_artifact(args.model)
    console.print(f"[green]Checkpoint directory:[/green] {checkpoint_dir}")

    # Verify checkpoint files exist
    model_files = list(checkpoint_dir.glob("model_*.pt"))
    if not model_files:
        console.print(f"[red]Error: No model_*.pt files found in {checkpoint_dir}[/red]")
        sys.exit(1)
    console.print(f"[dim]Found {len(model_files)} checkpoint file(s)[/dim]")

    # Ensure tokenizer is available
    ensure_tokenizer(checkpoint_dir.parent if checkpoint_dir.name != "checkpoints" else checkpoint_dir)

    # Set up nanochat directory structure
    source, model_tag = setup_nanochat_structure(checkpoint_dir)

    # Set environment variable for nanochat base directory
    os.environ["NANOCHAT_BASE_DIR"] = str(NANOCHAT_CACHE)

    # Build command to run chat_web.py
    chat_web_script = NANOCHAT_DIR / "scripts" / "chat_web.py"
    if not chat_web_script.exists():
        console.print(f"[red]Error: chat_web.py not found at {chat_web_script}[/red]")
        sys.exit(1)

    cmd = [
        sys.executable,
        str(chat_web_script),
        "--source", source,
        "--model-tag", model_tag,
        "--port", str(args.port),
        "--host", args.host,
        "--temperature", str(args.temperature),
        "--top-k", str(args.top_k),
        "--max-tokens", str(args.max_tokens),
        "--num-gpus", str(args.num_gpus),
        "--dtype", args.dtype,
    ]

    console.print()
    console.rule("[bold green]Starting Chat Server[/bold green]")
    console.print(f"[dim]Command: {' '.join(cmd)}[/dim]")
    console.print()

    # Run chat_web.py
    try:
        # Add nanochat to Python path
        env = os.environ.copy()
        python_path = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{NANOCHAT_DIR}:{python_path}" if python_path else str(NANOCHAT_DIR)

        subprocess.run(cmd, env=env, check=True)
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped[/yellow]")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Server exited with code {e.returncode}[/red]")
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
