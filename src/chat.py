#!/usr/bin/env python3
"""
Chat wrapper that loads a model and serves the chat UI.

This is a thin wrapper around third_party/nanochat/scripts/chat_web.py that:
1. Downloads the model from W&B artifact / Hugging Face (or uses a local checkpoint path)
2. Sets up the directory structure nanochat expects
3. Launches the FastAPI chat server

Usage:
    # Load from W&B artifact (default: morgy/fractal-llm/nanochat-fin-rl-artifact:v7)
    uv run src/chat.py

    # Specify a different W&B artifact
    uv run src/chat.py --model wandb:morgy/fractal-llm/my-finetuned-artifact:v1

    # Load from Hugging Face (e.g. karpathy/nanochat-d32)
    uv run src/chat.py --model hf:karpathy/nanochat-d32

    # Use a local checkpoint directory
    uv run src/chat.py --model /path/to/checkpoints

    # With custom settings
    uv run src/chat.py --port 8080 --temperature 0.6 --max-tokens 1024
"""

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import simple_parsing as sp
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

console = Console()

# Project paths
REPO_ROOT = Path(__file__).resolve().parent.parent
NANOCHAT_DIR = REPO_ROOT / "third_party" / "nanochat"

# Load .env if present (WANDB_API_KEY etc.)
ENV_PATH = REPO_ROOT / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH, override=False)

# Default storage for downloaded artifacts
STORAGE_ROOT = Path(os.environ.get("FRACTAL_STORAGE_DIR", Path.home() / ".cache" / "fractal-llm"))
MODEL_CACHE_ROOT = STORAGE_ROOT / "model_cache"
NANOCHAT_CACHE = STORAGE_ROOT / "nanochat"

# Keep HF caches out of tiny devpod workspaces by default.
_hf_home = STORAGE_ROOT / "huggingface"
os.environ.setdefault("HF_HOME", str(_hf_home))
os.environ.setdefault("HF_HUB_CACHE", str(_hf_home / "hub"))
Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["HF_HUB_CACHE"]).mkdir(parents=True, exist_ok=True)

SOURCE_TO_CHECKPOINT_DIR = {
    "sft": "chatsft_checkpoints",
    "mid": "mid_checkpoints",
    "rl": "chatrl_checkpoints",
}


@dataclass
class Args:
    """Chat server with W&B artifact model loading"""

    model: str = os.environ.get("MODEL_ARTIFACT", "morgy/fractal-llm/nanochat-fin-rl-artifact:v7")
    """Model reference: wandb:<artifact> | hf:<repo_id> | /local/path"""

    source: str = ""
    """Model source for nanochat loader: sft|mid|rl. If empty, auto-detect for W&B artifacts."""

    model_tag: str = "artifact"
    """Model tag name (symlink) used under $NANOCHAT_BASE_DIR/<source checkpoints>/"""

    device_type: str = ""
    """Device type: cuda|cpu|mps. If empty, nanochat auto-detects."""

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


def _has_model_files(checkpoint_dir: Path) -> bool:
    return bool(list(checkpoint_dir.glob("model_*.pt")))


def _has_tokenizer_files(tokenizer_dir: Path) -> bool:
    required = ["tokenizer.pkl", "token_bytes.pt"]
    return all((tokenizer_dir / name).exists() for name in required)


def _infer_source_from_wandb_artifact(artifact) -> str | None:
    raw_aliases = getattr(artifact, "aliases", [])
    aliases: set[str] = set()
    for alias in raw_aliases:
        if isinstance(alias, str):
            aliases.add(alias.lower())
        else:
            aliases.add(str(getattr(alias, "name", alias)).lower())
    for candidate in ("rl", "sft", "mid"):
        if candidate in aliases:
            return candidate
    return None


def _download_wandb_artifact(model_ref: str) -> tuple[Path, str | None]:
    console.print(f"[blue]Downloading artifact:[/blue] {model_ref}")
    import wandb

    api = wandb.Api()
    artifact = api.artifact(model_ref, type="model")

    MODEL_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    downloaded_dir = Path(artifact.download(root=str(MODEL_CACHE_ROOT)))
    console.print(f"[green]Artifact ready at:[/green] {downloaded_dir}")
    return downloaded_dir, _infer_source_from_wandb_artifact(artifact)


def _is_hf_model_ref(model_ref: str) -> bool:
    if model_ref.startswith("hf:"):
        return True
    if model_ref.startswith("wandb:"):
        return False
    # Heuristic: W&B artifacts almost always include a :version or :alias.
    if ":" in model_ref:
        return False
    return "/" in model_ref


def _parse_hf_model_ref(model_ref: str) -> tuple[str, str | None]:
    if model_ref.startswith("hf:"):
        model_ref = model_ref[len("hf:"):]
    if "@" in model_ref:
        repo_id, revision = model_ref.split("@", 1)
        return repo_id, revision
    return model_ref, None


def _infer_model_tag_from_hf_repo_id(repo_id: str) -> str | None:
    # e.g., "karpathy/nanochat-d32" -> "d32"
    leaf = repo_id.rsplit("/", 1)[-1]
    match = re.search(r"(?:^|-)d(\d+)$", leaf)
    if match:
        return f"d{match.group(1)}"
    return None


def _download_hf_model(model_ref: str) -> tuple[Path, Path, str | None, str | None]:
    repo_id, revision = _parse_hf_model_ref(model_ref)
    model_tag = _infer_model_tag_from_hf_repo_id(repo_id)
    inferred_source = "sft"  # HF nanochat uploads are SFT checkpoints by convention

    # Stage into our own cache dir so we can symlink a stable structure for nanochat.
    safe_repo = repo_id.replace("/", "__")
    target_root = MODEL_CACHE_ROOT / "hf" / safe_repo
    if revision:
        target_root = target_root / f"rev_{revision}"
    tokenizer_dir = target_root / "tokenizer"
    checkpoints_dir = target_root / "checkpoints" / (model_tag or "hf")
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    try:
        repo_files = api.list_repo_files(repo_id=repo_id, repo_type="model", revision=revision)
    except Exception:
        repo_files = []

    required_tokenizer = ["tokenizer.pkl", "token_bytes.pt"]
    for name in required_tokenizer:
        dest = tokenizer_dir / name
        if dest.exists():
            continue

        candidates = [name] + [p for p in repo_files if p.endswith(f"/{name}") or p == name]
        downloaded = None
        for candidate in candidates:
            try:
                downloaded = Path(
                    hf_hub_download(repo_id=repo_id, filename=candidate, repo_type="model", revision=revision)
                )
            except Exception:
                continue
            break
        if downloaded is None:
            raise FileNotFoundError(f"Missing required tokenizer file in HF repo {repo_id!r}: {name}")
        shutil.copy2(downloaded, dest)

    # Karpathy's nanochat uploads are "janky": single checkpoint pair at a specific step.
    # Try common names first (d32 uses 000650).
    checkpoint_candidates = [
        ("model_000650.pt", "meta_000650.json"),
    ]
    downloaded_any = False
    for model_name, meta_name in checkpoint_candidates:
        model_dest = checkpoints_dir / model_name
        meta_dest = checkpoints_dir / meta_name
        if model_dest.exists() and meta_dest.exists():
            downloaded_any = True
            break
        candidates_model = [model_name] + [p for p in repo_files if p.endswith(f"/{model_name}") or p == model_name]
        candidates_meta = [meta_name] + [p for p in repo_files if p.endswith(f"/{meta_name}") or p == meta_name]
        model_src = None
        meta_src = None
        for candidate in candidates_model:
            try:
                model_src = Path(
                    hf_hub_download(repo_id=repo_id, filename=candidate, repo_type="model", revision=revision)
                )
                break
            except Exception:
                continue
        for candidate in candidates_meta:
            try:
                meta_src = Path(
                    hf_hub_download(repo_id=repo_id, filename=candidate, repo_type="model", revision=revision)
                )
                break
            except Exception:
                continue
        if model_src is None or meta_src is None:
            continue
        shutil.copy2(model_src, model_dest)
        shutil.copy2(meta_src, meta_dest)
        downloaded_any = True
        break

    if not downloaded_any:
        model_steps: dict[int, str] = {}
        meta_steps: dict[int, str] = {}
        for path in repo_files:
            m = re.search(r"model_(\d+)\.pt$", path)
            if m:
                model_steps[int(m.group(1))] = path
            m = re.search(r"meta_(\d+)\.json$", path)
            if m:
                meta_steps[int(m.group(1))] = path

        common_steps = sorted(set(model_steps).intersection(meta_steps), reverse=True)
        if not common_steps:
            raise FileNotFoundError(
                f"Could not find nanochat checkpoint files in HF repo {repo_id!r}. "
                "Expected model_<step>.pt and meta_<step>.json."
            )
        step = common_steps[0]
        model_path = model_steps[step]
        meta_path = meta_steps[step]
        console.print(f"[dim]Using HF checkpoint step {step:06d} ({model_path}, {meta_path})[/dim]")

        model_src = Path(hf_hub_download(repo_id=repo_id, filename=model_path, repo_type="model", revision=revision))
        meta_src = Path(hf_hub_download(repo_id=repo_id, filename=meta_path, repo_type="model", revision=revision))
        shutil.copy2(model_src, checkpoints_dir / f"model_{step:06d}.pt")
        shutil.copy2(meta_src, checkpoints_dir / f"meta_{step:06d}.json")

    if not _has_model_files(checkpoints_dir):
        raise FileNotFoundError(f"HF checkpoint staging failed; no model_*.pt found in {checkpoints_dir}")
    if not _has_tokenizer_files(tokenizer_dir):
        raise FileNotFoundError(
            f"HF tokenizer staging failed; expected tokenizer.pkl + token_bytes.pt in {tokenizer_dir}"
        )

    return checkpoints_dir, tokenizer_dir, inferred_source, model_tag


def _find_checkpoint_dir(model_root: Path) -> Path:
    """Return a directory containing model_*.pt files."""
    candidates: list[Path] = []
    direct = model_root / "checkpoints"
    if direct.exists():
        candidates.append(direct)
    candidates.append(model_root)

    # Common W&B layout: model_root/<artifact-name>/checkpoints
    for nested in model_root.rglob("checkpoints"):
        if nested.is_dir():
            candidates.append(nested)

    for candidate in candidates:
        if candidate.exists() and _has_model_files(candidate):
            return candidate

    # Fall back to any directory that directly contains model_*.pt
    for model_file in model_root.rglob("model_*.pt"):
        return model_file.parent

    raise FileNotFoundError(
        f"Could not find any model_*.pt files under {model_root}. "
        "Expected a NanoChat checkpoint directory or an artifact containing a 'checkpoints/' folder."
    )


def _find_tokenizer_dir(model_root: Path, checkpoint_dir: Path) -> Path:
    """Return a directory containing tokenizer.pkl and token_bytes.pt."""
    search_roots: list[Path] = []
    for root in (model_root, model_root.parent, checkpoint_dir, checkpoint_dir.parent):
        if root.exists():
            search_roots.append(root)

    # Fast paths first
    for root in search_roots:
        direct = root / "tokenizer"
        if direct.exists() and _has_tokenizer_files(direct):
            return direct

    # More flexible search
    for root in search_roots:
        for tok_file in root.rglob("tokenizer.pkl"):
            tok_dir = tok_file.parent
            if _has_tokenizer_files(tok_dir):
                return tok_dir

    raise FileNotFoundError(
        "Tokenizer files not found. Expected tokenizer directory containing "
        "tokenizer.pkl and token_bytes.pt in the model artifact/path."
    )


def resolve_model_assets(model_ref: str) -> tuple[Path, Path, str | None, str | None]:
    """Resolve (checkpoint_dir, tokenizer_dir, inferred_source, inferred_model_tag)."""
    if _is_hf_model_ref(model_ref):
        return _download_hf_model(model_ref)

    # Strip wandb: prefix if present
    if model_ref.startswith("wandb:"):
        model_ref = model_ref[len("wandb:"):]

    # Check if it's a local path
    local_path = Path(model_ref).expanduser()
    if local_path.exists():
        model_root = local_path
        checkpoint_dir = _find_checkpoint_dir(model_root)
        tokenizer_dir = _find_tokenizer_dir(model_root, checkpoint_dir)
        return checkpoint_dir, tokenizer_dir, None, None

    model_root, inferred_source = _download_wandb_artifact(model_ref)
    checkpoint_dir = _find_checkpoint_dir(model_root)
    tokenizer_dir = _find_tokenizer_dir(model_root, checkpoint_dir)
    return checkpoint_dir, tokenizer_dir, inferred_source, None


def setup_tokenizer_structure(tokenizer_dir: Path) -> Path:
    """Point $NANOCHAT_BASE_DIR/tokenizer at the correct tokenizer directory."""
    if not _has_tokenizer_files(tokenizer_dir):
        raise FileNotFoundError(
            f"Tokenizer directory missing required files (tokenizer.pkl, token_bytes.pt): {tokenizer_dir}"
        )

    NANOCHAT_CACHE.mkdir(parents=True, exist_ok=True)
    tok_target = NANOCHAT_CACHE / "tokenizer"

    if tok_target.is_symlink():
        tok_target.unlink()
    elif tok_target.exists():
        shutil.rmtree(tok_target)

    tok_target.symlink_to(tokenizer_dir.resolve())
    console.print(f"[dim]Linked {tok_target} -> {tokenizer_dir}[/dim]")
    return tok_target


def setup_nanochat_structure(checkpoint_dir: Path, source: str, model_tag: str) -> tuple[str, str]:
    """Set up the directory structure nanochat expects.

    nanochat's load_model expects:
        $NANOCHAT_BASE_DIR/<source checkpoints dir>/{model_tag}/model_*.pt

    We create this structure by symlinking to the actual checkpoint directory.

    Returns (source, model_tag) to pass to chat_web.py
    """
    if source not in SOURCE_TO_CHECKPOINT_DIR:
        raise ValueError(f"Invalid source: {source}. Expected one of: {', '.join(SOURCE_TO_CHECKPOINT_DIR)}")

    checkpoints_base = NANOCHAT_CACHE / SOURCE_TO_CHECKPOINT_DIR[source]
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

    checkpoint_dir, tokenizer_dir, inferred_source, inferred_model_tag = resolve_model_assets(args.model)
    console.print(f"[green]Checkpoint directory:[/green] {checkpoint_dir}")
    console.print(f"[green]Tokenizer directory:[/green] {tokenizer_dir}")

    resolved_source = args.source or inferred_source or "sft"
    if args.source == "" and inferred_source is not None:
        console.print(f"[dim]Auto-detected source from W&B artifact aliases: {resolved_source}[/dim]")

    resolved_model_tag = args.model_tag
    if inferred_model_tag is not None and args.model_tag == "artifact":
        resolved_model_tag = inferred_model_tag
        console.print(f"[dim]Auto-detected model tag from HF repo id: {resolved_model_tag}[/dim]")

    # Ensure nanochat base dir points at the correct tokenizer/checkpoints
    setup_tokenizer_structure(tokenizer_dir)
    source, model_tag = setup_nanochat_structure(checkpoint_dir, resolved_source, resolved_model_tag)

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
    if args.device_type:
        cmd.extend(["--device-type", args.device_type])

    console.print(Panel.fit(
        f"Open: http://localhost:{args.port}\n"
        "If this is running on a remote machine (e.g. devpod), forward the port from your laptop:\n"
        f"  ssh -L {args.port}:localhost:{args.port} <host>",
        title="Connect",
    ))

    console.print()
    console.rule("[bold green]Starting Chat Server[/bold green]")
    console.print(f"[dim]Command: {' '.join(cmd)}[/dim]")
    console.print()

    # Run chat_web.py
    try:
        # Add nanochat to Python path
        env = os.environ.copy()
        python_path = env.get("PYTHONPATH", "")

        # nanochat imports rustbpe unconditionally, but inference doesn't require it.
        # If the extension isn't installed, provide a tiny shim so chat works out of the box.
        extra_paths: list[str] = []
        try:
            import rustbpe  # noqa: F401
        except ModuleNotFoundError:
            shim_dir = REPO_ROOT / "src" / "_nanochat_shims"
            extra_paths.append(str(shim_dir))
            console.print("[yellow]rustbpe not installed; using inference-only shim.[/yellow]")

        extra_paths.append(str(NANOCHAT_DIR))
        if python_path:
            extra_paths.append(python_path)
        env["PYTHONPATH"] = os.pathsep.join(extra_paths)

        subprocess.run(cmd, env=env, check=True, cwd=str(NANOCHAT_DIR))
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped[/yellow]")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Server exited with code {e.returncode}[/red]")
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
