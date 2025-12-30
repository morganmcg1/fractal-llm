"""
Run nanochat d20 training on Modal with 8×H100 and push artifacts to W&B.

This script clones the upstream nanochat repo inside the Modal container,
runs the speedrun (d20) script, and uploads the resulting checkpoint/report
to W&B (entity= morgan, project= fractal-llm).

Usage (from host):
    uv run modal run src/nanochat_modal.py --wandb-name my-d20-modal-run
"""

from dataclasses import dataclass
from pathlib import Path
import os
import tarfile

import modal
import simple_parsing as sp
from dotenv import load_dotenv

LOCAL_NANOCHAT_DIR = Path(__file__).resolve().parent.parent / "third_party" / "nanochat"
LOCAL_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

# Load .env from project root
if LOCAL_ENV_PATH.exists():
    load_dotenv(LOCAL_ENV_PATH, override=False)

# Configuration from .env (with defaults for backwards compatibility)
WANDB_ENTITY = os.getenv("WANDB_ENTITY", "morgan")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "fractal-llm")
NANOCHAT_REPO = "https://github.com/karpathy/nanochat.git"
BRANCH = ""  # use upstream default branch


# Modal image: CUDA 12.8 + Torch cu128 + git + uv (all installs via uv)
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "curl")
    .run_commands(
        # Install uv, then use uv pip (system) for everything
        "pip install uv>=0.4.0 "
        "&& uv pip install --system packaging ninja "
        "&& uv pip install --system --index-url https://download.pytorch.org/whl/cu128 "
        "torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 "
        "&& uv pip install --system wandb>=0.23.1 simple-parsing>=0.1.7 rich>=14.0.0 python-dotenv>=1.0.1"
    )
    # bundle vendored nanochat into the image so Mount API isn't needed
    .add_local_dir(LOCAL_NANOCHAT_DIR, "/workspace/nanochat_src")
)


app = modal.App("nanochat-modal")


@dataclass
class Args:
    """Modal entrypoint arguments."""

    wandb_name: str = "nanochat-d20-modal"
    repo_branch: str = BRANCH
    save_artifact_name: str = "nanochat-d20-speedrun"
    smoke: bool = False  # fast logging-only test path


def _pack_dir(src: Path, dest_tar: Path):
    """Tar.gz a directory if it exists."""
    if not src.exists():
        return None
    with tarfile.open(dest_tar, "w:gz") as tar:
        tar.add(src, arcname=src.name)
    return dest_tar


@app.function(
    gpu="H100:8",
    image=image,
    timeout=60 * 60 * 6,  # 6 hours cap
    secrets=[modal.Secret.from_name("wandb-secret")],
    volumes={"/results": modal.Volume.from_name("fractal-llm-results", create_if_missing=True)},
)
def train_d20(args: Args):
    import subprocess
    import json
    import shutil
    import wandb
    from rich.console import Console
    from rich.panel import Panel
    from dotenv import load_dotenv

    console = Console()
    console.rule("[bold blue]nanochat d20 on Modal (8×H100)")

    for candidate in ("/workspace/.env", "/root/.env", ".env"):
        if os.path.exists(candidate):
            load_dotenv(candidate, override=False)

    os.environ["WANDB_ENTITY"] = WANDB_ENTITY
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT
    os.environ["WANDB_NAME"] = args.wandb_name
    os.environ["WANDB_MODE"] = "online"
    os.environ["WANDB_RUN"] = args.wandb_name  # nanochat logging gate (defaults to 'dummy' if unset)
    os.environ["SAVE_ARTIFACT_NAME"] = args.save_artifact_name
    if args.smoke:
        os.environ["SMOKE"] = "1"

    workdir = Path("/workspace")
    repo_dir = workdir / "nanochat"

    # Use vendored nanochat (mounted from host) so our patches and pinned commit are used.
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    shutil.copytree("/workspace/nanochat_src", repo_dir, dirs_exist_ok=False)

    # Install dependencies via uv (editable install)
    subprocess.run(["uv", "pip", "install", "--system", "-e", "."], cwd=repo_dir, check=True)

    # Run speedrun (d20) with W&B env set and explicit run name to enable wandb logging
    console.print(Panel(f"Starting speedrun.sh (d20, 8×H100) run={args.wandb_name}", title="Train"))
    subprocess.run(["bash", "speedrun.sh", "--run", args.wandb_name], cwd=repo_dir, check=True)

    # Collect artifacts
    outputs = {}
    report = repo_dir / "report.md"
    if report.exists():
        outputs["report"] = str(report)

    # The speedrun writes checkpoints under out/, bundle them if present
    out_dir = repo_dir / "out"
    out_tar = workdir / "nanochat_out.tar.gz"
    out_bundle = _pack_dir(out_dir, out_tar)
    if out_bundle:
        outputs["out_tar"] = str(out_bundle)

    # Collect tokenizer assets to ensure downstream finetuning can load from artifact
    tokenizer_dir = workdir / "tokenizer"
    tokenizer_dir.mkdir(exist_ok=True)
    tokenizer_files = []
    for pattern in ["tokenizer.model", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]:
        tokenizer_files.extend(repo_dir.rglob(pattern))
    if tokenizer_files:
        for p in tokenizer_files:
            target = tokenizer_dir / p.name
            target.write_bytes(p.read_bytes())
        outputs["tokenizer_dir"] = str(tokenizer_dir)

    # Upload to W&B as artifact (skip here in smoke; handled in speedrun.sh to keep single run)
    if not args.smoke:
        run = wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            name=args.wandb_name,
            tags=["nanochat", "d20", "modal", "8xH100"],
            config={"repo": NANOCHAT_REPO, "branch": args.repo_branch, "gpus": 8, "model": "d20"},
            reinit=True,
        )

        artifact = wandb.Artifact(args.save_artifact_name, type="model")
        for label, path in outputs.items():
            if label == "tokenizer_dir":
                artifact.add_dir(path, name="tokenizer")
            else:
                artifact.add_file(path, name=Path(path).name if label == "out_tar" else label)
        run.log_artifact(artifact)

        # Save summary
        if report.exists():
            with open(report, "r", encoding="utf-8") as f:
                run.summary["report_md_head"] = f.read()[:2000]
        run.finish()

    # Persist artifacts to volume
    results_dir = Path("/results/nanochat")
    results_dir.mkdir(parents=True, exist_ok=True)
    if out_bundle:
        target = results_dir / out_bundle.name
        target.write_bytes(Path(out_bundle).read_bytes())
    if report.exists():
        target = results_dir / "report.md"
        target.write_bytes(report.read_bytes())

    console.print("[green]Completed. Artifacts pushed to W&B and /results volume.")


@app.local_entrypoint()
def main(
    wandb_name: str = "nanochat-d20-modal",
    repo_branch: str = BRANCH,
    save_artifact_name: str = "nanochat-d20-speedrun",
    smoke: bool = False,
):
    """Kick off a Modal run for nanochat d20."""
    train_d20.remote(Args(wandb_name=wandb_name, repo_branch=repo_branch, save_artifact_name=save_artifact_name, smoke=smoke))
