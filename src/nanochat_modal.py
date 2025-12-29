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

WANDB_ENTITY = "morgan"
WANDB_PROJECT = "fractal-llm"
NANOCHAT_REPO = "https://github.com/karpathy/nanochat.git"
BRANCH = "main"


# Modal image: CUDA 12.4 + Torch cu124 + flash-attn + git + uv
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "curl")
    .run_commands(
        "pip install --index-url https://download.pytorch.org/whl/cu124 "
        "torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 "
        "&& pip install flash-attn==2.6.3 --no-build-isolation "
        "&& pip install uv>=0.4.0 wandb>=0.19.0"
    )
)


app = modal.App("nanochat-modal")


@dataclass
class Args:
    """Modal entrypoint arguments."""

    wandb_name: str = "nanochat-d20-modal"
    repo_branch: str = BRANCH
    save_artifact_name: str = "nanochat-d20-speedrun"


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
    import wandb
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    console.rule("[bold blue]nanochat d20 on Modal (8×H100)")

    os.environ["WANDB_ENTITY"] = WANDB_ENTITY
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT
    os.environ["WANDB_NAME"] = args.wandb_name
    os.environ["WANDB_MODE"] = "online"

    workdir = Path("/workspace")
    repo_dir = workdir / "nanochat"

    # Clone repo
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", args.repo_branch, NANOCHAT_REPO, str(repo_dir)],
        check=True,
    )

    # Install dependencies via uv (respect upstream lock)
    subprocess.run(["uv", "pip", "install", "-r", "uv.lock"], cwd=repo_dir, check=True)

    # Run speedrun (d20) with W&B env set
    console.print(Panel("Starting speedrun.sh (d20, 8×H100)", title="Train"))
    subprocess.run(["bash", "speedrun.sh"], cwd=repo_dir, check=True)

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

    # Upload to W&B as artifact
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
):
    """Kick off a Modal run for nanochat d20."""
    train_d20.remote(Args(wandb_name=wandb_name, repo_branch=repo_branch, save_artifact_name=save_artifact_name))
