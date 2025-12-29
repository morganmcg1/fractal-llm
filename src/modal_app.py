"""
Modal app for fractal LLM training experiments.

Runs hyperparameter grid searches on H100 GPUs to visualize
the fractal boundary of neural network trainability.

Uses nanochat-d20 (561M) model from HuggingFace with SmolTalk SFT data.
"""

import modal

app = modal.App("fractal-llm", default_profile="weightsandbiases")

volume = modal.Volume.from_name("fractal-llm-results", create_if_missing=True)

# W&B configuration
WANDB_ENTITY = "morgan"
WANDB_PROJECT = "fractal-llm"

# H100-optimized image with CUDA support (Torch CU124 wheels)
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
    .run_commands(
        "pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1"
    )
    .pip_install(
        "transformers>=4.47.0",
        "datasets>=3.2.0",
        "accelerate>=1.2.0",
        "wandb>=0.23.1",
        "numpy>=2.0.0",
        "pandas>=2.2.0",
        "matplotlib>=3.9.0",
        "seaborn>=0.13.0",
        "scipy>=1.14.0",
        "tqdm>=4.67.0",
        "rich>=14.0.0",
        "huggingface-hub>=0.27.0",
        "lm-eval==0.4.2",
        "packaging",
        "ninja",
    )
    .run_commands("pip install flash-attn --no-build-isolation")
)

# Primary model for experiments (matches research plan)
# Pull from W&B artifact produced by nanochat_modal.py speedrun.
MODEL_ID = "wandb:morgan/fractal-llm/nanochat-d20-speedrun:latest"
DATASET_ID = "HuggingFaceTB/smoltalk"
MAX_SEQ_LEN = 128


@app.function(
    gpu="H100",
    image=image,
    volumes={"/results": volume},
    timeout=300,  # 5 min max per run
)
def train_single_run(
    learning_rate: float,
    num_tokens: int,
    grid_i: int,
    grid_j: int,
    seed: int = 42,
    max_steps: int | None = None,
    batch_size: int = 8,
) -> dict:
    """
    Run a single SFT training and return convergence metrics.

    Args:
        learning_rate: Learning rate for this run
        num_tokens: Approximate number of training tokens
        grid_i: Grid row index (for LR)
        grid_j: Grid column index (for dataset size)
        seed: Random seed for full reproducibility
        max_steps: Maximum training steps
        batch_size: Per-device batch size

    Returns:
        Dict with final_loss, converged, loss_trajectory
    """
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
        TrainerCallback,
    )
    from datasets import load_dataset
    import numpy as np
    import math
    import random
    import os
    from pathlib import Path
    import re
    import time

    # Enforce full reproducibility (following Stas Bekman's ml-engineering guide)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True)

    run_id = f"grid_{grid_i:03d}_{grid_j:03d}"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with bf16 for H100
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # Try to load a prebuilt shard matching the requested token budget (optional)
    shard_dir = Path("/results/smoltalk_shards")
    shard_path: Path | None = None
    if shard_dir.exists():
        shard_candidates = []
        for p in shard_dir.glob("smoltalk_*_tokens.jsonl"):
            m = re.search(r"smoltalk_(\d+)_tokens", p.name)
            if m:
                shard_tokens = int(m.group(1))
                shard_candidates.append((shard_tokens, p))
        shard_candidates = [c for c in shard_candidates if c[0] <= num_tokens]
        if shard_candidates:
            shard_tokens, shard_path = max(shard_candidates, key=lambda x: x[0])

    # Load and prepare SmolTalk dataset with deterministic ordering
    texts: list[str] = []
    total_tokens = 0

    if shard_path is not None:
        dataset = load_dataset("json", data_files=str(shard_path), split="train")
        for sample in dataset:
            text = sample.get("text", "")
            if not text:
                continue
            tokenized = tokenizer(text, truncation=True, max_length=MAX_SEQ_LEN, return_tensors="pt")
            input_ids = tokenized["input_ids"][0]
            text_len = input_ids.shape[0]
            texts.append(text)
            total_tokens += int(text_len)
            if total_tokens >= num_tokens:
                break
    else:
        dataset = load_dataset(DATASET_ID, "all", split="train", streaming=True)
        dataset = dataset.shuffle(seed=seed)

        def format_chat(sample):
            """Format SmolTalk conversation into training text."""
            messages = sample.get("messages", [])
            if not messages:
                return ""

            text_parts = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                text_parts.append(f"<|{role}|>\n{content}")
            text_parts.append("<|end|>")
            return "\n".join(text_parts)

        for sample in dataset:
            text = format_chat(sample)
            if not text:
                continue

            encoded = tokenizer(
                text,
                truncation=True,
                max_length=MAX_SEQ_LEN,
                return_tensors="pt",
            )
            text_len = encoded["input_ids"].shape[1]
            if text_len == 0:
                continue

            texts.append(text)
            total_tokens += int(text_len)
            if total_tokens >= num_tokens:
                break

    if not texts:
        return {
            "run_id": run_id,
            "grid_i": grid_i,
            "grid_j": grid_j,
            "learning_rate": learning_rate,
            "num_tokens": num_tokens,
            "seed": seed,
            "final_loss": float("inf"),
            "converged": False,
            "error": "No valid samples",
        }

    # Tokenize
    def tokenize_texts(texts):
        from torch.utils.data import Dataset

        class TextDataset(Dataset):
            def __init__(self, encodings):
                self.encodings = encodings

            def __len__(self):
                return len(self.encodings["input_ids"])

            def __getitem__(self, idx):
                return {
                    "input_ids": self.encodings["input_ids"][idx],
                    "attention_mask": self.encodings["attention_mask"][idx],
                    "labels": self.encodings["input_ids"][idx].clone(),
                }

        encodings = tokenizer(
            texts,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding="max_length",
            return_tensors="pt",
        )
        return TextDataset(encodings)

    train_dataset = tokenize_texts(texts)

    # Calculate steps based on tokens (respect requested token budget)
    tokens_per_step = batch_size * MAX_SEQ_LEN
    steps_target = max(1, math.ceil(num_tokens / tokens_per_step))
    calculated_steps = steps_target if max_steps is None else max(1, min(steps_target, max_steps))

    # Training arguments optimized for H100
    training_args = TrainingArguments(
        output_dir=f"/tmp/{run_id}",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        max_steps=calculated_steps,
        warmup_steps=max(1, calculated_steps // 10),
        weight_decay=0.1,
        logging_steps=max(1, calculated_steps // 10),
        save_strategy="no",
        report_to="none",
        bf16=True,
        gradient_checkpointing=False,
        dataloader_num_workers=0,
        optim="adamw_torch_fused",
        seed=seed,
        data_seed=seed,
        dataloader_drop_last=True,
    )

    # Data collator for causal LM
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Callback to capture loss trajectory
    loss_trajectory = []

    class LossCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                loss_trajectory.append(logs["loss"])

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[LossCallback()],
    )

    # Train and capture metrics
    final_loss = float("inf")
    converged = False
    error = None

    start_time = time.time()
    try:
        result = trainer.train()
        final_loss = result.training_loss
        converged = (
            not np.isnan(final_loss)
            and not np.isinf(final_loss)
            and final_loss < 10.0  # Threshold for "diverged"
        )
    except Exception as e:
        error = str(e)
    runtime_s = time.time() - start_time

    # Clean up GPU memory
    del model
    del trainer
    torch.cuda.empty_cache()

    return {
        "run_id": run_id,
        "grid_i": grid_i,
        "grid_j": grid_j,
        "learning_rate": learning_rate,
        "num_tokens": num_tokens,
        "tokens_used": total_tokens,
        "seed": seed,
        "final_loss": float(final_loss) if not math.isnan(final_loss) else float("inf"),
        "converged": converged,
        "runtime_s": runtime_s,
        "loss_trajectory": loss_trajectory[-10:] if loss_trajectory else [],  # Last 10 losses
        "error": error,
    }


@app.function(
    image=image,
    volumes={"/results": volume},
    timeout=86400,  # 24 hours for full grid
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def run_grid_search(
    lr_min: float = 1e-6,
    lr_max: float = 1e-3,
    tokens_min: int = 1000,
    tokens_max: int = 1000000,
    resolution: int = 128,
    checkpoint_every: int = 100,
    wandb_run_name: str | None = None,
    seed: int = 42,
    debug: bool = False,
) -> str:
    """
    Orchestrate a full grid search over learning rate and dataset size.
    Uses Modal's starmap for parallel execution.

    Args:
        seed: Global seed for reproducibility (passed to all training runs)
        debug: If True, add 'debug' tag to W&B run

    Returns path to results file.
    """
    import numpy as np
    import json
    from datetime import datetime
    import os
    import wandb

    # Initialize W&B
    config = {
        "lr_min": lr_min,
        "lr_max": lr_max,
        "tokens_min": tokens_min,
        "tokens_max": tokens_max,
        "resolution": resolution,
        "model_id": MODEL_ID,
        "dataset_id": DATASET_ID,
        "max_seq_len": MAX_SEQ_LEN,
        "total_runs": resolution ** 2,
        "seed": seed,
    }

    # Create unique sweep ID with datetime for filtering
    sweep_id = datetime.now().strftime("sweep-%Y%m%d-%H%M")
    run_name = wandb_run_name or f"fractal-grid-{resolution}x{resolution}"

    # Add sweep_id and debug to config for reference
    config["sweep_id"] = sweep_id
    config["debug"] = debug

    # Build tags list
    tags = ["fractal-llm", "grid-search", f"res-{resolution}", sweep_id]
    if debug:
        tags.append("debug")

    wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=run_name,
        config=config,
        tags=tags,
    )

    print(f"Sweep ID: {sweep_id} (use this tag to filter runs in W&B)")
    if debug:
        print("[DEBUG MODE] This is a debug run")

    # Generate grid coordinates
    learning_rates = np.logspace(np.log10(lr_min), np.log10(lr_max), resolution)
    token_counts = np.logspace(np.log10(tokens_min), np.log10(tokens_max), resolution).astype(int)

    # Create all grid points
    grid_points = []
    for i, lr in enumerate(learning_rates):
        for j, tokens in enumerate(token_counts):
            grid_points.append({
                "learning_rate": float(lr),
                "num_tokens": int(tokens),
                "grid_i": i,
                "grid_j": j,
            })

    total_runs = len(grid_points)
    print(f"Starting grid search: {resolution}x{resolution} = {total_runs} runs")
    wandb.log({"status": "started", "total_runs": total_runs})

    # Check for existing checkpoint
    checkpoint_path = f"/results/checkpoint_{resolution}x{resolution}.json"
    completed_runs = {}
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            completed_runs = {r["run_id"]: r for r in json.load(f)}
        print(f"Resuming from checkpoint: {len(completed_runs)} runs completed")
        wandb.log({"resumed_from_checkpoint": len(completed_runs)})

    # Filter out completed runs
    remaining_points = [
        p for p in grid_points
        if f"grid_{p['grid_i']:03d}_{p['grid_j']:03d}" not in completed_runs
    ]

    print(f"Remaining runs: {len(remaining_points)}")

    # Run in batches with starmap
    all_results = list(completed_runs.values())

    for batch_start in range(0, len(remaining_points), checkpoint_every):
        batch = remaining_points[batch_start : batch_start + checkpoint_every]

        # Use starmap for parallel execution
        batch_results = list(train_single_run.starmap(
            [
                (p["learning_rate"], p["num_tokens"], p["grid_i"], p["grid_j"], seed)
                for p in batch
            ]
        ))

        all_results.extend(batch_results)

        # Checkpoint
        with open(checkpoint_path, "w") as f:
            json.dump(all_results, f)
        volume.commit()

        completed = len(all_results)
        progress_pct = 100 * completed / total_runs

        # Log batch metrics to W&B
        batch_converged = sum(1 for r in batch_results if r.get("converged", False))
        batch_losses = [r["final_loss"] for r in batch_results if r.get("converged", False)]
        batch_tokens = [r.get("tokens_used", r.get("num_tokens", 0)) for r in batch_results]
        batch_runtime = [r.get("runtime_s") for r in batch_results if r.get("runtime_s") is not None]

        wandb.log({
            "completed_runs": completed,
            "progress_pct": progress_pct,
            "batch_converged": batch_converged,
            "batch_diverged": len(batch_results) - batch_converged,
            "batch_avg_loss": np.mean(batch_losses) if batch_losses else None,
            "batch_avg_tokens": np.mean(batch_tokens) if batch_tokens else None,
            "batch_avg_runtime_s": np.mean(batch_runtime) if batch_runtime else None,
        })

        print(f"Progress: {completed}/{total_runs} ({progress_pct:.1f}%)")

    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f"/results/grid_{resolution}x{resolution}_{timestamp}.json"

    final_data = {
        "config": config,
        "results": all_results,
    }

    with open(results_path, "w") as f:
        json.dump(final_data, f, indent=2)
    volume.commit()

    # Build fractal visualization following Sohl-Dickstein et al. approach:
    # - Blue = converged, intensity ∝ cumulative loss (higher = darker blue)
    # - Red = diverged, intensity ∝ time before divergence (1/final_loss)
    import matplotlib.pyplot as plt
    import matplotlib
    from matplotlib.colors import LinearSegmentedColormap

    matplotlib.use("Agg")

    # Create intensity grid: positive = converged (blue), negative = diverged (red)
    # Following paper: converged intensity ∝ Σ loss_t, diverged intensity ∝ Σ 1/loss_t
    fractal_grid = np.zeros((resolution, resolution))
    loss_grid = np.full((resolution, resolution), np.nan)

    converged_losses = []
    diverged_inv_losses = []

    for r in all_results:
        i, j = r.get("grid_i", 0), r.get("grid_j", 0)
        final_loss = r.get("final_loss", float("inf"))

        if r.get("converged", False) and final_loss < float("inf"):
            loss_grid[i, j] = final_loss
            converged_losses.append(final_loss)
        elif final_loss > 0 and final_loss < float("inf"):
            # For diverged, use inverse loss as intensity proxy
            diverged_inv_losses.append(1.0 / final_loss)

    # Normalize intensities
    if converged_losses:
        loss_min, loss_max = min(converged_losses), max(converged_losses)
        loss_range = loss_max - loss_min if loss_max > loss_min else 1.0
    else:
        loss_min, loss_max, loss_range = 0, 1, 1.0

    for r in all_results:
        i, j = r.get("grid_i", 0), r.get("grid_j", 0)
        final_loss = r.get("final_loss", float("inf"))

        if r.get("converged", False) and final_loss < float("inf"):
            # Converged: positive values (will be blue)
            # Higher loss = more intense blue (darker)
            intensity = (final_loss - loss_min) / loss_range
            fractal_grid[i, j] = 0.3 + 0.7 * intensity  # Range [0.3, 1.0]
        else:
            # Diverged: negative values (will be red)
            fractal_grid[i, j] = -1.0

    # Create custom blue-red diverging colormap (like Sohl-Dickstein)
    # Red for negative (diverged), Blue for positive (converged)
    colors_diverged = ["#8B0000", "#CD5C5C", "#FA8072"]  # Dark red to light red
    colors_converged = ["#ADD8E6", "#4169E1", "#00008B"]  # Light blue to dark blue
    colors = colors_diverged + ["#FFFFFF"] + colors_converged
    positions = [0.0, 0.15, 0.35, 0.5, 0.65, 0.85, 1.0]
    fractal_cmap = LinearSegmentedColormap.from_list("fractal", list(zip(positions, colors)))

    # Create visualizations
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: Fractal-style convergence boundary (main visualization)
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

    # Plot 2: Loss heatmap (converged runs only)
    im2 = axes[1].imshow(
        loss_grid,
        origin="lower",
        aspect="auto",
        cmap="viridis_r",  # Reversed so lower loss = brighter
        extent=[np.log10(tokens_min), np.log10(tokens_max), np.log10(lr_min), np.log10(lr_max)],
    )
    axes[1].set_xlabel("log₁₀(tokens)")
    axes[1].set_ylabel("log₁₀(learning rate)")
    axes[1].set_title("Final Loss\n(converged runs only)")
    cbar2 = plt.colorbar(im2, ax=axes[1])
    cbar2.set_label("Loss")

    # Plot 3: Binary convergence mask
    convergence_grid = np.zeros((resolution, resolution))
    for r in all_results:
        i, j = r.get("grid_i", 0), r.get("grid_j", 0)
        convergence_grid[i, j] = 1 if r.get("converged", False) else 0

    im3 = axes[2].imshow(
        convergence_grid,
        origin="lower",
        aspect="auto",
        cmap="RdBu",  # Red=0 (diverged), Blue=1 (converged)
        vmin=0,
        vmax=1,
        extent=[np.log10(tokens_min), np.log10(tokens_max), np.log10(lr_min), np.log10(lr_max)],
    )
    axes[2].set_xlabel("log₁₀(tokens)")
    axes[2].set_ylabel("log₁₀(learning rate)")
    axes[2].set_title("Binary Convergence\n(for fractal dimension)")
    cbar3 = plt.colorbar(im3, ax=axes[2])
    cbar3.set_ticks([0, 1])
    cbar3.set_ticklabels(["Diverged", "Converged"])

    plt.tight_layout()
    fig_path = f"/results/grid_{resolution}x{resolution}_{timestamp}.png"
    plt.savefig(fig_path, dpi=150, facecolor="white")
    plt.close()

    # Log to W&B
    total_converged = int(convergence_grid.sum())
    converged_losses = [r["final_loss"] for r in all_results if r.get("converged", False)]

    wandb.log({
        "convergence_grid": wandb.Image(fig_path, caption="LR × Tokens Grid"),
        "total_converged": total_converged,
        "total_diverged": total_runs - total_converged,
        "convergence_rate": total_converged / total_runs,
        "mean_converged_loss": np.mean(converged_losses) if converged_losses else None,
        "min_converged_loss": np.min(converged_losses) if converged_losses else None,
    })

    # Log results table
    results_table = wandb.Table(
        columns=["grid_i", "grid_j", "learning_rate", "num_tokens", "tokens_used", "final_loss", "converged", "runtime_s", "error"],
        data=[
            [
                r.get("grid_i"),
                r.get("grid_j"),
                r.get("learning_rate"),
                r.get("num_tokens"),
                r.get("tokens_used"),
                r.get("final_loss"),
                r.get("converged"),
                r.get("runtime_s"),
                r.get("error"),
            ]
            for r in all_results
        ],
    )
    wandb.log({"results_table": results_table})

    wandb.finish()

    print(f"Results saved to {results_path}")
    return results_path


@app.function(
    image=image,
    volumes={"/results": volume},
    timeout=3600,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def compute_fractal_dimension(results_path: str, wandb_run_id: str | None = None) -> dict:
    """
    Compute the box-counting fractal dimension of the convergence boundary.
    Logs results to the existing W&B run if run_id provided, otherwise creates new run.
    """
    import numpy as np
    import json
    import wandb
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use("Agg")

    with open(results_path, "r") as f:
        data = json.load(f)

    results = data["results"]
    config = data["config"]
    resolution = config["resolution"]

    # Initialize W&B (resume if run_id provided)
    if wandb_run_id:
        wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            id=wandb_run_id,
            resume="must",
        )
    else:
        wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            name=f"fractal-analysis-{resolution}x{resolution}",
            config=config,
            tags=["fractal-llm", "fractal-analysis"],
        )

    # Build convergence grid
    grid = np.zeros((resolution, resolution), dtype=bool)
    for r in results:
        if "grid_i" in r and "grid_j" in r:
            grid[r["grid_i"], r["grid_j"]] = r["converged"]

    # Find boundary (edge between converged and diverged)
    from scipy import ndimage
    boundary = ndimage.binary_dilation(grid) ^ ndimage.binary_erosion(grid)

    # Box counting
    def box_count(binary_image, box_size):
        """Count boxes containing the boundary."""
        h, w = binary_image.shape
        count = 0
        for i in range(0, h, box_size):
            for j in range(0, w, box_size):
                box = binary_image[i:i+box_size, j:j+box_size]
                if box.any():
                    count += 1
        return count

    # Compute for multiple box sizes
    sizes = [2, 4, 8, 16, 32]
    sizes = [s for s in sizes if s < resolution]
    counts = [box_count(boundary, s) for s in sizes]

    # Fit line on log-log plot
    log_sizes = np.log(sizes)
    log_counts = np.log(np.array(counts) + 1)  # +1 to avoid log(0)
    coeffs = np.polyfit(log_sizes, log_counts, 1)
    fractal_dimension = -coeffs[0]

    # Create box-counting plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Boundary visualization
    axes[0].imshow(boundary, origin="lower", cmap="binary")
    axes[0].set_title(f"Convergence Boundary ({boundary.sum()} pixels)")
    axes[0].set_xlabel("Token dimension (j)")
    axes[0].set_ylabel("LR dimension (i)")

    # Plot 2: Log-log plot for fractal dimension
    axes[1].scatter(log_sizes, log_counts, s=100, zorder=5)
    fit_line = coeffs[0] * log_sizes + coeffs[1]
    axes[1].plot(log_sizes, fit_line, "r--", label=f"D = {fractal_dimension:.3f}")
    axes[1].set_xlabel("log(box size)")
    axes[1].set_ylabel("log(box count)")
    axes[1].set_title(f"Box-Counting Fractal Dimension: {fractal_dimension:.3f}")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fractal_fig_path = results_path.replace(".json", "_fractal.png")
    plt.savefig(fractal_fig_path, dpi=150)
    plt.close()

    # Save analysis
    analysis = {
        "fractal_dimension": float(fractal_dimension),
        "box_sizes": sizes,
        "box_counts": [int(c) for c in counts],
        "boundary_pixels": int(boundary.sum()),
        "converged_ratio": float(grid.sum() / grid.size),
    }

    analysis_path = results_path.replace(".json", "_fractal.json")
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    volume.commit()

    # Log to W&B
    wandb.log({
        "fractal_dimension": fractal_dimension,
        "boundary_pixels": int(boundary.sum()),
        "fractal_plot": wandb.Image(fractal_fig_path, caption="Fractal Analysis"),
    })

    # Log box counting data as table
    box_table = wandb.Table(
        columns=["box_size", "box_count", "log_size", "log_count"],
        data=[[s, c, np.log(s), np.log(c + 1)] for s, c in zip(sizes, counts)],
    )
    wandb.log({"box_counting_table": box_table})

    wandb.finish()

    return analysis


@app.local_entrypoint()
def main(
    resolution: int = 32,
    lr_min: float = 1e-6,
    lr_max: float = 1e-3,
    tokens_min: int = 1000,
    tokens_max: int = 1000000,
    test_only: bool = False,
    benchmark: bool = False,
    debug: bool = False,
):
    """
    Local entrypoint for running fractal grid search.

    Args:
        resolution: Grid resolution (32, 64, or 128)
        lr_min: Minimum learning rate
        lr_max: Maximum learning rate
        tokens_min: Minimum training tokens
        tokens_max: Maximum training tokens
        test_only: If True, just run a single test
        benchmark: If True, run 6 parallel tests to estimate timing
        debug: If True, add 'debug' tag to W&B runs
    """
    from rich.console import Console
    from rich.panel import Panel
    import time

    console = Console()
    console.rule("[bold blue]Fractal LLM Training")

    if benchmark:
        console.print(Panel("Running 6 parallel training runs to estimate timing (including 1M tokens)...", title="Benchmark"))
        # Test across the full range: 1K, 10K, 100K, 1M tokens
        test_runs = [
            (1e-4, 1000, 0, 0),     # Min tokens
            (1e-4, 10000, 0, 1),    # 10K tokens
            (1e-5, 10000, 0, 2),    # Low LR
            (1e-3, 10000, 0, 3),    # High LR
            (1e-4, 100000, 1, 0),   # 100K tokens
            (1e-4, 1000000, 2, 0),  # Max tokens (1M)
        ]
        start = time.time()
        results = list(train_single_run.starmap(test_runs))
        elapsed = time.time() - start

        console.print("\n[bold]Results:[/bold]")
        for r in results:
            status = "[green]✓[/green]" if r["converged"] else "[red]✗[/red]"
            console.print(f"  {status} LR={r['learning_rate']:.0e}, tokens={r['num_tokens']:,}: loss={r['final_loss']:.3f}")

        console.rule("[bold]Timing Estimates")
        avg_per_run = elapsed / len(test_runs)
        console.print(f"{len(test_runs)} runs completed in [cyan]{elapsed:.1f}s[/cyan]")
        console.print(f"Average per run (parallel): [cyan]{avg_per_run:.1f}s[/cyan]")
        console.print(f"\n[bold]Full grid search estimates (sequential):[/bold]")
        console.print(f"  32×32   (1,024 runs):  [yellow]{1024 * avg_per_run / 60:.1f} min[/yellow]")
        console.print(f"  64×64   (4,096 runs):  [yellow]{4096 * avg_per_run / 60:.1f} min[/yellow]")
        console.print(f"  128×128 (16,384 runs): [yellow]{16384 * avg_per_run / 60:.1f} min[/yellow]")
        return

    if test_only:
        console.print(Panel("Running single test training (seed=42)...", title="Test"))
        result = train_single_run.remote(
            learning_rate=1e-4,
            num_tokens=10000,
            grid_i=0,
            grid_j=0,
            seed=42,
            max_steps=10,
        )
        console.print(f"Result: {result}")
        return

    debug_str = " [DEBUG]" if debug else ""
    console.print(Panel(
        f"Resolution: {resolution}x{resolution} ({resolution**2} runs)\n"
        f"LR range: {lr_min:.0e} to {lr_max:.0e}\n"
        f"Tokens range: {tokens_min:,} to {tokens_max:,}\n"
        f"Debug mode: {debug}",
        title=f"Grid Search Configuration{debug_str}",
    ))

    results_path = run_grid_search.remote(
        lr_min=lr_min,
        lr_max=lr_max,
        tokens_min=tokens_min,
        tokens_max=tokens_max,
        resolution=resolution,
        debug=debug,
    )

    console.print(f"[green]Results saved to: {results_path}")

    # Compute fractal dimension
    console.print("[yellow]Computing fractal dimension...")
    analysis = compute_fractal_dimension.remote(results_path)
    console.print(f"[green]Fractal dimension: {analysis['fractal_dimension']:.3f}")
    console.print(f"[green]Converged ratio: {analysis['converged_ratio']:.1%}")
