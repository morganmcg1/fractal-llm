"""
Fine-tune nanochat-d20 on CUAD-QA legal contract dataset.

Minimalistic nanochat-style implementation with explicit training loop,
8×H100 DDP support, SFT loss masking, and cosine decay LR schedule.

Usage:
    # Test run (small sample)
    uv run modal run src/finetune_cuad.py --test-only

    # Full training
    uv run modal run src/finetune_cuad.py --epochs 3 --lr 2e-5

    # With evaluation
    uv run modal run src/finetune_cuad.py --epochs 3 --eval-samples 500
"""

import math
import modal

# ============================================================================
# Configuration Globals (nanochat style)
# Override via Modal local_entrypoint args
# ============================================================================

# Training
lr = 2e-5
warmup_frac = 0.1
min_lr = 1e-6
weight_decay = 0.01
grad_clip = 1.0
num_epochs = 3
batch_size = 4
grad_accum_steps = 4

# Data
max_seq_len = 1024
max_context_chars = 3000
max_samples = None

# Model
model_artifact = "wandb:morgan/fractal-llm/nanochat-d20-speedrun:latest"

# Eval
eval_samples = None
eval_every = 500

# Logging
log_every = 10
save_every = 1000

# W&B
wandb_entity = "morgan"
wandb_project = "fractal-llm"

# Reproducibility
seed = 42

# ============================================================================
# Modal Setup
# ============================================================================

app = modal.App("cuad-finetune-ddp", default_profile="weightsandbiases")


# ============================================================================
# Dummy W&B (nanochat pattern for non-logging runs)
# ============================================================================


class DummyWandb:
    """Dummy wandb object for non-logging runs (non-master process or disabled)."""

    def log(self, *args, **kwargs):
        pass

    def finish(self, *args, **kwargs):
        pass


volume = modal.Volume.from_name("fractal-llm-results", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .run_commands(
        "pip install uv>=0.4.0 "
        "&& uv pip install --system --index-url https://download.pytorch.org/whl/cu128 "
        "torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 "
        "&& uv pip install --system "
        "transformers>=4.47.0 "
        "datasets>=3.2.0 "
        "wandb>=0.23.1 "
        "python-dotenv>=1.0.1 "
        "numpy>=2.0.0 "
        "tqdm>=4.67.0 "
        "rich>=14.0.0 "
        "huggingface-hub>=0.27.0"
    )
)


# ============================================================================
# Determinism
# ============================================================================


def set_seed(seed_value: int):
    """Set all seeds for full reproducibility."""
    import os
    import random
    import numpy as np
    import torch

    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True)


# ============================================================================
# DDP Initialization
# ============================================================================


def compute_init():
    """Initialize DDP and return (ddp, rank, local_rank, world_size, device)."""
    import os
    import torch
    import torch.distributed as dist

    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        device = "cuda" if torch.cuda.is_available() else "cpu"

    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, device


# ============================================================================
# Model Loading from W&B Artifact
# ============================================================================


def resolve_model_path(model_id: str) -> str:
    """
    Resolve model identifier to local path.

    Supports:
    - W&B artifact paths: "wandb:<entity>/<project>/<name>:<version>"
    - HuggingFace Hub paths: "org/model-name"
    - Local paths: "/path/to/model"
    """
    from pathlib import Path

    if model_id.startswith("wandb:"):
        import wandb

        artifact_path = model_id[6:]
        api = wandb.Api()
        artifact = api.artifact(artifact_path, type="model")

        cache_root = Path("/results/model_cache")
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_dir = cache_root / artifact.name.replace(":", "_")
        if not cache_dir.exists():
            artifact.download(root=str(cache_dir))

        artifact_path = Path(cache_dir)

        # Extract tar.gz if present
        for tar_file in artifact_path.glob("*.tar.gz"):
            import tarfile

            extract_dir = artifact_path / "extracted"
            if not extract_dir.exists():
                with tarfile.open(tar_file, "r:gz") as tar:
                    tar.extractall(extract_dir)

            for subdir in extract_dir.rglob("config.json"):
                return str(subdir.parent)

        for config in artifact_path.rglob("config.json"):
            return str(config.parent)

        return str(cache_dir)

    return model_id


# ============================================================================
# Loss Masking for SFT
# ============================================================================


def create_labels_with_mask(input_ids, assistant_token_id: int, end_token_id: int):
    """
    Create labels tensor with -100 for non-assistant tokens.
    Only supervise tokens after <|assistant|> until <|end|>.
    """
    import torch

    labels = input_ids.clone()
    batch_size, seq_len = input_ids.shape

    for i in range(batch_size):
        in_assistant = False
        for j in range(seq_len):
            token_id = input_ids[i, j].item()

            if token_id == assistant_token_id:
                labels[i, j] = -100
                in_assistant = True
            elif token_id == end_token_id:
                in_assistant = False
            elif not in_assistant:
                labels[i, j] = -100

    return labels


# ============================================================================
# Data Generator
# ============================================================================


def cuad_data_generator(
    split: str,
    tokenizer,
    assistant_token_id: int,
    end_token_id: int,
    max_context: int,
    max_len: int,
    max_samp: int | None,
    data_seed: int,
    rank: int,
    world_size: int,
):
    """
    Infinite generator yielding tokenized CUAD samples with loss masking.
    Yields: (input_ids, labels, attention_mask) as tensors.
    Loops forever through epochs (nanochat style).
    """
    from datasets import load_dataset

    dataset = load_dataset(
        "theatticusproject/cuad-qa",
        revision="refs/pr/6",
        split=split,
    )

    if max_samp:
        dataset = dataset.select(range(min(max_samp, len(dataset))))

    dataset = dataset.shard(num_shards=world_size, index=rank)

    epoch = 0
    while True:  # Infinite loop over epochs (nanochat style)
        shuffled = dataset.shuffle(seed=data_seed + epoch)
        for sample in shuffled:
            context = sample["context"]
            question = sample["question"]
            answers = sample.get("answers", {})
            answer_texts = answers.get("text", [])

            if len(context) > max_context:
                context = context[:max_context] + "..."

            if not answer_texts or all(not a.strip() for a in answer_texts):
                answer = "No relevant clause found."
            else:
                answer = answer_texts[0].strip() or "No relevant clause found."

            text = (
                f"<|system|>\n"
                f"You are a legal document analyst. Answer questions about contracts by extracting relevant text spans.\n"
                f"<|user|>\n"
                f"Contract:\n{context}\n\n"
                f"Question: {question}\n"
                f"<|assistant|>\n"
                f"{answer}\n"
                f"<|end|>"
            )

            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_len,
                padding="max_length",
                return_tensors="pt",
            )

            input_ids = encoded["input_ids"].squeeze(0)
            attention_mask = encoded["attention_mask"].squeeze(0)
            labels = create_labels_with_mask(
                input_ids.unsqueeze(0), assistant_token_id, end_token_id
            ).squeeze(0)

            yield input_ids, labels, attention_mask
        epoch += 1


# ============================================================================
# Evaluation
# ============================================================================


def evaluate_model(
    model,
    tokenizer,
    max_samples_eval: int,
    max_context: int,
    max_new_tokens: int = 256,
) -> dict:
    """Evaluate model on CUAD test set with EM/F1 metrics."""
    import torch
    import re
    import string
    from collections import Counter
    from datasets import load_dataset
    from tqdm import tqdm

    def normalize_answer(s: str) -> str:
        s = s.lower()
        s = "".join(ch for ch in s if ch not in string.punctuation)
        s = re.sub(r"\b(a|an|the)\b", " ", s)
        s = " ".join(s.split())
        return s

    def compute_em(pred: str, gts: list[str]) -> float:
        pred_norm = normalize_answer(pred)
        for gt in gts:
            if normalize_answer(gt) == pred_norm:
                return 1.0
        return 0.0

    def compute_f1(pred: str, gts: list[str]) -> float:
        pred_tokens = normalize_answer(pred).split()
        max_f1 = 0.0
        for gt in gts:
            gt_tokens = normalize_answer(gt).split()
            if not pred_tokens and not gt_tokens:
                max_f1 = max(max_f1, 1.0)
                continue
            if not pred_tokens or not gt_tokens:
                continue
            common = Counter(pred_tokens) & Counter(gt_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                continue
            precision = num_same / len(pred_tokens)
            recall = num_same / len(gt_tokens)
            f1 = 2 * precision * recall / (precision + recall)
            max_f1 = max(max_f1, f1)
        return max_f1

    test_dataset = load_dataset(
        "theatticusproject/cuad-qa",
        revision="refs/pr/6",
        split="test",
    )
    test_dataset = test_dataset.select(range(min(max_samples_eval, len(test_dataset))))

    raw_model = model.module if hasattr(model, "module") else model
    raw_model.eval()

    em_total, f1_total = 0.0, 0.0

    for sample in tqdm(test_dataset, desc="Evaluating", disable=False):
        context = sample["context"]
        if len(context) > max_context:
            context = context[:max_context] + "..."

        question = sample["question"]
        ground_truths = sample["answers"]["text"]
        if not ground_truths or all(not gt.strip() for gt in ground_truths):
            ground_truths = ["No relevant clause found."]

        prompt = (
            f"<|system|>\n"
            f"You are a legal document analyst. Answer questions about contracts by extracting relevant text spans.\n"
            f"<|user|>\n"
            f"Contract:\n{context}\n\n"
            f"Question: {question}\n"
            f"<|assistant|>\n"
        )

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(raw_model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = raw_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "<|assistant|>" in full_output:
            prediction = full_output.split("<|assistant|>")[-1].strip()
        else:
            prediction = full_output[len(prompt):].strip()

        if "<|end|>" in prediction:
            prediction = prediction.split("<|end|>")[0].strip()

        em_total += compute_em(prediction, ground_truths)
        f1_total += compute_f1(prediction, ground_truths)

    n = len(test_dataset)
    raw_model.train()

    return {
        "exact_match": em_total / n if n > 0 else 0,
        "f1": f1_total / n if n > 0 else 0,
        "eval_samples": n,
    }


# ============================================================================
# Training Function
# ============================================================================


def _train_worker():
    """
    Worker function called by torchrun for DDP training.
    Reads config from environment variables set by the launcher.
    Nanochat-style: step-based while True loop, LR multiplier pattern, token tracking.
    """
    import os
    import json
    import math
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    import wandb
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datetime import datetime
    from rich.console import Console
    from dotenv import load_dotenv

    console = Console()

    for candidate in ("/workspace/.env", "/root/.env", ".env"):
        if os.path.exists(candidate):
            load_dotenv(candidate, override=False)

    # Read config from env (set by launcher)
    config = json.loads(os.environ.get("TRAIN_CONFIG", "{}"))
    lr_val = config.get("lr", lr)
    epochs_val = config.get("epochs", num_epochs)
    batch_size_val = config.get("batch_size", batch_size)
    max_samples_val = config.get("max_samples", max_samples)
    eval_samples_val = config.get("eval_samples", eval_samples)
    wandb_run_name = config.get("wandb_run_name")

    # DDP init
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
    master_process = ddp_rank == 0

    # Set seed with rank offset for different data per rank
    set_seed(seed + ddp_rank)

    if master_process:
        console.rule("[bold blue]CUAD Fine-tuning (nanochat style, 8×H100)")

    # Load model
    model_path = resolve_model_path(model_artifact)
    if master_process:
        console.print(f"Loading model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = model.to(device)

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # Get special token IDs
    assistant_token_id = tokenizer.convert_tokens_to_ids("<|assistant|>")
    end_token_id = tokenizer.convert_tokens_to_ids("<|end|>")

    # Optimizer with initial_lr tracking (nanochat pattern)
    raw_model = model.module if ddp else model
    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=lr_val,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
    )
    # Store initial_lr for multiplier pattern
    for param_group in optimizer.param_groups:
        param_group["initial_lr"] = param_group["lr"]

    # Calculate total steps (num_iterations in nanochat terminology)
    from datasets import load_dataset as ld_count

    full_dataset = ld_count("theatticusproject/cuad-qa", revision="refs/pr/6", split="train")
    dataset_size = len(full_dataset)
    if max_samples_val:
        dataset_size = min(max_samples_val, dataset_size)
    del full_dataset

    samples_per_rank = dataset_size // ddp_world_size
    steps_per_epoch = samples_per_rank // (batch_size_val * grad_accum_steps)
    num_iterations = steps_per_epoch * epochs_val

    if master_process:
        console.print(f"Dataset size: {dataset_size}")
        console.print(f"Samples per rank: {samples_per_rank}")
        console.print(f"Steps per epoch: {steps_per_epoch}")
        console.print(f"Total iterations: {num_iterations}")
        console.print(f"LR: {lr_val}, warmup: {warmup_frac}, min_lr: {min_lr}")

    # LR scheduler - returns multiplier (nanochat pattern)
    min_lr_frac = min_lr / lr_val

    def get_lr_multiplier(it: int) -> float:
        warmup_iters = int(warmup_frac * num_iterations)
        if it < warmup_iters:
            return (it + 1) / warmup_iters
        progress = (it - warmup_iters) / max(1, num_iterations - warmup_iters)
        return min_lr_frac + 0.5 * (1.0 - min_lr_frac) * (1 + math.cos(math.pi * progress))

    # W&B init with DummyWandb pattern (nanochat style)
    run_name = wandb_run_name or f"cuad-sft-{datetime.now().strftime('%Y%m%d-%H%M')}"
    use_dummy_wandb = run_name == "dummy" or not master_process
    wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(
        entity=wandb_entity,
        project=wandb_project,
        name=run_name,
        config={
            "model_artifact": model_artifact,
            "lr": lr_val,
            "warmup_frac": warmup_frac,
            "min_lr": min_lr,
            "num_epochs": epochs_val,
            "batch_size": batch_size_val,
            "grad_accum_steps": grad_accum_steps,
            "max_seq_len": max_seq_len,
            "max_context_chars": max_context_chars,
            "weight_decay": weight_decay,
            "grad_clip": grad_clip,
            "ddp_world_size": ddp_world_size,
            "num_iterations": num_iterations,
            "dataset_size": dataset_size,
            "seed": seed,
        },
        tags=["cuad", "sft", "nanochat-style", "8xH100"],
    )

    # Create infinite data loader (nanochat pattern)
    train_loader = cuad_data_generator(
        split="train",
        tokenizer=tokenizer,
        assistant_token_id=assistant_token_id,
        end_token_id=end_token_id,
        max_context=max_context_chars,
        max_len=max_seq_len,
        max_samp=max_samples_val,
        data_seed=seed,
        rank=ddp_rank,
        world_size=ddp_world_size,
    )

    # -------------------------------------------------------------------------
    # Training loop (nanochat style: step-based while True)
    # -------------------------------------------------------------------------
    model.train()
    step = 0
    total_tokens_trained = 0  # Track supervised tokens (where labels != -100)
    train_loss = 0.0
    val_loss = float("inf")
    final_eval = {}

    # Batch accumulation buffers
    batch_input_ids = []
    batch_labels = []
    batch_attention_mask = []

    while True:
        last_step = step == num_iterations

        # Eval at start of loop (nanochat pattern - ensures final eval)
        if eval_samples_val and (last_step or (step > 0 and step % eval_every == 0)):
            model.eval()

            # Compute validation loss (nanochat pattern: forward pass on val data)
            val_loader = cuad_data_generator(
                split="test",
                tokenizer=tokenizer,
                assistant_token_id=assistant_token_id,
                end_token_id=end_token_id,
                max_context=max_context_chars,
                max_len=max_seq_len,
                max_samp=eval_samples_val,
                data_seed=seed,
                rank=ddp_rank,
                world_size=ddp_world_size,
            )
            val_losses = []
            val_tokens = 0
            eval_batches = eval_samples_val // (batch_size_val * ddp_world_size)
            val_batch = []
            for _ in range(eval_batches):
                while len(val_batch) < batch_size_val:
                    val_batch.append(next(val_loader))
                vx = torch.stack([b[0] for b in val_batch[:batch_size_val]]).to(device)
                vy = torch.stack([b[1] for b in val_batch[:batch_size_val]]).to(device)
                va = torch.stack([b[2] for b in val_batch[:batch_size_val]]).to(device)
                val_batch = val_batch[batch_size_val:]
                with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(input_ids=vx, attention_mask=va, labels=vy)
                val_losses.append(outputs.loss)
                val_tokens += (vy != -100).sum().item()

            if val_losses:
                val_loss = torch.stack(val_losses).mean()
                if ddp:
                    dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
                val_loss_item = val_loss.item()
            else:
                val_loss_item = float("inf")

            # Log val loss (all ranks computed, master logs)
            if master_process:
                wandb_run.log({
                    "val/loss": val_loss_item,
                    "val/step": step,
                })
                console.print(f"step {step:05d} | val loss: {val_loss_item:.4f}")

            # Compute EM/F1 metrics (master only, slower)
            if master_process:
                eval_results = evaluate_model(
                    model, tokenizer, eval_samples_val, max_context_chars
                )
                wandb_run.log({
                    "eval/exact_match": eval_results["exact_match"],
                    "eval/f1": eval_results["f1"],
                    "eval/step": step,
                })
                console.print(
                    f"step {step:05d} | eval EM: {eval_results['exact_match']:.2%} | "
                    f"F1: {eval_results['f1']:.2%}"
                )
                if last_step:
                    final_eval = eval_results
                    final_eval["val_loss"] = val_loss_item

            model.train()

        # Save checkpoint
        if master_process and step > 0 and step % save_every == 0:
            save_dir = f"/results/cuad_checkpoint_step_{step}"
            raw_model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            console.print(f"step {step:05d} | [green]checkpoint saved: {save_dir}")

        # Termination
        if last_step:
            break

        # ---------------------------------------------------------------------
        # Single training step with gradient accumulation
        # ---------------------------------------------------------------------
        tokens_this_step = torch.tensor(0, device=device, dtype=torch.long)

        for micro_step in range(grad_accum_steps):
            # Accumulate batch
            while len(batch_input_ids) < batch_size_val:
                input_ids, labels, attention_mask = next(train_loader)
                batch_input_ids.append(input_ids)
                batch_labels.append(labels)
                batch_attention_mask.append(attention_mask)

            # Stack batch
            x = torch.stack(batch_input_ids[:batch_size_val]).to(device)
            y = torch.stack(batch_labels[:batch_size_val]).to(device)
            attn = torch.stack(batch_attention_mask[:batch_size_val]).to(device)

            # Clear used samples
            batch_input_ids = batch_input_ids[batch_size_val:]
            batch_labels = batch_labels[batch_size_val:]
            batch_attention_mask = batch_attention_mask[batch_size_val:]

            # Forward + backward
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(input_ids=x, attention_mask=attn, labels=y)
                loss = outputs.loss / grad_accum_steps

            train_loss = loss.detach() * grad_accum_steps  # For logging (unscaled)
            loss.backward()

            # Count supervised tokens (where labels != -100)
            tokens_this_step += (y != -100).sum()

        # Aggregate tokens across ranks
        if ddp:
            dist.all_reduce(tokens_this_step, op=dist.ReduceOp.SUM)
        total_tokens_trained += tokens_this_step.item()

        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), grad_clip)

        # LR schedule (multiplier pattern)
        lrm = get_lr_multiplier(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["initial_lr"] * lrm

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Logging
        train_loss_item = train_loss.item()
        current_lr = optimizer.param_groups[0]["lr"]

        if master_process and step % log_every == 0:
            wandb_run.log({
                "train/loss": train_loss_item,
                "train/lr": current_lr,
                "train/lrm": lrm,
                "train/step": step,
                "train/tokens": total_tokens_trained,
            })
            console.print(
                f"step {step:05d}/{num_iterations:05d} | "
                f"loss: {train_loss_item:.4f} | lrm: {lrm:.4f} | "
                f"tokens: {total_tokens_trained:,}"
            )

        step += 1

    # -------------------------------------------------------------------------
    # Final save and cleanup
    # -------------------------------------------------------------------------
    output_dir = f"/results/cuad_sft_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if master_process:
        raw_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        console.print(f"\n[green]Model saved to: {output_dir}")
        console.print(f"Total tokens trained: {total_tokens_trained:,}")

        if final_eval:
            wandb_run.log({
                "final/exact_match": final_eval["exact_match"],
                "final/f1": final_eval["f1"],
                "final/val_loss": final_eval.get("val_loss", float("inf")),
                "final/tokens": total_tokens_trained,
            })
            console.print(f"[bold green]Final val loss: {final_eval.get('val_loss', 'N/A'):.4f}")
            console.print(f"[bold green]Final EM: {final_eval['exact_match']:.2%}")
            console.print(f"[bold green]Final F1: {final_eval['f1']:.2%}")

        # Write results to file for launcher to read
        results = {
            "final_loss": train_loss_item,
            "model_path": output_dir,
            "total_steps": step,
            "total_tokens_trained": total_tokens_trained,
            **final_eval,
        }
        with open("/results/train_results.json", "w") as f:
            json.dump(results, f)

    # Cleanup (nanochat pattern)
    wandb_run.finish()
    if ddp:
        dist.destroy_process_group()


@app.function(
    gpu="H100:8",
    image=image,
    volumes={"/results": volume},
    timeout=60 * 60 * 4,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train_cuad_ddp(
    lr_override: float | None = None,
    epochs_override: int | None = None,
    batch_size_override: int | None = None,
    max_samples_override: int | None = None,
    eval_samples_override: int | None = None,
    wandb_run_name: str | None = None,
) -> dict:
    """
    Launch DDP training via torchrun for 8×H100.
    """
    import subprocess
    import json
    import os
    from rich.console import Console

    console = Console()
    console.rule("[bold blue]Launching CUAD Fine-tuning (8×H100 DDP)")

    # Build config for workers
    config = {
        "lr": lr_override if lr_override is not None else lr,
        "epochs": epochs_override if epochs_override is not None else num_epochs,
        "batch_size": batch_size_override if batch_size_override is not None else batch_size,
        "max_samples": max_samples_override if max_samples_override is not None else max_samples,
        "eval_samples": eval_samples_override if eval_samples_override is not None else eval_samples,
        "wandb_run_name": wandb_run_name,
    }

    # Set config as env var for workers
    os.environ["TRAIN_CONFIG"] = json.dumps(config)

    console.print(f"Config: {config}")

    # Launch via torchrun
    # Note: This script must be available in the container
    # We use -c to inline the worker call
    worker_code = """
import sys
sys.path.insert(0, '/root')
from finetune_cuad import _train_worker
_train_worker()
"""

    # Write worker script
    with open("/root/run_worker.py", "w") as f:
        f.write(f"""
import sys
import os
# Add path where Modal places the module
sys.path.insert(0, '/root')
os.chdir('/root')

# Import and run
import importlib.util
spec = importlib.util.spec_from_file_location("finetune_cuad", "/root/finetune_cuad.py")
if spec is None:
    # Fallback: the module might be importable directly
    from finetune_cuad import _train_worker
else:
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._train_worker()
""")

    # Copy this file to /root for import
    import shutil
    shutil.copy(__file__, "/root/finetune_cuad.py")

    # Run torchrun
    cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node=8",
        "/root/run_worker.py",
    ]

    console.print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)

    # Read results
    results_path = "/results/train_results.json"
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        volume.commit()
        return results

    volume.commit()
    return {"error": "Training completed but no results file found"}


# ============================================================================
# Local Entrypoint
# ============================================================================


@app.local_entrypoint()
def main(
    lr: float = 2e-5,
    epochs: int = 3,
    batch_size: int = 4,
    max_samples: int | None = None,
    eval_samples: int | None = None,
    test_only: bool = False,
    wandb_name: str | None = None,
):
    """
    Fine-tune nanochat-d20 on CUAD (8×H100 DDP, nanochat style).

    Args:
        lr: Learning rate
        epochs: Training epochs
        batch_size: Per-device batch size
        max_samples: Limit training samples
        eval_samples: Evaluation samples (None = skip)
        test_only: Run small test
        wandb_name: W&B run name
    """
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    console.rule("[bold blue]CUAD Fine-tuning (nanochat style)")

    if test_only:
        console.print(Panel("Test mode: 100 samples, 1 epoch", title="Config"))
        result = train_cuad_ddp.remote(
            lr_override=lr,
            epochs_override=1,
            batch_size_override=batch_size,
            max_samples_override=100,
            eval_samples_override=50,
            wandb_run_name="cuad-test-ddp",
        )
    else:
        console.print(Panel(
            f"LR: {lr}\nEpochs: {epochs}\nBatch size: {batch_size}\n"
            f"Max samples: {max_samples or 'all (~22k)'}\nEval samples: {eval_samples or 'skip'}",
            title="Training Configuration",
        ))
        result = train_cuad_ddp.remote(
            lr_override=lr,
            epochs_override=epochs,
            batch_size_override=batch_size,
            max_samples_override=max_samples,
            eval_samples_override=eval_samples,
            wandb_run_name=wandb_name,
        )

    tokens_trained = result.get('total_tokens_trained', 0)
    eval_section = ""
    if 'exact_match' in result:
        eval_section = (
            f"Val loss: {result.get('val_loss', 'N/A'):.4f}\n"
            f"Eval EM: {result.get('exact_match', 0):.2%}\n"
            f"Eval F1: {result.get('f1', 0):.2%}"
        )
    console.print(Panel(
        f"Final train loss: {result.get('final_loss', 'N/A'):.4f}\n"
        f"Total tokens: {tokens_trained:,}\n"
        f"Model path: {result.get('model_path', 'N/A')}\n"
        + eval_section,
        title="[green]Training Complete",
    ))
