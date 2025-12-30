#!/bin/bash

# This script is the "Best ChatGPT clone that $100 can buy",
# It is designed to run in ~4 hours on 8XH100 node at $3/GPU/hour.

# 1) Example launch (simplest):
# bash speedrun.sh
# 2) Example launch in a screen session (because the run takes ~4 hours):
# screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh
# 3) Example launch with wandb logging, but see below for setting up wandb first:
# WANDB_RUN=speedrun screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Python venv setup with uv

# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies
uv sync --extra gpu
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup
# If you wish to use wandb for logging (it's nice!, recommended).
# 1) Make sure to first log in to wandb, e.g. run:
#    `wandb login`
# 2) Set the WANDB_RUN environment variable when running this script, e.g.:
#    `WANDB_RUN=d26 bash speedrun.sh`
if [ -z "$WANDB_RUN" ]; then
    # by default use "dummy" : it's handled as a special case, skips logging to wandb
    WANDB_RUN=dummy
fi

# Allow callers (e.g., Modal launcher) to route logs to their own project/entity.
# Defaults remain nanochat to preserve upstream behavior when unset.
if [ -z "$WANDB_PROJECT" ]; then
    WANDB_PROJECT=nanochat
fi
if [ -z "$WANDB_ENTITY" ]; then
    WANDB_ENTITY=""
fi
export WANDB_RUN WANDB_PROJECT WANDB_ENTITY
echo "wandb target: ${WANDB_ENTITY:+$WANDB_ENTITY/}$WANDB_PROJECT run=$WANDB_RUN"

# Allow callers (e.g., Modal launcher) to route logs to their own project/entity.
# Defaults remain nanochat to preserve upstream behavior when unset.
if [ -z "$WANDB_PROJECT" ]; then
    WANDB_PROJECT=nanochat
fi
if [ -z "$WANDB_ENTITY" ]; then
    WANDB_ENTITY=""
fi
export WANDB_RUN WANDB_PROJECT WANDB_ENTITY
echo "wandb target: ${WANDB_ENTITY:+$WANDB_ENTITY/}$WANDB_PROJECT run=$WANDB_RUN"

# Allow callers (e.g., Modal launcher) to route logs to their own project/entity.
# Defaults remain nanochat to preserve upstream behavior when unset.
if [ -z "$WANDB_PROJECT" ]; then
    WANDB_PROJECT=nanochat
fi
if [ -z "$WANDB_ENTITY" ]; then
    WANDB_ENTITY=""
fi
export WANDB_RUN WANDB_PROJECT WANDB_ENTITY
echo "wandb target: ${WANDB_ENTITY:+$WANDB_ENTITY/}$WANDB_PROJECT run=$WANDB_RUN"

# -----------------------------------------------------------------------------
# Smoke test mode (fast, no downloads). Set SMOKE=1 to enable.
if [ "${SMOKE:-0}" = "1" ]; then
    echo "SMOKE=1 enabled: running 3-layer, 10-step distributed smoke train with logging and artifacts."
    repo_root=/workspace/nanochat
    mkdir -p "$repo_root/out" "$repo_root/tokenizer"
    cat > "$repo_root/smoke_train.py" <<'PY'
import os, time, json, pathlib, math
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import wandb

def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(42 + rank)

    vocab_size = 256
    seq_len = 64
    batch = 8
    steps = 10

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, 256)
            encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=512, batch_first=True)
            self.enc = nn.TransformerEncoder(encoder_layer, num_layers=3)
            self.head = nn.Linear(256, vocab_size)
        def forward(self, x):
            h = self.embed(x)
            h = self.enc(h)
            return self.head(h)

    model = TinyModel().to(device)
    ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    opt = torch.optim.AdamW(ddp_model.parameters(), lr=1e-3)

    project = os.environ.get("WANDB_PROJECT", "nanochat")
    entity = os.environ.get("WANDB_ENTITY") or None
    run_name = os.environ.get("WANDB_RUN", "smoke")
    artifact_name = os.environ.get("SAVE_ARTIFACT_NAME", "smoke-artifact")
    if rank == 0:
        wb = wandb.init(project=project, entity=entity, name=run_name, config={"smoke": True, "layers":3,"steps":steps})

    for step in range(steps):
        x = torch.randint(0, vocab_size, (batch, seq_len), device=device)
        y = torch.randint(0, vocab_size, (batch, seq_len), device=device)
        logits = ddp_model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        dist.barrier()
        if rank == 0:
            wb.log({"train/loss": loss.item(), "step": step})

    # Save checkpoint and tokenizer stub on rank 0
    if rank == 0:
        out_dir = pathlib.Path(os.environ.get("REPO_ROOT", "/workspace/nanochat")) / "out"
        out_dir.mkdir(exist_ok=True)
        ckpt_path = out_dir / "smoke.pt"
        torch.save(model.state_dict(), ckpt_path)

        tok_dir = pathlib.Path(os.environ.get("REPO_ROOT", "/workspace/nanochat")) / "tokenizer"
        tok_dir.mkdir(exist_ok=True)
        (tok_dir / "tokenizer.json").write_text(json.dumps({"vocab_size": vocab_size, "notes":"smoke stub"}))
        (tok_dir / "vocab.json").write_text(json.dumps({str(i):i for i in range(vocab_size)}))
        (tok_dir / "merges.txt").write_text("# dummy merges\n")
        (tok_dir / "tokenizer_config.json").write_text(json.dumps({"model_type":"smoke"}))

        report = pathlib.Path(os.environ.get("REPO_ROOT", "/workspace/nanochat")) / "report.md"
        report.write_text("# Smoke Report\n\nRan 3-layer tiny model for 10 steps on random data.\n")

        # Upload artifact within the same run
        art = wandb.Artifact(artifact_name, type="model")
        art.add_file(ckpt_path, name="smoke.pt")
        art.add_dir(tok_dir, name="tokenizer")
        if report.exists():
            art.add_file(report, name="report.md")
        wb.log_artifact(art, aliases=[run_name, "latest"])
        wb.log({"artifact_logged": True})

        wb.finish()

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
PY
    REPO_ROOT=$repo_root torchrun --standalone --nproc_per_node=8 "$repo_root/smoke_train.py"
    exit 0
fi

# -----------------------------------------------------------------------------
# During the course of the run, we will be writing markdown reports to the report/
# directory in the base dir. This command clears it out and writes a header section
# with a bunch of system info and a timestamp that marks the start of the run.
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer

# Install Rust / Cargo
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Build the rustbpe Tokenizer
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# Download the first ~2B characters of pretraining dataset
# look at dev/repackage_data_reference.py for details on how this data was prepared
# each data shard is ~250M chars
# so we download 2e9 / 250e6 = 8 data shards at this point
# each shard is ~100MB of text (compressed), so this is about ~800MB of data on disk
python -m nanochat.dataset -n 8
# Immediately also kick off downloading more shards in the background while tokenizer trains
# See comment below for why 240 is the right number here
python -m nanochat.dataset -n 240 &
DATASET_DOWNLOAD_PID=$!
# train the tokenizer with vocab size 2**16 = 65536 on ~2B characters of data
python -m scripts.tok_train --max_chars=2000000000
# evaluate the tokenizer (report compression ratio etc.)
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining)

# The d20 model is 561M parameters.
# Chinchilla says #tokens = 20X #params, so we need 561e6 * 20 = 11.2B tokens.
# Assume our tokenizer is 4.8 chars/token, this is 11.2B * 4.8 ~= 54B chars.
# At 250M chars/shard, this is 54B / 250M ~= 216 shards needed for pretraining.
# Round up to 240 for safety. At ~100MB/shard, this downloads ~24GB of data to disk.
# (The total number of shards available in the entire dataset is 1822.)
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# Number of processes/GPUs to use
NPROC_PER_NODE=8

# pretrain the d20 model
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- --depth=20 --run=$WANDB_RUN
# evaluate the model on a larger chunk of train/val data and draw some samples
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_loss
# evaluate the model on CORE tasks
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_eval

# -----------------------------------------------------------------------------
# Midtraining (teach the model conversation special tokens, tool use, multiple choice)

# download 2.3MB of synthetic identity conversations to impart a personality to nanochat
# see dev/gen_synthetic_data.py for details on how this data was prepared and to get a sense of how you can easily tune it
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# run midtraining and eval the model
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.mid_train -- --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i mid

# -----------------------------------------------------------------------------
# Supervised Finetuning (domain adaptation to each sequence all by itself per row)

# train sft and re-eval right away (should see a small bump)
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_sft -- --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i sft

# chat with the model over CLI! Leave out the -p to chat interactively
# python -m scripts.chat_cli -p "Why is the sky blue?"

# even better, chat with your model over a pretty WebUI ChatGPT style
# python -m scripts.chat_web

# -----------------------------------------------------------------------------
# Reinforcement Learning. GSM8K
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_rl -- --run=$WANDB_RUN
# eval the RL model only on GSM8K
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i rl -a GSM8K

# -----------------------------------------------------------------------------
# Generate the full report by putting together all the sections
# report.md is the output and will be copied to current directory for convenience
python -m nanochat.report generate
