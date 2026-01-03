# Project: fractal-LLM

This project explores the fractal nature of training dynamics in neural networks, specifically LLMs. Inspired by Jascha Sohl-Dickstein's work showing that the boundary between convergent and divergent training is fractal.

**Key References:**
- Blog: https://sohl-dickstein.github.io/2024/02/12/fractal.html
- Paper: https://arxiv.org/abs/2402.06184
- colab: https://colab.research.google.com/github/Sohl-Dickstein/fractal/blob/main/the_boundary_of_neural_network_trainability_is_fractal.ipynb

Our resarch question: will fractal boundaries still emerge with more realistic adaptive optimizers and fine-tuning on the nanochat LLM?

## Project Goal

Run hyperparameter grid searches over LR × dataset size (and other axes) during SFT to visualize fractal trainability boundaries. Goal is to discover practical rules of thumb for fine-tuning LLMs.

## Definition: stable vs trainable ("converged")

For grids/visualizations we use two related but distinct labels:

- **stable**: training completed without exceptions and the final training loss is finite
- **trainable** (recorded as `converged`): `mean(last K train losses) / first_train_loss < trainable_loss_ratio_threshold` (defaults: `K=20`, threshold `=1.0`)

This mirrors the original Sohl-Dickstein notebook approach: average over the last window to smooth oscillatory behavior, and call it trainable if it ends lower than it started.

## Dependency management and running python

Always us `uv` for everything, `uv sync` for syncing dependencies, `uv run` for running files etc. 

## Infrastructure

### CoreWeave (Kubernetes GPU Cluster) - DevPod
Remote development environment on CoreWeave's Kubernetes cluster with GPU access.

- **Cluster**: `cks-wb3` (CoreWeave Kubernetes Service)
- **GPU**: H200 (configurable 1-8 GPUs per workspace)
- **Container**: `us-docker.pkg.dev/colab-images/public/runtime` (Google Colab image with CUDA, uv, Jupyter)
- **Provider**: `kubernetes-crwv` (DevPod provider pointing to CoreWeave)

**Prerequisites:**
```bash
# Install kubectl and devpod
brew install kubectl devpod
```

**One-time setup:**
```bash
# 1. Download kubeconfig from https://console.coreweave.com/tokens
# 2. Run the setup script (merges kubeconfig, creates devpod provider, adds alias)
./crwv_cli/setup.sh ~/Downloads/CWKubeconfig

# Or manually create provider with custom GPU count:
devpod provider add kubernetes \
  --name kubernetes-crwv \
  -o KUBERNETES_CONTEXT=cks-wb3 \
  -o RESOURCES="limits.nvidia.com/gpu=8" \
  -o LABELS="devpod.sh/user=$(whoami)" \
  -o INACTIVITY_TIMEOUT=1d
```

**Commands:**
```bash
# Start GPU dev environment (opens VSCode connected to remote container)
corepod .
# Or explicitly:
devpod up --devcontainer-image us-docker.pkg.dev/colab-images/public/runtime --provider kubernetes-crwv .

# SSH into running workspace
ssh fractal-llm.devpod

# List workspaces
devpod list

# Stop/delete workspace
devpod stop fractal-llm
devpod delete fractal-llm

# Check cluster GPU usage
kubectl top nodes --context cks-wb3
kubectl get pods --all-namespaces --context cks-wb3 -o custom-columns='NS:.metadata.namespace,NAME:.metadata.name,GPU:.spec.containers[*].resources.limits.nvidia\.com/gpu' | grep -v "<none>"
```

**Repro script in tmux (from local machine):**
```bash
ssh fractal-llm.devpod 'bash -lc "cd /workspaces/fractal-llm && source .env && tmux new-session -d -s repro_check \"./scripts/repro_check.sh\" && tmux ls"'
```
Attach to the session:
```bash
ssh -t fractal-llm.devpod 'tmux attach -t repro_check'
```

**Changing GPU count:**
```bash
# Delete provider and recreate with new GPU count
devpod provider delete kubernetes-crwv
devpod provider add kubernetes --name kubernetes-crwv \
  -o KUBERNETES_CONTEXT=cks-wb3 \
  -o RESOURCES="limits.nvidia.com/gpu=4" \
  -o LABELS="devpod.sh/user=$(whoami)" \
  -o INACTIVITY_TIMEOUT=1d
```

**Disk Storage**
Coreweave devpod workspace (`/workspaces/fractal-llm`) is tiny. Always store checkpoints, W&B files, HF caches, and datasets under `/var/tmp/fractal-llm` (set `FRACTAL_STORAGE_DIR=/var/tmp/fractal-llm`). The training code now defaults to that location, but double-check when adding new scripts.

**Run a single finetune on devpod GPU (uses /var/tmp)**
```bash
cd /workspaces/fractal-llm && source .env && FRACTAL_STORAGE_DIR=/var/tmp/fractal-llm \
CUDA_VISIBLE_DEVICES=0 MAX_SEQ_LEN=1024 TOKENIZER_ARTIFACT="$MODEL_ARTIFACT" \
uv run python -m src.finetune --run devpod-default --eval_every 0 --log_every 20 --save_artifacts False
```
Notes:
- `src/finetune.py` now defaults to freezing token embeddings: `--trainable_param_groups=matrix,unembedding`
- To train everything (including embeddings), pass `--trainable_param_groups=all`

**Run a multi-devpod grid sweep (recommended for fractal grids)**
```bash
# from your laptop (local repo), launches tmux workers on each devpod and returns immediately
DEVPODS="fractal-llm-1 fractal-llm-2 fractal-llm-3" \
SWEEP_AXES=matrix_unembedding TOKENS_PER_RUN=5e5 \
RES=5 RUN_PREFIX=5x5-trial2 GRID_SWEEP_ID=5x5-trial2 \
./scripts/grid_sweep.sh

# monitor
devpod ssh fractal-llm-1   # then: tmux attach -t grid_5x5-trial2
```

**Notes:**
- Workspace syncs local directory to `/workspaces/fractal-llm` in container (~1.2GB with data/)
- Container auto-deletes after 1 day of inactivity
- Do NOT create `.devcontainer.json` with a different image; the `corepod` alias specifies the GPU image

---

### Modal (Cloud GPU) - VERIFIED WORKING
- **Profile**: `weightsandbiases`
- **GPU**: H100 (single or up to 8× per node)
- **Volume**: `fractal-llm-results` for persistent storage
- **Image**: `nvidia/cuda:12.8.0-devel-ubuntu22.04` (Torch 2.8.0+cu128 via uv pip; rich + python-dotenv included)
- **Model code**: vendored `third_party/nanochat` (commit `8f979a8bdab491c4c152ce5c87f90c2ec31d0845`, 2025-12-28). Keep this copy in sync if you update upstream. Commit info lives in `third_party/nanochat/COMMIT_INFO.txt`.
- **Model artifact**: nanochat-d20 (561M) from W&B artifact `morgy/fractal-llm/nanochat-fin-rl-artifact:v7`

**Commands:**
```bash
# Test single training run on H100
uv run modal run src/modal_app.py --test-only

# Run full grid search (e.g., 32x32)
uv run modal run src/modal_app.py --resolution 32

# Deploy for production
uv run modal deploy src/modal_app.py

# Check profile
uv run modal profile current

# OOD eval snapshot (HellaSwag + ARC)
uv run eval/run_lmeval.py --model nanochat-students/nanochat-d20 --tasks hellaswag,arc_challenge --max-samples 500

# Create Modal env + token + secret (one time)
uv run modal environment create fractal-llm
uv run modal token set --token-id $MODAL_TOKEN_ID --token-secret $MODAL_TOKEN_SECRET --profile=weightsandbiases
uv run modal secret create --env fractal-llm wandb-secret WANDB_API_KEY="$WANDB_API_KEY"

# Run nanochat d20 speedrun on 8×H100 (logs artifact to W&B; detached so laptop can sleep)
# Uses vendored nanochat at third_party/nanochat (pinned commit above) so our wandb patches apply.
MODAL_ENVIRONMENT=fractal-llm uv run modal run --detach src/nanochat_modal.py \
  --wandb-name nanochat-d20-modal \
  --save-artifact-name nanochat-d20-speedrun
# WANDB_RUN is set from wandb-name to avoid 'dummy' runs; artifact packs model_out.tar.gz + tokenizer/* + report.md

# Local finetune using a specific W&B artifact (no Modal; torchrun on 1–8 GPUs)
torchrun --standalone --nproc_per_node=1 -m src.finetune \
  --model_id="wandb:morgy/fractal-llm/nanochat-fin-rl-artifact:v7" \
  --run=smoke --learning_rate=3e-4 --num_tokens=20000 --log_every=1 --eval_every=0

# Smoke test (fast, validates logging + artifact in a single run)
# 3-layer tiny model, 10 steps, logs every step and uploads smoke-mini-artifact in the same W&B run.
MODAL_ENVIRONMENT=fractal-llm uv run modal run src/nanochat_modal.py \
  --wandb-name smoke-mini \
  --save-artifact-name smoke-mini-artifact \
  --smoke
```

**Test Results (verified 2024-12-29):**
- H100 training: 9 steps in 1.27s
- Loss: 3.39 → 3.00 (converged)
- Dataset: DocVQA (morgan/docvqa-nanochat)

### Project Structure
```
fractal-llm/
├── src/
│   ├── modal_app.py        # Modal H100 grid search training
│   ├── nanochat_modal.py   # Train nanochat-d20 on 8×H100, push to W&B
│   └── visualize.py        # Post-hoc visualization and fractal analysis
├── data/
│   ├── prepare_docvqa.py        # Process DocVQA from source dataset
│   ├── push_docvqa_hub.py       # Push DocVQA to HuggingFace Hub
│   └── download_docvqa_hub.py   # Download DocVQA from HF Hub → JSONL
├── eval/
│   └── run_lmeval.py       # OOD evaluation (HellaSwag, ARC)
├── third_party/
│   └── nanochat/           # Vendored nanochat (pinned commit)
├── claude-research.md      # Research notes and experiment design
├── .env                    # Modal credentials (gitignored)
└── pyproject.toml          # Dependencies (uv, torch 2.8.0 cu128)
```

### Datasets

#### DocVQA (Document QA)
Single-page document QA dataset for nanochat fine-tuning.
- **HF Hub**: [morgan/docvqa-nanochat](https://huggingface.co/datasets/morgan/docvqa-nanochat)
- **Source**: [pixparse/docvqa-single-page-questions](https://huggingface.co/datasets/pixparse/docvqa-single-page-questions)
- **Stats**: 39,455 train / 5,349 val samples, ~17.7M tokens total
- **Tokenizer**: tiktoken cl100k_base (GPT-4 style BPE)

**Processing features:**
- Answer-priority truncation: OCR lines containing the answer are always included
- Max 1750 tokens per example (for 2048 context window)
- Short answers only (≤150 chars)
- Page numbers from `other_metadata['ucsf_document_page_no']`
- Match types tracked: exact, fuzzy, none

**Commands:**
```bash
# Download from HF Hub (auto-skips if files exist with correct counts)
uv run data/download_docvqa_hub.py --all --out_dir data/

# Force re-download
uv run data/download_docvqa_hub.py --all --force

# Regenerate from source (slow, ~45k samples with parallel tokenization)
uv run data/prepare_docvqa.py --out_path data/docvqa_train.jsonl --split train --workers 8
uv run data/prepare_docvqa.py --out_path data/docvqa_val.jsonl --split validation --workers 8

# Push to HF Hub (requires HF token)
uv run data/push_docvqa_hub.py --train data/docvqa_train.hub.jsonl --val data/docvqa_val.hub.jsonl --repo morgan/docvqa-nanochat --token $HF_TOKEN
```

**Use in training scripts:**
```python
from data.download_docvqa_hub import ensure_docvqa_jsonl

# Auto-downloads from HF Hub if missing or count mismatch
train_path = ensure_docvqa_jsonl("train", "data/docvqa_train.jsonl")
val_path = ensure_docvqa_jsonl("validation", "data/docvqa_val.jsonl")

# Use with nanochat CustomJSON
from tasks.customjson import CustomJSON
train_ds = CustomJSON(filepath=str(train_path))
```

---

## Coding guidelines and philosophy
- Generate code that is simple and readable, avoid unnecessary abstractions and complexity. This is a research codebase so we want to be maintainable and readable.
- Avoid overly defensive coding, no need for a lot of `try, except` patterns, I want the code to fail if something is wrong so that I can fix it.
- Do not add demo-only flags or placeholder CLI options that gate real functionality (e.g., `--run` just to toggle execution); scripts should run their main logic directly.

## Dependency management
This project uses uv as dependency manager for python. Run scripts using `uv run script.py` instead of calling python directly. This is also true for tools like `uv run pytest`

## Argument parsing
Use `simple_parsing` as an argument parser for the scripts. Like this

```python
import simple_parsing as sp

@dataclass
class Args:
    """ Help string for this group of command-line arguments """
    arg1: str       # Help string for a required str argument
    arg2: int = 1   # Help string for arg2

args = sp.parse(Args)
```

## Typing
We are using modern python (3.12+) so no need to import annotations, you can also use `dict` and `list` and `a | b` or `a | None` instead of Optional, Union, Dict, List, etc...

## Printing and logging
Use rich.Console to print stuff on scripts, use Panel and console.rule to make stuff organized

## Debugging
When running scripts, use the `debug` flags if available, and ask to run the full pipeline (this enables faster iteration)

## Running Analysis
Ensure to always use performant code for running analysis, always use pandas best practices for speed and efficiency.

## Working with Weights & Biases - project and entity to use
When logging to `wandb` or `weave` from Weights & Biases, always log to the `morgy` entity and the `fractal-llm` project, unless specifically asked to log elsewhere
**Always enable W&B metric logging for all runs**. Do not disable logging (e.g., `WANDB_MODE=disabled`) unless explicitly asked.

### WandB terminal UI (beta leet)
- Inspect runs locally with the new TUI: `uv run wandb beta leet https://wandb.ai/morgan/fractal-llm/runs/<run_id>`
- Useful for monitoring long Modal jobs without opening a browser.

## Working with Jupyter notebooks
### Reading / visualizing pandas dataframes
When working with jupyter notebooks, remove truncation so we can print full outputs
```python
import pandas as pd
pd.set_option('display.max_columns', None)   # no column truncation
pd.set_option('display.width', None)         # keep each row on one line
pd.set_option('display.max_colwidth', None)  # don't truncate long string cells
```

### Autoreload
Prefer adding autoreload at the top cell of the notebook so that we don't have to restart the notebook when we make changes to our library
```python
%load_ext autoreload
%autoreload 2
```

## Running commands
Avoid asking the user to run commands unless its strictly necesary for the user to run it. Its fine to educate them and tell them the commands that are being run and why, but if you've been asked to achieve a task and there isn't a strong reason why you can't just run the command yourself, just run the command.
