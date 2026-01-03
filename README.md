# fractal-llm

Fractal analysis of LLM fine-tuning trainability boundaries using nanochat-d20 and DocVQA on Modal H100s. We vendor nanochat under `third_party/nanochat` (commit `8f979a8bdab491c4c152ce5c87f90c2ec31d0845`, documented in `third_party/nanochat/COMMIT_INFO.txt`) so training uses our patched copy.

## Quickstart

1) **Test a single run**
`uv run modal run src/modal_app.py --test-only`

2) **Run a pilot grid (32×32)**
`uv run modal run src/modal_app.py --resolution 32`

3) **Visualize**
`uv run python src/visualize.py --results-path /results/grid_32x32_YYYYMMDD_HHMMSS.json --output-dir results/figures`

4) **OOD eval (HellaSwag + ARC)**
`uv run eval/run_lmeval.py --model nanochat-students/nanochat-d20 --tasks hellaswag,arc_challenge --max-samples 500`

5) **Local finetune (nanochat-style, torchrun on 8 GPUs)**
   - Fast smoke: `torchrun --standalone --nproc_per_node=1 -m src.finetune --run=smoke --learning_rate=3e-4 --num_tokens=20000 --log_every=1 --eval_every=0`
   - Full grid + visuals (writes JSON+PNG+fractal JSON to `results/`, logs W&B if `WANDB_RUN` set):  
     `torchrun --standalone --nproc_per_node=8 -m src.finetune --grid=True --run=fractal-grid --resolution=16 --lr_min=1e-5 --lr_max=1e-3 --tokens_min=5e3 --tokens_max=5e5`
   - Use a specific W&B artifact as the model source:  
     `torchrun --standalone --nproc_per_node=1 -m src.finetune --model_id="wandb:morgy/fractal-llm/nanochat-fin-rl-artifact:v7" --run=smoke --learning_rate=3e-4 --num_tokens=20000 --log_every=1 --eval_every=0`

### Local Grid Sweep (parallel single-GPU)
1) Cache model/tokenizer locally (e.g., `${FRACTAL_STORAGE_DIR:-/var/tmp/fractal-llm}/results/model_cache/.../checkpoints`). Cache DocVQA once, then set `HF_DATASETS_OFFLINE=1` for repeatable sweeps.
2) Select GPUs: `GPUS="0 1 2 3 4 5 6 7"` (one run per ID).
3) Launch sweep with `scripts/grid_sweep.sh` (logs → `${FRACTAL_STORAGE_DIR:-/var/tmp/fractal-llm}/results/grid_logs/<RUN_PREFIX>/`):
   ```bash
   RUN_PREFIX=grid-smoke \
   FRACTAL_STORAGE_DIR=/var/tmp/fractal-llm \
   GPUS="0 1 2 3 4 5 6 7" \
   SWEEP_AXES=matrix_unembedding \
   TOKENS_PER_RUN=5e5 \
   MATRIX_LR_MIN=1e-6 MATRIX_LR_MAX=3e-1 \
   UNEMBEDDING_LR_MIN=1e-6 UNEMBEDDING_LR_MAX=1e-1 \
   TRAINABLE_PARAM_GROUPS=matrix,unembedding \
   RES=16 \
   MODEL_ID=/var/tmp/fractal-llm/results/model_cache/nanochat-d20-20251230-r3-sft-artifact_v0/checkpoints \
   DATASET_ID=morgan/docvqa-nanochat \
   # optional: pin HF commit
   DATASET_REVISION=main \
   MAX_SEQ_LEN=1024 \
   HF_DATASETS_OFFLINE=1 \
   ./scripts/grid_sweep.sh
   ```
   - Overrides: `SWEEP_AXES=matrix_unembedding` sweeps matrix_lr×unembedding_lr at fixed `TOKENS_PER_RUN`; `SWEEP_AXES=lr_tokens` sweeps learning_rate×num_tokens using `LR_MIN..LR_MAX` and `TOK_MIN..TOK_MAX`.
   - `TRAINABLE_PARAM_GROUPS` controls which model groups are updated (default freezes embeddings); `GRID_SWEEP_ID` groups runs; `RUN_PREFIX` (or `WANDB_RUN_PREFIX`) names outputs; `LOG_DIR` changes destination.
   - Output: per-point logs `run_<i>_<j>.log`; JSON summary prints the parsed final loss for each point.
4) Probe the max per-GPU batch size for `src/finetune.py`:
   `GPU=0 BS_START=8 BS_MAX=256 ./scripts/probe_batch_size.sh`

### Multi-DevPod Grid Sweep (CoreWeave)
Launch a single sweep across multiple devpods (each devpod runs its own set of points; each uses all local GPUs):
```bash
DEVPODS="fractal-llm-1 fractal-llm-2 fractal-llm-3" \
SWEEP_AXES=matrix_unembedding TOKENS_PER_RUN=5e5 \
RES=5 RUN_PREFIX=5x5-trial2 GRID_SWEEP_ID=5x5-trial2 \
./scripts/grid_sweep.sh
```
Monitor: `devpod ssh fractal-llm-1` then `tmux attach -t grid_5x5-trial2`.

### NanoChat

To run a full nanochat run, including RL, as well as artifact and tokenizer saving for each stage, run this:
```
export NANOCHAT_BASE_DIR=/var/tmp/nanochat && WANDB_RUN=nanochat-fin WANDB_PROJECT=fractal-llm WANDB_ENTITY=morgy NPROC_PER_NODE=8 bash speedrun.sh
```

### CoreWeave DevPod storage quota workaround (torch install)
If torch wheels blow your workspace quota, put the venv on `/var/tmp` and keep the torch install to one copy:
```bash
# from third_party/nanochat
rm -rf .venv /workspaces/.uv-cache
python3.10 -m venv /var/tmp/nanochat-venv
ln -sfn /var/tmp/nanochat-venv .venv
source .venv/bin/activate
python -m ensurepip --upgrade || true
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu126 \
  torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
UV_CACHE_DIR=/workspaces/.uv-cache UV_LINK_MODE=symlink uv sync --extra gpu
```
If checkpoints/tokenizer are filling the small workspace volume, relocate nanochat data by exporting:
`export NANOCHAT_BASE_DIR=/var/tmp/nanochat` (or any larger mount) before running `speedrun.sh`.

## Modal setup (8×H100 nanochat training)
1) Create env (once): `uv run modal environment create fractal-llm`
2) Set token (once):  
   `uv run modal token set --token-id <token> --token-secret <secret> --profile=weightsandbiases`  
   (token currently in `.env` as `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`)
3) Create W&B secret in env (once):  
   `uv run modal secret create --env fractal-llm wandb-secret WANDB_API_KEY="$WANDB_API_KEY"` (load from `.env`)
4) Train nanochat d20 on 8×H100 using the vendored nanochat copy and log artifact to W&B (runs remotely; safe to close laptop after launch):  
   `MODAL_ENVIRONMENT=fractal-llm uv run modal run --detach src/nanochat_modal.py --wandb-name nanochat-d20-modal --save-artifact-name nanochat-d20-speedrun`
   (Sets `WANDB_RUN=<wandb-name>` so nanochat wandb logging is enabled; artifact includes model_out.tar.gz + tokenizer/* + report.md; uses `third_party/nanochat` pinned commit noted above)

Smoke test (fast logging + artifact in one run)
`MODAL_ENVIRONMENT=fractal-llm uv run modal run src/nanochat_modal.py --wandb-name smoke-mini --save-artifact-name smoke-mini-artifact --smoke`
- Runs a 3-layer tiny model for 10 steps on 8 GPUs, logs every step, and uploads a smoke artifact (checkpoint + tokenizer stub + report) to the same W&B run.

## Datasets

### DocVQA (Document QA)
Single-page document QA dataset for nanochat fine-tuning, published at [morgan/docvqa-nanochat](https://huggingface.co/datasets/morgan/docvqa-nanochat).

| Split | Samples | Tokens |
|-------|---------|--------|
| Train | 39,455 | 15.5M |
| Val | 5,349 | 2.2M |

**Key features:**
- Answer-priority truncation (answer always in context)
- Max 1750 tokens per example (fits 2048 context window)
- Short answers only (≤150 chars)
- Page numbers from document metadata

**Download and use:**
```bash
# Download from HF Hub (skips if files exist)
uv run data/download_docvqa_hub.py --all --out_dir data/

# Or in training scripts:
from data.download_docvqa_hub import ensure_docvqa_jsonl
train_path = ensure_docvqa_jsonl("train", "data/docvqa_train.jsonl")
```

**Regenerate from source** (pixparse/docvqa-single-page-questions):
```bash
uv run data/prepare_docvqa.py --out_path data/docvqa_train.jsonl --split train --workers 8
uv run data/prepare_docvqa.py --out_path data/docvqa_val.jsonl --split validation --workers 8
```


## CoreWeave DevPod (Remote GPU Development)

Remote development on CoreWeave's Kubernetes cluster with H200 GPUs. Workspace disk is tiny; store all checkpoints, datasets, HF caches, and W&B files under `/var/tmp/fractal-llm` (set `FRACTAL_STORAGE_DIR=/var/tmp/fractal-llm`).

**Prerequisites:** `brew install kubectl devpod`

**One-time setup:**
```bash
# Download kubeconfig from https://console.coreweave.com/tokens, then:
./crwv_cli/setup.sh ~/Downloads/CWKubeconfig
```

**Usage:**
```bash
# Start 8×H200 dev environment (opens VSCode)
corepod .

# SSH into workspace
ssh fractal-llm.devpod

# Manage workspaces
devpod list
devpod stop fractal-llm
devpod delete fractal-llm
```

**One-off local finetune on devpod GPU (saves to /var/tmp)**  
```bash
cd /workspaces/fractal-llm && source .env && FRACTAL_STORAGE_DIR=/var/tmp/fractal-llm \
CUDA_VISIBLE_DEVICES=0 MAX_SEQ_LEN=1024 TOKENIZER_ARTIFACT="$MODEL_ARTIFACT" \
uv run python -m src.finetune --run devpod-default --eval_every 0 --log_every 1 --save_artifacts False
```
Defaults to freezing token embeddings (`--trainable_param_groups=matrix,unembedding`). Use `--trainable_param_groups=all` to train everything.

**Check cluster workloads:**
```bash
# Node resource usage
kubectl top nodes --context cks-wb3

# GPU allocations across cluster
kubectl get pods --all-namespaces --context cks-wb3 \
  -o custom-columns='NS:.metadata.namespace,NAME:.metadata.name,GPU:.spec.containers[*].resources.limits.nvidia\.com/gpu' \
  | grep -v "<none>"
```

**Change GPU count:** Delete provider and recreate with different `RESOURCES="limits.nvidia.com/gpu=N"` (see CLAUDE.md for details).

## Visualization

### Trainability Boundary Chart (3-Panel Grid)

The grid sweep generates a 3-panel visualization:

1. **Trainability Boundary** (left): Diverging red-white-blue colormap showing trainable vs not-trainable
2. **Final Loss** (center): Loss values for *trainable* runs only (viridis colormap)
3. **Binary Trainable** (right): Simple 0/1 trainability mask

### Definition: stable vs trainable ("converged")

We separate two concepts:

- **stable**: training completed without exceptions and the final training loss is finite
- **trainable** (this is what we record as `converged`): `mean(last K train losses) / first_train_loss < trainable_loss_ratio_threshold` (defaults: `K=20`, threshold `=1.0`)

This matches the original Sohl-Dickstein notebook idea: average the last window to smooth oscillations, and call it trainable if it ends lower than it started.

#### Color Scheme (Trainability Boundary)

The colormap uses a diverging red-white-blue scheme with values from -1.0 to +1.0:

| Value | Color | Meaning |
|-------|-------|---------|
| -1.0 | Dark Red (#8B0000) | Not trainable (includes unstable failures + stable-but-not-trainable) |
| -0.5 to 0 | Pink → White | Unused (all not-trainable runs map to -1.0) |
| 0.3 | Light Blue (#ADD8E6) | Trainable, but **highest loss** among trainable |
| 0.65 | Royal Blue (#4169E1) | Trainable, medium loss |
| 1.0 | Dark Blue (#00008B) | Trainable, **lowest loss** (best) |

**Key insight**: Among trainable runs, the loss is normalized to [0.3, 1.0]. Lower loss → darker blue → better training outcome. This lets you see not just *if* training was trainable, but *how well* it trained.

## Notes
- W&B: entity `morgy`, project `fractal-llm`. Fractal sweeps load the model from W&B artifact `nanochat-d20-speedrun:latest`.
- Always enable W&B metric logging for all runs (do not use `WANDB_MODE=disabled` unless explicitly requested).
- Modal training image: CUDA 12.8, torch 2.8.0+cu128, installs via `uv pip`; `python-dotenv` and `rich` included; flash-attn omitted.
- Token budget per grid point is respected (`steps = ceil(tokens / (bs*seq_len))`).
- WandB terminal UI (beta leet): inspect any run locally via `uv run wandb beta leet https://wandb.ai/morgy/fractal-llm/runs/<run_id>` (handy for Modal jobs).

## Reproducibility
- Deterministic CUDA everywhere: `CUBLAS_WORKSPACE_CONFIG=:4096:8`, TF32 off, `torch.use_deterministic_algorithms(True)`, `torch.backends.cudnn.deterministic=True`, `torch.backends.cudnn.benchmark=False`, `TORCH_NUM_THREADS=1`, `CUDA_DEVICE_ORDER=PCI_BUS_ID`, and NCCL fixed (`NCCL_ALGO=Ring`, `NCCL_PROTO=Simple`, `NCCL_MIN_NRINGS=1`).
- Seeds: base `seed` applies to model init for all ranks; each grid point gets `run_seed = seed + grid_i*1000 + grid_j`, reused for data shuffle and logging. Results JSON/W&B config now record `run_seed` plus reproducibility metadata (git commit, torch/cuda/nccl versions, env flags).
- Data determinism: DocVQA streaming loader accepts optional `DATASET_REVISION` and reuses the same shuffle buffer/seed per run; set `HF_DATASETS_OFFLINE=1` after caching to avoid remote variance.
- GPU scaling: For fractal grids, prefer 8 independent single-GPU jobs (one per H200) for maximum throughput and bitwise repeatability. DDP paths keep identical initial weights across ranks; NCCL topology is pinned to ring for stable reductions.
- Test harness: `./scripts/repro_check.sh` launches 8 single-GPU smoke trainings and asserts identical final losses across GPUs. Example:  
  `RUN_PREFIX=repro-smoke TOKENS=2000 HF_DATASETS_OFFLINE=0 ./scripts/repro_check.sh`
- Run the repro script in a tmux session on the devpod (from local machine):  
  `ssh fractal-llm.devpod 'bash -lc "cd /workspaces/fractal-llm && source .env && tmux new-session -d -s repro_check \"./scripts/repro_check.sh\" && tmux ls"'`  
  Attach with: `ssh -t fractal-llm.devpod 'tmux attach -t repro_check'`
