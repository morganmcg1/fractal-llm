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
   - Fast smoke: `torchrun --standalone --nproc_per_node=1 -m src.finetune_modal_app --run=smoke --learning_rate=3e-4 --num_tokens=20000 --log_every=10 --eval_every=0`
   - Full grid + visuals (writes JSON+PNG+fractal JSON to `results/`, logs W&B if `WANDB_RUN` set):  
     `torchrun --standalone --nproc_per_node=8 -m src.finetune_modal_app --grid=True --run=fractal-grid --resolution=16 --lr_min=1e-5 --lr_max=1e-3 --tokens_min=5e3 --tokens_max=5e5`
   - Use a specific W&B artifact as the model source:  
     `torchrun --standalone --nproc_per_node=1 -m src.finetune_modal_app --model_id="wandb:morgan/fractal-llm/nanochat-d20-20251230-r3-sft-artifact:v0" --run=smoke --learning_rate=3e-4 --num_tokens=20000 --log_every=1 --eval_every=0`

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

Remote development on CoreWeave's Kubernetes cluster with H200 GPUs.

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

## Notes
- W&B: entity `morgan`, project `fractal-llm`. Fractal sweeps load the model from W&B artifact `nanochat-d20-speedrun:latest`.
- Modal training image: CUDA 12.8, torch 2.8.0+cu128, installs via `uv pip`; `python-dotenv` and `rich` included; flash-attn omitted.
- Token budget per grid point is respected (`steps = ceil(tokens / (bs*seq_len))`).
- WandB terminal UI (beta leet): inspect any run locally via `uv run wandb beta leet https://wandb.ai/morgan/fractal-llm/runs/<run_id>` (handy for Modal jobs).
