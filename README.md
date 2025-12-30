# fractal-llm

Fractal analysis of LLM fine-tuning trainability boundaries using nanochat-d20 and SmolTalk on Modal H100s.

## Quickstart

1) **Prep shards (optional but faster)**  
`uv run data/prepare_smoltalk.py --budgets 1000 10000 100000 1000000 --out-dir /results/smoltalk_shards`

2) **Test a single run**  
`uv run modal run src/modal_app.py --test-only`

3) **Run a pilot grid (32×32)**  
`uv run modal run src/modal_app.py --resolution 32`

4) **Visualize**  
`uv run python src/visualize.py --results-path /results/grid_32x32_YYYYMMDD_HHMMSS.json --output-dir results/figures`

5) **OOD eval (HellaSwag + ARC)**  
`uv run eval/run_lmeval.py --model nanochat-students/nanochat-d20 --tasks hellaswag,arc_challenge --max-samples 500`

## Modal setup (8×H100 nanochat training)
1) Create env (once): `uv run modal environment create fractal-llm`
2) Set token (once):  
   `uv run modal token set --token-id <token> --token-secret <secret> --profile=weightsandbiases`  
   (token currently in `.env` as `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`)
3) Create W&B secret in env (once):  
   `uv run modal secret create --env fractal-llm wandb-secret WANDB_API_KEY="$WANDB_API_KEY"` (load from `.env`)
4) Train nanochat d20 on 8×H100 and log artifact to W&B (runs remotely; safe to close laptop after launch):  
   `MODAL_ENVIRONMENT=fractal-llm uv run modal run src/nanochat_modal.py --wandb-name nanochat-d20-modal --save-artifact-name nanochat-d20-speedrun`

## Notes
- W&B: entity `morgan`, project `fractal-llm`. Fractal sweeps load the model from W&B artifact `nanochat-d20-speedrun:latest`.
- Modal training image: CUDA 12.8, torch 2.8.0+cu128, installs via `uv pip`; `python-dotenv` and `rich` included; flash-attn omitted.
- Token budget per grid point is respected (`steps = ceil(tokens / (bs*seq_len))`).
