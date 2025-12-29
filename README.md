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

## Notes
- Modal profile is set to `weightsandbiases`; W&B logs to `morgan/fractal-llm`.
- Image uses CUDA 12.4 with Torch 2.5.1 CU124 wheels and flash-attn.
- Token budget per grid point is respected (`steps = ceil(tokens / (bs*seq_len))`).
