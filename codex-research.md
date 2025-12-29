# NanoChat Fractal-SFT feasibility (H100, 128 ctx)

## Goal
Estimate wall-clock and outline a minimal experiment to probe the “fractal” training frontier using only SFT on karpathy/nanochat, targeting short-context (128) runs on a single H100.

## Evidence (recent)
- Fractal training blog + paper: training dynamics near a fractal-shaped stability boundary; small LR/dataset tweaks can flip convergence/ divergence. Blog: Sohl-Dickstein 2024-02-12; Paper: arXiv:2402.06184.
- NanoChat quick tutorial: 15–30 min fine-tune on a single 24 GB GPU with the default 124M model and small chat dataset (Jose D. Baena, 2024-03). Shows sub-hour runs are realistic for tiny models.
- H100 speedrun baseline: AWS 8×H100 completes the full nanochat pipeline (pretrain+SFT+inference) in ~1 hr; the SFT stage alone is ~24 min on 8 GPUs (AWS Machine Learning Blog, 2024-10). Linear scaling suggests ~3.2 hr on one H100 for the same config.

## Runtime estimate for our target
- Start from the 124M nanochat model, context 128, SFT only, modest chat set (≤2M tokens).
- Scale the 8×H100 SFT figure (~24 min) by 8 → ~3.2 hr on one H100 for that larger setup.
- Reducing model size/context and dataset to the above yields roughly 45–90 minutes wall clock on a single H100 (batch ~8–16 seq, fp16/bf16). Very small runs (≤500k tokens) should land in the 15–30 minute band similar to the 24 GB GPU tutorial, but with higher throughput headroom.

## Proposed experiment recipe
- Model: nanochat 124M (or shrink depth to 8 for faster sweeps). If you want an RL-start, convert a small RLHF checkpoint (e.g., Zephyr-β 0.5–1B) into nanochat format, but 124M keeps runs <1.5h.
- Context len: 128; pack short turns to keep tokens/step high.
- Dataset: 1–2M tokens of concise dialogs (e.g., OpenHermes filtered to ≤128 tokens/turn). Keep a held-out 50–100k-token slice for early stopping.
- OOD eval: ARC-Challenge, PIQA, Winogrande, TruthfulQA (all fit in 128 tokens per item). For chat style, MT-Bench-lite single-turn prompts also fit.
- Optimizer: AdamW, weight decay 0.1, cosine decay, warmup 1–2% of steps, grad clip 1.0.
- Mixed precision: bf16; enable grad accumulation to keep per-step tokens ≈ context_len × micro_batch × grad_accum.

## Sweep knobs (guided by the fractal boundary idea)
- Learning rate: log sweep 3e-5 → 6e-4 (expect sharp cliff; keep 3–5 points).
- Effective batch tokens: vary grad_accum to try ~16k, 32k tokens/step.
- Dataset size: 0.25M, 0.5M, 1M, 2M tokens to see where convergence flips.
- Optional: label smoothing 0.05 vs 0; dropout 0 vs 0.1 for stability margin.

## Rule-of-thumb takeaway
- On a single H100, sub-billion-parameter chat models can finish a useful 128-ctx SFT sweep in well under 2 hours; allocate ~1 hour per LR point when using ≤2M tokens. The fractal-boundary behavior implies the “best” LR sits just below the largest LR that stays stable—so a coarse LR sweep plus one smaller step is usually enough.

## Next actions
1) Prepare a 0.5–2M token short-context chat shard and OOD eval scripts (ARC, PIQA, Winogrande, TruthfulQA).
2) Run 3–5 LR points with fixed batch tokens (~32k) and record loss/accuracy vs wall clock.
3) If divergence appears, halve LR or reduce batch tokens; if stable, try +25% LR to approach the boundary.
