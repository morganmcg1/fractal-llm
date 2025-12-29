# Fractal Training Dynamics for LLM Fine-Tuning: Research Notes

## Background: Fractal Boundaries in Neural Network Training

The key insight from [Sohl-Dickstein's work](https://sohl-dickstein.github.io/2024/02/12/fractal.html) ([arXiv:2402.06184](https://arxiv.org/abs/2402.06184)) is that **the boundary between convergent and divergent training is fractal**. This emerges because:

1. Neural network training iterates a fixed function repeatedly (gradient descent steps)
2. Similar to Mandelbrot/Julia set generation, this iteration creates fractal boundaries
3. The fractals persist across **10+ decades of scale** until floating-point precision limits
4. **Best hyperparameters are usually near the edge of stability** - the largest learning rate that still converges

### Original Experiment Setup
- **Model**: 2-layer fully connected network, 16 hidden units, no bias
- **Hyperparameters varied**: Learning rates for input layer vs output layer (2D grid)
- **Resolution**: Up to 10^-16 (floating point precision limit)
- **Key finding**: Small hyperparameter changes → large training dynamics changes at all scales

### Extension to Transformers ([arXiv:2501.04286](https://arxiv.org/html/2501.04286))
Recent work applied this to decoder-only transformers (~96K params, 2 layers):
- Varied: attention layer LR (η_att) vs FC layer LR (η_fc)
- Found fractal dimensions 1.54-1.98 across different regions
- Confirms self-similar chaotic boundaries in transformer architectures

---

## NanoGPT/Nanochat Speed Capabilities

### NanoGPT Speedrun Records
The [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) community has achieved remarkable speeds:

| Record Holder | Time | Cost | Notes |
|--------------|------|------|-------|
| Karpathy (original) | 45 min | ~$6 | GPT-2 124M baseline |
| [Franz Cesista (leloykun)](https://x.com/leloykun/status/1885640350368420160) | **2.93 min** | ~$0.40 | Current record |

**Key optimizations enabling this:**
- Muon optimizer with Newton-Schulz iteration (~1.5x sample efficiency)
- Flash Attention 3 with sliding window pattern
- FP8 matmuls for specific layers
- Rotary embeddings, QK-Norm, ReLU²
- Batch size scheduling + partial model freezing
- Multi-token prediction
- Skip connections from embedding to every block

**Hardware**: 8×H100 GPUs (~$8/hour on cloud)

### Nanochat (Full ChatGPT Clone)
[Nanochat](https://github.com/karpathy/nanochat) extends this to full SFT:

| Tier | Time | Cost | Model Size | Notes |
|------|------|------|------------|-------|
| $100 speedrun | ~4 hours | ~$100 | 560M (d20) | 11.2B tokens |
| $300 tier | ~12 hours | ~$300 | ~GPT-2 | Surpasses GPT-2 |
| $1000 tier | ~41.6 hours | ~$1000 | 1.9B (d32) | Coherent math/code |

---

## Proposed Experiment: Fractal Analysis of LLM Fine-Tuning

### Core Hypothesis
The trainability boundary for SFT has fractal structure, and understanding this could yield **practical rules of thumb** for:
- Optimal learning rate selection
- Dataset size requirements
- When fine-tuning will succeed vs fail

### Experimental Design

#### Model Selection
**Recommendation**: Start with a mid-trained checkpoint to study SFT dynamics:
- **Option A**: Use modded-nanogpt's 124M checkpoint (fastest iteration)
- **Option B**: Use nanochat's d20 (~560M) checkpoint
- **Option C**: Use an open RL-trained model like Qwen2.5-0.5B-Instruct or SmolLM-135M

**Rationale**: Starting from a pretrained/instruction-tuned checkpoint lets us study the SFT-specific dynamics rather than pretraining.

#### Hyperparameter Axes to Explore

**Primary 2D Grid (for fractal visualization):**

| Axis | Range | Rationale |
|------|-------|-----------|
| **Learning Rate** | 1e-7 to 1e-3 (log scale) | Primary instability driver |
| **Dataset Size** | 100 to 100K samples (log scale) | Affects convergence basin |

**Alternative 2D Grids:**
1. **LR × Batch Size**: Classic stability trade-off
2. **LR × Warmup Ratio**: Studies edge-of-stability dynamics
3. **Attention LR × FC LR**: Following the transformer fractal paper
4. **LR × Weight Decay**: Regularization vs speed trade-off

#### Resolution & Compute Estimates

For a **128×128 resolution** grid:

| Component | Estimate |
|-----------|----------|
| Total training runs | 16,384 |
| Per-run time (nanochat $100 config, 1 epoch SFT) | ~30 sec - 2 min |
| Total H100 hours (conservative) | 100-500 hours |
| Cost at $3/H100-hour | $300 - $1,500 |

**Optimization strategies:**
1. Use smaller model (124M) for initial exploration → ~10x faster
2. Reduce training tokens per run (1K-10K) → ~10x faster
3. Run at 32×32 resolution first, zoom into interesting regions
4. Parallelize across 8×H100 node efficiently

**Aggressive estimate with optimizations:**
- 124M model, 1K tokens per run, 32×32 initial resolution
- ~2 sec per run × 1024 runs = ~35 minutes for initial grid
- Zoom 4× into interesting regions: additional ~2 hours
- **Total: 3-4 hours on 8×H100 (~$30)**

---

## Out-of-Distribution (OOD) Datasets for Evaluation

The key is finding datasets that measure **generalization quality** rather than just training loss.

### Recommended OOD Benchmarks

| Dataset | Type | Why It's Good for This |
|---------|------|----------------------|
| **HellaSwag** | Commonsense completion | Tests reasoning transfer, ~70K questions |
| **ARC-Challenge** | Grade-school science | Tests knowledge generalization |
| **MMLU** | Multi-domain knowledge | 57 subjects, good breadth measure |
| **TruthfulQA** | Factual accuracy | Measures hallucination tendency |
| **GSM8K** | Math reasoning | Tests logical generalization |

### For Fine-Tuning Specifically

Consider training on one domain, evaluating on another:

| Train Domain | Eval Domain (OOD) | Measures |
|--------------|-------------------|----------|
| General chat (ShareGPT) | Code (HumanEval) | Code transfer |
| Code | Math (GSM8K) | Reasoning transfer |
| Math | General QA (MMLU) | Knowledge retention |
| Instruction following | Safety (ToxiGen) | Alignment stability |

### Practical Choices

**For speed (recommended for 16K+ runs):**
- Subset of HellaSwag (1K samples) - fast to evaluate
- Validation perplexity on held-out domain

**For quality:**
- Full HellaSwag + ARC-Challenge
- Multiple checkpoints per run to track dynamics

---

## What We Might Discover

### Potential "Rules of Thumb" for Fine-Tuning

1. **Critical Learning Rate Threshold**
   - At what LR does the trainability boundary become fractal?
   - Is there a "safe zone" below which fine-tuning always works?

2. **Dataset Size Phase Transitions**
   - Minimum viable dataset size as function of LR
   - Does the fractal dimension change with dataset size?

3. **Optimal Operating Point**
   - Best LR is at edge of stability → fractal analysis could reveal this
   - Potentially find that optimal LR = f(dataset_size, model_size)

4. **Early Warning Signs**
   - Gradient norm patterns that predict divergence
   - Loss trajectory signatures of "near-fractal-edge" training

### Connection to Existing Research

Recent papers ([arXiv:2412.13337](https://arxiv.org/html/2412.13337v1)) found:
- Larger batch + lower LR → better generalization
- Lower perplexity + moderate sequence length > sheer data volume
- Different abilities (math, code, general) have distinct scaling laws

**Our fractal analysis could explain WHY these patterns exist.**

---

## Recommended Experiment Plan

### Phase 1: Validation (1 day, ~$30)
1. Clone modded-nanogpt, get 124M pretrained checkpoint
2. Implement 32×32 LR × dataset_size grid
3. Train for 1K tokens each, measure final loss + HellaSwag subset
4. Visualize: do we see fractal-like boundaries?

### Phase 2: High-Resolution Scan (1-2 days, ~$100)
1. Zoom into interesting boundary regions at 128×128
2. Add OOD evaluation (full HellaSwag + ARC)
3. Compute box-counting fractal dimension

### Phase 3: Scaling Study (3-5 days, ~$300)
1. Repeat at 560M (nanochat d20) checkpoint
2. Compare fractal dimensions across model sizes
3. Look for universal patterns

### Phase 4: Practical Rules (1 week)
1. Derive empirical formulas: optimal_LR = f(dataset_size, model_size)
2. Test on new model/dataset combinations
3. Write up findings

---

## Current Implementation Status

### Infrastructure (COMPLETE)
- **Platform**: Modal (cloud GPU)
- **GPU**: H100 (single or 8×)
- **Profile**: `weightsandbiases`
- **Volume**: `fractal-llm-results` for persistent storage
- **Logging**: W&B → `morgan/fractal-llm`

### Code Structure (IMPLEMENTED)
```
fractal-llm/
├── src/
│   └── modal_app.py        # Modal H100 training + grid search
├── claude-research.md      # This file
├── CLAUDE.md               # Project guidelines
├── .env                    # Modal credentials (gitignored)
└── pyproject.toml          # Dependencies (uv)
```

### Key Functions in `modal_app.py`
| Function | Purpose |
|----------|---------|
| `train_single_run()` | Single SFT run on H100, returns loss + convergence |
| `run_grid_search()` | Orchestrates parallel grid search with W&B logging |
| `compute_fractal_dimension()` | Box-counting fractal dimension analysis |

### Current Configuration
- **Model**: GPT-2 124M (HuggingFace `gpt2`)
- **Dataset**: SmolTalk (`HuggingFaceTB/smoltalk`)
- **Sequence length**: 128 tokens
- **Grid axes**: Learning rate (1e-6 to 1e-3) × Training tokens (1K to 1M)

### Commands
```bash
# Test single run on H100
uv run modal run src/modal_app.py --test-only

# Run 32×32 grid search (~1K runs)
uv run modal run src/modal_app.py --resolution 32

# Run 128×128 grid search (~16K runs)
uv run modal run src/modal_app.py --resolution 128
```

---

## Fractal Visualization Approach

Following [Sohl-Dickstein et al.](https://github.com/Sohl-Dickstein/fractal):

### Color Scheme
- **Blue** = Converged training
- **Red** = Diverged training

### Intensity Mapping
- **Converged**: Intensity ∝ cumulative loss (Σ loss_t) - darker blue = higher loss before convergence
- **Diverged**: Intensity ∝ time before divergence (Σ 1/loss_t) - more intense red = longer low-loss period

### Plots Generated (logged to W&B)
1. **Trainability Boundary**: Fractal-style blue/red heatmap (main visualization)
2. **Loss Heatmap**: Final loss for converged runs only
3. **Binary Convergence**: Clean mask for fractal dimension calculation

### Box-Counting Fractal Dimension
- Computed on the binary convergence boundary
- Uses box sizes: [2, 4, 8, 16, 32]
- Dimension D estimated from log-log slope: N(ε) ∝ ε^(-D)

---

## Key Sources

### Fractal Training Dynamics
- [Neural network training makes beautiful fractals](https://sohl-dickstein.github.io/2024/02/12/fractal.html) - Sohl-Dickstein's blog
- [arXiv:2402.06184](https://arxiv.org/abs/2402.06184) - Original paper
- [arXiv:2501.04286](https://arxiv.org/html/2501.04286) - Extension to transformers
- [GitHub: Sohl-Dickstein/fractal](https://github.com/Sohl-Dickstein/fractal) - Original code

### NanoGPT/Nanochat
- [GitHub: karpathy/nanochat](https://github.com/karpathy/nanochat) - Full ChatGPT training
- [GitHub: KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) - Speed-optimized
- [Franz Cesista's 2.93 min record](https://x.com/leloykun/status/1885640350368420160)

### Fine-Tuning Research
- [arXiv:2412.13337](https://arxiv.org/html/2412.13337v1) - SFT recipe for small LLMs
- [arXiv:2506.14681](https://arxiv.org/html/2506.14681v1) - Massive SFT experiments
- [Neptune.ai: Hyperparameter Optimization for LLMs](https://neptune.ai/blog/hyperparameter-optimization-for-llms)

### OOD Evaluation
- [arXiv:2306.04618](https://arxiv.org/abs/2306.04618) - OOD robustness in NLP
- [ThinkBench](https://arxiv.org/html/2502.16268v1) - Dynamic OOD evaluation
- [GitHub: mlabonne/llm-datasets](https://github.com/mlabonne/llm-datasets) - Curated datasets

---

## Time Estimates Summary

| Phase | Duration | Cost | Status |
|-------|----------|------|--------|
| Setup + validation | 1 day | ~$30 | ✅ COMPLETE |
| 32×32 grid search | ~1 hour | ~$10 | Ready to run |
| 128×128 high-res | 1-2 days | ~$100 | Ready to run |
| Scale to 560M | 3-5 days | ~$300 | Planned |
| Rule derivation | 1 week | - | Planned |

**Infrastructure is ready.** Next step: run `uv run modal run src/modal_app.py --resolution 32`

**Quick proof of concept**: ~1 hour, ~$10 using GPT-2 124M at 32×32 resolution.
