# Fractal LLM Fine-Tuning: Final Research Plan

## Executive Summary

This experiment explores the **fractal structure of SFT trainability boundaries** using nanochat on a single H100. We'll create a 128×128 grid varying **learning rate × dataset size**, visualize the convergence/divergence boundary, and compute its fractal dimension. The goal is to discover practical rules of thumb for LLM fine-tuning.

**Total estimated time**: 4-8 hours on 1×H100 (~$25-50)
**Expected output**: Fractal visualization + empirical formula for optimal LR = f(dataset_size)

---

## 1. Background & Motivation

The [Sohl-Dickstein fractal training paper](https://arxiv.org/abs/2402.06184) shows that the boundary between convergent and divergent training is **fractal** - small hyperparameter changes cause large, unpredictable training outcome changes at all scales down to floating-point precision.

Key insight: **Optimal hyperparameters are near the edge of stability** - the largest learning rate that still converges. This fractal boundary explains why hyperparameter tuning is so finicky.

Recent work ([arXiv:2501.04286](https://arxiv.org/html/2501.04286)) extended this to transformers, finding fractal dimensions of 1.54-1.98 when varying attention vs FC layer learning rates.

**Our contribution**: Apply this analysis to **LLM fine-tuning** specifically, mapping the LR × dataset_size boundary to derive practical guidelines.

---

## 2. Experimental Setup

### 2.1 Model & Checkpoint

**Primary choice**: [nanochat-students/nanochat-d20](https://huggingface.co/nanochat-students/nanochat-d20) (561M params)
- Pre-trained on FineWeb-EDU, mid-trained on MMLU/GSM8K
- Already instruction-tuned, perfect for studying SFT dynamics
- Available on HuggingFace for easy download  

*(We standardize on nanochat-d20 for all runs.)*

### 2.2 Hardware & Runtime

| Config | Per-run tokens | Est. time/run | 128×128 grid total |
|--------|---------------|---------------|---------------------|
| Fast sweep | 1K tokens | ~2 sec | ~9 hours |
| **Standard** | 10K tokens | ~15 sec | ~9 hours |
| Thorough | 100K tokens | ~2 min | ~72 hours |

**Recommendation**: Run full 128×128 grid with 10K tokens/run (~40 hours). Can potentially identify multiple zoom-worthy boundary regions for even higher resolution analysis.

### 2.3 Hyperparameter Axes

**Primary 2D Grid (LR × Dataset Size)**:

| Axis | Range | Points | Rationale |
|------|-------|--------|-----------|
| **Learning Rate** | 1e-6 to 1e-3 | 128 (log scale) | Primary instability driver |
| **Dataset Size** | 1K to 1M tokens | 128 (log scale) | Affects convergence basin |

Implementation detail: each run uses `steps = ceil(tokens / (batch_size * seq_len))` so the requested token budget is actually consumed (was previously capped).
**Alternative grids to explore**:
1. **LR × Batch Size**: batch_tokens from 4K to 128K
2. **Attention LR × FC LR**: Following [arXiv:2501.04286](https://arxiv.org/html/2501.04286)
3. **LR × Warmup Ratio**: 0% to 10% of steps

### 2.4 SFT Dataset

**Training data**: [DocVQA](https://huggingface.co/datasets/morgan/docvqa-nanochat) document QA dataset
- 39,455 training samples, ~15.5M tokens
- Streamed from HuggingFace Hub (morgan/docvqa-nanochat)

**Held-out validation**: 5,349 validation samples from DocVQA

### 2.5 Convergence Classification

For each (LR, dataset_size) pair, classify as:
- **Converged**: Final loss < threshold AND no NaN/Inf
- **Diverged**: Loss exploded (NaN/Inf) or final loss > 10.0
- **Marginal**: Near-threshold behavior (for boundary identification)

**Primary metric**: Final training loss after fixed steps

### 2.6 Reproducibility

All training runs are fully reproducible with fixed seeds, following [Stas Bekman's ML Engineering guide](https://github.com/stas00/ml-engineering):

- **Global seed**: 42 (default, configurable)
- **RNG sources controlled**: Python `random`, NumPy, PyTorch CPU/CUDA
- **Deterministic CUDA**: `torch.use_deterministic_algorithms(True)` with `CUBLAS_WORKSPACE_CONFIG=:16:8`
- **Data ordering**: Fixed shuffle seed in dataset loading
- **Trainer seed**: HuggingFace `seed` and `data_seed` parameters

**Verification**: Run the same grid point twice and confirm identical loss trajectories. The 5-15% performance cost of deterministic mode is acceptable for research reproducibility.

---

## 3. OOD Evaluation

For a subset of grid points (e.g., 10×10 across the converged region), evaluate:

| Benchmark | Samples | Context | Purpose |
|-----------|---------|---------|---------|
| **HellaSwag** (subset) | 1000 | ~40 tokens | Fast commonsense eval |
| **ARC-Challenge** | 1172 | ~60 tokens | Reasoning transfer |
| **PIQA** | 1838 | ~40 tokens | Physical intuition |
| **Winogrande** | 1267 | ~30 tokens | Coreference resolution |

Using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for standardized evaluation.

**Goal**: Does OOD performance correlate with distance from fractal boundary?

---

## 4. Fractal Dimension Calculation

### 4.1 Box-Counting Method

Following [Francesco Turci's NumPy implementation](https://francescoturci.net/2016/03/31/box-counting-in-numpy/):

```python
def box_counting_dimension(binary_image):
    """Compute fractal dimension via box counting."""
    sizes = 2 ** np.arange(1, 8)  # Box sizes: 2, 4, 8, ..., 128
    counts = []
    for size in sizes:
        # Count boxes containing the boundary
        boxes = binary_image.reshape(
            binary_image.shape[0] // size, size,
            binary_image.shape[1] // size, size
        ).any(axis=(1, 3)).sum()
        counts.append(boxes)

    # Fractal dimension = -slope of log(count) vs log(size)
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]
```

### 4.2 Expected Results

- **Fractal dimension 1.0**: Smooth boundary (trivial)
- **Fractal dimension 1.5-2.0**: Fractal boundary (expected based on prior work)
- **Higher roughness** at smaller LR × smaller dataset regions

---

## 5. Implementation Plan

**Phase 1: Setup (fast)**
1. Confirm Modal profile `weightsandbiases`, secret `wandb-secret`.
2. `uv run modal run src/modal_app.py --test-only` (nanochat-d20, H100).

**Phase 2: Pilot Sweep (32×32)**  
1. `uv run modal run src/modal_app.py --resolution 32` (tokens respected via steps=ceil(tokens/(bs*seq_len))).  
2. Visualize with `src/visualize.py`.

**Phase 3: High-Res Sweep (128×128)**  
1. `uv run modal run src/modal_app.py --resolution 128`  
2. Checkpointing auto-resumes in `/results/checkpoint_128x128.json`.

**Phase 4: Fractal Analysis + OOD**  
1. `compute_fractal_dimension` auto-runs after the grid.  
2. Run `uv run eval/run_lmeval.py --model nanochat-students/nanochat-d20 --tasks hellaswag,arc_challenge --max-samples 500` for OOD snapshot.

**Phase 5: Rules of Thumb**  
1. Fit `optimal_LR = f(dataset_size)` from converged runs.  
2. Report safe-zone and boundary proximity insights.

---

## 6. Code Structure

```
fractal-llm/
├── src/
│   ├── modal_app.py      # Modal grid search + fractal viz + W&B logging
│   └── visualize.py      # Offline visualizations
├── data/
│   ├── prepare_docvqa.py        # Process DocVQA from source
│   └── download_docvqa_hub.py   # Download DocVQA from HF Hub
├── eval/
│   └── run_lmeval.py     # lm-eval-harness wrapper (HellaSwag/ARC)
├── results/              # Figures, grids (created at runtime)
└── research.md           # This file
```

---

## 7. Expected Discoveries

### 7.1 Quantitative

- **Fractal dimension** of LR × dataset_size boundary (expect 1.5-1.9)
- **Critical LR threshold**: LR_max(dataset_size) before divergence
- **Minimum viable dataset**: tokens_min(LR) for stable training

### 7.2 Practical Rules of Thumb

1. **Safe LR formula**: `LR_safe = 0.5 × LR_critical(dataset_size)`
2. **Dataset scaling**: Does 10× more data allow 2× higher LR?
3. **Early warning signs**: Loss trajectory patterns near boundary

### 7.3 Potential Paper Contributions

- First fractal analysis of LLM SFT dynamics
- Empirical formula for optimal_LR = f(dataset_size, model_size)
- Explanation for why fine-tuning is so hyperparameter-sensitive

---

## 8. Resource Estimates

| Phase | H100 Hours | Cost ($3/hr) | Output |
|-------|------------|--------------|--------|
| Setup | 0.5 | $1.50 | Working pipeline |
| 128×128 grid | 40 | $120.00 | Full fractal visualization |
| Fractal analysis | 1 | $3.00 | Dimension calculation |
| OOD eval | 4 | $12.00 | Performance correlation |
| **Total** | **~46** | **~$140** | Full analysis |

**Note**: 40 hours assumes sequential runs. With efficient batching/parallelization on H100, actual wall-clock could be reduced.

---

## 9. Key Dependencies

- [nanochat](https://github.com/karpathy/nanochat) - Training framework
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) - OOD benchmarks
- [wandb](https://wandb.ai) - Experiment tracking (entity: milieu, project: radio_analysis)
- NumPy/Pandas - Analysis
- Matplotlib/Seaborn - Visualization

---

## 10. Success Criteria

1. **Visual**: Clear fractal-like boundary in convergence heatmap
2. **Quantitative**: Fractal dimension between 1.3 and 2.0
3. **Practical**: Derived formula predicts optimal LR within 2× for new dataset sizes
4. **OOD insight**: Discovered correlation between boundary proximity and generalization

---

## References

### Fractal Training
- [Sohl-Dickstein Blog](https://sohl-dickstein.github.io/2024/02/12/fractal.html)
- [arXiv:2402.06184](https://arxiv.org/abs/2402.06184) - Original paper
- [arXiv:2501.04286](https://arxiv.org/html/2501.04286) - Transformer extension
- [arXiv:2406.13971](https://arxiv.org/abs/2406.13971) - Fractal boundaries from non-convexity

### NanoGPT/Nanochat
- [nanochat GitHub](https://github.com/karpathy/nanochat)
- [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) - Speed records
- [nanochat-d20 checkpoint](https://huggingface.co/nanochat-students/nanochat-d20)

### Box-Counting
- [Francesco Turci's NumPy implementation](https://francescoturci.net/2016/03/31/box-counting-in-numpy/)
- [PoreSpy fractal dimension docs](https://porespy.org/examples/metrics/tutorials/computing_fractal_dim.html)

### Evaluation
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
