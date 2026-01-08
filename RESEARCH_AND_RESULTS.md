# Fractal-LLM Research Results

Investigation of fractal trainability boundaries in LLM fine-tuning, inspired by [Sohl-Dickstein et al. (2024)](https://arxiv.org/abs/2402.06184).

## Experimental Setup

| Parameter | Value |
|-----------|-------|
| **Model** | nanochat-d20 (561M params) |
| **Task** | SFT on DocVQA |
| **Grid Resolution** | 64×64 (4,096 configurations) |
| **Axes Swept** | Matrix LR × Unembedding LR |
| **Tokens/Run** | 250,000 |
| **Optimizer** | AdamW |
| **Trainable Params** | matrix + unembedding layers only |

---

## Individual Grid Results

### 1. With Cosine Annealing (Default Schedule)

| Metric | Value |
|--------|-------|
| **Converged Ratio** | 77.5% (3,174/4,096) |
| **Fractal Dimension** | 1.21 |
| **Stable Ratio** | ~100% |
| **Boundary Pixels** | 333 |

**W&B:** [morgy/fractal-llm/runs/3acujwrk](https://wandb.ai/morgy/fractal-llm/runs/3acujwrk)
**Run Name:** `64x64-250k-ALL-ft-grid-summary`
**Sweep ID:** `64x64-250k-ALL-DEVPODS`

---

### 2. No Annealing (Constant LR)

| Metric | Value |
|--------|-------|
| **Converged Ratio** | 72.5% (2,969/4,096) |
| **Fractal Dimension** | 1.20 |
| **Stable Ratio** | 100% |
| **Boundary Pixels** | 322 |

**W&B:** [morgy/fractal-llm/runs/fm4jhhbs](https://wandb.ai/morgy/fractal-llm/runs/fm4jhhbs)
**Run Name:** `64x64-no-anneal-250k-20260105_151802-ft-grid-summary`
**Sweep ID:** `64x64-no-anneal-250k-20260105_151802-ft`

---

## Comparison: Annealing vs No Annealing

| Metric | With Annealing | No Annealing | Δ |
|--------|----------------|--------------|---|
| Converged Ratio | 77.5% | 72.5% | **+5.0%** |
| Fractal Dimension | 1.21 | 1.20 | +0.01 |
| Stable Ratio | ~100% | 100% | — |

**W&B Comparison Run:** [morgy/fractal-llm/runs/dbpdfjdf](https://wandb.ai/morgy/fractal-llm/runs/dbpdfjdf)
**Run Name:** `64x64-anneal-vs-constant-lr-summary_analysis`

### Key Finding

**Cosine annealing improves trainability by ~5% without altering fractal geometry.** The boundary shape (D ≈ 1.2) is invariant to LR schedule—suggesting the fractal structure is determined by model architecture and loss landscape, not optimization dynamics. Annealing rescues configurations near the boundary that would otherwise diverge late in training.

---

## Comparison with Original Research

### Sohl-Dickstein et al. Fractal Dimensions

| Configuration | Fractal Dim |
|--------------|-------------|
| Deep linear (full batch) | 1.17 |
| ReLU (full batch) | 1.20 |
| tanh (dataset size=1) | 1.41 |
| tanh (minibatch) | 1.55 |
| tanh (full batch) | 1.66 |
| Parameter init offset | 1.98 |

*Source: [arXiv:2402.06184](https://arxiv.org/abs/2402.06184), Table 1*

### Our Results in Context

Our observed **D ≈ 1.20** aligns closely with Sohl-Dickstein's ReLU full-batch result (1.20) and deep linear networks (1.17). This is notably **lower** than their tanh experiments (1.41–1.66).

**Possible explanations:**

1. **Adaptive optimizer effect**: We use AdamW vs. their steepest gradient descent. Adam's momentum and adaptive learning rates may smooth the trainability boundary, reducing fractal complexity.

2. **Model scale**: Our 561M-param transformer vs. their 2-layer FCN with 16 hidden units. Higher-dimensional parameter spaces may constrain boundary geometry.

3. **Pre-training**: Fine-tuning a pre-trained model starts in a "good" basin, potentially simplifying the boundary vs. training from scratch.

4. **Activation function**: Transformers use GeLU/SiLU activations, which may behave more like ReLU (D=1.20) than tanh (D=1.66) in terms of boundary complexity.

### Fractality Confirmed

Despite the lower dimension, we observe **clear fractal structure** at 64×64 resolution. The boundary between trainable (blue) and non-trainable (red) regions shows characteristic jagged, self-similar patterns—consistent with the original work's finding that "fractal boundaries persist across more than ten decades of scale."

---

## References

- **Paper:** Sohl-Dickstein, J. (2024). *The boundary of neural network trainability is fractal.* [arXiv:2402.06184](https://arxiv.org/abs/2402.06184)
- **Blog:** [Neural network training makes beautiful fractals](https://sohl-dickstein.github.io/2024/02/12/fractal.html)
- **Original Code:** [github.com/Sohl-Dickstein/fractal](https://github.com/Sohl-Dickstein/fractal)
- **Colab:** [the_boundary_of_neural_network_trainability_is_fractal.ipynb](https://colab.research.google.com/github/Sohl-Dickstein/fractal/blob/main/the_boundary_of_neural_network_trainability_is_fractal.ipynb)

---

## W&B Run Index

| Run | ID | Link |
|-----|-----|------|
| With Annealing Grid | `3acujwrk` | [View](https://wandb.ai/morgy/fractal-llm/runs/3acujwrk) |
| No Annealing Grid | `fm4jhhbs` | [View](https://wandb.ai/morgy/fractal-llm/runs/fm4jhhbs) |
| Comparison Analysis | `dbpdfjdf` | [View](https://wandb.ai/morgy/fractal-llm/runs/dbpdfjdf) |
