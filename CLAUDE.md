# Project: fractal-LLM

This project explores the fractal nature of training dynamics in neural networks, specifically LLMs. Inspired by Jascha Sohl-Dickstein's work showing that the boundary between convergent and divergent training is fractal.

**Key References:**
- Blog: https://sohl-dickstein.github.io/2024/02/12/fractal.html
- Paper: https://arxiv.org/abs/2402.06184

## Project Goal

Run hyperparameter grid searches over LR × dataset size (and other axes) during SFT to visualize fractal trainability boundaries. Goal is to discover practical rules of thumb for fine-tuning LLMs.

## Infrastructure

### Modal (Cloud GPU) - VERIFIED WORKING
- **Profile**: `weightsandbiases`
- **GPU**: H100 (single or up to 8× per node)
- **Volume**: `fractal-llm-results` for persistent storage
- **Image**: `nvidia/cuda:12.8.0-devel-ubuntu22.04` (Torch 2.8.0+cu128 via uv pip; rich + python-dotenv included)
- **Model**: nanochat-d20 (561M) from W&B artifact `morgan/fractal-llm/nanochat-d20-speedrun:latest`

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

# Build SmolTalk shards (optional, faster startup)
uv run data/prepare_smoltalk.py --budgets 1000 10000 100000 1000000 --out-dir /results/smoltalk_shards

# OOD eval snapshot (HellaSwag + ARC)
uv run eval/run_lmeval.py --model nanochat-students/nanochat-d20 --tasks hellaswag,arc_challenge --max-samples 500

# Create Modal env + token + secret (one time)
uv run modal environment create fractal-llm
uv run modal token set --token-id $MODAL_TOKEN_ID --token-secret $MODAL_TOKEN_SECRET --profile=weightsandbiases
uv run modal secret create --env fractal-llm wandb-secret WANDB_API_KEY="$WANDB_API_KEY"

# Run nanochat d20 speedrun on 8×H100 (logs artifact to W&B)
MODAL_ENVIRONMENT=fractal-llm uv run modal run src/nanochat_modal.py \
  --wandb-name nanochat-d20-modal \
  --save-artifact-name nanochat-d20-speedrun
# (Runs remotely; safe to close laptop after launch)
```

**Test Results (verified 2024-12-29):**
- H100 training: 9 steps in 1.27s
- Loss: 3.39 → 3.00 (converged)
- Dataset: SmolTalk (HuggingFaceTB/smoltalk)

### Project Structure
```
fractal-llm/
├── src/
│   └── modal_app.py     # Modal H100 training functions
├── claude-research.md   # Research notes and experiment design
├── .env                 # Modal credentials (gitignored)
└── pyproject.toml       # Dependencies (uv)
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
When logging to `wandb` or `weave` from Weights & Biases, always log to the `morgan` entity and the `fractal-llm` project, unless specifically asked to log elsewhere

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
