# Repository Snapshot: fractal-llm

> **Generated:** 2026-01-05 14:06:22
>
> **Description:** Exploring fractal training dynamics in LLMs

This file contains the complete source code of the repository, formatted for LLM consumption.

## Repository Structure

```
./
  .gitignore
  .python-version
  CLAUDE.md
  LICENSE
  README.md
  main.py
  pyproject.toml
eval/
  run_lmeval.py
scripts/
  chat_devpod.sh
  grid_sweep.sh
src/
  eval_samples.py
  finetune.py
  grid_sweep_summary.py
  visualize.py
    third_party/nanochat/nanochat/
      adamw.py
      gpt.py
      muon.py
      tokenizer.py
```

---

## File Contents


### `.gitignore`

```
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[codz]
*$py.class

.DS_Store

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
#  Usually these files are written by a python script from a template
#  before PyInstaller builds the exe, so as to inject date/other infos into it.
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py.cover
.hypothesis/
.pytest_cache/
cover/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
.pybuilder/
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
#   For a library or package, you might want to ignore these files since the code is
#   intended to run in multiple environments; otherwise, check them in:
# .python-version

# pipenv
#   According to pypa/pipenv#598, it is recommended to include Pipfile.lock in version control.
#   However, in case of collaboration, if having platform-specific dependencies or dependencies
#   having no cross-platform support, pipenv may install dependencies that don't work, or not
#   install all needed dependencies.
#Pipfile.lock

# UV
#   Similar to Pipfile.lock, it is generally recommended to include uv.lock in version control.
#   This is especially recommended for binary packages to ensure reproducibility, and is more
#   commonly ignored for libraries.
#uv.lock

# poetry
#   Similar to Pipfile.lock, it is generally recommended to include poetry.lock in version control.
#   This is especially recommended for binary packages to ensure reproducibility, and is more
#   commonly ignored for libraries.
#   https://python-poetry.org/docs/basic-usage/#commit-your-poetrylock-file-to-version-control
#poetry.lock
#poetry.toml

# pdm
#   Similar to Pipfile.lock, it is generally recommended to include pdm.lock in version control.
#   pdm recommends including project-wide configuration in pdm.toml, but excluding .pdm-python.
#   https://pdm-project.org/en/latest/usage/project/#working-with-version-control
#pdm.lock
#pdm.toml
.pdm-python
.pdm-build/

# pixi
#   Similar to Pipfile.lock, it is generally recommended to include pixi.lock in version control.
#pixi.lock
#   Pixi creates a virtual environment in the .pixi directory, just like venv module creates one
#   in the .venv directory. It is recommended not to include this directory in version control.
.pixi

# PEP 582; used by e.g. github.com/David-OConnor/pyflow and github.com/pdm-project/pdm
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.envrc
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# pytype static type analyzer
.pytype/

# Cython debug symbols
cython_debug/

# PyCharm
#  JetBrains specific template is maintained in a separate JetBrains.gitignore that can
#  be found at https://github.com/github/gitignore/blob/main/Global/JetBrains.gitignore
#  and can be added to the global gitignore or merged into this file.  For a more nuclear
#  option (not recommended) you can uncomment the following to ignore the entire idea folder.
#.idea/

# Abstra
# Abstra is an AI-powered process automation framework.
# Ignore directories containing user credentials, local state, and settings.
# Learn more at https://abstra.io/docs
.abstra/

# Visual Studio Code
#  Visual Studio Code specific template is maintained in a separate VisualStudioCode.gitignore 
#  that can be found at https://github.com/github/gitignore/blob/main/Global/VisualStudioCode.gitignore
#  and can be added to the global gitignore or merged into this file. However, if you prefer, 
#  you could uncomment the following to ignore the entire vscode folder
# .vscode/

# Ruff stuff:
.ruff_cache/

# PyPI configuration file
.pypirc

# Cursor
#  Cursor is an AI-powered code editor. `.cursorignore` specifies files/directories to
#  exclude from AI features like autocomplete and code analysis. Recommended for sensitive data
#  refer to https://docs.cursor.com/context/ignore-files
.cursorignore
.cursorindexingignore

# Marimo
marimo/_static/
marimo/_lsp/
__marimo__/

# Project-specific
results/
data/
*.pkl
*.pt

# Modal
.modal/

```


### `.python-version`

```
3.12

```


### `CLAUDE.md`

```markdown
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
uv run python -m src.finetune --run devpod-default --eval_every 0 --log_every 1 --save_artifacts False
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

```


### `LICENSE`

```
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

```


### `README.md`

```markdown
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

## Chat UI (Web)

`src/chat.py` is a small wrapper around nanochat’s `third_party/nanochat/scripts/chat_web.py` that can load a model + tokenizer from a W&B artifact and serve the web chat UI/API.

**Run locally (laptop)**
```bash
cd /path/to/fractal-llm && source .env  # or export WANDB_API_KEY=...
uv run src/chat.py --model wandb:morgy/fractal-llm/nanochat-fin-rl-artifact:v7 --host 127.0.0.1 --port 8000
```
Open `http://localhost:8000`.

**Run Karpathy’s HF d32 checkpoint**
```bash
uv run src/chat.py --model hf:karpathy/nanochat-d32 --host 127.0.0.1 --port 8000
```

**Run on DevPod (GPU)**
```bash
ssh fractal-llm.devpod
cd /workspaces/fractal-llm && source .env
./scripts/chat_devpod.sh \
  --model wandb:morgy/fractal-llm/nanochat-fin-rl-artifact:v7 \
  --port 8000 --num-gpus 1
```

**Connect from your laptop**
```bash
ssh -L 8000:localhost:8000 fractal-llm.devpod
```
Then open `http://localhost:8000` in your browser.

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

```


### `eval/run_lmeval.py`

```python
"""
Run lm-evaluation-harness on a given model/checkpoint for quick OOD evaluation.

Example:
    uv run eval/run_lmeval.py --model nanochat-students/nanochat-d20 --tasks hellaswag,arc_challenge
"""

from dataclasses import dataclass
import simple_parsing as sp
from rich.console import Console
from rich.panel import Panel
from lm_eval import evaluator

console = Console()


@dataclass
class Args:
    """Evaluate a model with lm-evaluation-harness."""

    model: str  # HF model ID or local path
    tasks: str = "hellaswag,arc_challenge"  # Comma-separated task list
    batch_size: int = 8  # LM Eval batch size
    max_samples: int | None = 500  # Optional cap per task for speed
    device: str = "cuda"  # cuda or cpu


def main(args: Args):
    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    console.rule("[bold blue]lm-eval-harness")
    console.print(Panel(f"Model: {args.model}\nTasks: {task_list}\nBatch: {args.batch_size}", title="Config"))

    results = evaluator.simple_evaluate(
        model="hf",
        model_args=f"pretrained={args.model},device={args.device}",
        tasks=task_list,
        batch_size=args.batch_size,
        limit=args.max_samples,
    )

    console.print(Panel(str(results["results"]), title="Scores"))
    console.print("[green]Complete.")


if __name__ == "__main__":
    main(sp.parse(Args))

```


### `main.py`

```python
def main():
    print("Hello from fractal-llm!")


if __name__ == "__main__":
    main()

```


### `pyproject.toml`

```toml
[project]
name = "fractal-llm"
version = "0.1.0"
description = "Fractal analysis of LLM fine-tuning trainability boundaries"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.117.1",
    "matplotlib>=3.9.0",
    "modal>=1.3.0.post1",
    "numpy>=2.0.0",
    "rich>=14.0.0",
    "simple-parsing>=0.1.7",
    "tqdm>=4.67.0",
    "pandas>=2.2.0",
    "seaborn>=0.13.0",
    "scipy>=1.14.0",
    "torch==2.8.0",
    "torchvision==0.23.0",
    "torchaudio==2.8.0",
    "tiktoken>=0.11.0",
    "tokenizers>=0.22.0",
    "transformers>=4.47.0",
    "datasets>=3.2.0",
    "accelerate>=1.2.0",
    "uvicorn>=0.36.0",
    "wandb>=0.23.1",
    "huggingface-hub>=0.27.0",
    "lm-eval==0.4.2",
    "python-dotenv>=1.0.1",
]

[tool.uv]
# Use PyTorch cu128 index for torch 2.8.0 wheels
# Use unsafe-best-match to prefer PyPI for non-torch packages
extra-index-url = ["https://download.pytorch.org/whl/cu128"]
index-strategy = "unsafe-best-match"

```


### `scripts/chat_devpod.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# DevPod workspace disk is tiny; default to /var/tmp on remote.
export FRACTAL_STORAGE_DIR="${FRACTAL_STORAGE_DIR:-/var/tmp/fractal-llm}"

exec uv run src/chat.py "$@"


```


### `scripts/grid_sweep.sh`

```bash
#!/usr/bin/env bash
# Parallel single-GPU grid sweep launcher for fractal grids.
# One worker per GPU runs its assigned points sequentially (no GPU oversubscription).
# Logs per-run output and summarizes final losses.

set -euo pipefail
[[ ${DEBUG_GRID:-0} -eq 1 ]] && set -x

RUN_PREFIX=${RUN_PREFIX:-grid-$(date +%Y%m%d_%H%M%S)}
WANDB_RUN_PREFIX=${WANDB_RUN_PREFIX:-}
if [[ -n "${WANDB_RUN_PREFIX}" ]]; then
  RUN_PREFIX="${WANDB_RUN_PREFIX}"
fi
GRID_SWEEP_ID=${GRID_SWEEP_ID:-${RUN_PREFIX}}  # constant tag across all points in this sweep
SWEEP_AXES=${SWEEP_AXES:-matrix_unembedding}  # matrix_unembedding | lr_tokens
RES=${RES:-4}                        # grid resolution per axis (RES x RES points)
LR_MIN=${LR_MIN:-1e-6}
LR_MAX=${LR_MAX:-3e-1}
TOK_MIN=${TOK_MIN:-5e3}
TOK_MAX=${TOK_MAX:-5e5}
MATRIX_LR_MIN=${MATRIX_LR_MIN:-1e-6}
MATRIX_LR_MAX=${MATRIX_LR_MAX:-3e-1}
UNEMBEDDING_LR_MIN=${UNEMBEDDING_LR_MIN:-1e-6}
UNEMBEDDING_LR_MAX=${UNEMBEDDING_LR_MAX:-1e-1}
GPU_IDS_STR=${GPUS-}
FRACTAL_STORAGE_DIR=${FRACTAL_STORAGE_DIR:-/var/tmp/fractal-llm}
LOG_DIR=${LOG_DIR:-${FRACTAL_STORAGE_DIR}/results/grid_logs/${RUN_PREFIX}}
MODEL_ID=${MODEL_ID:-${MODEL_ARTIFACT:-}}
DATASET_ID=${DATASET_ID:-}
DATASET_REVISION=${DATASET_REVISION:-}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-1024}
TOKENS_PER_RUN=${TOKENS_PER_RUN:-}  # optional override: fixed num_tokens instead of TOK_MIN..MAX grid
LR_FIXED=${LR_FIXED:-}              # optional override: fixed learning_rate instead of LR_MIN..MAX grid
SEED=${SEED:-999}
WANDB_PROJECT=${WANDB_PROJECT:-fractal-llm}
WANDB_ENTITY=${WANDB_ENTITY:-morgy}
FINETUNE_WANDB_TAGS=${FINETUNE_WANDB_TAGS:-fractal-grid}
TRAINABLE_PARAM_GROUPS=${TRAINABLE_PARAM_GROUPS:-matrix,unembedding}  # passed to src.finetune --trainable_param_groups
MAX_RETRIES=${MAX_RETRIES:-3}       # per-point retries for transient failures (total attempts = 1 + MAX_RETRIES)
RETRY_BACKOFF_S=${RETRY_BACKOFF_S:-5}  # base backoff seconds (exponential-ish via multiplier)
RETRY_PATTERNS=${RETRY_PATTERNS:-"ReadTimeout|Read timed out|ConnectionPool|ConnectionError|Temporary failure in name resolution|502|503|504"}
SKIP_COMPLETED=${SKIP_COMPLETED:-1} # if 1, skip points whose log already contains a Final: line
LOG_SUMMARY=${LOG_SUMMARY:-1}       # if 1, log a W&B grid-summary run at the end (single-pod/local only by default)

# Multi-devpod orchestration (run this script locally with DEVPODS set).
# Example:
#   DEVPODS="fractal-llm-1 fractal-llm-2 fractal-llm-3" RES=16 TOKENS_PER_RUN=250000 \
#   SWEEP_AXES=matrix_unembedding MATRIX_LR_MIN=1e-6 MATRIX_LR_MAX=3e-1 \
#   UNEMBEDDING_LR_MIN=1e-6 UNEMBEDDING_LR_MAX=1e-1 \
#   ./scripts/grid_sweep.sh
DEVPODS_STR=${DEVPODS:-}            # space/comma-separated devpod workspace names (e.g., "fractal-llm-1 fractal-llm-2")
GRID_SWEEP_ROLE=${GRID_SWEEP_ROLE:-}  # orchestrator|worker (internal)
POD_INDEX=${POD_INDEX:-0}           # worker pod shard index in [0, NUM_PODS)
NUM_PODS=${NUM_PODS:-1}             # number of pods participating in the sweep
DEVPOD_NAME=${DEVPOD_NAME:-}        # optional label for logs/tags
DEVPOD_WORKDIR=${DEVPOD_WORKDIR:-}  # if unset in orchestrator mode, defaults per pod to /workspaces/<devpod-name>
DEVPOD_TMUX_SESSION=${DEVPOD_TMUX_SESSION:-grid_${RUN_PREFIX}}
AUTO_PULL=${AUTO_PULL:-1}           # if 1, run git pull --ff-only on each devpod before starting
AUTO_RESET=${AUTO_RESET:-1}         # if 1, run git reset --hard HEAD after pulling (restores missing tracked files)
AUTO_UV_SYNC=${AUTO_UV_SYNC:-1}     # if 1, run uv sync --frozen on each devpod before starting
WAIT_FOR_COMPLETION=${WAIT_FOR_COMPLETION:-0}  # deprecated: multi-devpod always waits + summarizes
POLL_INTERVAL_S=${POLL_INTERVAL_S:-30}
COLLECT_LOGS=${COLLECT_LOGS:-0}     # deprecated: multi-devpod always collects logs + summarizes
SUMMARY_AFTER_COLLECT=${SUMMARY_AFTER_COLLECT:-0}  # deprecated: multi-devpod always collects logs + summarizes

_quote() { printf "%q" "$1"; }
_assign() {
  local key=$1
  local val=${2-}
  if [[ -z "${val}" ]]; then
    echo "${key}="
  else
    echo "${key}=$(_quote "${val}")"
  fi
}

if [[ "${SWEEP_AXES}" == "matrix_unembedding" ]] && [[ -z "${TOKENS_PER_RUN}" ]]; then
  # In matrix_unembedding mode we sweep only the optimizer LRs; token budget stays fixed.
  # Default to the max token budget from the legacy lr_tokens axis.
  TOKENS_PER_RUN="${TOK_MAX}"
fi

if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "[grid] git pull --ff-only"
  git pull --ff-only
fi

if [[ -n "${DEVPODS_STR}" ]] && [[ "${GRID_SWEEP_ROLE}" != "worker" ]]; then
  # Multi-devpod orchestrator: always wait, collect logs, and log a single combined W&B summary run.
  WAIT_FOR_COMPLETION=1
  COLLECT_LOGS=1
  SUMMARY_AFTER_COLLECT=1

  if ! command -v devpod >/dev/null 2>&1; then
    echo "[grid] DEVPODS is set but devpod CLI not found in PATH" >&2
    exit 2
  fi

  # Normalize separators (commas -> spaces), then split.
  DEVPODS_STR="${DEVPODS_STR//,/ }"
  # shellcheck disable=SC2206
  DEVPODS=(${DEVPODS_STR})
  NUM_PODS=${#DEVPODS[@]}
  if [[ "${NUM_PODS}" -le 0 ]]; then
    echo "[grid] No devpods specified. Set DEVPODS=\"fractal-llm-1 fractal-llm-2 ...\"" >&2
    exit 2
  fi

  echo "[grid] multi-devpod orchestrator mode"
  echo "[grid] devpods: ${DEVPODS[*]}"
  echo "[grid] sweep_axes=${SWEEP_AXES} res=${RES} tokens_per_run=${TOKENS_PER_RUN:-<grid>}"
  echo "[grid] run_prefix=${RUN_PREFIX} sweep_id=${GRID_SWEEP_ID} tmux_session=${DEVPOD_TMUX_SESSION}"

  # Launch a worker tmux session on each devpod.
  pids=()
  for pod_idx in "${!DEVPODS[@]}"; do
    pod="${DEVPODS[$pod_idx]}"
    echo "[grid] launching worker ${pod_idx}/${NUM_PODS} on devpod=${pod}"
    pod_workdir="${DEVPOD_WORKDIR:-/workspaces/${pod}}"

    # Build env assignments for the tmux command (shell-escaped).
    env_assign=(
      "GRID_SWEEP_ROLE=worker"
      "DEVPODS="
      "$(_assign POD_INDEX "${pod_idx}")"
      "$(_assign NUM_PODS "${NUM_PODS}")"
      "$(_assign DEVPOD_NAME "${pod}")"
      "$(_assign DEVPOD_WORKDIR "${pod_workdir}")"
      "$(_assign DEVPOD_TMUX_SESSION "${DEVPOD_TMUX_SESSION}")"
      "$(_assign RUN_PREFIX "${RUN_PREFIX}")"
      "$(_assign GRID_SWEEP_ID "${GRID_SWEEP_ID}")"
      "$(_assign SWEEP_AXES "${SWEEP_AXES}")"
      "$(_assign RES "${RES}")"
      "$(_assign LR_MIN "${LR_MIN}")"
      "$(_assign LR_MAX "${LR_MAX}")"
      "$(_assign TOK_MIN "${TOK_MIN}")"
      "$(_assign TOK_MAX "${TOK_MAX}")"
      "$(_assign MATRIX_LR_MIN "${MATRIX_LR_MIN}")"
      "$(_assign MATRIX_LR_MAX "${MATRIX_LR_MAX}")"
      "$(_assign UNEMBEDDING_LR_MIN "${UNEMBEDDING_LR_MIN}")"
      "$(_assign UNEMBEDDING_LR_MAX "${UNEMBEDDING_LR_MAX}")"
      "$(_assign FRACTAL_STORAGE_DIR "${FRACTAL_STORAGE_DIR}")"
      "$(_assign LOG_DIR "${LOG_DIR}")"
      "$(_assign MAX_SEQ_LEN "${MAX_SEQ_LEN}")"
      "$(_assign TOKENS_PER_RUN "${TOKENS_PER_RUN}")"
      "$(_assign LR_FIXED "${LR_FIXED}")"
      "$(_assign SEED "${SEED}")"
      "$(_assign WANDB_PROJECT "${WANDB_PROJECT}")"
      "$(_assign WANDB_ENTITY "${WANDB_ENTITY}")"
      "$(_assign FINETUNE_WANDB_TAGS "${FINETUNE_WANDB_TAGS}")"
      "$(_assign TRAINABLE_PARAM_GROUPS "${TRAINABLE_PARAM_GROUPS}")"
      "$(_assign MAX_RETRIES "${MAX_RETRIES}")"
      "$(_assign RETRY_BACKOFF_S "${RETRY_BACKOFF_S}")"
      "$(_assign RETRY_PATTERNS "${RETRY_PATTERNS}")"
      "$(_assign SKIP_COMPLETED "${SKIP_COMPLETED}")"
      "LOG_SUMMARY=0"
    )
    [[ -n "${MODEL_ID}" ]] && env_assign+=("$(_assign MODEL_ID "${MODEL_ID}")")
    [[ -n "${DATASET_ID}" ]] && env_assign+=("$(_assign DATASET_ID "${DATASET_ID}")")
    [[ -n "${DATASET_REVISION}" ]] && env_assign+=("$(_assign DATASET_REVISION "${DATASET_REVISION}")")
    tmux_cmd="${env_assign[*]} ./scripts/grid_sweep.sh"

    devpod --silent ssh "${pod}" --command "bash -lc '
      set -euo pipefail
      cd ${pod_workdir}
      if [[ -f .env ]]; then source .env; fi
      if [[ ${AUTO_PULL} -eq 1 ]]; then
        git pull --ff-only
      fi
      if [[ ${AUTO_RESET} -eq 1 ]]; then
        git reset --hard HEAD || echo \"[grid] WARNING: git reset failed on ${pod}\"
      fi
      if [[ ${AUTO_UV_SYNC} -eq 1 ]]; then
        if ! uv sync --frozen; then
          echo \"[grid] WARNING: uv sync --frozen failed on ${pod}; removing .venv and retrying\"
          rm -rf .venv || true
          uv sync --frozen
        fi
      fi
      if tmux has-session -t ${DEVPOD_TMUX_SESSION} 2>/dev/null; then
        echo \"[grid] ERROR: tmux session already exists on ${pod}: ${DEVPOD_TMUX_SESSION}\" >&2
        exit 3
      fi
      tmux new-session -d -s ${DEVPOD_TMUX_SESSION} -c ${pod_workdir} \"${tmux_cmd}\"
      echo \"[grid] started ${pod}: tmux attach -t ${DEVPOD_TMUX_SESSION}\"
    '" &
    pids+=($!)
  done

  launch_rc=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      launch_rc=1
    fi
  done
  if [[ "${launch_rc}" -ne 0 ]]; then
    echo "[grid] ERROR: one or more devpods failed to launch" >&2
    exit 3
  fi

  echo "[grid] all devpods launched"
  echo "[grid] monitor:"
  for pod in "${DEVPODS[@]}"; do
    echo "  devpod ssh ${pod}   # then: tmux attach -t ${DEVPOD_TMUX_SESSION}"
  done

  if [[ "${WAIT_FOR_COMPLETION}" == "1" ]]; then
    echo "[grid] waiting for completion (poll=${POLL_INTERVAL_S}s)"
    while true; do
      alive=0
      for pod in "${DEVPODS[@]}"; do
        status="$(
          devpod --silent ssh "${pod}" \
            --command "bash -lc 'if tmux has-session -t ${DEVPOD_TMUX_SESSION} 2>/dev/null; then echo RUNNING; else echo DONE; fi'" \
            || echo ERROR
        )"
        if [[ "${status}" == *RUNNING* ]]; then
          alive=$((alive + 1))
        elif [[ "${status}" == *DONE* ]]; then
          true
        else
          echo "[grid] WARNING: failed to poll ${pod} (status=${status}); treating as RUNNING" >&2
          alive=$((alive + 1))
        fi
      done
      if [[ "${alive}" -eq 0 ]]; then
        break
      fi
      echo "[grid] ${alive}/${NUM_PODS} devpods still running..."
      sleep "${POLL_INTERVAL_S}"
    done
    echo "[grid] all devpods complete"
  fi

  if [[ "${COLLECT_LOGS}" == "1" ]]; then
    mkdir -p "${LOG_DIR}"
    echo "[grid] collecting logs into ${LOG_DIR}"
    for pod in "${DEVPODS[@]}"; do
      echo "[grid] collecting from ${pod}"
      if ! devpod --silent ssh "${pod}" --command "bash -lc '
          set -euo pipefail
          if [[ -d ${LOG_DIR} ]]; then
            cd ${LOG_DIR}
            # Only copy the canonical per-point logs; these are all we need for grid_sweep_summary.
            tar --exclude=\"*.attempt*.log\" -cf - run_*.log 2>/dev/null || tar -cf - --files-from /dev/null
          else
            tar -cf - --files-from /dev/null
          fi
        '" | tar -C "${LOG_DIR}" -xf -; then
        echo "[grid] WARNING: failed to collect logs from ${pod}; continuing" >&2
      fi
    done
  fi

  if [[ "${SUMMARY_AFTER_COLLECT}" == "1" ]]; then
    if [[ "${COLLECT_LOGS}" != "1" ]]; then
      echo "[grid] SUMMARY_AFTER_COLLECT=1 requires COLLECT_LOGS=1" >&2
      exit 2
    fi
    echo "[grid] logging combined grid summary from ${LOG_DIR}"
    uv run python -m src.grid_sweep_summary \
      --log_dir "${LOG_DIR}" \
      --run_prefix "${RUN_PREFIX}" \
      --grid_sweep_id "${GRID_SWEEP_ID}" \
      --sweep_axes "${SWEEP_AXES}" \
      --resolution "${RES}" \
      --wandb_project "${WANDB_PROJECT}" \
      --wandb_entity "${WANDB_ENTITY}" \
      --wandb_tags "${FINETUNE_WANDB_TAGS}" \
      --storage_dir "${FRACTAL_STORAGE_DIR}"
  fi

  exit 0
fi

mkdir -p "${LOG_DIR}"
echo "[grid] logging to ${LOG_DIR}"
echo "[grid] W&B project=${WANDB_PROJECT} entity=${WANDB_ENTITY} tags=${FINETUNE_WANDB_TAGS} sweep_id=${GRID_SWEEP_ID}"
echo "[grid] sweep_axes=${SWEEP_AXES} res=${RES} tokens_per_run=${TOKENS_PER_RUN:-<grid>}"

# Auto-detect GPUs when GPUS isn't set (safe default for devpod 1×GPU workspaces).
GPU_IDS=()
if [[ -n "${GPU_IDS_STR}" ]]; then
  # shellcheck disable=SC2206
  GPU_IDS=(${GPU_IDS_STR})
else
  gpu_count=0
  if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_count=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
  fi
  if [[ "${gpu_count}" -le 0 ]]; then
    gpu_count=$(uv run python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)
  fi
  if [[ "${gpu_count}" -le 0 ]]; then
    echo "[grid] No GPUs detected. Set GPUS=\"0\" (CPU runs are not supported here)." >&2
    exit 2
  fi
  for ((i=0; i<gpu_count; i++)); do
    GPU_IDS+=("${i}")
  done
fi

# Build grid points
mapfile -t GRID_POINTS < <(uv run python - <<PY
import numpy as np, os
mode = os.environ.get("SWEEP_AXES", "lr_tokens")
res = int(os.environ.get("RES", "4"))
tokens_per_run = os.environ.get("TOKENS_PER_RUN")

if mode == "lr_tokens":
    lr_fixed = os.environ.get("LR_FIXED")
    if lr_fixed:
        lrs = np.array([float(lr_fixed)])
    else:
        lr_min = float(os.environ.get("LR_MIN", "1e-6"))
        lr_max = float(os.environ.get("LR_MAX", "3e-1"))
        lrs = np.logspace(np.log10(lr_min), np.log10(lr_max), res)

    if tokens_per_run:
        toks = np.array([int(float(tokens_per_run))])
    else:
        tok_min = float(os.environ.get("TOK_MIN", "5e3"))
        tok_max = float(os.environ.get("TOK_MAX", "5e5"))
        toks = np.logspace(np.log10(tok_min), np.log10(tok_max), res).astype(int)

    for i, lr in enumerate(lrs):
        for j, tok in enumerate(toks):
            print(f"{i},{j},{lr:.6g},{int(tok)}")

elif mode == "matrix_unembedding":
    if not tokens_per_run:
        raise SystemExit("TOKENS_PER_RUN is required for SWEEP_AXES=matrix_unembedding")
    tok = int(float(tokens_per_run))

    mmin = float(os.environ.get("MATRIX_LR_MIN", "1e-6"))
    mmax = float(os.environ.get("MATRIX_LR_MAX", "3e-1"))
    umin = float(os.environ.get("UNEMBEDDING_LR_MIN", "1e-6"))
    umax = float(os.environ.get("UNEMBEDDING_LR_MAX", "1e-1"))
    matrix_lrs = np.logspace(np.log10(mmin), np.log10(mmax), res)
    unembedding_lrs = np.logspace(np.log10(umin), np.log10(umax), res)

    for i, mlr in enumerate(matrix_lrs):
        for j, ulr in enumerate(unembedding_lrs):
            print(f"{i},{j},{mlr:.6g},{ulr:.6g},{tok}")

else:
    raise SystemExit(f"Unknown SWEEP_AXES={mode!r} (expected lr_tokens or matrix_unembedding)")
PY
)

if [[ ${#GRID_POINTS[@]} -eq 0 ]]; then
  echo "[grid] No grid points generated; check inputs" >&2
  exit 1
fi
echo "[grid] grid points: ${#GRID_POINTS[@]}"

extra_args=()
[[ -n "${MODEL_ID}" ]] && extra_args+=(--model_id "${MODEL_ID}")
[[ -n "${DATASET_ID}" ]] && extra_args+=(--dataset_id "${DATASET_ID}")
[[ -n "${DATASET_REVISION}" ]] && extra_args+=(--dataset_revision "${DATASET_REVISION}")

num_gpus=${#GPU_IDS[@]}
if [[ ${num_gpus} -eq 0 ]]; then
  echo "[grid] No GPUs specified; set GPUS=\"0 1 2 ...\"" >&2
  exit 2
fi

if [[ "${NUM_PODS}" -le 0 ]]; then
  echo "[grid] NUM_PODS must be >= 1; got ${NUM_PODS}" >&2
  exit 2
fi
if [[ "${POD_INDEX}" -lt 0 ]] || [[ "${POD_INDEX}" -ge "${NUM_PODS}" ]]; then
  echo "[grid] POD_INDEX must satisfy 0 <= POD_INDEX < NUM_PODS; got POD_INDEX=${POD_INDEX} NUM_PODS=${NUM_PODS}" >&2
  exit 2
fi

echo "[grid] dispatching ${#GRID_POINTS[@]} points across ${num_gpus} GPU workers (pod ${POD_INDEX}/${NUM_PODS} ${DEVPOD_NAME})"
pids=()
for gpu_idx in "${!GPU_IDS[@]}"; do
  gpu=${GPU_IDS[$gpu_idx]}
  (
    set +e
    worker_rc=0
    start_idx=$((POD_INDEX + NUM_PODS * gpu_idx))
    stride=$((NUM_PODS * num_gpus))
    for ((point_idx=start_idx; point_idx<${#GRID_POINTS[@]}; point_idx+=stride)); do
      point=${GRID_POINTS[$point_idx]}
      lr_args=()
      if [[ "${SWEEP_AXES}" == "matrix_unembedding" ]]; then
        IFS=',' read -r gi gj mlr ulr tok <<<"${point}"
        lr_desc="matrix_lr=${mlr} unembedding_lr=${ulr}"
        lr_args+=(--matrix_lr "${mlr}" --unembedding_lr "${ulr}")
      else
        IFS=',' read -r gi gj lr tok <<<"${point}"
        lr_desc="learning_rate=${lr}"
        lr_args+=(--learning_rate "${lr}")
      fi
      log="${LOG_DIR}/run_${gi}_${gj}.log"
      echo "[grid] GPU ${gpu} -> (${gi},${gj}) ${lr_desc} tok=${tok} :: ${log}"

      if [[ "${SKIP_COMPLETED}" == "1" ]] && [[ -f "${log}" ]] && grep -q "^Final: loss=" "${log}"; then
        echo "[grid] SKIP (${gi},${gj}) already has Final: in ${log}"
        continue
      fi

      attempt=0
      point_ok=0
      while [[ "${attempt}" -le "${MAX_RETRIES}" ]]; do
        attempt=$((attempt + 1))
        attempt_log="${LOG_DIR}/run_${gi}_${gj}.attempt${attempt}.log"
        echo "[grid] GPU ${gpu} -> (${gi},${gj}) attempt ${attempt}/$((MAX_RETRIES + 1)) :: ${attempt_log}"

        CUDA_VISIBLE_DEVICES=${gpu} HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-0} \
          FRACTAL_STORAGE_DIR=${FRACTAL_STORAGE_DIR} \
          WANDB_PROJECT=${WANDB_PROJECT} WANDB_ENTITY=${WANDB_ENTITY} \
          PYTHONUNBUFFERED=1 \
          uv run python -m src.finetune \
            --run "${RUN_PREFIX}" \
            --grid_sweep_id "${GRID_SWEEP_ID}" \
            --grid_i "${gi}" \
            --grid_j "${gj}" \
            "${lr_args[@]}" \
            --num_tokens "${tok}" \
            --eval_every 0 \
            --eval_batches 0 \
            --log_every 1 \
            --save_artifacts False \
            --deterministic True \
            --seed "${SEED}" \
            --max_seq_len "${MAX_SEQ_LEN}" \
            --wandb_tags "${FINETUNE_WANDB_TAGS}" \
            --trainable_param_groups "${TRAINABLE_PARAM_GROUPS}" \
            "${extra_args[@]}" \
            2>&1 | tee "${attempt_log}"
        cmd_rc=${PIPESTATUS[0]}

        # Canonical log always points at the last attempt (success or final failure).
        cp -f "${attempt_log}" "${log}"

        if [[ "${cmd_rc}" -eq 0 ]]; then
          point_ok=1
          break
        fi

        if grep -Eq "${RETRY_PATTERNS}" "${attempt_log}"; then
          if [[ "${attempt}" -le "${MAX_RETRIES}" ]]; then
            sleep_s=$((RETRY_BACKOFF_S * attempt))
            echo "[grid] RETRY (${gi},${gj}) in ${sleep_s}s (rc=${cmd_rc}; matched RETRY_PATTERNS)"
            sleep "${sleep_s}"
            continue
          fi
        fi

        echo "[grid] FAILED gpu=${gpu} point=(${gi},${gj}) rc=${cmd_rc} log=${log}" >&2
        worker_rc=1
        break
      done
    done
    exit "${worker_rc}"
  ) &
  pids+=($!)
done

rc=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    rc=1
  fi
done
echo "[grid] workers complete (rc=${rc})"

# Summarize results + log a single W&B "grid-summary" run with the image/table.
summary_rc=0
if [[ "${LOG_SUMMARY}" == "1" ]]; then
  if ! uv run python -m src.grid_sweep_summary \
    --log_dir "${LOG_DIR}" \
    --run_prefix "${RUN_PREFIX}" \
    --grid_sweep_id "${GRID_SWEEP_ID}" \
    --sweep_axes "${SWEEP_AXES}" \
    --resolution "${RES}" \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_entity "${WANDB_ENTITY}" \
    --wandb_tags "${FINETUNE_WANDB_TAGS}" \
    --storage_dir "${FRACTAL_STORAGE_DIR}"; then
    summary_rc=1
    echo "[grid] WARNING: failed to log grid summary to W&B (log_dir=${LOG_DIR})" >&2
  fi
fi

if [[ ${summary_rc} -ne 0 ]]; then
  rc=1
fi
exit ${rc}

```


### `src/eval_samples.py`

```python
"""
Small copy/paste-able evaluation samples for the chat UI.

These come from the validation split used in `src/finetune.py`:
  dataset_id = "morgan/docvqa-nanochat"
  split = "validation"

Usage:
  - Copy `EVAL_SAMPLES["docvqa_val_0"]["user"]` into the chat UI input box.
  - The `expected` field is the ground-truth answer from the dataset.
"""

EVAL_SAMPLES: dict[str, dict[str, str]] = {
    "docvqa_val_0": {
        "expected": "SARGASSO SEA TEMPERATURE",
        "user": """Document:
Page 4
Unsettled Science
Knowing that weather forecasts are reliable for a
Moreover, computer models relied upon
few days at best, we should recognize the enor-
by climate scientists predict that lower atmos-
mous challenge facing scientists seeking to pre-
pheric temperatures will rise as fast as or faster
dict climate change and its impact over the next
than temperatures at the surface. However, only
century. In spite of everyone's desire for clear
within the last 20 years have reliable global
answers, it is not surprising that fundamental
measurements of temperatures in the lower at-
gaps in knowledge leave scientists unable to
mosphere been available through the use of
make reliable predictions about future changes.
satellite technology. These measurements show
A recent report from the National Re-
little if any warming.
search Council (NRC) raises important issues,
Even less is known about the potential
including these still-unanswered questions:
positive or negative impacts of climate change.
(1) Has human activity al-
In fact, many academic
ready begun to change
Sargasso Sea Temperature
studies and field experi-
temperature and the cli-
78-
ments have demonstrated
mate, and (2) How signifi-
77
Medieval
that increased levels of car-
cant will future change be?
varm period
76 -
Little
bon dioxide can
The NRC report con-
ice age
crop and forest growth.
firms that Earth's surface
So, while some argue
temperature has risen by
that the science debate is
about 1 degree Fahrenheit
73
settled and governments
over the past 150 years.
should focus only on near-
Some use this result to
72
71 .
OF
term policies-that is empty
claim that humans are
rhetoric. Inevitably, future
causing global warming,
70 -
scientific research will help
and they point to storms or
1000
500
0
- B.C. A.D. -
500 1000 1500 2000
us understand how human
floods to say that danger-
actions and natural climate
ous impacts are already
Source: Science (1996)
change may affect the world
under way. Yet scientists remain unable to con-
and will help determine what actions may be de-
firm either contention.
sirable to address the long-term.
Geological evidence indicates that climate
Science has given us enough information
and greenhouse gas levels experience significant
to know that climate changes may pose long-
natural variability for reasons having nothing to
term risks. Natural variability and human activity
do with human activity. Historical records and
may lead to climate change that could be signif-
current scientific evidence show that Europe and
North America experienced a medieval warm
cant and perhaps both positive and negative.
Consequently, people, companies and govern-
period one thousand years ago, followed cen-
ments should take responsible actions now to
turies later by a little ice age. The geological
address the issue.
record shows even larger changes throughout
One essential step is to encourage devel-
Earth's history. Against this backdrop of large,
opment of lower-emission technologies to meet
poorly understood natural variability, it is impos-
our future needs for energy. We'll next look at
sible for scientists to attribute the recent small
the promise of technology and what is being
surface temperature increase to human causes.
done today.
ExonMobil'
www.exxon.mobil.com
2000 Exxon Mobil Corporation
Source: https://www.industrydocuments.ucsf.edu/docs/ttw10228

Question: What is the title of the chart?""",
    },
    "docvqa_val_1": {
        "expected": "Exxonmobil",
        "user": """Document:
Page 4
Unsettled Science
Knowing that weather forecasts are reliable for a
Moreover, computer models relied upon
few days at best, we should recognize the enor-
by climate scientists predict that lower atmos-
mous challenge facing scientists seeking to pre-
pheric temperatures will rise as fast as or faster
dict climate change and its impact over the next
than temperatures at the surface. However, only
century. In spite of everyone's desire for clear
within the last 20 years have reliable global
answers, it is not surprising that fundamental
measurements of temperatures in the lower at-
gaps in knowledge leave scientists unable to
mosphere been available through the use of
make reliable predictions about future changes.
satellite technology. These measurements show
A recent report from the National Re-
little if any warming.
search Council (NRC) raises important issues,
Even less is known about the potential
including these still-unanswered questions:
positive or negative impacts of climate change.
(1) Has human activity al-
In fact, many academic
ready begun to change
Sargasso Sea Temperature
studies and field experi-
temperature and the cli-
78-
ments have demonstrated
mate, and (2) How signifi-
77
Medieval
that increased levels of car-
cant will future change be?
varm period
76 -
Little
bon dioxide can
The NRC report con-
ice age
crop and forest growth.
firms that Earth's surface
So, while some argue
temperature has risen by
that the science debate is
about 1 degree Fahrenheit
73
settled and governments
over the past 150 years.
should focus only on near-
Some use this result to
72
71 .
OF
term policies-that is empty
claim that humans are
rhetoric. Inevitably, future
causing global warming,
70 -
scientific research will help
and they point to storms or
1000
500
0
- B.C. A.D. -
500 1000 1500 2000
us understand how human
floods to say that danger-
actions and natural climate
ous impacts are already
Source: Science (1996)
change may affect the world
under way. Yet scientists remain unable to con-
and will help determine what actions may be de-
firm either contention.
sirable to address the long-term.
Geological evidence indicates that climate
Science has given us enough information
and greenhouse gas levels experience significant
to know that climate changes may pose long-
natural variability for reasons having nothing to
term risks. Natural variability and human activity
do with human activity. Historical records and
may lead to climate change that could be signif-
current scientific evidence show that Europe and
North America experienced a medieval warm
cant and perhaps both positive and negative.
Consequently, people, companies and govern-
period one thousand years ago, followed cen-
ments should take responsible actions now to
turies later by a little ice age. The geological
address the issue.
record shows even larger changes throughout
One essential step is to encourage devel-
Earth's history. Against this backdrop of large,
opment of lower-emission technologies to meet
poorly understood natural variability, it is impos-
our future needs for energy. We'll next look at
sible for scientists to attribute the recent small
the promise of technology and what is being
surface temperature increase to human causes.
done today.
ExonMobil'
www.exxon.mobil.com
2000 Exxon Mobil Corporation
Source: https://www.industrydocuments.ucsf.edu/docs/ttw10228

Question: Which company name is mentioned at the bottom?""",
    },
    "docvqa_val_2": {
        "expected": "18 million",
        "user": """Document:
Page 4
DOMESTIC PRODUCT DEVELOPMENT (cont'd.)
POL 0911, B&H Menthol versus Salem 100 - B&H Menthol, without print down rod, are
being produced in Cabarrus this week.
HTI 1723, Marlboro Lights Menthol versus Salem Lights 100's samples are being
produced in Louisville this week.
Market Research
HTI 2526 and HTI 2532, Marlboro 80 Box versus Camel 80 Box - These samples have
been approved for shipment on 6/4/90.
INTERNATIONAL PRODUCT DEVELOPMENT
PM Super Lights (Hong Kong)
Production start-up of Philip Morris Super Lights Menthol began the 6th of June at the
Manufacturing Center. The 18 million order is to be shipped to Hong Kong in preparation for a
July 1 launch.
Project Ring (Korea)
Cigarettes for PMI test #13 (Parliament KS 9mg versus 88 Lights) have been approved and
shipped to the warehouse.
Seoul Consumer Panel Testing (Korea)
Cigarettes for SCP #9 (88 Lights versus PM Super Lights carbon loading study) have been
approved and shipped to the warehouse. Filters have been made and combined for SCP #10
Parliament filter study).
Merit Lights (Hong Kong)
Cigarettes for PMI testing of Merit Lights prototype versus Kent have been produced and
are under analysis.
4
2022155854
Source: https://www.industrydocuments.ucsf.edu/docs/khxj0037

Question: how much order is to be shipped to hong kong?""",
    },
    "docvqa_val_3": {
        "expected": "Philip Morris Super Lights",
        "user": """Document:
Page 4
DOMESTIC PRODUCT DEVELOPMENT (cont'd.)
POL 0911, B&H Menthol versus Salem 100 - B&H Menthol, without print down rod, are
being produced in Cabarrus this week.
HTI 1723, Marlboro Lights Menthol versus Salem Lights 100's samples are being
produced in Louisville this week.
Market Research
HTI 2526 and HTI 2532, Marlboro 80 Box versus Camel 80 Box - These samples have
been approved for shipment on 6/4/90.
INTERNATIONAL PRODUCT DEVELOPMENT
PM Super Lights (Hong Kong)
Production start-up of Philip Morris Super Lights Menthol began the 6th of June at the
Manufacturing Center. The 18 million order is to be shipped to Hong Kong in preparation for a
July 1 launch.
Project Ring (Korea)
Cigarettes for PMI test #13 (Parliament KS 9mg versus 88 Lights) have been approved and
shipped to the warehouse.
Seoul Consumer Panel Testing (Korea)
Cigarettes for SCP #9 (88 Lights versus PM Super Lights carbon loading study) have been
approved and shipped to the warehouse. Filters have been made and combined for SCP #10
Parliament filter study).
Merit Lights (Hong Kong)
Cigarettes for PMI testing of Merit Lights prototype versus Kent have been produced and
are under analysis.
4
2022155854
Source: https://www.industrydocuments.ucsf.edu/docs/khxj0037

Question: full form of PM super lights""",
    },
    "docvqa_val_4": {
        "expected": "INTER-OFFICE CORRESPONDENCE",
        "user": """Document:
Page 1
PHILIP MORRIS. U. S.A.
INTER - OFFICE CORRESPONDENCE
Richmond, Virginia
To:
.Dr. Richard Carchman
Date: May 9, 1990
From:
.Maria Shulleeta
Subject:
.Prospective Alternate Preservatives List for Phase I Screening
After examining pertinant literature and discussing with knowledgeable PM personnel the
company's continuing need for an alternate preservative for the RL process , a number of
compounds have been identified for screening in Phase I preservative assays. Some of these
compounds are known tobacco constituents whose structures are similiar to other compounds
which have demonstrated significant antimicrobial activity in our assays. Other compounds on
the proposed list are essential oils or essential oil components which are known to have
antimicrobial activity in other test systems. The prospective test compounds are listed below
with their CAS numbers (where known). Please comment on the acceptability of the use of
these compounds in our processes. It is important to consider that any compound that is would
have to be effective (complete inhibition of bacterial growth for 24 hours) at low dose (<300
ug/ml) in Phase I screening before subsequent testing in the Phase III fermentor-scale assay or
subjective screening would be suggested. In evaluating the listed compounds, please indicate a
priority for screening by rating the compounds for acceptability (e.g., very acceptable
Mono
compounds would be rated "1" and consequently tested first):
CA
18
RTECS:
Compound
CAS number
HSAB
ATECS
MDNO Caryophyllene
87-44-5
V
Sclareol
515-03-7 7
Sclareolide
564-20-5
HSOB
RTECS
Fumaric Acid
X
110-82-2 110-17-8
Taxnets
2-phenylethyl valerate
7460-74-4 /
Send to OMaria
HSDP
Moro Phenyl acetic acid
103-82-2. J
Abietic acid
514-10-3 /
# 1902
KTECS
Xanthophyll
127- 440 -2
RTECS
MciJo Basil oil
8015-73-4
RTECS MONO: Bay oil
8006- 78-8
ASDe
PTECS
MONO Cumin oil."
8014-13-9 7
RTECS
MONO Lemongrass oil
8007-02-1
ITECS
1. 1010
Caraway oil
8000-42-8 /
H.DB
RTECE
MONO
Orange oil
208- 57-9
Mero Oakmoss oil
9000-50-4
VTECS MONO Phenylacetaldehyde
122-78:1
2022156519
Source: https://www.industrydocuments.ucsf.edu/docs/ljxj0037

Question: What kind of a communication/letter  is this?""",
    },
}


```


### `src/finetune.py`

```python
"""
Finetune nanochat-style on a single 8×GPU node (no Modal) and optionally sweep LR×token grids.

Usage:
  # single run on 8 GPUs (DDP, data parallel)
  torchrun --standalone --nproc_per_node=8 -m src.finetune -- --run myrun --learning_rate=3e-4 --num_tokens=200000

  # debug/smoke test (minimal data, fast iteration, saves artifacts)
  torchrun --standalone --nproc_per_node=1 -m src.finetune -- --debug

  # grid search (runs sequentially, all 8 GPUs work on each point together)
  torchrun --standalone --nproc_per_node=8 -m src.finetune -- --grid --resolution=16 --lr_min=1e-5 --lr_max=1e-3 --tokens_min=5e3 --tokens_max=5e5

  # PARALLEL grid search: run 8 independent single-GPU experiments (RECOMMENDED for fractal grids)
  # This gives 8x throughput for grid search by running different (lr, tokens) combinations in parallel
  for gpu in {0..7}; do
    CUDA_VISIBLE_DEVICES=$gpu python -m src.finetune --run grid-$gpu --learning_rate=<lr_$gpu> --num_tokens=<tokens_$gpu> &
  done
  wait

Scaling Strategy for Fractal Grid Search:
  - For maximum throughput: Run 8 independent single-GPU experiments in parallel
  - Each GPU runs a different (learning_rate, num_tokens) combination
  - Use a launcher script to distribute grid points across GPUs
  - This is 8x faster than sequential grid search with DDP

Reproducibility:
  - deterministic=True (default) enables full reproducibility
  - Same seed + same GPU = identical results
  - CUBLAS_WORKSPACE_CONFIG and torch.use_deterministic_algorithms are set
  - Grid points get unique seeds: seed + grid_i * 1000 + grid_j

Style notes:
- Mirrors nanochat/scripts/chat_sft.py: flat config globals + configurator overrides, compute_init/cleanup, DummyWandb, print0.
- Uses DocVQA dataset with proper conversation masking (assistant tokens only).
- Runs entirely locally; no Modal objects or volumes.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import subprocess
from dotenv import load_dotenv
from contextlib import nullcontext
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List
import shutil
import itertools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage

import torch
import torch.distributed as dist
import numpy as np
import wandb
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from torch.amp import GradScaler

# ---------------------------------------------------------------------------
# Reproducibility settings (critical for fractal grid experiments)
# Must be set before any CUDA operations
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # Deterministic cuBLAS
os.environ["PYTHONHASHSEED"] = "0"  # Deterministic Python hashing
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("NCCL_ALGO", "Ring")  # Stable reduction order (deterministic for single-node)
os.environ.setdefault("NCCL_PROTO", "Simple")
os.environ.setdefault("NCCL_MIN_NRINGS", "1")
os.environ.setdefault("NCCL_DEBUG", os.environ.get("NCCL_DEBUG", "WARN"))
os.environ.setdefault("HF_DATASETS_OFFLINE", os.environ.get("HF_DATASETS_OFFLINE", "0"))


def set_seed(seed: int, rank: int = 0):
    """Set all random seeds for full reproducibility.

    Model initialization uses the base seed for all ranks so weights match
    exactly; rank offset is applied only to RNG streams that affect data order.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Optional per-rank jitter for data-only randomness can be layered on top
    # by callers if needed without changing model initialization.


def enable_deterministic_mode():
    """Enable PyTorch deterministic algorithms for reproducibility.

    Note: This may have a small performance impact (~5-10%).
    """
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Disable auto-tuning for reproducibility
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))


def repro_context():
    """Collect reproducibility-relevant metadata for logging."""
    import socket

    def _git_rev():
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                cwd=REPO_ROOT,
            )
            return result.stdout.strip()
        except Exception:
            return None

    # Get hostname and devpod name for multi-pod sweep identification
    hostname = socket.gethostname()
    # DEVPOD_NAME env var is set by the multi-devpod launcher; fallback to hostname
    devpod_name = os.environ.get("DEVPOD_NAME", hostname)

    ctx = {
        "seed": seed,
        "deterministic": deterministic,
        "model_id": model_id,
        "dataset_id": dataset_id,
        "dataset_revision": dataset_revision,
        "tokenizer_artifact": tokenizer_artifact,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "nccl": ".".join(map(str, torch.cuda.nccl.version())) if torch.cuda.is_available() else None,
        "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "nccl_algo": os.environ.get("NCCL_ALGO"),
        "nccl_proto": os.environ.get("NCCL_PROTO"),
        "nccl_min_nrings": os.environ.get("NCCL_MIN_NRINGS"),
        "hf_datasets_offline": os.environ.get("HF_DATASETS_OFFLINE"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "tf32": {
            "matmul": torch.backends.cuda.matmul.allow_tf32,
            "cudnn": torch.backends.cudnn.allow_tf32,
        },
        "torch_num_threads": torch.get_num_threads(),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "git_commit": _git_rev(),
        "hostname": hostname,
        "devpod_name": devpod_name,
    }
    return ctx

# ---------------------------------------------------------------------------
# Wire nanochat helpers (style compatibility)
REPO_ROOT = Path(__file__).resolve().parent.parent

# Default storage: push results, checkpoints, and HF caches to /var/tmp to avoid small workspace quotas.
STORAGE_ROOT = Path(os.environ.get("FRACTAL_STORAGE_DIR", "/var/tmp/fractal-llm"))
RESULTS_ROOT = STORAGE_ROOT / "results"
MODEL_CACHE_ROOT = RESULTS_ROOT / "model_cache"
CHECKPOINTS_ROOT = RESULTS_ROOT / "checkpoints"
for _path in [STORAGE_ROOT, RESULTS_ROOT, MODEL_CACHE_ROOT, CHECKPOINTS_ROOT]:
    _path.mkdir(parents=True, exist_ok=True)

# HuggingFace caches (datasets + hub) and nanochat base dir => /var/tmp by default.
_storage_env_defaults = {
    # Hugging Face caches
    "HF_HOME": "/var/tmp/huggingface",
    "HF_DATASETS_CACHE": "/var/tmp/huggingface/datasets",
    "HF_HUB_CACHE": "/var/tmp/huggingface/hub",
    # nanochat base dir
    "NANOCHAT_BASE_DIR": "/var/tmp/nanochat",
    # WandB local files/config/cache
    "WANDB_DIR": str(STORAGE_ROOT / "wandb"),
    "WANDB_CONFIG_DIR": str(STORAGE_ROOT / "wandb" / "config"),
    "WANDB_CACHE_DIR": str(STORAGE_ROOT / "wandb" / "cache"),
}
for _k, _v in _storage_env_defaults.items():
    os.environ.setdefault(_k, _v)
for _p in [
    Path(os.environ["HF_HOME"]),
    Path(os.environ["HF_DATASETS_CACHE"]),
    Path(os.environ["HF_HUB_CACHE"]),
    Path(os.environ["NANOCHAT_BASE_DIR"]),
    Path(os.environ["WANDB_DIR"]),
    Path(os.environ["WANDB_CONFIG_DIR"]),
    Path(os.environ["WANDB_CACHE_DIR"]),
]:
    _p.mkdir(parents=True, exist_ok=True)

NANOCHAT_DIR = REPO_ROOT / "third_party" / "nanochat"
DOTENV_CANDIDATES = [
    REPO_ROOT / ".env",
    Path(".env"),
    Path("/workspace/.env"),
    Path("/root/.env"),
]
for _cand in DOTENV_CANDIDATES:
    if _cand.exists():
        load_dotenv(_cand, override=False)

if NANOCHAT_DIR.exists() and str(NANOCHAT_DIR) not in sys.path:
    sys.path.insert(0, str(NANOCHAT_DIR))

# nanochat imports (after sys.path wiring)
from nanochat.checkpoint_manager import build_model, find_last_step, save_checkpoint
from nanochat.tokenizer import get_tokenizer
from nanochat.common import get_base_dir

try:
    from nanochat.common import (
        compute_init,
        compute_cleanup,
        print0,
        DummyWandb,
        autodetect_device_type,
    )
except Exception:
    # Fallback if nanochat isn't importable (e.g., trimmed checkout). Minimal shims.
    def print0(s: str = "", **kwargs):
        if int(os.environ.get("RANK", 0)) == 0:
            print(s, **kwargs)

    def autodetect_device_type():
        return "cuda" if torch.cuda.is_available() else "cpu"

    def compute_init(device_type="cuda"):
        is_ddp = all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world = int(os.environ.get("WORLD_SIZE", 1))
        device = torch.device(device_type if device_type != "cuda" else f"cuda:{local_rank}")
        if is_ddp and device_type == "cuda":
            torch.cuda.set_device(device)
            dist.init_process_group("nccl", device_id=device)
            dist.barrier()
        return is_ddp, rank, local_rank, world, device

    def compute_cleanup():
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

    class DummyWandb:
        def log(self, *args, **kwargs): ...
        def finish(self): ...

# ---------------------------------------------------------------------------
# Config (nanochat-style flat globals + configurator)
run = "dummy"  # wandb run name; "dummy" disables logging
model_id = os.environ.get("MODEL_ARTIFACT", "morgy/fractal-llm/nanochat-fin-rl-artifact:v7")
dataset_id = os.environ.get("DATASET_ID", "morgan/docvqa-nanochat")
dataset_revision = os.environ.get("DATASET_REVISION", None)
max_seq_len = int(os.environ.get("MAX_SEQ_LEN", "1024"))  # nanochat default=2048, DocVQA avg ~400 tokens
dtype = "bfloat16"  # float32 | bfloat16 | float16 (bfloat16 matches model weights)
device_batch_size = 8  # aim for the largest batch that fits; override via CLI/ENV if needed
# H200 / ~140GiB: probed stable device_batch_size=48 for this config (MAX_SEQ_LEN=1024),
# running 100 steps with 3 eval rounds (eval_every=33, eval_batches=4).
if torch.cuda.is_available():
    _total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if _total_gb >= 120:
        device_batch_size = 48
# Set 0 to force grad_accum_steps=1; otherwise acts as a target effective batch size.
target_examples_per_step = 0
learning_rate = 3e-4
# Differential LRs (nanochat/scripts/chat_sft.py style):
# - If *_lr is >0, use it directly.
# - If *_lr is <=0, derive from `learning_rate` via the multipliers below.
matrix_lr = -1.0
embedding_lr = -1.0
unembedding_lr = -1.0
embedding_lr_mult = 10.0
unembedding_lr_mult = 0.2
# Parameter groups to train (comma-separated): matrix, embedding, unembedding.
# Aliases: head/lm_head -> unembedding; wte/emb -> embedding; blocks -> matrix; "all"/"none" supported.
# Default freezes embeddings to reduce optimizer state + improve stability on small-token runs.
trainable_param_groups = "matrix,unembedding"

# Note: With nanochat optimizers, weight decay applies only to the AdamW groups
# (embeddings + unembedding), not the Muon matrix parameters.
weight_decay = 0.1

# Global scale factor applied to all per-group base LRs before warmup/cosine.
init_lr_frac = 1.0
warmup_frac = 0.06
final_lr_frac = 0.0  # cosine anneals to lr * final_lr_frac (0.0 = full decay to zero)
num_tokens = 200_000  # approximate global tokens to see
num_iterations = -1  # derived from num_tokens if -1
eval_every = 200
eval_batches = 4  # number of validation batches per eval pass
log_every = 1  # always log W&B metrics every optimizer step
seed = 999
deterministic = True  # if True, enable full reproducibility (slight perf hit ~5-10%)
debug = False  # if True, run a quick smoke test with minimal data
save_artifacts = False  # if True, save checkpoint and upload to W&B (auto-enabled in debug mode)
visualize = False  # if True, show model predictions before/after training
gradient_checkpointing = False
source_stage = "sft"  # nanochat checkpoint family: base|mid|sft|rl
tokenizer_artifact = os.environ.get("TOKENIZER_ARTIFACT", None)
# "Converged" == "trainable" (Sohl-Dickstein-style):
# - stable: training ran to completion without exceptions and with finite loss
# - trainable: mean(last K losses) < first loss (averaging smooths oscillations)
trainable_window_steps = 20
trainable_loss_ratio_threshold = 1.0  # mean(last_K)/loss[0] < threshold ⇒ trainable
wandb_tags = "finetune"  # comma-separated wandb tags (e.g., "finetune,debug")
grid_sweep_id = ""  # sweep identifier tag; auto-generated as YYYY-MM-DD_HH-MM if empty

# Grid search knobs
grid = False
resolution = 8
lr_min = 1e-5
lr_max = 1e-3
tokens_min = 5_000
tokens_max = 500_000
checkpoint_every = 32
grid_i = 0
grid_j = 0

# Derived/configurable keys list
config_keys = [
    k
    for k, v in list(globals().items())
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]

# Allow CLI overrides using nanochat/configurator.py semantics if available.
configurator = NANOCHAT_DIR / "nanochat" / "configurator.py"
if configurator.exists():
    # Allow both "--key=value" and "--key value" forms; normalize before running configurator.
    if len(sys.argv) > 1:
        new_args = []
        args = sys.argv[1:]
        i = 0
        while i < len(args):
            arg = args[i]
            if arg.startswith("--") and "=" not in arg and i + 1 < len(args) and not args[i + 1].startswith("--"):
                new_args.append(f"{arg}={args[i+1]}")
                i += 2
            else:
                new_args.append(arg)
                i += 1
        sys.argv = [sys.argv[0]] + new_args
    exec(configurator.read_text())
# Always log every optimizer step (prints + W&B) for reproducible fractal grids.
log_every = 1
user_config = {k: globals()[k] for k in config_keys}

# Respect WANDB_RUN override and attach suffix similar to chat_sft
env_run = os.environ.get("WANDB_RUN")
if env_run:
    run = env_run
if run != "dummy" and not run.endswith("-ft"):
    run = f"{run}-ft"

# Debug mode: override to minimal settings for a quick smoke test
if debug:
    num_tokens = 2000
    log_every = 1
    save_artifacts = True
    if run == "dummy":
        run = "debug-finetune"
    print0(f"[DEBUG MODE] num_tokens={num_tokens}, log_every={log_every}, save_artifacts={save_artifacts}")

# Grid sweeps should not save model checkpoints/artifacts (disk + time overhead).
# Enforce this even if the caller accidentally enables `save_artifacts`.
if (grid or grid_sweep_id) and save_artifacts:
    print0("[grid] Disabling save_artifacts for grid sweeps")
    save_artifacts = False

WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "fractal-llm")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "morgy")


# ---------------------------------------------------------------------------
# Data utilities


class DocVQADataset(IterableDataset):
    """Streaming iterable dataset sharded across ranks."""

    def __init__(self, tokenizer, split: str, seed: int, world_size: int, rank: int):
        self.tokenizer = tokenizer
        self.split = split
        self.seed = seed
        self.world_size = world_size
        self.rank = rank

    def __iter__(self):
        load_kwargs = {"split": self.split, "streaming": True}
        if dataset_revision:
            load_kwargs["revision"] = dataset_revision
        # HF Hub can occasionally time out on repo metadata fetches (especially under parallel sweeps).
        # Retry a few times with exponential backoff to avoid losing sweep points to transient network blips.
        max_retries = int(os.environ.get("HF_LOAD_DATASET_RETRIES", "3"))
        base_sleep_s = float(os.environ.get("HF_LOAD_DATASET_RETRY_SLEEP_S", "2"))
        ds = None
        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                ds = load_dataset(dataset_id, **load_kwargs)
                last_exc = None
                break
            except Exception as exc:  # noqa: BLE001 - intentional: robustness against transient hub/network errors
                last_exc = exc
                msg = str(exc)
                retryable = any(
                    s in msg
                    for s in (
                        "ReadTimeout",
                        "Read timed out",
                        "ConnectionPool",
                        "ConnectionError",
                        "Temporary failure in name resolution",
                        "502",
                        "503",
                        "504",
                    )
                )
                if not retryable or attempt >= max_retries:
                    raise
                sleep_s = base_sleep_s * (2**attempt)
                if self.rank == 0:
                    print0(
                        f"[WARN] load_dataset({dataset_id!r}, split={self.split!r}) failed: {msg} "
                        f"(attempt {attempt+1}/{max_retries+1}); retrying in {sleep_s:.1f}s"
                    )
                import time

                time.sleep(sleep_s)
        if ds is None:
            # Should be unreachable, but keeps type-checkers happy.
            assert last_exc is not None
            raise last_exc
        # Keep deterministic ordering to mirror fractal-boundary experiments.
        # Avoid shuffle buffers that introduce nondeterminism across runs.
        if not deterministic:
            ds = ds.shuffle(seed=self.seed, buffer_size=2048)
        if self.world_size > 1:
            ds = ds.shard(num_shards=self.world_size, index=self.rank)
        for sample in ds:
            # Pass messages directly - dataset already has proper {user, assistant} structure
            messages = sample.get("messages", [])
            if not messages:
                continue
            conversation = {"messages": messages}
            ids, mask = self.tokenizer.render_conversation(conversation)
            # Truncate to max_seq_len (keeping room for shift)
            orig_len = len(ids)
            ids = ids[:max_seq_len]
            mask = mask[:max_seq_len]
            if orig_len > max_seq_len and self.rank == 0:
                # Log truncation warning once per sample on rank 0
                print0(f"[WARN] Truncated sample from {orig_len} tokens to {max_seq_len} tokens")
            if len(ids) < 2:
                continue  # Need at least 2 tokens for next-token prediction

            # Shift for next-token prediction: input predicts next token
            # input_ids[i] should predict ids[i+1], so:
            #   input_ids = ids[:-1] (all but last)
            #   labels = ids[1:] (all but first, shifted by 1)
            input_ids = torch.tensor(ids[:-1], dtype=torch.long)
            labels = torch.tensor(ids[1:], dtype=torch.long)
            # Shift mask to align with labels (mask[i] corresponds to ids[i])
            mask_shifted = mask[1:]
            if sum(mask_shifted) == 0:
                # No assistant tokens to train on; skip to avoid NaN loss
                continue
            # Apply mask: positions where mask==0 should not contribute to loss
            # Use -1 as ignore_index (matches gpt.py F.cross_entropy ignore_index=-1)
            labels = torch.where(
                torch.tensor(mask_shifted, dtype=torch.long) == 1,
                labels,
                torch.tensor(-1, dtype=torch.long),
            )
            yield input_ids, labels


def collate_sft(batch, pad_token_id: int):
    """Collate variable-length sequences with dynamic padding to longest in batch."""
    input_ids_list, labels_list = zip(*batch)
    max_len = max(len(ids) for ids in input_ids_list)

    padded_input_ids = []
    padded_labels = []

    for ids, labels in zip(input_ids_list, labels_list):
        pad_len = max_len - len(ids)
        if pad_len > 0:
            padded_input_ids.append(torch.cat([ids, torch.full((pad_len,), pad_token_id, dtype=ids.dtype)]))
            padded_labels.append(torch.cat([labels, torch.full((pad_len,), -1, dtype=labels.dtype)]))
        else:
            padded_input_ids.append(ids)
            padded_labels.append(labels)

    return (
        torch.stack(padded_input_ids),
        torch.stack(padded_labels),
    )


def build_dataloader(tokenizer, seed, world_size, rank, split: str = "train") -> DataLoader:
    dataset = DocVQADataset(tokenizer, split=split, seed=seed, world_size=world_size, rank=rank)
    pad_token_id = tokenizer.encode_special("<|assistant_end|>")

    def collate_fn(batch):
        return collate_sft(batch, pad_token_id)

    return DataLoader(dataset, batch_size=device_batch_size, pin_memory=True, collate_fn=collate_fn)


# ---------------------------------------------------------------------------
# Model helpers

def ensure_tokenizer(tokenizer_id: str | None, artifact_root: Path | None) -> Path:
    """
    Ensure nanochat tokenizer files are available under ~/.cache/nanochat/tokenizer.
    Strategy:
      1) If tokenizer already exists there, use it.
      2) Else, if artifact_root contains tokenizer files, copy them in.
      3) Else, if tokenizer_id is provided (wandb:...), download and copy.
      4) Else, raise.
    """
    base_dir = Path(get_base_dir())
    tok_dir = base_dir / "tokenizer"
    tok_dir.mkdir(parents=True, exist_ok=True)
    # RustBPETokenizer only needs tokenizer.pkl; get_token_bytes needs token_bytes.pt
    required = ["tokenizer.pkl", "token_bytes.pt"]

    def has_required(path: Path) -> bool:
        return all((path / r).exists() for r in required)

    if has_required(tok_dir):
        return tok_dir

    def copy_from(src_dir: Path):
        for r in required:
            for cand in src_dir.rglob(r):
                target = tok_dir / cand.name
                target.write_bytes(cand.read_bytes())

    if artifact_root:
        # Look for tokenizer assets inside the already-downloaded model artifact
        for sub in [artifact_root, artifact_root / "tokenizer"]:
            if sub.exists():
                copy_from(sub)
        if has_required(tok_dir):
            return tok_dir

    if tokenizer_id:
        import wandb as _wandb

        art_path = tokenizer_id[len("wandb:") :] if tokenizer_id.startswith("wandb:") else tokenizer_id
        api = _wandb.Api()
        art = api.artifact(art_path, type="model")
        dl_root = tok_dir / "tmp_download"
        if dl_root.exists():
            shutil.rmtree(dl_root)
        dl_root.mkdir(parents=True, exist_ok=True)
        art_local = Path(art.download(root=str(dl_root)))
        copy_from(art_local)
        shutil.rmtree(dl_root, ignore_errors=True)
        if has_required(tok_dir):
            return tok_dir

    raise FileNotFoundError(
        "Tokenizer not found. Provide TOKENIZER_ARTIFACT (wandb:<entity>/<project>/<name>:<ver>) "
        "or place tokenizer files under ~/.cache/nanochat/tokenizer."
    )


def resolve_checkpoint_dir(model_ref: str) -> Path:
    """
    Download W&B artifact if needed, otherwise treat as local path.
    Expects nanochat checkpoint layout: checkpoints/model_*.pt + meta_*.json (or those files directly in root).
    """
    if model_ref.startswith("wandb:"):
        import wandb as _wandb
        art_path = model_ref[len("wandb:") :]
        api = _wandb.Api()
        art = api.artifact(art_path, type="model")
        cache_root = MODEL_CACHE_ROOT
        cache_root.mkdir(parents=True, exist_ok=True)

        def has_checkpoints(path: Path) -> bool:
            if not path.exists():
                return False
            return any(path.glob("model_*.pt")) and any(path.glob("meta_*.json"))

        # Multiple sweep workers can try to download the same artifact concurrently.
        # Use a simple file lock and validate the cache dir before trusting it.
        lock_path = cache_root / f".{art.name.replace(':', '_')}.lock"
        try:
            import fcntl  # type: ignore[attr-defined]

            with open(lock_path, "w", encoding="utf-8") as f:
                fcntl.flock(f, fcntl.LOCK_EX)

                target = cache_root / art.name.replace(":", "_")
                ckpt_subdir = target / "checkpoints"

                cache_ok = has_checkpoints(ckpt_subdir) or has_checkpoints(target)
                if target.exists() and not cache_ok:
                    shutil.rmtree(target, ignore_errors=True)

                if not target.exists():
                    art.download(root=str(target))

                if has_checkpoints(ckpt_subdir):
                    return ckpt_subdir
                if has_checkpoints(target):
                    return target
        except Exception:
            # Fallback: best-effort without a lock (e.g., non-POSIX filesystems).
            target = cache_root / art.name.replace(":", "_")
            if not target.exists():
                art.download(root=str(target))
            if (target / "checkpoints").exists():
                return target / "checkpoints"
            return target

        raise FileNotFoundError(f"Downloaded artifact is missing checkpoints under {target}")
    return Path(model_ref)


def load_model_and_tok(device_type: str):
    ckpt_dir = resolve_checkpoint_dir(model_id)
    tok_dir = ensure_tokenizer(tokenizer_artifact, ckpt_dir.parent if ckpt_dir.name == "checkpoints" else ckpt_dir)
    step = find_last_step(ckpt_dir)
    model, tokenizer, meta = build_model(ckpt_dir, step=step, device=torch.device(device_type), phase="train")
    model.to(device_type)
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return model, tokenizer


def parse_trainable_param_groups(spec: str) -> set[str]:
    allowed = {"matrix", "embedding", "unembedding"}
    aliases = {
        "*": "all",
        "all": "all",
        "none": "none",
        "": "none",
        "head": "unembedding",
        "lm_head": "unembedding",
        "unembed": "unembedding",
        "embed": "embedding",
        "emb": "embedding",
        "wte": "embedding",
        "blocks": "matrix",
        "trunk": "matrix",
    }
    norm = (spec or "").strip().lower()
    norm = aliases.get(norm, norm)
    if norm == "all":
        return set(allowed)
    if norm == "none":
        return set()
    parts = [aliases.get(p.strip().lower(), p.strip().lower()) for p in norm.split(",") if p.strip()]
    groups = set(parts)
    unknown = groups - allowed
    if unknown:
        raise ValueError(f"Unknown trainable_param_groups: {sorted(unknown)} (allowed: {sorted(allowed)})")
    return groups


def apply_trainable_param_groups(model, trainable_groups: set[str]) -> dict[str, int]:
    """Set requires_grad for nanochat GPT parameter groups and return param counts."""
    if not (hasattr(model, "transformer") and hasattr(model, "lm_head")):
        raise AttributeError(f"Expected nanochat GPT with transformer + lm_head, got {type(model)}")
    if not (hasattr(model.transformer, "h") and hasattr(model.transformer, "wte")):
        raise AttributeError(f"Expected nanochat GPT with transformer.h + transformer.wte, got {type(model)}")

    def _set_requires_grad(module, enabled: bool):
        for p in module.parameters():
            p.requires_grad = enabled

    train_matrix = "matrix" in trainable_groups
    train_embedding = "embedding" in trainable_groups
    train_unembedding = "unembedding" in trainable_groups

    _set_requires_grad(model.transformer.h, train_matrix)
    _set_requires_grad(model.transformer.wte, train_embedding)
    _set_requires_grad(model.lm_head, train_unembedding)

    counts = {}
    counts["total"] = sum(p.numel() for p in model.parameters())
    counts["trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    counts["matrix"] = sum(p.numel() for p in model.transformer.h.parameters() if p.requires_grad)
    counts["embedding"] = sum(p.numel() for p in model.transformer.wte.parameters() if p.requires_grad)
    counts["unembedding"] = sum(p.numel() for p in model.lm_head.parameters() if p.requires_grad)
    return counts


# ---------------------------------------------------------------------------
# Training core

@dataclass
class RunResult:
    run_name: str
    learning_rate: float
    num_tokens_target: int
    tokens_seen: int
    avg_loss: float
    final_loss: float
    trainable_loss0: float
    trainable_mean_last_k: float
    trainable_ratio: float
    stable: bool
    converged: bool
    steps: int
    runtime_s: float
    seed: int
    world_size: int
    device_type: str
    error: str | None = None
    grid_i: int | None = None
    grid_j: int | None = None


# ---------------------------------------------------------------------------
# Visualization and analysis helpers

def capture_baseline_samples(dataloader, device, num_samples: int = 5):
    """Capture fixed samples for before/after comparison."""
    samples = []
    for batch in dataloader:
        input_ids, labels = [b.to(device) for b in batch]
        for i in range(input_ids.shape[0]):
            if len(samples) >= num_samples:
                break
            samples.append({
                "input_ids": input_ids[i].clone(),
                "labels": labels[i].clone(),
            })
        if len(samples) >= num_samples:
            break
    return samples


def generate_sample_predictions(model, tokenizer, samples, device, cast_ctx, step: int, max_gen_tokens: int = 250):
    """Generate predictions for fixed samples and return table data.

    Args:
        model: The model to generate from
        tokenizer: Tokenizer for decoding
        samples: List of dicts with 'input_ids' and 'labels' tensors
        device: Device to run on
        cast_ctx: Autocast context function
        step: Training step (for logging)
        max_gen_tokens: Maximum tokens to generate per sample

    Returns:
        List of [step, sample_idx, prompt_text, generated_text, expected_text]
    """
    model.eval()
    table_data = []

    with torch.no_grad():
        for idx, sample in enumerate(samples):
            input_ids = sample["input_ids"].unsqueeze(0)
            labels = sample["labels"]

            # Find where assistant response starts (first non-masked token)
            non_masked = (labels >= 0).nonzero(as_tuple=True)[0]
            if len(non_masked) == 0:
                print0(f"[WARN] No non-masked tokens found in sample {idx} during sample generation.")
                continue

            prompt_end = int(non_masked[0].item())
            prompt_tokens = input_ids[0, :prompt_end]
            expected_tokens = labels[prompt_end:]
            expected_tokens = expected_tokens[expected_tokens >= 0]

            # Generate from model (greedy)
            gen_input = prompt_tokens.unsqueeze(0).to(device)
            generated = []
            max_gen_len = min(max_gen_tokens, len(expected_tokens) + 20)
            for _ in range(max_gen_len):
                with cast_ctx():
                    logits = model(gen_input)
                    if isinstance(logits, dict):
                        logits = logits.get("logits", logits)
                    if hasattr(logits, "shape") and len(logits.shape) == 3:
                        logits = logits[:, -1, :]
                next_token = logits.argmax(dim=-1)
                generated.append(next_token.item())
                gen_input = torch.cat([gen_input, next_token.unsqueeze(0)], dim=1)
                # Stop on EOS or assistant_end token
                if next_token.item() in [tokenizer.encode_special("<|assistant_end|>"), 0]:
                    break

            # Decode for table
            prompt_text = tokenizer.decode(prompt_tokens.tolist())
            expected_text = tokenizer.decode(expected_tokens.tolist()) if len(expected_tokens) > 0 else ""
            generated_text = tokenizer.decode(generated) if generated else ""

            table_data.append([
                step,
                idx,
                prompt_text,
                generated_text,
                expected_text,
            ])

    model.train()
    return table_data


def print_generation_table(
    table_data,
    stage: str,
    max_prompt: int = 200,
    max_expected: int = 100,
    max_generated: int = 100,
):
    """Print a generation table (as produced by generate_sample_predictions)."""
    if not table_data:
        print0(f"[WARN] No samples to display for {stage} predictions.")
        return 0

    def truncate(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[:limit] + "..."

    print0("=" * 60)
    print0(f"MODEL PREDICTIONS ({stage.upper()} training)")
    print0("=" * 60)

    for row in table_data:
        _, idx, prompt_text, generated_text, expected_text = row

        prompt_display = truncate(prompt_text, max_prompt)
        expected_display = truncate(expected_text or "", max_expected)
        generated_display = truncate(generated_text or "", max_generated)

        print0(f"\n--- Sample {idx + 1} ---")
        print0(f"Prompt: {prompt_display}")
        print0(f"Expected: {expected_display if expected_display else '(empty)'}")
        print0(f"Generated: {generated_display if generated_display else '(empty)'}")

    print0("=" * 60)
    return len(table_data)


def visualize_predictions(model, tokenizer, dataloader, device, cast_ctx, num_samples: int = 3, stage: str = "before"):
    """Generate and display model predictions on validation samples."""
    # Capture samples from dataloader
    samples = capture_baseline_samples(dataloader, device, num_samples=num_samples)

    # Generate predictions (shorter generation for display)
    table_data = generate_sample_predictions(
        model, tokenizer, samples, device, cast_ctx,
        step=0,  # step not used for display
        max_gen_tokens=50,
    )

    return print_generation_table(table_data, stage=stage)


def build_grids(results: List[RunResult], resolution: int):
    """Construct convergence, loss, and intensity grids."""
    fractal_grid = np.zeros((resolution, resolution))
    loss_grid = np.full((resolution, resolution), np.nan)
    convergence_grid = np.zeros((resolution, resolution))

    converged_losses: List[float] = []

    for r in results:
        i = r.grid_i or 0
        j = r.grid_j or 0
        if r.converged and math.isfinite(r.final_loss):
            loss_grid[i, j] = r.final_loss
            converged_losses.append(r.final_loss)
            convergence_grid[i, j] = 1
        else:
            convergence_grid[i, j] = 0

    if converged_losses:
        loss_min, loss_max = min(converged_losses), max(converged_losses)
        loss_range = loss_max - loss_min if loss_max > loss_min else 1.0
    else:
        loss_min, loss_max, loss_range = 0.0, 1.0, 1.0

    for r in results:
        i = r.grid_i or 0
        j = r.grid_j or 0
        if r.converged and math.isfinite(r.final_loss):
            # Invert intensity: lower loss → darker blue (value closer to 1.0)
            intensity = (r.final_loss - loss_min) / loss_range
            fractal_grid[i, j] = 1.0 - 0.7 * intensity  # Range: 0.3 (worst) to 1.0 (best)
        else:
            fractal_grid[i, j] = -1.0  # Red for diverged / failed

    return fractal_grid, loss_grid, convergence_grid


def save_visualizations(
    results: List[RunResult],
    resolution: int,
    lr_min: float,
    lr_max: float,
    tokens_min: float,
    tokens_max: float,
    out_prefix: Path,
):
    """Create and save three-panel visualization. Returns image path."""
    fractal_grid, loss_grid, convergence_grid = build_grids(results, resolution)

    # Diverging red-white-blue colormap for trainability visualization
    # Color mapping (vmin=-1, vmax=1):
    #   -1.0: Dark red - Diverged/failed runs
    #    0.0: White - Boundary (unused, converged starts at 0.3)
    #    0.3: Light blue - Converged, highest loss among converged
    #    1.0: Dark blue - Converged, lowest loss (best)
    colors = [
        "#8B0000",  # Dark red (diverged)
        "#B22222",  # Firebrick
        "#CD5C5C",  # Indian red
        "#FA8072",  # Salmon
        "#FFC0CB",  # Pink (light diverged)
        "#FFFFFF",  # White (boundary)
        "#E0FFFF",  # Light cyan
        "#ADD8E6",  # Light blue (worst converged)
        "#87CEEB",  # Sky blue
        "#4169E1",  # Royal blue
        "#0000CD",  # Medium blue
        "#00008B",  # Dark blue (best converged)
    ]
    positions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.65, 0.75, 0.85, 0.92, 1.0]
    fractal_cmap = LinearSegmentedColormap.from_list("fractal", list(zip(positions, colors)))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    im1 = axes[0].imshow(
        fractal_grid,
        origin="lower",
        aspect="auto",
        cmap=fractal_cmap,
        vmin=-1.0,
        vmax=1.0,
        extent=[np.log10(tokens_min), np.log10(tokens_max), np.log10(lr_min), np.log10(lr_max)],
    )
    axes[0].set_xlabel("log₁₀(tokens)")
    axes[0].set_ylabel("log₁₀(learning rate)")
    axes[0].set_title("Trainability Boundary\n(Blue=Trainable, Red=Not trainable)")
    cbar1 = plt.colorbar(im1, ax=axes[0])
    cbar1.set_label("Trainability")

    im2 = axes[1].imshow(
        loss_grid,
        origin="lower",
        aspect="auto",
        cmap="viridis_r",
        extent=[np.log10(tokens_min), np.log10(tokens_max), np.log10(lr_min), np.log10(lr_max)],
    )
    axes[1].set_xlabel("log₁₀(tokens)")
    axes[1].set_ylabel("log₁₀(learning rate)")
    axes[1].set_title("Final Loss (trainable)")
    cbar2 = plt.colorbar(im2, ax=axes[1])
    cbar2.set_label("Loss")

    im3 = axes[2].imshow(
        convergence_grid,
        origin="lower",
        aspect="auto",
        cmap="RdBu",
        vmin=0,
        vmax=1,
        extent=[np.log10(tokens_min), np.log10(tokens_max), np.log10(lr_min), np.log10(lr_max)],
    )
    axes[2].set_xlabel("log₁₀(tokens)")
    axes[2].set_ylabel("log₁₀(learning rate)")
    axes[2].set_title("Binary Trainable")
    cbar3 = plt.colorbar(im3, ax=axes[2])
    cbar3.set_ticks([0, 1])
    cbar3.set_ticklabels(["Not trainable", "Trainable"])

    plt.tight_layout()
    out_path = out_prefix.with_suffix(".png")
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    return out_path, convergence_grid


def compute_fractal(convergence_grid: np.ndarray):
    """Compute box-counting fractal dimension of boundary."""
    boundary = ndimage.binary_dilation(convergence_grid) ^ ndimage.binary_erosion(convergence_grid)

    def box_count(binary_image, box_size):
        h, w = binary_image.shape
        count = 0
        for i in range(0, h, box_size):
            for j in range(0, w, box_size):
                if binary_image[i : i + box_size, j : j + box_size].any():
                    count += 1
        return count

    sizes = [s for s in [2, 4, 8, 16, 32, 64] if s < boundary.shape[0]]
    if sizes:
        counts = [box_count(boundary, s) for s in sizes]
        log_sizes = np.log(sizes)
        log_counts = np.log(np.array(counts) + 1)
        coeffs = np.polyfit(log_sizes, log_counts, 1)
        fractal_dimension = -coeffs[0]
    else:
        counts = []
        fractal_dimension = float("nan")
    return {
        "fractal_dimension": float(fractal_dimension),
        "box_sizes": sizes,
        "box_counts": [int(c) for c in counts],
        "boundary_pixels": int(boundary.sum()),
        "converged_ratio": float(convergence_grid.sum() / convergence_grid.size),
    }

def derive_num_iterations(tokens_per_microbatch: int, grad_accum_steps: int, tokens_goal: int) -> int:
    """
    Derive number of optimizer steps to reach `tokens_goal`.

    `tokens_per_microbatch` is the *global* (all ranks) number of supervised tokens in one
    microbatch (i.e., one forward/backward pass), so this stays consistent under torchrun.
    """
    tokens_per_step = max(1, tokens_per_microbatch * grad_accum_steps)
    return max(1, math.ceil(tokens_goal / tokens_per_step))


def train_once(
    learning_rate_override: float | None = None,
    num_tokens_override: int | None = None,
    grid_i: int | None = None,
    grid_j: int | None = None,
    runtime=None,
) -> RunResult:
    import time

    lr = float(learning_rate_override or learning_rate)
    tokens_goal = int(num_tokens_override or num_tokens)

    # Init compute/DDP once; reuse runtime to avoid re-init errors during grid sweeps
    if runtime is None:
        device_type = autodetect_device_type()
        ddp, rank, local_rank, world_size, device = compute_init(device_type)
        runtime = (ddp, rank, local_rank, world_size, device, device_type)
    else:
        ddp, rank, local_rank, world_size, device, device_type = runtime
    master = rank == 0

    # Reproducibility: set all seeds and enable deterministic mode
    set_seed(seed, rank=rank)
    if deterministic:
        enable_deterministic_mode()
    run_repro = repro_context() | {
        "seed": seed,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "device": str(device),
    }
    if master:
        print0(f"[REPRO] {json.dumps(run_repro, indent=2)}")

    # wandb
    use_dummy = run == "dummy" or not master
    run_name = run
    if (grid or grid_i != 0 or grid_j != 0) and run:
        run_name = f"{run}-g{grid_i}-{grid_j}"

    # Build tags list from wandb_tags + grid_sweep_id; always include base "finetune" tag
    user_tags = [t.strip() for t in wandb_tags.split(",") if t.strip()]
    if not any(t.lower() == "finetune" for t in user_tags):
        user_tags.append("finetune")
    sweep_id = grid_sweep_id or datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_tags = list(dict.fromkeys(user_tags + [sweep_id]))  # preserve order, de-dup

    # Effective differential LRs (chat_sft-style) derived from the sweep axis.
    eff_matrix_lr = float(matrix_lr) if matrix_lr > 0 else lr
    eff_embedding_lr = float(embedding_lr) if embedding_lr > 0 else eff_matrix_lr * float(embedding_lr_mult)
    eff_unembedding_lr = float(unembedding_lr) if unembedding_lr > 0 else eff_matrix_lr * float(unembedding_lr_mult)
    trainable_groups = parse_trainable_param_groups(trainable_param_groups)
    bad_lr = (
        ("matrix" in trainable_groups and eff_matrix_lr <= 0)
        or ("embedding" in trainable_groups and eff_embedding_lr <= 0)
        or ("unembedding" in trainable_groups and eff_unembedding_lr <= 0)
    )
    if bad_lr:
        raise ValueError(
            f"Non-positive learning rates for enabled groups: matrix_lr={eff_matrix_lr} embedding_lr={eff_embedding_lr} "
            f"unembedding_lr={eff_unembedding_lr} (trainable_param_groups={trainable_param_groups!r})"
        )
    log_matrix_lr = eff_matrix_lr if "matrix" in trainable_groups else 0.0
    log_embedding_lr = eff_embedding_lr if "embedding" in trainable_groups else 0.0
    log_unembedding_lr = eff_unembedding_lr if "unembedding" in trainable_groups else 0.0

    wb = (
        DummyWandb()
        if use_dummy
        else wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=run_name,
            config=user_config
            | {
                "learning_rate": eff_matrix_lr,
                "num_tokens": tokens_goal,
                "grid_i": grid_i,
                "grid_j": grid_j,
                "seed": seed,
                "repro": run_repro,
                "grid_sweep_id": sweep_id,
                "matrix_lr": log_matrix_lr,
                "embedding_lr": log_embedding_lr,
                "unembedding_lr": log_unembedding_lr,
                "trainable_groups": ",".join(sorted(trainable_groups)),
                # Multi-devpod identification (top-level for easy W&B filtering)
                "hostname": run_repro.get("hostname"),
                "devpod_name": run_repro.get("devpod_name"),
            },
            tags=run_tags,
            save_code=True,
            settings=wandb.Settings(init_timeout=300, _service_wait=300),
        )
    )

    global_step = 0
    if not use_dummy:
        wandb.define_metric("global_step")
        wandb.define_metric("*", step_metric="global_step")

    def wb_log(payload: dict, step: int):
        step_i = int(step)
        wb.log(({"global_step": step_i} | payload), step=step_i)

    model, tokenizer = load_model_and_tok(device_type)
    # Cast model to float32 if dtype is float32, fp32 will have higher precision for granular sweeps
    if dtype == "float32":
        model = model.float()
    model.to(device)
    param_counts = apply_trainable_param_groups(model, trainable_groups)
    if master:
        print0(
            f"[PARAMS] trainable_param_groups={trainable_param_groups!r} "
            f"trainable={param_counts['trainable']:,}/{param_counts['total']:,} "
            f"(matrix={param_counts['matrix']:,} emb={param_counts['embedding']:,} unemb={param_counts['unembedding']:,})"
        )
    wb_log(
        {
            "model/params_total": int(param_counts["total"]),
            "model/params_trainable": int(param_counts["trainable"]),
            "model/params_trainable_matrix": int(param_counts["matrix"]),
            "model/params_trainable_embedding": int(param_counts["embedding"]),
            "model/params_trainable_unembedding": int(param_counts["unembedding"]),
        },
        step=global_step,
    )
    if device_type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    dataloader = build_dataloader(tokenizer, seed=seed, world_size=world_size, rank=rank)
    pad_token_id = tokenizer.encode_special("<|assistant_end|>")
    data_iter = iter(dataloader)

    # Peek first batch to estimate tokens/batch (then reuse it)
    first_batch = next(data_iter)
    _, labels_peek = [x.to(device) for x in first_batch]
    tokens_per_microbatch = int((labels_peek >= 0).sum().item())
    tokens_per_microbatch_t = torch.tensor(tokens_per_microbatch, device=device)
    if ddp:
        dist.all_reduce(tokens_per_microbatch_t, op=dist.ReduceOp.SUM)
    tokens_per_microbatch_global = int(tokens_per_microbatch_t.item())
    data_iter = itertools.chain([first_batch], data_iter)

    examples_per_step = device_batch_size * world_size
    if target_examples_per_step <= 0:
        grad_accum_steps = 1
    else:
        assert (
            target_examples_per_step % examples_per_step == 0
        ), "Target examples per step must be divisible by examples per step"
        grad_accum_steps = target_examples_per_step // examples_per_step
    total_steps = num_iterations if num_iterations > 0 else derive_num_iterations(
        tokens_per_microbatch_global, grad_accum_steps, tokens_goal
    )
    warmup_steps = max(1, int(total_steps * warmup_frac))
    # Optimizers: match nanochat/scripts/chat_sft.py (AdamW for embedding+lm_head, Muon for matrix weights).
    if not hasattr(model, "setup_optimizers"):
        raise AttributeError(
            f"Expected a nanochat GPT model with setup_optimizers(), got {type(model)}. "
            "If you intended to use a different model, add an optimizer implementation for it."
        )
    optimizers = model.setup_optimizers(
        unembedding_lr=eff_unembedding_lr,
        embedding_lr=eff_embedding_lr,
        matrix_lr=eff_matrix_lr,
        weight_decay=weight_decay,
    )
    # DistAdamW/DistMuon assume all params they see have grads, so strip frozen params from each group.
    for opt in optimizers:
        for group in opt.param_groups:
            group["params"] = [p for p in group["params"] if p.requires_grad]
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["lr"] * init_lr_frac
            group["initial_lr"] = group["lr"]
            if len(group["params"]) == 0:
                group["lr"] = 0.0
                group["initial_lr"] = 0.0

    def lr_mult(step_idx: int) -> float:
        # linear warmup
        if step_idx < warmup_steps:
            return (step_idx + 1) / warmup_steps
        # cosine decay from 1.0 to final_lr_frac
        progress = (step_idx - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return final_lr_frac + (1.0 - final_lr_frac) * cosine_decay

    # metrics
    losses: List[float] = []
    tokens_seen = 0
    start_time = time.time()
    error = None
    final_loss = float("inf")
    trainable_loss0 = float("nan")
    trainable_mean_last_k = float("nan")
    trainable_ratio = float("nan")

    bf16_ok = device_type == "cuda" and torch.cuda.is_bf16_supported()
    use_bf16 = dtype == "bfloat16" and bf16_ok
    use_fp16 = dtype == "float16" and device_type == "cuda"
    use_autocast = use_bf16 or use_fp16
    autocast_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)
    scaler = GradScaler(enabled=use_fp16)  # disabled for fp32/bf16

    if use_autocast:
        def cast_ctx():
            return torch.amp.autocast(device_type=device_type, dtype=autocast_dtype)
    else:
        def cast_ctx():
            return nullcontext()

    # Log baseline samples for before/after comparison (use validation data)
    baseline_samples = []
    generations_table = None
    before_data = None
    if master:
        baseline_loader = build_dataloader(tokenizer, seed=seed + 999, world_size=1, rank=0, split="validation")
        baseline_samples = capture_baseline_samples(baseline_loader, device, num_samples=5)
        print0(f"Captured {len(baseline_samples)} validation samples for before/after comparison")

        # Generate predictions BEFORE training and log to wandb
        if baseline_samples and not use_dummy:
            generations_table = wandb.Table(
                columns=["step", "sample_idx", "prompt", "generated", "expected"],
                log_mode="INCREMENTAL",
            )
            before_data = generate_sample_predictions(model, tokenizer, baseline_samples, device, cast_ctx, step=0)
            if before_data:
                for row in before_data:
                    generations_table.add_data(*row)
                wb_log({"samples/generations_incr": generations_table}, step=global_step)
                print0(f"Logged {len(before_data)} validation sample predictions (before training) to wandb table (incremental)")
                print_generation_table(before_data, stage="before")

    # Diagnostic: compute initial loss before any training
    if master:
        print0("=" * 60)
        print0("DIAGNOSTIC: Initial state before training")
        print0("=" * 60)
        model.eval()
        with torch.no_grad():
            diag_batch = next(iter(dataloader))
            diag_input, diag_labels = [b.to(device) for b in diag_batch]
            with cast_ctx():
                init_loss = model(diag_input, diag_labels).item()
            # Token statistics (exclude padding)
            nonpad_mask = (diag_input != pad_token_id) | (diag_labels >= 0)
            total_tokens = int(nonpad_mask.sum().item())
            trained_tokens = int((diag_labels >= 0).sum().item())
            masked_tokens = total_tokens - trained_tokens
            print0(f"Initial loss (before training): {init_loss:.4f}")
            print0(f"Batch shape: {diag_input.shape}")
            print0(f"Total non-pad tokens in batch: {total_tokens}")
            pct_trained = 0.0 if total_tokens == 0 else 100 * trained_tokens / total_tokens
            print0(f"Trained tokens (non-masked): {trained_tokens} ({pct_trained:.1f}%)")
            print0(f"Masked tokens (non-pad): {masked_tokens} ({100 - pct_trained:.1f}%)")
            wb_log({"diagnostic/initial_loss": init_loss, "diagnostic/trained_token_pct": pct_trained}, step=global_step)
        model.train()
        print0("=" * 60)

    # Visualization: show predictions before training (fallback if no table data)
    if master and visualize and before_data is None:
        val_loader = build_dataloader(tokenizer, seed=seed + 999, world_size=1, rank=0, split="validation")
        visualize_predictions(model, tokenizer, val_loader, device, cast_ctx, num_samples=3, stage="before")

    # Training loop
    print0("=" * 60)
    print0("Training loop")
    print0("=" * 60)
    try:
        for step_idx in range(total_steps):
            model.zero_grad(set_to_none=True)
            num_tokens_step = 0
            running_loss = 0.0
            for _ in range(grad_accum_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    batch = next(data_iter)
                batch = [b.to(device) for b in batch]
                b_input, b_labels = batch
                with cast_ctx():
                    loss = model(b_input, b_labels)
                scaled_loss = loss / grad_accum_steps
                if scaler.is_enabled():
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                running_loss += loss.detach()
                num_tokens_step += int((b_labels >= 0).sum().item())

            # lr schedule
            mult = lr_mult(step_idx)
            for opt in optimizers:
                for group in opt.param_groups:
                    group["lr"] = group["initial_lr"] * mult
            if scaler.is_enabled():
                for opt in optimizers:
                    scaler.step(opt)
                scaler.update()
            else:
                for opt in optimizers:
                    opt.step()

            tokens_step_t = torch.tensor(num_tokens_step, device=device)
            if ddp:
                dist.all_reduce(tokens_step_t, op=dist.ReduceOp.SUM)
            tokens_seen += int(tokens_step_t.item())

            # reduce loss for logging
            loss_item = running_loss / grad_accum_steps
            if ddp:
                dist.all_reduce(loss_item, op=dist.ReduceOp.AVG)
            loss_scalar = loss_item.item()
            final_loss = loss_scalar
            if not math.isfinite(loss_scalar):
                error = "non-finite loss detected"
                break
            global_step += 1
            losses.append(loss_scalar)

            if master and (step_idx % log_every == 0 or step_idx + 1 == total_steps):
                if len(optimizers) != 2:
                    raise ValueError(f"Expected 2 optimizers (AdamW, Muon), got {len(optimizers)}")
                adamw_opt, muon_opt = optimizers
                if len(adamw_opt.param_groups) < 2:
                    raise ValueError(
                        f"Expected AdamW to have >=2 param groups (lm_head, embedding), got {len(adamw_opt.param_groups)}"
                    )
                lr_matrix_now = float(muon_opt.param_groups[0]["lr"])
                lr_unembedding_now = float(adamw_opt.param_groups[0]["lr"])
                lr_embedding_now = float(adamw_opt.param_groups[1]["lr"])
                lr_log = {
                    "step": step_idx,
                    "train/loss": loss_scalar,
                    "train/lr_mult": mult,
                    "train/lr": lr_matrix_now,
                    "train/lr_matrix": lr_matrix_now,
                    "train/lr_embedding": lr_embedding_now,
                    "train/lr_unembedding": lr_unembedding_now,
                    "train/tokens_seen": tokens_seen,
                }

                lr_str = (
                    f"lr_matrix={lr_log['train/lr_matrix']:.2e} "
                    f"lr_emb={lr_log['train/lr_embedding']:.2e} "
                    f"lr_unemb={lr_log['train/lr_unembedding']:.2e}"
                )
                print0(f"step {step_idx+1:05d}/{total_steps:05d} loss={loss_scalar:.4f} {lr_str} tokens={tokens_seen:,}")
                wb_log(lr_log, step=global_step)

            if master and eval_every > 0 and (step_idx + 1) % eval_every == 0:
                # Lightweight validation on a few batches (rank 0 only)
                model.eval()
                val_loader = build_dataloader(tokenizer, seed=seed + 999, world_size=1, rank=0, split="validation")
                val_losses = []
                with torch.no_grad():
                    for b_idx, (v_input, v_labels) in enumerate(val_loader):
                        if b_idx >= eval_batches:
                            break
                        v_input, v_labels = v_input.to(device), v_labels.to(device)
                        with cast_ctx():
                            v_loss = model(v_input, v_labels)
                        val_losses.append(v_loss.item())
                if val_losses:
                    val_loss_mean = float(np.mean(val_losses))
                    wb_log({"val/loss": val_loss_mean, "val/batches": len(val_losses)}, step=global_step)
                    print0(f"eval loss={val_loss_mean:.4f} over {len(val_losses)} batches")
                model.train()

    except Exception as exc:  # pylint: disable=broad-except
        error = str(exc)
        print0(f"[ERROR] training failed: {error}")

    runtime_s = time.time() - start_time
    stable = error is None and bool(losses) and math.isfinite(final_loss)
    if losses:
        trainable_loss0 = float(losses[0])
        k = min(int(trainable_window_steps), len(losses))
        trainable_mean_last_k = float(np.mean(losses[-k:]))
        trainable_ratio = (
            float("inf")
            if trainable_loss0 <= 0 or not math.isfinite(trainable_loss0)
            else trainable_mean_last_k / trainable_loss0
        )
    trainable = stable and math.isfinite(trainable_ratio) and trainable_ratio < float(trainable_loss_ratio_threshold)
    converged = trainable
    avg_loss = float(np.mean(losses)) if losses else float("inf")

    if device_type == "cuda" and master:
        max_alloc_gb = torch.cuda.max_memory_allocated() / (1024**3)
        max_reserved_gb = torch.cuda.max_memory_reserved() / (1024**3)
        print0(f"[MEM] max_allocated_gb={max_alloc_gb:.2f} max_reserved_gb={max_reserved_gb:.2f}")
        wb_log({"memory/max_allocated_gb": max_alloc_gb, "memory/max_reserved_gb": max_reserved_gb}, step=global_step)

    # Generate predictions AFTER training on same baseline samples and log to wandb
    after_data = None
    if master and generations_table is not None and baseline_samples and not use_dummy:
        after_data = generate_sample_predictions(model, tokenizer, baseline_samples, device, cast_ctx, step=global_step)
        if after_data:
            for row in after_data:
                generations_table.add_data(*row)
            wb_log({"samples/generations_incr": generations_table}, step=global_step)
            print0(f"Logged {len(after_data)} validation sample predictions (after training, step {global_step}) to wandb table (incremental)")
            print_generation_table(after_data, stage="after")

    # Log a final immutable table for stable viewing in the W&B UI
    if master and generations_table is not None and not use_dummy and generations_table.data:
        final_table = wandb.Table(
            columns=generations_table.columns,
            data=generations_table.data,
            log_mode="IMMUTABLE",
        )
        wb_log({"samples/generations": final_table}, step=global_step)
        print0(f"Logged final generations table with {len(generations_table.data)} rows to wandb")

    # Display: show predictions after training (fallback if no table data)
    if master and visualize and after_data is None:
        val_loader = build_dataloader(tokenizer, seed=seed + 999, world_size=1, rank=0, split="validation")
        visualize_predictions(model, tokenizer, val_loader, device, cast_ctx, num_samples=3, stage="after")

    # Save checkpoint and upload artifact (single runs only, never during grid sweeps)
    if master and save_artifacts and not (grid or grid_sweep_id):
        raw_model = getattr(model, "module", model)

        # Save checkpoint locally
        checkpoint_dir = CHECKPOINTS_ROOT / run_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model_config_kwargs = raw_model.config.__dict__ if hasattr(raw_model, "config") else {}
        save_checkpoint(
            str(checkpoint_dir),
            global_step,
            raw_model.state_dict(),
            None,  # don't save optimizer state
            {
                "step": global_step,
                "final_loss": final_loss,
                "tokens_seen": tokens_seen,
                "converged": converged,
                "stable": stable,
                "trainable_ratio": trainable_ratio,
                "learning_rate": eff_matrix_lr,
                "matrix_lr": eff_matrix_lr,
                "embedding_lr": eff_embedding_lr,
                "unembedding_lr": eff_unembedding_lr,
                "model_config": model_config_kwargs,
            },
            rank=0,
        )
        print0(f"Saved checkpoint to {checkpoint_dir}")

        # Upload to W&B as artifact
        if not use_dummy:
            artifact_name = f"{run_name}-artifact" if not run_name.endswith("-artifact") else run_name
            art = wandb.Artifact(artifact_name, type="model")
            art.add_dir(str(checkpoint_dir), name="checkpoints")
            # Bundle tokenizer if available
            tok_dir = Path(get_base_dir()) / "tokenizer"
            if tok_dir.is_dir():
                art.add_dir(str(tok_dir), name="tokenizer")
            wb.log_artifact(art, aliases=["finetune", run_name, "latest"])
            print0(f"Uploaded artifact: {artifact_name}")

    if master and not use_dummy:
        wb.summary["final_train_loss"] = final_loss
        wb.summary["converged"] = converged
        wb.summary["stable"] = stable
        wb.summary["trainable_loss0"] = trainable_loss0
        wb.summary["trainable_mean_last_k"] = trainable_mean_last_k
        wb.summary["trainable_ratio"] = trainable_ratio
        wb.summary["tokens_seen"] = tokens_seen
        wb.summary["runtime_s"] = runtime_s
        wb.finish()

    # cleanup
    del optimizers
    del model
    torch.cuda.empty_cache()

    return RunResult(
        run_name=run,
        learning_rate=eff_matrix_lr,
        num_tokens_target=tokens_goal,
        tokens_seen=tokens_seen,
        avg_loss=avg_loss,
        final_loss=final_loss,
        trainable_loss0=trainable_loss0,
        trainable_mean_last_k=trainable_mean_last_k,
        trainable_ratio=trainable_ratio,
        stable=stable,
        converged=converged,
        steps=global_step,
        runtime_s=runtime_s,
        seed=seed,
        world_size=world_size,
        device_type=device_type,
        error=error,
        grid_i=grid_i,
        grid_j=grid_j,
    )


# ---------------------------------------------------------------------------
# Grid search orchestrator

def logspace(min_v: float, max_v: float, n: int):
    return np.logspace(np.log10(min_v), np.log10(max_v), n)


def run_grid_search():
    learning_rates = logspace(lr_min, lr_max, resolution)
    token_counts = logspace(tokens_min, tokens_max, resolution).astype(int)

    # initialize once and reuse across grid points
    device_type = autodetect_device_type()
    runtime = compute_init(device_type)
    runtime = runtime + (device_type,)

    if int(os.environ.get("RANK", 0)) == 0:
        print0(f"Grid search {resolution}x{resolution} ({resolution*resolution} runs)")
    results: List[RunResult] = []

    for i, lr_val in enumerate(learning_rates):
        for j, tok_val in enumerate(token_counts):
            if int(os.environ.get("RANK", 0)) == 0:
                print0(f"\n=== Grid ({i},{j}) lr={lr_val:.2e} tokens={tok_val:,} ===")
            res = train_once(
                learning_rate_override=float(lr_val),
                num_tokens_override=int(tok_val),
                grid_i=i,
                grid_j=j,
                runtime=runtime,
            )
            if int(os.environ.get("RANK", 0)) == 0:
                results.append(res)
                # checkpoint to disk
                ckpt_dir = RESULTS_ROOT
                ckpt_dir.mkdir(exist_ok=True, parents=True)
                ckpt_path = ckpt_dir / f"finetune_grid_{resolution}x{resolution}.json"
                with open(ckpt_path, "w", encoding="utf-8") as f:
                    json.dump([asdict(r) for r in results], f, indent=2)
                print0(f"Checkpointed {len(results)} results to {ckpt_path}")

    # Rank 0: finalize artifacts
    if int(os.environ.get("RANK", 0)) == 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = RESULTS_ROOT
        results_dir.mkdir(exist_ok=True, parents=True)
        prefix = results_dir / f"finetune_grid_{resolution}x{resolution}_{timestamp}"
        results_path = prefix.with_suffix(".json")

        config_dump = {
            "resolution": resolution,
            "lr_min": lr_min,
            "lr_max": lr_max,
            "tokens_min": tokens_min,
            "tokens_max": tokens_max,
            "model_id": model_id,
            "dataset_id": dataset_id,
            "max_seq_len": max_seq_len,
            "trainable_window_steps": trainable_window_steps,
            "trainable_loss_ratio_threshold": trainable_loss_ratio_threshold,
            "repro": repro_context(),
        }

        with open(results_path, "w", encoding="utf-8") as f:
            json.dump({"config": config_dump, "results": [asdict(r) for r in results]}, f, indent=2)

        img_path, convergence_grid = save_visualizations(
            results,
            resolution,
            lr_min,
            lr_max,
            tokens_min,
            tokens_max,
            prefix,
        )
        fractal = compute_fractal(convergence_grid)
        fractal_path = prefix.with_suffix(".fractal.json")
        with open(fractal_path, "w", encoding="utf-8") as f:
            json.dump(fractal, f, indent=2)

        print0(f"Saved results: {results_path}")
        print0(f"Saved visualization: {img_path}")
        print0(f"Fractal dimension: {fractal['fractal_dimension']:.3f}")

        if run != "dummy":
            summary_tags = ["fractal-grid", "finetune"]
            summary_run = wandb.init(
                project=WANDB_PROJECT,
                entity=WANDB_ENTITY,
                name=f"{run}-grid-summary",
                config=config_dump | {"fractal_dimension": fractal["fractal_dimension"]},
                tags=summary_tags,
            )

            summary_run.log(
                {
                    "fractal/image": wandb.Image(str(img_path), caption="LR × Tokens Grid"),
                    "fractal/dimension": fractal["fractal_dimension"],
                    "fractal/boundary_pixels": fractal["boundary_pixels"],
                    "fractal/converged_ratio": fractal["converged_ratio"],
                }
            )

            results_table = wandb.Table(
                columns=[
                    "grid_i",
                    "grid_j",
                    "learning_rate",
                    "num_tokens",
                    "tokens_seen",
                    "final_loss",
                    "trainable_ratio",
                    "stable",
                    "converged",
                    "runtime_s",
                    "error",
                ],
                data=[
                    [
                        r.grid_i,
                        r.grid_j,
                        r.learning_rate,
                        r.num_tokens_target,
                        r.tokens_seen,
                        r.final_loss,
                        r.trainable_ratio,
                        r.stable,
                        r.converged,
                        r.runtime_s,
                        r.error,
                    ]
                    for r in results
                ],
            )
            summary_run.log({"results_table": results_table})
            summary_run.finish()

    return results


# ---------------------------------------------------------------------------
# Entry

def main():
    if grid:
        run_grid_search()
    else:
        device_type = autodetect_device_type()
        runtime = compute_init(device_type)
        runtime = runtime + (device_type,)
        ddp, rank, local_rank, world_size, device, _device_type = runtime
        res = train_once(runtime=runtime, grid_i=grid_i, grid_j=grid_j)
        if int(os.environ.get("RANK", 0)) == 0:
            print0(
                f"\nFinal: loss={res.final_loss:.4f} tokens_seen={res.tokens_seen:,} "
                f"stable={res.stable} converged={res.converged} trainable_ratio={res.trainable_ratio:.4f}"
            )

            # Save results JSON (single run artifact)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = RESULTS_ROOT
            results_dir.mkdir(exist_ok=True, parents=True)
            results_path = results_dir / f"finetune_single_{run}_{timestamp}.json"
            repro = repro_context() | {
                "seed": res.seed,
                "world_size": res.world_size,
                "device_type": res.device_type,
            }
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump({"config": user_config, "result": asdict(res), "repro": repro}, f, indent=2)
            print0(f"Saved results: {results_path}")

        # Treat non-finite loss as an expected "diverged" outcome for sweep grids.
        # Only crash the process for unexpected exceptions (e.g., OOM).
        exit_code = 1 if (res.error is not None and res.error != "non-finite loss detected") else 0
        if ddp and dist.is_available() and dist.is_initialized():
            # Sync failure across ranks so torchrun sees a consistent exit status.
            flag = torch.tensor(exit_code, device=device)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            exit_code = int(flag.item())

        compute_cleanup()
        if exit_code != 0:
            raise SystemExit(exit_code)
        return

    compute_cleanup()


if __name__ == "__main__":
    main()

```


### `src/grid_sweep_summary.py`

```python
"""
Summarize a parallel grid sweep launched by scripts/grid_sweep.sh.

This reads the per-point log files (run_<i>_<j>.log), reconstructs the 2D grid,
renders the same 3-panel grid visualization as src/finetune.py, and logs a single
W&B summary run with the image + a results table.

Example:
  uv run python -m src.grid_sweep_summary \
    --log_dir /var/tmp/fractal-llm/results/grid_logs/my-sweep \
    --run_prefix my-sweep \
    --grid_sweep_id my-sweep \
    --sweep_axes matrix_unembedding \
    --resolution 2
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import simple_parsing as sp
import wandb
from matplotlib.colors import LinearSegmentedColormap
from rich.console import Console
from rich.panel import Panel
from scipy import ndimage

console = Console()


@dataclass
class Args:
    """Summarize a grid sweep and log a W&B summary run."""

    log_dir: Path  # Directory containing run_<i>_<j>.log files
    run_prefix: str  # Same prefix passed to finetune as --run
    resolution: int = 4  # Grid resolution per axis (RES)
    sweep_axes: str = "matrix_unembedding"  # matrix_unembedding | lr_tokens
    grid_sweep_id: str = ""  # Tag shared across all points (defaults to run_prefix)
    wandb_project: str = os.environ.get("WANDB_PROJECT", "fractal-llm")
    wandb_entity: str = os.environ.get("WANDB_ENTITY", "morgy")
    wandb_tags: str = os.environ.get("FINETUNE_WANDB_TAGS", "fractal-grid")
    storage_dir: Path = Path(os.environ.get("FRACTAL_STORAGE_DIR", "/var/tmp/fractal-llm"))


@dataclass
class PointResult:
    grid_i: int
    grid_j: int
    num_tokens: int
    tokens_seen: int | None
    final_loss: float | None
    trainable_ratio: float | None
    stable: bool | None
    converged: bool | None
    error: str | None
    log: str
    learning_rate: float | None = None
    matrix_lr: float | None = None
    unembedding_lr: float | None = None


def _parse_overrides(text: str) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for m in re.finditer(r"^Overriding: ([a-zA-Z0-9_]+) = (.+)$", text, flags=re.MULTILINE):
        overrides[m.group(1)] = m.group(2).strip()
    return overrides


def _parse_final_line(text: str) -> tuple[float | None, int | None, bool | None, bool | None, float | None]:
    # New (2026-01): includes stable + trainable_ratio.
    m = re.search(
        r"Final: loss=([0-9.]+) tokens_seen=([0-9,]+) stable=(True|False) converged=(True|False) trainable_ratio=([0-9.]+)",
        text,
    )
    if m:
        loss = float(m.group(1))
        tokens_seen = int(m.group(2).replace(",", ""))
        stable = m.group(3) == "True"
        converged = m.group(4) == "True"
        trainable_ratio = float(m.group(5))
        return loss, tokens_seen, stable, converged, trainable_ratio

    # Old format (backwards compatible)
    m = re.search(
        r"Final: loss=([0-9.]+) tokens_seen=([0-9,]+) converged=(True|False)",
        text,
    )
    if not m:
        return None, None, None, None, None
    loss = float(m.group(1))
    tokens_seen = int(m.group(2).replace(",", ""))
    converged = m.group(3) == "True"
    return loss, tokens_seen, None, converged, None


def _parse_error(text: str) -> str | None:
    errs = re.findall(r"^\[ERROR\] training failed: (.+)$", text, flags=re.MULTILINE)
    return errs[-1].strip() if errs else None


def _fractal_cmap():
    """Create a diverging red-white-blue colormap for trainability visualization.

    Color mapping (vmin=-1, vmax=1):
      -1.0: Dark red (#8B0000) - Diverged/failed runs
      -0.5: Salmon (#FA8072) - (unused in practice, all diverged → -1.0)
       0.0: White (#FFFFFF) - Boundary (unused, converged starts at 0.3)
       0.3: Light blue (#ADD8E6) - Converged, highest loss among converged
       0.65: Royal blue (#4169E1) - Converged, medium loss
       1.0: Dark blue (#00008B) - Converged, lowest loss (best)
    """
    # More granular color stops for smoother transitions
    colors = [
        "#8B0000",  # Dark red (diverged)
        "#B22222",  # Firebrick
        "#CD5C5C",  # Indian red
        "#FA8072",  # Salmon
        "#FFC0CB",  # Pink (light diverged)
        "#FFFFFF",  # White (boundary)
        "#E0FFFF",  # Light cyan
        "#ADD8E6",  # Light blue (worst converged)
        "#87CEEB",  # Sky blue
        "#4169E1",  # Royal blue
        "#0000CD",  # Medium blue
        "#00008B",  # Dark blue (best converged)
    ]
    positions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.65, 0.75, 0.85, 0.92, 1.0]
    return LinearSegmentedColormap.from_list("fractal", list(zip(positions, colors)))


def _build_grids(points: list[PointResult], resolution: int):
    fractal_grid = np.zeros((resolution, resolution))
    loss_grid = np.full((resolution, resolution), np.nan)
    convergence_grid = np.zeros((resolution, resolution))

    converged_losses: list[float] = []
    for p in points:
        if p.converged and p.final_loss is not None and math.isfinite(p.final_loss):
            loss_grid[p.grid_i, p.grid_j] = p.final_loss
            converged_losses.append(p.final_loss)
            convergence_grid[p.grid_i, p.grid_j] = 1
        else:
            convergence_grid[p.grid_i, p.grid_j] = 0

    if converged_losses:
        loss_min, loss_max = min(converged_losses), max(converged_losses)
        loss_range = loss_max - loss_min if loss_max > loss_min else 1.0
    else:
        loss_min, loss_max, loss_range = 0.0, 1.0, 1.0

    for p in points:
        if p.converged and p.final_loss is not None and math.isfinite(p.final_loss):
            # Invert intensity: lower loss → darker blue (value closer to 1.0)
            intensity = (p.final_loss - loss_min) / loss_range
            fractal_grid[p.grid_i, p.grid_j] = 1.0 - 0.7 * intensity  # Range: 0.3 (worst) to 1.0 (best)
        else:
            fractal_grid[p.grid_i, p.grid_j] = -1.0

    return fractal_grid, loss_grid, convergence_grid


def _compute_fractal(convergence_grid: np.ndarray):
    boundary = ndimage.binary_dilation(convergence_grid) ^ ndimage.binary_erosion(convergence_grid)

    def box_count(binary_image, box_size):
        h, w = binary_image.shape
        count = 0
        for i in range(0, h, box_size):
            for j in range(0, w, box_size):
                if binary_image[i : i + box_size, j : j + box_size].any():
                    count += 1
        return count

    sizes = [s for s in [2, 4, 8, 16, 32, 64] if s < boundary.shape[0]]
    if sizes:
        counts = [box_count(boundary, s) for s in sizes]
        log_sizes = np.log(sizes)
        log_counts = np.log(np.array(counts) + 1)
        coeffs = np.polyfit(log_sizes, log_counts, 1)
        fractal_dimension = -coeffs[0]
    else:
        counts = []
        fractal_dimension = float("nan")
    return {
        "fractal_dimension": float(fractal_dimension),
        "box_sizes": sizes,
        "box_counts": [int(c) for c in counts],
        "boundary_pixels": int(boundary.sum()),
        "converged_ratio": float(convergence_grid.sum() / convergence_grid.size),
    }


def _safe_extent(vmin: float, vmax: float) -> tuple[float, float]:
    if vmin <= 0 or vmax <= 0:
        raise ValueError(f"Extent must be positive; got vmin={vmin} vmax={vmax}")
    if vmin == vmax:
        vmin = vmin * 0.9
        vmax = vmax * 1.1
    return math.log10(vmin), math.log10(vmax)


def summarize_and_log(args: Args) -> tuple[Path, Path, str]:
    log_dir = args.log_dir.expanduser().resolve()
    if not log_dir.exists():
        raise FileNotFoundError(str(log_dir))

    sweep_id = args.grid_sweep_id or args.run_prefix

    # Keep W&B files off the workspace.
    os.environ.setdefault("WANDB_DIR", str(args.storage_dir / "wandb"))
    os.environ.setdefault("WANDB_CONFIG_DIR", str(args.storage_dir / "wandb" / "config"))
    os.environ.setdefault("WANDB_CACHE_DIR", str(args.storage_dir / "wandb" / "cache"))

    log_files = sorted(log_dir.glob("run_*.log"))
    if not log_files:
        raise RuntimeError(f"No logs found in {log_dir}")

    points: list[PointResult] = []
    name_re = re.compile(r"run_(\d+)_(\d+)\.log$")
    for log_path in log_files:
        m = name_re.search(log_path.name)
        if not m:
            continue
        gi = int(m.group(1))
        gj = int(m.group(2))
        txt = log_path.read_text(errors="replace")
        overrides = _parse_overrides(txt)
        err = _parse_error(txt)
        final_loss, tokens_seen, stable, converged, trainable_ratio = _parse_final_line(txt)
        if stable is None:
            stable = err is None and final_loss is not None and math.isfinite(final_loss)

        num_tokens = int(str(overrides.get("num_tokens", "0")).replace("_", ""))
        learning_rate = float(overrides["learning_rate"]) if "learning_rate" in overrides else None
        matrix_lr = float(overrides["matrix_lr"]) if "matrix_lr" in overrides else None
        unembedding_lr = float(overrides["unembedding_lr"]) if "unembedding_lr" in overrides else None

        points.append(
            PointResult(
                grid_i=gi,
                grid_j=gj,
                num_tokens=num_tokens,
                tokens_seen=tokens_seen,
                final_loss=final_loss,
                trainable_ratio=trainable_ratio,
                stable=stable,
                converged=converged,
                error=err,
                log=log_path.name,
                learning_rate=learning_rate,
                matrix_lr=matrix_lr,
                unembedding_lr=unembedding_lr,
            )
        )

    expected = args.resolution * args.resolution
    if len(points) != expected:
        console.print(
            Panel(
                f"Expected {expected} points (resolution={args.resolution}), found {len(points)} logs in {log_dir}",
                title="grid-summary warning",
            )
        )

    fractal_grid, loss_grid, convergence_grid = _build_grids(points, args.resolution)
    fractal = _compute_fractal(convergence_grid)

    if args.sweep_axes == "matrix_unembedding":
        xs = [p.unembedding_lr for p in points if p.unembedding_lr is not None]
        ys = [p.matrix_lr for p in points if p.matrix_lr is not None]
        x_label = "log₁₀(unembedding lr)"
        y_label = "log₁₀(matrix lr)"
        caption = "Matrix LR × Unembedding LR Grid"
    elif args.sweep_axes == "lr_tokens":
        xs = [float(p.num_tokens) for p in points if p.num_tokens and p.num_tokens > 0]
        ys = [p.learning_rate for p in points if p.learning_rate is not None]
        x_label = "log₁₀(tokens)"
        y_label = "log₁₀(learning rate)"
        caption = "LR × Tokens Grid"
    else:
        raise ValueError(f"Unknown sweep_axes={args.sweep_axes!r} (expected lr_tokens or matrix_unembedding)")

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    ex0, ex1 = _safe_extent(xmin, xmax)
    ey0, ey1 = _safe_extent(ymin, ymax)
    extent = [ex0, ex1, ey0, ey1]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    im1 = axes[0].imshow(
        fractal_grid,
        origin="lower",
        aspect="auto",
        cmap=_fractal_cmap(),
        vmin=-1.0,
        vmax=1.0,
        extent=extent,
    )
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel(y_label)
    axes[0].set_title("Trainability Boundary\n(Blue=Converged, Red=Diverged)")
    plt.colorbar(im1, ax=axes[0]).set_label("Convergence")

    im2 = axes[1].imshow(
        loss_grid,
        origin="lower",
        aspect="auto",
        cmap="viridis_r",
        extent=extent,
    )
    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel(y_label)
    axes[1].set_title("Final Loss (converged)")
    plt.colorbar(im2, ax=axes[1]).set_label("Loss")

    im3 = axes[2].imshow(
        convergence_grid,
        origin="lower",
        aspect="auto",
        cmap="RdBu",
        vmin=0,
        vmax=1,
        extent=extent,
    )
    axes[2].set_xlabel(x_label)
    axes[2].set_ylabel(y_label)
    axes[2].set_title("Binary Convergence")
    cbar3 = plt.colorbar(im3, ax=axes[2])
    cbar3.set_ticks([0, 1])
    cbar3.set_ticklabels(["Diverged", "Converged"])

    plt.tight_layout()
    out_prefix = log_dir / f"grid_summary_{args.run_prefix}"
    img_path = out_prefix.with_suffix(".png")
    json_path = out_prefix.with_suffix(".json")
    fig.savefig(img_path, dpi=150, facecolor="white")
    plt.close(fig)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "run_prefix": args.run_prefix,
                    "grid_sweep_id": sweep_id,
                    "sweep_axes": args.sweep_axes,
                    "resolution": args.resolution,
                    "wandb_project": args.wandb_project,
                    "wandb_entity": args.wandb_entity,
                },
                "fractal": fractal,
                "points": [asdict(p) for p in points],
            },
            f,
            indent=2,
        )

    tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
    if not any(t.lower() == "finetune" for t in tags):
        tags.append("finetune")
    tags += [sweep_id, "grid-summary"]
    tags = list(dict.fromkeys([t for t in tags if t]))

    run_base = args.run_prefix if args.run_prefix.endswith("-ft") else f"{args.run_prefix}-ft"
    summary_name = f"{run_base}-grid-summary"

    summary_run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=summary_name,
        config={
            "run_prefix": args.run_prefix,
            "grid_sweep_id": sweep_id,
            "sweep_axes": args.sweep_axes,
            "resolution": args.resolution,
            "fractal_dimension": fractal["fractal_dimension"],
            "boundary_pixels": fractal["boundary_pixels"],
            "converged_ratio": fractal["converged_ratio"],
        },
        tags=tags,
        settings=wandb.Settings(init_timeout=300, _service_wait=300),
    )
    summary_run.log(
        {
            "fractal/image": wandb.Image(str(img_path), caption=caption),
            "fractal/dimension": fractal["fractal_dimension"],
            "fractal/boundary_pixels": fractal["boundary_pixels"],
            "fractal/converged_ratio": fractal["converged_ratio"],
        }
    )

    if args.sweep_axes == "matrix_unembedding":
        columns = [
            "grid_i",
            "grid_j",
            "matrix_lr",
            "unembedding_lr",
            "num_tokens",
            "tokens_seen",
            "final_loss",
            "trainable_ratio",
            "stable",
            "converged",
            "error",
            "log",
        ]
        data = [
            [
                p.grid_i,
                p.grid_j,
                p.matrix_lr,
                p.unembedding_lr,
                p.num_tokens,
                p.tokens_seen,
                p.final_loss,
                p.trainable_ratio,
                p.stable,
                p.converged,
                p.error,
                p.log,
            ]
            for p in points
        ]
    else:
        columns = [
            "grid_i",
            "grid_j",
            "learning_rate",
            "num_tokens",
            "tokens_seen",
            "final_loss",
            "trainable_ratio",
            "stable",
            "converged",
            "error",
            "log",
        ]
        data = [
            [
                p.grid_i,
                p.grid_j,
                p.learning_rate,
                p.num_tokens,
                p.tokens_seen,
                p.final_loss,
                p.trainable_ratio,
                p.stable,
                p.converged,
                p.error,
                p.log,
            ]
            for p in points
        ]
    summary_run.log({"results_table": wandb.Table(columns=columns, data=data)})
    summary_run.finish()

    return img_path, json_path, summary_name


def main():
    args = sp.parse(Args)
    console.rule("[bold]grid sweep summary[/bold]")
    img_path, json_path, summary_name = summarize_and_log(args)
    console.print(
        Panel(
            f"[bold]Logged:[/bold] {args.wandb_entity}/{args.wandb_project} :: {summary_name}\n"
            f"[bold]Image:[/bold] {img_path}\n"
            f"[bold]JSON:[/bold]  {json_path}",
            title="grid-summary",
        )
    )


if __name__ == "__main__":
    main()

```


### `src/visualize.py`

```python
"""
Visualization tools for fractal LLM training experiments.

Creates heatmaps of convergence/divergence boundaries and fractal analysis plots.
"""

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from scipy import ndimage
from rich.console import Console
import simple_parsing as sp

console = Console()


def load_results(results_path: str | Path) -> tuple[dict, pd.DataFrame]:
    """Load grid search results from JSON file."""
    with open(results_path, "r") as f:
        data = json.load(f)

    config = data["config"]
    df = pd.DataFrame(data["results"])
    if "stable" not in df.columns:
        # Backwards compatibility: infer "stable" for older result files.
        # This matches the common-sense definition: no error and finite final loss.
        err_ok = ~df.get("error", pd.Series([None] * len(df))).notna()
        finite_ok = np.isfinite(df.get("final_loss", np.nan))
        df["stable"] = err_ok & finite_ok
    return config, df


def build_grid(df: pd.DataFrame, resolution: int, metric: str = "converged") -> np.ndarray:
    """Build a 2D grid from results DataFrame."""
    grid = np.full((resolution, resolution), np.nan)

    for _, row in df.iterrows():
        i, j = int(row["grid_i"]), int(row["grid_j"])
        if 0 <= i < resolution and 0 <= j < resolution:
            if metric in {"converged", "stable"}:
                v = row.get(metric, np.nan)
                if pd.isna(v):
                    continue
                grid[i, j] = 1.0 if bool(v) else 0.0
            elif metric == "loss":
                loss = row["final_loss"]
                grid[i, j] = min(loss, 20.0)  # Cap for visualization
            else:
                grid[i, j] = row.get(metric, np.nan)

    return grid


def plot_convergence_heatmap(
    config: dict,
    df: pd.DataFrame,
    output_path: str | Path | None = None,
    title: str = "Trainable/Not Trainable Boundary",
) -> plt.Figure:
    """
    Plot heatmap showing trainable ("converged") vs not trainable regions.
    """
    resolution = config["resolution"]
    grid = build_grid(df, resolution, metric="converged")

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Custom colormap: red (diverged) -> yellow (boundary) -> green (converged)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "convergence", ["#d62728", "#ffff00", "#2ca02c"]
    )

    # Create axis labels
    lr_min, lr_max = config["lr_min"], config["lr_max"]
    tokens_min, tokens_max = config["tokens_min"], config["tokens_max"]

    # Plot heatmap
    im = ax.imshow(
        grid,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=0,
        vmax=1,
        extent=[
            np.log10(tokens_min),
            np.log10(tokens_max),
            np.log10(lr_min),
            np.log10(lr_max),
        ],
    )

    ax.set_xlabel("log₁₀(Training Tokens)", fontsize=12)
    ax.set_ylabel("log₁₀(Learning Rate)", fontsize=12)
    ax.set_title(title, fontsize=14)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label="Trainable")
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["Not trainable", "Boundary", "Trainable"])

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        console.print(f"[green]Saved heatmap to {output_path}")

    return fig


def plot_stability_heatmap(
    config: dict,
    df: pd.DataFrame,
    output_path: str | Path | None = None,
    title: str = "Stable/Unstable Boundary",
) -> plt.Figure:
    """Plot heatmap showing stable (green) vs unstable (red) runs."""
    resolution = config["resolution"]
    grid = build_grid(df, resolution, metric="stable")

    fig, ax = plt.subplots(figsize=(10, 8))

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "stability", ["#d62728", "#ffff00", "#2ca02c"]
    )

    lr_min, lr_max = config["lr_min"], config["lr_max"]
    tokens_min, tokens_max = config["tokens_min"], config["tokens_max"]

    im = ax.imshow(
        grid,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=0,
        vmax=1,
        extent=[
            np.log10(tokens_min),
            np.log10(tokens_max),
            np.log10(lr_min),
            np.log10(lr_max),
        ],
    )

    ax.set_xlabel("log₁₀(Training Tokens)", fontsize=12)
    ax.set_ylabel("log₁₀(Learning Rate)", fontsize=12)
    ax.set_title(title, fontsize=14)

    cbar = plt.colorbar(im, ax=ax, label="Stable")
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["Unstable", "Boundary", "Stable"])

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        console.print(f"[green]Saved stability heatmap to {output_path}")

    return fig


def plot_loss_heatmap(
    config: dict,
    df: pd.DataFrame,
    output_path: str | Path | None = None,
    title: str = "Final Training Loss",
) -> plt.Figure:
    """Plot heatmap of final loss values."""
    resolution = config["resolution"]
    grid = build_grid(df, resolution, metric="loss")

    fig, ax = plt.subplots(figsize=(10, 8))

    lr_min, lr_max = config["lr_min"], config["lr_max"]
    tokens_min, tokens_max = config["tokens_min"], config["tokens_max"]

    # Use log scale for loss with viridis colormap
    im = ax.imshow(
        np.log10(grid + 0.1),  # +0.1 to avoid log(0)
        origin="lower",
        aspect="auto",
        cmap="viridis_r",
        extent=[
            np.log10(tokens_min),
            np.log10(tokens_max),
            np.log10(lr_min),
            np.log10(lr_max),
        ],
    )

    ax.set_xlabel("log₁₀(Training Tokens)", fontsize=12)
    ax.set_ylabel("log₁₀(Learning Rate)", fontsize=12)
    ax.set_title(title, fontsize=14)

    cbar = plt.colorbar(im, ax=ax, label="log₁₀(Loss)")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        console.print(f"[green]Saved loss heatmap to {output_path}")

    return fig


def plot_boundary(
    config: dict,
    df: pd.DataFrame,
    output_path: str | Path | None = None,
    title: str = "Trainability Boundary (Trainable vs Not Trainable)",
) -> plt.Figure:
    """Plot just the boundary between converged and diverged regions."""
    resolution = config["resolution"]
    grid = build_grid(df, resolution, metric="converged")

    # Find boundary using morphological operations
    grid_bool = grid > 0.5
    boundary = ndimage.binary_dilation(grid_bool) ^ ndimage.binary_erosion(grid_bool)

    fig, ax = plt.subplots(figsize=(10, 8))

    lr_min, lr_max = config["lr_min"], config["lr_max"]
    tokens_min, tokens_max = config["tokens_min"], config["tokens_max"]

    ax.imshow(
        boundary.astype(float),
        origin="lower",
        aspect="auto",
        cmap="binary",
        extent=[
            np.log10(tokens_min),
            np.log10(tokens_max),
            np.log10(lr_min),
            np.log10(lr_max),
        ],
    )

    ax.set_xlabel("log₁₀(Training Tokens)", fontsize=12)
    ax.set_ylabel("log₁₀(Learning Rate)", fontsize=12)
    ax.set_title(title, fontsize=14)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        console.print(f"[green]Saved boundary plot to {output_path}")

    return fig


def plot_fractal_analysis(
    config: dict,
    df: pd.DataFrame,
    output_path: str | Path | None = None,
) -> tuple[plt.Figure, float]:
    """
    Plot box-counting fractal dimension analysis.
    Returns the figure and computed fractal dimension.
    """
    resolution = config["resolution"]
    grid = build_grid(df, resolution, metric="converged")

    # Find boundary
    grid_bool = grid > 0.5
    boundary = ndimage.binary_dilation(grid_bool) ^ ndimage.binary_erosion(grid_bool)

    # Box counting
    def box_count(binary_image: np.ndarray, box_size: int) -> int:
        h, w = binary_image.shape
        count = 0
        for i in range(0, h, box_size):
            for j in range(0, w, box_size):
                box = binary_image[i : i + box_size, j : j + box_size]
                if box.any():
                    count += 1
        return count

    # Compute for multiple box sizes (include 64 for higher resolution grids)
    sizes = [2, 4, 8, 16, 32, 64]
    sizes = [s for s in sizes if s < resolution]
    counts = [box_count(boundary, s) for s in sizes]

    # Filter out zero counts
    valid_idx = [i for i, c in enumerate(counts) if c > 0]
    sizes = [sizes[i] for i in valid_idx]
    counts = [counts[i] for i in valid_idx]

    if len(sizes) < 2:
        console.print("[yellow]Not enough data points for fractal analysis")
        return None, float("nan")

    # Fit line on log-log plot
    log_sizes = np.log(sizes)
    log_counts = np.log(counts)
    coeffs = np.polyfit(log_sizes, log_counts, 1)
    fractal_dim = -coeffs[0]

    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: boundary visualization
    lr_min, lr_max = config["lr_min"], config["lr_max"]
    tokens_min, tokens_max = config["tokens_min"], config["tokens_max"]

    ax1.imshow(
        boundary.astype(float),
        origin="lower",
        aspect="auto",
        cmap="hot",
        extent=[
            np.log10(tokens_min),
            np.log10(tokens_max),
            np.log10(lr_min),
            np.log10(lr_max),
        ],
    )
    ax1.set_xlabel("log₁₀(Training Tokens)", fontsize=12)
    ax1.set_ylabel("log₁₀(Learning Rate)", fontsize=12)
    ax1.set_title("Trainability Boundary (Trainable vs Not Trainable)", fontsize=14)

    # Right: log-log plot with fit
    ax2.scatter(log_sizes, log_counts, s=100, c="blue", zorder=5)
    fit_x = np.linspace(min(log_sizes), max(log_sizes), 100)
    fit_y = coeffs[0] * fit_x + coeffs[1]
    ax2.plot(fit_x, fit_y, "r--", linewidth=2, label=f"Fit (D = {fractal_dim:.3f})")

    ax2.set_xlabel("log(Box Size)", fontsize=12)
    ax2.set_ylabel("log(Box Count)", fontsize=12)
    ax2.set_title(f"Box Counting: Fractal Dimension = {fractal_dim:.3f}", fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        console.print(f"[green]Saved fractal analysis to {output_path}")

    return fig, fractal_dim


def create_all_visualizations(results_path: str | Path, output_dir: str | Path):
    """Create all visualization outputs from a results file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    console.rule("[bold blue]Fractal LLM Visualization")

    config, df = load_results(results_path)
    console.print(f"Loaded {len(df)} results from {results_path}")
    console.print(f"Resolution: {config['resolution']}x{config['resolution']}")

    # Convergence stats (trainable == converged)
    stable = int(df["stable"].sum()) if "stable" in df.columns else 0
    converged = int(df["converged"].sum())
    total = len(df)
    console.print(f"Stable: {stable}/{total} ({100*stable/total:.1f}%)")
    console.print(f"Trainable (converged): {converged}/{total} ({100*converged/total:.1f}%)")

    # Generate plots
    console.rule("Generating Plots")

    plot_convergence_heatmap(
        config, df, output_dir / "convergence_heatmap.png"
    )

    plot_stability_heatmap(
        config, df, output_dir / "stability_heatmap.png"
    )

    plot_loss_heatmap(
        config, df, output_dir / "loss_heatmap.png"
    )

    plot_boundary(
        config, df, output_dir / "boundary.png"
    )

    fig, fractal_dim = plot_fractal_analysis(
        config, df, output_dir / "fractal_analysis.png"
    )

    if not np.isnan(fractal_dim):
        console.print(f"\n[bold green]Fractal Dimension: {fractal_dim:.3f}")

        # Interpretation
        if fractal_dim < 1.2:
            console.print("[yellow]Boundary appears relatively smooth (low fractal dimension)")
        elif fractal_dim < 1.6:
            console.print("[green]Moderate fractal structure detected")
        else:
            console.print("[bold magenta]Strong fractal structure! Boundary is highly irregular")

    console.rule("[bold green]Complete")
    console.print(f"All outputs saved to {output_dir}/")


@dataclass
class Args:
    """Visualize fractal LLM training results."""

    results_path: str  # Path to grid search results JSON
    output_dir: str = "results/figures"  # Output directory for plots


if __name__ == "__main__":
    args = sp.parse(Args)
    create_all_visualizations(args.results_path, args.output_dir)

```


### `third_party/nanochat/nanochat/adamw.py`

```python
"""
Borrowed from modded-nanogpt. By Keller, @vagrawal, et al.
Not a general optimizer! But works for our specific use.
"""
import torch
import torch.distributed as dist
from torch import Tensor


class DistAdamW(torch.optim.Optimizer):
    """
    Distributed AdamW optimizer.
    In the style of ZeRO-2, i.e. sharded optimizer states and gradient reduction
    """
    def __init__(self, param_groups, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(param_groups, defaults)

    @torch.compile
    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        reduce_scatter_futures: list[torch.Future] = []
        all_reduce_futures: list[torch.Future] = []
        grad_slices = []
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            for base_i in range(len(params)):
                assert params[base_i].shape[0] % world_size == 0, f"First dim of parameter shape {params[base_i].shape} must be divisible by world size {world_size}"
                grad = params[base_i].grad
                rank_size = grad.shape[0] // world_size
                grad_slice = torch.empty_like(grad[:rank_size])
                reduce_scatter_futures.append(dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future())
                grad_slices.append(grad_slice)

        idx = 0
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            params = group['params']
            for base in range(len(params)):
                reduce_scatter_futures[idx].wait()
                p = params[base]
                rank_size = p.shape[0] // world_size
                p_slice = p[rank * rank_size:(rank + 1) * rank_size]
                lr = group['lr'] * getattr(p, "lr_mul", 1.0)
                state = self.state[p]
                g_slice = grad_slices[idx]
                # State init
                if not state:
                    state['step'] = torch.tensor(0, dtype=torch.int64, device=p.device)
                    state['exp_avg'] = torch.zeros_like(p_slice)
                    state['exp_avg_sq'] = torch.zeros_like(p_slice)
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                state['step'] += 1
                t = state['step']
                # weight decay
                if wd != 0:
                    eff_weight_decay = lr * wd * getattr(p, "wd_mul", 1.0)
                    p_slice.mul_(1 - eff_weight_decay)
                # update running averages
                exp_avg.mul_(beta1).add_(g_slice, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g_slice, g_slice, value=1 - beta2)
                # bias corrections
                bias1 = 1 - beta1 ** t
                bias2 = 1 - beta2 ** t
                # compute step
                denom = exp_avg_sq.sqrt().add_(eps)
                step_size = lr * (torch.sqrt(bias2) / bias1)
                update = exp_avg.div(denom).mul_(step_size)
                p_slice.add_(other=update, alpha=-1.0)
                idx += 1
                all_reduce_futures.append(dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future())
        torch.futures.collect_all(all_reduce_futures).wait()

```


### `third_party/nanochat/nanochat/gpt.py`

```python
"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
"""

import math
from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0
from nanochat.muon import Muon, DistMuon
from nanochat.adamw import DistAdamW

@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768


def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin, kv_cache):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin) # QK rotary embedding
        q, k = norm(q), norm(k) # QK norm
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2) # make head be batch dim, i.e. (B, T, H, D) -> (B, H, T, D)

        # Apply KV cache: insert current k,v into cache, get the full view so far
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        Tq = q.size(2) # number of queries in this forward pass
        Tk = k.size(2) # number of keys/values in total (in the cache + current forward pass)

        # Attention: queries attend to keys/values autoregressively. A few cases to handle:
        enable_gqa = self.n_head != self.n_kv_head # Group Query Attention (GQA): duplicate key/value heads to match query heads if desired
        if kv_cache is None or Tq == Tk:
            # During training (no KV cache), attend as usual with causal attention
            # And even if there is KV cache, we can still use this simple version when Tq == Tk
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
        elif Tq == 1:
            # During inference but with a single query in this forward pass:
            # The query has to attend to all the keys/values in the cache
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)
        else:
            # During inference AND we have a chunk of queries in this forward pass:
            # First, each query attends to all the cached keys/values (i.e. full prefix)
            attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=q.device) # True = keep, False = mask
            prefix_len = Tk - Tq
            attn_mask[:, :prefix_len] = True
            # Then, causal attention within this chunk
            attn_mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq), dtype=torch.bool, device=q.device))
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa)

        # Re-assemble the heads side by side and project back to residual stream
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, kv_cache):
        x = x + self.attn(norm(x), cos_sin, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__()
        self.config = config
        # For DDP, we want vocab_size divisible by world_size. Also, there are potential performance benefits, see:
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} to be divisible by {pad_vocab_size_to}")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, padded_vocab_size, bias=False)
        # To support meta device initialization, we init the rotary embeddings here, but it's fake
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    def init_weights(self):
        self.apply(self._init_weights)
        # zero out classifier weights
        torch.nn.init.zeros_(self.lm_head.weight)
        # zero out c_proj weights in all blocks
        for block in self.transformer.h:
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
        # init the rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # Cast the embeddings from fp32 to bf16: optim can tolerate it and it saves memory: both in the model and the activations
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # https://arxiv.org/pdf/2310.17813
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)

    # TODO: bump base theta more, e.g. 100K is more common more recently
    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16() # keep them in bfloat16
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """ Return the estimated FLOPs per token for the model. Ref: https://arxiv.org/abs/2204.02311 """
        nparams = sum(p.numel() for p in self.parameters())
        nparams_embedding = self.transformer.wte.weight.numel()
        l, h, q, t = self.config.n_layer, self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t
        return num_flops_per_token

    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()
        # Separate out all parameters into 3 groups (matrix, embedding, lm_head)
        matrix_params = list(self.transformer.h.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params)
        # Create the AdamW optimizer for the embedding and lm_head
        # Scale the LR for the AdamW parameters by ∝1/√dmodel (having tuned the LRs for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
        ]
        adamw_kwargs = dict(betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay)
        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
        # Create the Muon optimizer for the linear layers
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95)
        MuonFactory = DistMuon if ddp else Muon
        muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)
        # Combine them the two optimizers into one list
        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        return optimizers

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        # Forward the trunk of the Transformer
        x = self.transformer.wte(idx)
        x = norm(x)
        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache)
        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]
        logits = self.lm_head(x) # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
        logits = logits[..., :self.config.vocab_size] # slice to remove padding
        logits = logits.float() # switch to fp32 for logit softcap and loss computation
        logits = softcap * torch.tanh(logits / softcap) # squash the logits

        if targets is not None:
            # training: given the targets, compute and return the loss
            # TODO experiment with chunked cross-entropy?
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            # inference: just return the logits directly
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token

```


### `third_party/nanochat/nanochat/muon.py`

```python
"""
Muon optimizer from Keller et al.
Also a lot of borrowing of ideas from modded-nanogpt.
"""
import torch
from torch import Tensor
import torch.distributed as dist

@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iteration steps to use.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params: list[Tensor] = [*params]
        param_groups = []
        for size in {p.numel() for p in params}:
            group = dict(params=[p for p in params if p.numel() == size])
            param_groups.append(group)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            for p in params:
                g = p.grad
                assert g is not None
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf: Tensor = state["momentum_buffer"]
                buf.lerp_(g, 1 - group["momentum"])
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                p.add_(g, alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1))**0.5)


class DistMuon(torch.optim.Optimizer):
    """
    Muon: SGD-momentum + (optional) Nesterov, then orthogonalize the 2D update via Newton–Schulz,
    finally apply aspect-ratio scaled step. Performs its own distributed synchronization:
      - reduce_scatter(AVG) for gradient averaging
      - all_gather to replicate updated weights

    Notes:
      * Designed for 2D parameters (e.g., linear/conv kernels reshaped to 2D). Do not use for 0D/1D
        params like embeddings or scalars.
      * Momentum buffers are maintained only on the 'owner' rank for each parameter (rank chosen
        by block-cyclic assignment below). If you checkpoint optimizer state on a single rank,
        consolidate states beforehand.

    Args:
        params: iterable of Tensors
        lr: learning rate
        momentum: momentum coefficient in [0,1)
        nesterov: if True, Nesterov-style update (g <- lerp(g, buf, momentum)); else use buf
        ns_steps: number of Newton–Schulz iterations for the orthogonalization
    """
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params = list(params)
        assert all(p.ndim == 2 for p in params), "Muon expects 2D parameters only"
        rank = dist.get_rank()
        # Group all parameters by their shape
        shapes = sorted({p.shape for p in params}) # sort to ensure consistent / deterministic ordering
        param_groups = []
        for shape in shapes:
            group_params = [p for p in params if p.shape == shape]
            device, dtype = group_params[0].device, group_params[0].dtype
            assert all(p.device == device for p in group_params)
            assert all(p.dtype == dtype for p in group_params)
            if rank == 0:
                print(f"Muon: Grouping {len(group_params)} params of shape {shape}, device {device}, dtype {dtype}")
            param_groups.append(dict(params=group_params, zero_buffer=torch.zeros_like(group_params[0])))
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Ensure all grads exist
        assert all(p.grad is not None for group in self.param_groups for p in group["params"]), "All params must have grads"

        # Kick off all the reduce scatter operations to average up the gradients across all ranks
        all_reduce_futures = []
        for group in self.param_groups:
            params = group["params"]
            zero_buffer = group["zero_buffer"]
            # Go through params in groups of world_size.
            for base_i in range(0, len(params), world_size):
                # The compute owner of each param is rank i % world_size
                owner_idx = base_i + rank
                # each rank stacks up its chunk of world_size params into a list
                rs_input = [p.grad for p in params[base_i:base_i + world_size]]
                # pad rs_input with the zero buffer to complete the group
                rs_input.extend([zero_buffer] * (world_size - len(rs_input)))
                # the output buffer gets strided across the group based on the rank
                rs_output = params[owner_idx].grad if owner_idx < len(params) else torch.empty_like(zero_buffer)
                # reduce scatter the gradients within this group of world_size params
                work = dist.reduce_scatter(rs_output, rs_input, op=dist.ReduceOp.AVG, async_op=True).get_future()
                all_reduce_futures.append(work)

        # Now each rank computes the update and gathers
        future_idx = 0
        all_gather_futures = []
        for group in self.param_groups:
            params = group["params"]
            zero_buffer = group["zero_buffer"]
            # Go through params in groups of world_size.
            for base_i in range(0, len(params), world_size):
                # The compute owner of each param is rank i % world_size
                owner_idx = base_i + rank # calculate the index of the param that this rank owns
                # Wait for the reduce scatter to complete
                all_reduce_futures[future_idx].wait() # possibly later we could use wait_any polling instead
                future_idx += 1
                # Owner computes the Muon update, result is in its param
                if owner_idx < len(params):
                    p = params[owner_idx]
                    g = p.grad  # now averaged across ranks
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf: Tensor = state["momentum_buffer"]
                    buf.lerp_(g, 1.0 - group["momentum"])
                    g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                    g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                    scale = (max(1.0, p.size(-2) / p.size(-1)) ** 0.5)
                    p.add_(g, alpha=-group["lr"] * scale)
                # Replicate updated parameters to all ranks
                ag_input = params[owner_idx] if owner_idx < len(params) else zero_buffer
                ag_output = params[base_i:base_i + world_size]
                ag_output.extend([torch.empty_like(zero_buffer) for _ in range(world_size - len(ag_output))]) # pad
                work = dist.all_gather(ag_output, ag_input, async_op=True).get_future()
                all_gather_futures.append(work)

        # Wait for all work to finish
        torch.futures.collect_all(all_gather_futures).wait()

```


### `third_party/nanochat/nanochat/tokenizer.py`

```python
"""
BPE Tokenizer in the style of GPT-4.

Two implementations are available:
1) HuggingFace Tokenizer that can do both training and inference but is really confusing
2) Our own RustBPE Tokenizer for training and tiktoken for efficient inference
"""

import os
import copy
from functools import lru_cache

SPECIAL_TOKENS = [
    # every document begins with the Beginning of Sequence (BOS) token that delimits documents
    "<|bos|>",
    # tokens below are only used during finetuning to render Conversations into token ids
    "<|user_start|>", # user messages
    "<|user_end|>",
    "<|assistant_start|>", # assistant messages
    "<|assistant_end|>",
    "<|python_start|>", # assistant invokes python REPL tool
    "<|python_end|>",
    "<|output_start|>", # python REPL outputs back to assistant
    "<|output_end|>",
]

# NOTE: this split pattern deviates from GPT-4 in that we use \p{N}{1,2} instead of \p{N}{1,3}
# I did this because I didn't want to "waste" too many tokens on numbers for smaller vocab sizes.
# I haven't validated that this is actually a good idea, TODO.
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# -----------------------------------------------------------------------------
# Generic GPT-4-style tokenizer based on HuggingFace Tokenizer
from tokenizers import Tokenizer as HFTokenizer
from tokenizers import pre_tokenizers, decoders, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

class HuggingFaceTokenizer:
    """Light wrapper around HuggingFace Tokenizer for some utilities"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, hf_path):
        # init from a HuggingFace pretrained tokenizer (e.g. "gpt2")
        tokenizer = HFTokenizer.from_pretrained(hf_path)
        return cls(tokenizer)

    @classmethod
    def from_directory(cls, tokenizer_dir):
        # init from a local directory on disk (e.g. "out/tokenizer")
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        tokenizer = HFTokenizer.from_file(tokenizer_path)
        return cls(tokenizer)

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        # train from an iterator of text
        # Configure the HuggingFace Tokenizer
        tokenizer = HFTokenizer(BPE(
            byte_fallback=True, # needed!
            unk_token=None,
            fuse_unk=False,
        ))
        # Normalizer: None
        tokenizer.normalizer = None
        # Pre-tokenizer: GPT-4 style
        # the regex pattern used by GPT-4 to split text into groups before BPE
        # NOTE: The pattern was changed from \p{N}{1,3} to \p{N}{1,2} because I suspect it is harmful to
        # very small models and smaller vocab sizes, because it is a little bit wasteful in the token space.
        # (but I haven't validated this! TODO)
        gpt4_split_regex = Regex(SPLIT_PATTERN) # huggingface demands that you wrap it in Regex!!
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(pattern=gpt4_split_regex, behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
        ])
        # Decoder: ByteLevel (it pairs together with the ByteLevel pre-tokenizer)
        tokenizer.decoder = decoders.ByteLevel()
        # Post-processor: None
        tokenizer.post_processor = None
        # Trainer: BPE
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            show_progress=True,
            min_frequency=0, # no minimum frequency
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=SPECIAL_TOKENS,
        )
        # Kick off the training
        tokenizer.train_from_iterator(text_iterator, trainer)
        return cls(tokenizer)

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def get_special_tokens(self):
        special_tokens_map = self.tokenizer.get_added_tokens_decoder()
        special_tokens = [w.content for w in special_tokens_map.values()]
        return special_tokens

    def id_to_token(self, id):
        return self.tokenizer.id_to_token(id)

    def _encode_one(self, text, prepend=None, append=None):
        # encode a single string
        # prepend/append can be either a string of a special token or a token id directly.
        assert isinstance(text, str)
        ids = []
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
            ids.append(prepend_id)
        ids.extend(self.tokenizer.encode(text, add_special_tokens=False).ids)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)
            ids.append(append_id)
        return ids

    def encode_special(self, text):
        # encode a single special token via exact match
        return self.tokenizer.token_to_id(text)

    def get_bos_token_id(self):
        bos = self.encode_special("<|bos|>")
        return bos

    def encode(self, text, *args, **kwargs):
        if isinstance(text, str):
            return self._encode_one(text, *args, **kwargs)
        elif isinstance(text, list):
            return [self._encode_one(t, *args, **kwargs) for t in text]
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def save(self, tokenizer_dir):
        # save the tokenizer to disk
        os.makedirs(tokenizer_dir, exist_ok=True)
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        self.tokenizer.save(tokenizer_path)
        print(f"Saved tokenizer to {tokenizer_path}")

# -----------------------------------------------------------------------------
# Tokenizer based on rustbpe + tiktoken combo
import pickle
import rustbpe
import tiktoken

class RustBPETokenizer:
    """Light wrapper around tiktoken (for efficient inference) but train with rustbpe"""

    def __init__(self, enc, bos_token):
        self.enc = enc
        self.bos_token_id = self.encode_special(bos_token)

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        # 1) train using rustbpe
        tokenizer = rustbpe.Tokenizer()
        # the special tokens are inserted later in __init__, we don't train them here
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        assert vocab_size_no_special >= 256, f"vocab_size_no_special must be at least 256, got {vocab_size_no_special}"
        tokenizer.train_from_iterator(text_iterator, vocab_size_no_special, pattern=SPLIT_PATTERN)
        # 2) construct the associated tiktoken encoding for inference
        pattern = tokenizer.get_pattern()
        mergeable_ranks_list = tokenizer.get_mergeable_ranks()
        mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
        tokens_offset = len(mergeable_ranks)
        special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
        enc = tiktoken.Encoding(
            name="rustbpe",
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks, # dict[bytes, int] (token bytes -> merge priority rank)
            special_tokens=special_tokens, # dict[str, int] (special token name -> token id)
        )
        return cls(enc, "<|bos|>")

    @classmethod
    def from_directory(cls, tokenizer_dir):
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "rb") as f:
            enc = pickle.load(f)
        return cls(enc, "<|bos|>")

    @classmethod
    def from_pretrained(cls, tiktoken_name):
        # https://github.com/openai/tiktoken/blob/eedc8563/tiktoken_ext/openai_public.py
        enc = tiktoken.get_encoding(tiktoken_name)
        # tiktoken calls the special document delimiter token "<|endoftext|>"
        # yes this is confusing because this token is almost always PREPENDED to the beginning of the document
        # it most often is used to signal the start of a new sequence to the LLM during inference etc.
        # so in nanoChat we always use "<|bos|>" short for "beginning of sequence", but historically it is often called "<|endoftext|>".
        return cls(enc, "<|endoftext|>")

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_special_tokens(self):
        return self.enc.special_tokens_set

    def id_to_token(self, id):
        return self.enc.decode([id])

    @lru_cache(maxsize=32)
    def encode_special(self, text):
        return self.enc.encode_single_token(text)

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, append=None, num_threads=8):
        # text can be either a string or a list of strings

        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)

        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id) # TODO: slightly inefficient here? :( hmm
            if append is not None:
                ids.append(append_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for ids_row in ids:
                    ids_row.insert(0, prepend_id) # TODO: same
            if append is not None:
                for ids_row in ids:
                    ids_row.append(append_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

        return ids

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        return self.enc.decode(ids)

    def save(self, tokenizer_dir):
        # save the encoding object to disk
        os.makedirs(tokenizer_dir, exist_ok=True)
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(self.enc, f)
        print(f"Saved tokenizer encoding to {pickle_path}")

    def render_conversation(self, conversation, max_tokens=2048):
        """
        Tokenize a single Chat conversation (which we call a "doc" or "document" here).
        Returns:
        - ids: list[int] is a list of token ids of this rendered conversation
        - mask: list[int] of same length, mask = 1 for tokens that the Assistant is expected to train on.
        """
        # ids, masks that we will return and a helper function to help build them up.
        ids, mask = [], []
        def add_tokens(token_ids, mask_val):
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))

        # sometimes the first message is a system message...
        # => just merge it with the second (user) message
        if conversation["messages"][0]["role"] == "system":
            # some conversation surgery is necessary here for now...
            conversation = copy.deepcopy(conversation) # avoid mutating the original
            messages = conversation["messages"]
            assert messages[1]["role"] == "user", "System message must be followed by a user message"
            messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
            messages = messages[1:]
        else:
            messages = conversation["messages"]
        assert len(messages) >= 1, f"Conversation has less than 1 message: {messages}"

        # fetch all the special tokens we need
        bos = self.get_bos_token_id()
        user_start, user_end = self.encode_special("<|user_start|>"), self.encode_special("<|user_end|>")
        assistant_start, assistant_end = self.encode_special("<|assistant_start|>"), self.encode_special("<|assistant_end|>")
        python_start, python_end = self.encode_special("<|python_start|>"), self.encode_special("<|python_end|>")
        output_start, output_end = self.encode_special("<|output_start|>"), self.encode_special("<|output_end|>")

        # now we can tokenize the conversation
        add_tokens(bos, 0)
        for i, message in enumerate(messages):

            # some sanity checking here around assumptions, to prevent footguns
            must_be_from = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == must_be_from, f"Message {i} is from {message['role']} but should be from {must_be_from}"

            # content can be either a simple string or a list of parts (e.g. containing tool calls)
            content = message["content"]

            if message["role"] == "user":
                assert isinstance(content, str), "User messages are simply expected to be strings"
                value_ids = self.encode(content)
                add_tokens(user_start, 0)
                add_tokens(value_ids, 0)
                add_tokens(user_end, 0)
            elif message["role"] == "assistant":
                add_tokens(assistant_start, 0)
                if isinstance(content, str):
                    # simple string => simply add the tokens
                    value_ids = self.encode(content)
                    add_tokens(value_ids, 1)
                elif isinstance(content, list):
                    for part in content:
                        value_ids = self.encode(part["text"])
                        if part["type"] == "text":
                            # string part => simply add the tokens
                            add_tokens(value_ids, 1)
                        elif part["type"] == "python":
                            # python tool call => add the tokens inside <|python_start|> and <|python_end|>
                            add_tokens(python_start, 1)
                            add_tokens(value_ids, 1)
                            add_tokens(python_end, 1)
                        elif part["type"] == "python_output":
                            # python output => add the tokens inside <|output_start|> and <|output_end|>
                            # none of these tokens are supervised because the tokens come from Python at test time
                            add_tokens(output_start, 0)
                            add_tokens(value_ids, 0)
                            add_tokens(output_end, 0)
                        else:
                            raise ValueError(f"Unknown part type: {part['type']}")
                else:
                    raise ValueError(f"Unknown content type: {type(content)}")
                add_tokens(assistant_end, 1)

        # truncate to max_tokens tokens MAX (helps prevent OOMs)
        ids = ids[:max_tokens]
        mask = mask[:max_tokens]
        return ids, mask

    def visualize_tokenization(self, ids, mask, with_token_id=False):
        """Small helper function useful in debugging: visualize the tokenization of render_conversation"""
        RED = '\033[91m'
        GREEN = '\033[92m'
        RESET = '\033[0m'
        GRAY = '\033[90m'
        tokens = []
        for i, (token_id, mask_val) in enumerate(zip(ids, mask)):
            token_str = self.decode([token_id])
            color = GREEN if mask_val == 1 else RED
            tokens.append(f"{color}{token_str}{RESET}")
            if with_token_id:
                tokens.append(f"{GRAY}({token_id}){RESET}")
        return '|'.join(tokens)

    def render_for_completion(self, conversation):
        """
        Used during Reinforcement Learning. In that setting, we want to
        render the conversation priming the Assistant for a completion.
        Unlike the Chat SFT case, we don't need to return the mask.
        """
        # We have some surgery to do: we need to pop the last message (of the Assistant)
        conversation = copy.deepcopy(conversation) # avoid mutating the original
        messages = conversation["messages"]
        assert messages[-1]["role"] == "assistant", "Last message must be from the Assistant"
        messages.pop() # remove the last message (of the Assistant) inplace

        # Now tokenize the conversation
        ids, mask = self.render_conversation(conversation)

        # Finally, to prime the Assistant for a completion, append the Assistant start token
        assistant_start = self.encode_special("<|assistant_start|>")
        ids.append(assistant_start)
        return ids

# -----------------------------------------------------------------------------
# nanochat-specific convenience functions

def get_tokenizer():
    from nanochat.common import get_base_dir
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    # return HuggingFaceTokenizer.from_directory(tokenizer_dir)
    return RustBPETokenizer.from_directory(tokenizer_dir)

def get_token_bytes(device="cpu"):
    import torch
    from nanochat.common import get_base_dir
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
    assert os.path.exists(token_bytes_path), f"Token bytes not found at {token_bytes_path}? It gets written by tok_train.py"
    with open(token_bytes_path, "rb") as f:
        token_bytes = torch.load(f, map_location=device)
    return token_bytes

```
