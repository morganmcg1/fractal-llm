#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# DevPod workspace disk is tiny; default to /var/tmp on remote.
export FRACTAL_STORAGE_DIR="${FRACTAL_STORAGE_DIR:-/var/tmp/fractal-llm}"

exec uv run src/chat.py "$@"

