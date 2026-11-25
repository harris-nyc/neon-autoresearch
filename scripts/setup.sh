#!/usr/bin/env bash
# Phase 1 (each box): install, data, pre-flight checks + noise floor. ~45 min.
#   ./scripts/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v uv >/dev/null; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
export PATH="$HOME/.local/bin:$PATH"
nvidia-smi --query-gpu=name,memory.total --format=csv
uv sync
uv run python -m orchestrator.data_setup
mkdir -p runs && uv run python -m orchestrator.preflight --seeds ${SEEDS:-5} 2>&1 | tee runs/preflight.txt
echo; echo "===== SETUP DONE: paste everything from 'leakage test' down to Claude ====="
