#!/usr/bin/env bash
# Phase 2: 1 real experiment per model (after its baseline) to check the model, the loop and the cost. ~20 min.
#   ./scripts/dry_run.sh neon      then      ./scripts/dry_run.sh opus     (or both at once: they share the GPU)
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
source ~/.ai_keys
ARM=${1:?usage: dry_run.sh neon|opus|fable}
case "$ARM" in
  neon)  PROVIDER=gemini;    MODEL=gemini-4-neon ;;
  opus)  PROVIDER=anthropic; MODEL=claude-opus-5-5 ;;
  fable) PROVIDER=anthropic; MODEL=claude-fable-5-2 ;;
esac
# Keep threshold from the first night's 5-seed noise floor, so every model is judged by the same bar.
EPS=${EPSILON:-0.001881}
rm -rf runs/dry-$ARM .compile_cache/dry-$ARM
uv run python -m orchestrator.run_arm --arm dry-$ARM --provider "$PROVIDER" --model "$MODEL" --max-exps 1 --epsilon "$EPS"
echo; cut -f1,7,8,14,23,6 runs/dry-$ARM/log.tsv
rm -rf runs/dry-$ARM
echo; echo "===== DRY RUN DONE: paste the table above to Claude ====="
