#!/usr/bin/env bash
# Phase 3: start one model's overnight run in the background. Run it once per model, on the same box:
#   ./scripts/start.sh fable
# Both share the GPU (they take turns training). Safe to close the laptop after.
# Re-running it resumes where it stopped. Check progress: tail -n 5 runs/logs/<arm>.log
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
ARM=${1:?usage: start.sh neon|opus|fable}
case "$ARM" in
  neon)  PROVIDER=gemini;    MODEL=gemini-4-neon ;;
  opus)  PROVIDER=anthropic; MODEL=claude-opus-5-5 ;;
  fable) PROVIDER=anthropic; MODEL=claude-fable-5-2 ;;
esac
# Keep threshold from the first night's 5-seed noise floor, so every model is judged by the same bar.
EPS=${EPSILON:-0.001881}
HOURS=${HOURS:-10}; MAX_USD=${MAX_USD:-70}
mkdir -p runs/logs
CMD="source ~/.ai_keys && export PATH=$HOME/.local/bin:\$PATH && cd $PWD && uv run python -m orchestrator.run_arm --arm $ARM --provider $PROVIDER --model $MODEL --hours $HOURS --max-usd $MAX_USD --epsilon $EPS 2>&1 | tee -a runs/logs/$ARM.log"
tmux kill-session -t "race-$ARM" 2>/dev/null || true
tmux new-session -d -s "race-$ARM" "$CMD"
sleep 90
tail -n 5 "runs/logs/$ARM.log"
echo; echo "===== STARTED: '$ARM' is running in the background (tmux session 'race-$ARM'). Paste the lines above to Claude. ====="
