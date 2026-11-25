#!/usr/bin/env bash
# Phase 4: held-out test, dashboard, video. ~75 min.
#   ./scripts/morning.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
ARMS=${ARMS:-"fable"}
for a in $ARMS; do [ -f runs/$a/state.json ] || { echo "runs/$a is missing"; exit 1; }; tmux kill-session -t race-$a 2>/dev/null || true; done
uv run python -m orchestrator.final_test --arms $ARMS --seeds 3 2>&1 | tee runs/final_test.txt
uv run python -m analysis.build_dashboard --arms "fable:Claude Fable 5.2"
uv sync --extra video && uv run playwright install --with-deps chromium >/dev/null
sudo apt-get install -y -qq ffmpeg >/dev/null
uv run python -m analysis.record_replay
ffmpeg -loglevel error -y -i runs/replay.webm -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p runs/replay.mp4
tar czf ~/results.tgz --exclude='runs/_final/*/seed*/work' --exclude='runs/_preflight/*/out' runs
ls -la ~/results.tgz
echo; echo "===== MORNING DONE: results are in ~/results.tgz. Tell Claude. ====="
