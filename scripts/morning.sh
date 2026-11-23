#!/usr/bin/env bash
# Phase 4: held-out test, dashboard, video. ~75 min.
#   ./scripts/morning.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
[ -f runs/neon/state.json ] && [ -f runs/opus/state.json ] || { echo "runs/neon or runs/opus is missing"; exit 1; }
tmux kill-session -t race-neon 2>/dev/null || true
tmux kill-session -t race-opus 2>/dev/null || true
uv run python -m orchestrator.final_test --arms neon opus --seeds 3 2>&1 | tee runs/final_test.txt
uv run python -m analysis.build_dashboard --arms "neon:Gemini 4 Neon" "opus:Claude Opus 5.5"
uv sync --extra video && uv run playwright install --with-deps chromium >/dev/null
sudo apt-get install -y -qq ffmpeg >/dev/null
uv run python -m analysis.record_replay
ffmpeg -loglevel error -y -i runs/replay.webm -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p runs/replay.mp4
tar czf ~/results.tgz --exclude='runs/_final/*/seed*/work' --exclude='runs/_preflight/*/out' runs
ls -la ~/results.tgz
echo; echo "===== MORNING DONE: results are in ~/results.tgz. Tell Claude. ====="
