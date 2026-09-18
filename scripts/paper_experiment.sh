#!/usr/bin/env bash
# Launch paper-experiment detached from the parent shell.
#
# POSIX counterpart to paper_experiment.bat. Same motivation - decouple
# the experiment's process tree from whatever launched this script so a
# Claude Code memory reaper (or a terminal you accidentally close)
# doesn't kill the whole thing.
#
# Uses `nohup ... &` + explicit stdout/stderr redirection so:
#   1. The experiment survives the launching shell exiting.
#   2. SIGHUP from a closed terminal doesn't propagate.
#   3. All output goes to out/logs/experiment.out for `tail -F`.
#
# Usage:
#   scripts/paper_experiment.sh reasoning-quality
#   scripts/paper_experiment.sh reasoning-quality --dashboard-port 8001

set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)

if [ "$#" -lt 1 ]; then
  echo "Usage: $(basename "$0") <experiment-name> [extra paper-experiment args]" >&2
  echo "Example: $(basename "$0") reasoning-quality" >&2
  exit 2
fi

EXP=$1
shift

# Belt + suspenders in case something downstream inherits and does
# spawn under a Claude Code parent.
export CLAUDE_CODE_DISABLE_BG_SHELL_PRESSURE_REAP=1

cd "$REPO"

# Prefer the project venv's entry point; fall back to `uv run`.
if [ -x ".venv/bin/agentic-investor" ]; then
  BIN=".venv/bin/agentic-investor"
elif [ -x ".venv/Scripts/agentic-investor.exe" ]; then
  BIN=".venv/Scripts/agentic-investor.exe"
else
  BIN="uv run agentic-investor"
fi

mkdir -p out/logs
: > out/logs/experiment.out  # truncate previous run so tail -F starts clean

# `nohup` guards against SIGHUP if the launching shell closes; `&`
# backgrounds; `disown` (if the shell supports it) prevents the shell
# from tracking the process at all.
nohup $BIN paper-experiment "$EXP" \
  --serve-dashboard --dashboard-port 8000 \
  --paper-loop-args --auto --top-n 8 --regen-mode event \
  "$@" > out/logs/experiment.out 2>&1 &

PID=$!
disown "$PID" 2>/dev/null || true

echo "Launched paper-experiment $EXP (pid $PID) detached from this shell."
echo "Log:       $REPO/out/logs/experiment.out"
echo "Dashboard: http://localhost:8000/"
echo "Stop:      kill $PID    # or 'pkill -f paper-experiment'"
