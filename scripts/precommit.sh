#!/usr/bin/env bash
# Pre-commit hook: regenerate FLAGS.md from the flags registry and stage
# it if it changed. Blocks the commit with a helpful message when a
# newly-added flag hasn't been captured yet.
#
# Install (one-time, from repo root):
#   ln -sf ../../scripts/precommit.sh .git/hooks/pre-commit
#   chmod +x .git/hooks/pre-commit
# Windows (Git Bash):
#   cp scripts/precommit.sh .git/hooks/pre-commit
#
# The hook is scoped narrowly - it only regenerates FLAGS.md and defers
# tests/ruff to CI. Adding more work here has a habit of making local
# commits slow enough that people --no-verify around it, defeating
# the point.

set -e

# Locate repo root even if the hook was invoked from a subdir.
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"

# Prefer the venv python so we don't hit a system-wide interp that lacks
# the project's deps. Fall back to `python` if the venv layout differs.
if [ -x ".venv/Scripts/python.exe" ]; then
    PY=".venv/Scripts/python.exe"
elif [ -x ".venv/bin/python" ]; then
    PY=".venv/bin/python"
else
    PY="python"
fi

# Regenerate; script exits 0 whether or not there was a diff.
"$PY" scripts/regen_flags_md.py

# If FLAGS.md was updated, add it to the current commit so the change
# ships together with whatever flag registry edit prompted it.
if ! git diff --quiet -- FLAGS.md; then
    echo "[pre-commit] FLAGS.md regenerated; staging."
    git add FLAGS.md
fi
