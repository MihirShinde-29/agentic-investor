#!/usr/bin/env python
"""Regenerate FLAGS.md from the flags registry (task #158).

Run manually or wire as a pre-commit hook so `FLAGS.md` can never
drift from `src/agentic_investor/flags.py`. Exits non-zero when the
regenerated content differs from what's on disk - the exit code lets
a pre-commit hook block a commit that forgot to run this.

Usage:
  python scripts/regen_flags_md.py         # regenerate + report
  python scripts/regen_flags_md.py --check # exit 1 if drift, don't write

Pre-commit hook setup (one-time):
  ln -s ../../scripts/precommit.sh .git/hooks/pre-commit
  chmod +x .git/hooks/pre-commit
Or on Windows:
  copy scripts\\precommit.sh .git\\hooks\\pre-commit
"""

from __future__ import annotations

import sys
from pathlib import Path

_TARGET = Path(__file__).resolve().parent.parent / "FLAGS.md"

_HEADER = (
    "<!--\n"
    "This file is auto-generated from src/agentic_investor/flags.py.\n"
    "Do NOT edit by hand: run `python scripts/regen_flags_md.py`\n"
    "after adding or changing a flag. Task #158 wires this into a\n"
    "pre-commit hook (see scripts/precommit.sh) so drift can't ship.\n"
    "-->\n\n"
    "# Feature flags\n\n"
    "Every `AGENTIC_*` env var the paper-loop reads, sourced from the\n"
    "central `flags` registry. Values shown are defaults.\n\n"
)


def _render() -> str:
    from agentic_investor.flags import format_flags_table
    return _HEADER + format_flags_table() + "\n"


def main() -> int:
    check_only = "--check" in sys.argv[1:]
    fresh = _render()
    if _TARGET.exists():
        current = _TARGET.read_text(encoding="utf-8")
    else:
        current = ""
    if fresh == current:
        print(f"{_TARGET.name}: up to date")
        return 0
    if check_only:
        print(
            f"{_TARGET.name}: DRIFT - regenerate with "
            f"`python scripts/regen_flags_md.py`",
            file=sys.stderr,
        )
        return 1
    _TARGET.write_text(fresh, encoding="utf-8")
    print(f"{_TARGET.name}: regenerated ({len(fresh)} chars)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
