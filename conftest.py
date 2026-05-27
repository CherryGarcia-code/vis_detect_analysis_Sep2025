"""Worktree-local conftest.

Ensure pytest imports `visdetect` from THIS worktree's ``src/`` rather than
from the main-repo editable install. Without this, the editable install at
the project root would shadow any new code added in the worktree, defeating
the purpose of isolation.

Affects only pytest invocations from this worktree (does not modify the
main repo's environment).
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).parent / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
