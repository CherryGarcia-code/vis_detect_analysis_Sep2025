"""Ensure tests in this worktree load source from the worktree itself.

The shared .venv uses an editable install that resolves visdetect to
E:/python_analysis/git_repos/vis_detect_analysis_Sep2025/src/. Prepending
this worktree's src/ directory to sys.path overrides that so all tests
here run against the code being developed in this branch.
"""
import sys
from pathlib import Path

_WORKTREE_SRC = str(Path(__file__).parent / "src")
if _WORKTREE_SRC not in sys.path:
    sys.path.insert(0, _WORKTREE_SRC)
