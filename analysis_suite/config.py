"""Configuration wrapper for the analysis_suite.

Re-exports everything from the canonical project config
(:mod:`visdetect.analysis.config`) and adds suite-specific output
directories (FIGURE_DIR, CACHE_DIR).

IMPORTANT: This module assumes the visdetect package is importable.
Recommended setup:
    cd /path/to/project
    pip install -e .

If not using editable install, this module falls back to sys.path
manipulation, but this is less reliable for CI and testing.
"""

import os
import sys

# Try importing visdetect package directly (editable install)
try:
    from visdetect.analysis.config import *  # noqa: F401, F403
except ImportError:
    # Fallback: Add src to sys.path (less reliable)
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _src = os.path.join(_root, "src")
    if _src not in sys.path:
        sys.path.insert(0, _src)

    # Try import again
    try:
        from visdetect.analysis.config import *  # noqa: F401, F403
    except ImportError as e:
        raise ImportError(
            f"Cannot import visdetect package. "
            f"Please run 'pip install -e .' from project root. "
            f"Original error: {e}"
        )

# ── Suite-specific output directories ─────────────────────────────────
FIGURE_DIR = os.path.join(ROOT, "analysis_suite", "figures")  # noqa: F405
CACHE_DIR  = os.path.join(ROOT, "analysis_suite", "cache")    # noqa: F405

os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
