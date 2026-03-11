"""Configuration wrapper for the analysis_suite.

Re-exports everything from the canonical project config
(:mod:`visdetect.analysis.config`) and adds suite-specific output
directories (FIGURE_DIR, CACHE_DIR).

Existing ``from config import ...`` lines in analysis_suite scripts
continue to work unchanged.
"""

import os
import sys

# Ensure visdetect is importable (editable install or sys.path fallback)
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src = os.path.join(_root, "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

# Re-export everything from the canonical config
from visdetect.analysis.config import *  # noqa: F401, F403

# ── Suite-specific output directories ─────────────────────────────────
FIGURE_DIR = os.path.join(ROOT, "analysis_suite", "figures")  # noqa: F405
CACHE_DIR  = os.path.join(ROOT, "analysis_suite", "cache")    # noqa: F405

os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
