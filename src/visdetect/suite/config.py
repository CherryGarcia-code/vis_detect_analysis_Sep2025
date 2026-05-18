"""Configuration for the analysis suite.

Re-exports the canonical project config (:mod:`visdetect.analysis.config`)
and adds suite-specific output directories (FIGURE_DIR, CACHE_DIR).
"""

import os

from visdetect.analysis.config import *  # noqa: F401, F403

# ── Suite-specific output directories ─────────────────────────────────
FIGURE_DIR = os.path.join(ROOT, "analysis_suite", "figures")  # noqa: F405
CACHE_DIR  = os.path.join(ROOT, "analysis_suite", "cache")    # noqa: F405

os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
