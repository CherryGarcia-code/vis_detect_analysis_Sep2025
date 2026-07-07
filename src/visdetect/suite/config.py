"""Configuration for the analysis suite.

Re-exports the canonical project config (:mod:`visdetect.analysis.config`)
and adds suite-specific output directories (FIGURE_DIR, CACHE_DIR).
"""

import os

from visdetect.analysis.config import *  # noqa: F401, F403

# ── Suite-specific output directories ─────────────────────────────────
# analysis_suite/ was archived (2026-07-01). These compatibility paths now resolve
# to the repo-root convention so any lingering suite.* consumer (e.g. plotting.save_figure)
# writes to a VALID location and never recreates analysis_suite/ at the repo root.
# New work should use data/cache/<topic>/ + FIGURES/<topic>/ directly, not these.
FIGURE_DIR = os.path.join(ROOT, "FIGURES")            # was analysis_suite/figures  # noqa: F405
CACHE_DIR  = os.path.join(ROOT, "data", "cache")      # was analysis_suite/cache    # noqa: F405

os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
