"""Per-UID tracking QC: metrics, badge logic, and extraction primitives.

This module is library code (no I/O orchestration). The
`scripts/pipelines/tracking/build_qc_sheets.py` driver wires it up.

See docs/superpowers/specs/2026-05-21-tracking-qc-sheets-design.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

# ─── Badge thresholds (tweakable; documented in spec §7) ──────────────
ISI_PASS: float = 0.75
ISI_WARN: float = 0.65

DEPTH_PASS_UM: float = 15.0
DEPTH_WARN_UM: float = 30.0

WAVE_PASS_R: float = 0.95
WAVE_WARN_R: float = 0.90

FR_CV_PASS: float = 0.35
FR_CV_WARN: float = 0.60

# ─── Change-size pools for Change_ON heatmaps ─────────────────────────
# Spec excludes 1.5× from heatmaps (ambiguous mid).
BIG_POOL: Set[float] = {2.0, 4.0}
SMALL_POOL: Set[float] = {1.25, 1.35}

# ─── Footprint extraction ─────────────────────────────────────────────
# How many channels above/below the peak to include in the footprint snippet.
FOOTPRINT_HALFWIDTH_CHANS: int = 8
