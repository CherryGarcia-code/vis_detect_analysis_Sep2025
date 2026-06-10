"""User-defined behavioral state labeling — data model, raster, queue, rendering.

See docs/superpowers/specs/2026-06-10-behavioral-state-labeler-design.md.
States are anchored to the experimenter's sparse labels on the outcome raster,
not to a latent HMM. Color encodes the *lick decision's valence*.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from visdetect.analysis.behavior import get_trial_dataframe
from visdetect.analysis.config import LICK_VALENCE_COLORS, STAGE_ORDER, parse_session_date


def classify_lick_valence(outcome: str, is_go: bool, is_catch: bool) -> str:
    """Map a trial outcome to its lick-valence class.

    appropriate_lick   : go-trial hit (licked to a real change)
    inappropriate_lick : early lick ('fa', any trial) OR catch-trial 'hit' (SDT false alarm)
    nolick             : 'miss' (covers go-miss AND catch correct-rejection)
    abort / ref        : as-is ('ref' is excluded from fractions downstream)
    """
    o = (outcome or "").lower()
    if o == "abort":
        return "abort"
    if o == "ref":
        return "ref"
    if o == "fa":
        return "inappropriate_lick"
    if o == "hit":
        return "appropriate_lick" if is_go else "inappropriate_lick"
    if o == "miss":
        return "nolick"
    return "ref"  # unknown -> excluded from fractions
