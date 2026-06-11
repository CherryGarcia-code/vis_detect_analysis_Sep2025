"""Feature extraction, decision-tree calibration, and tagging for behavioral states.

See docs/superpowers/specs/2026-06-10-behavioral-state-labeler-design.md.
Features are local outcome-composition fractions (lick-valence + difficulty-aware)
over a symmetric window W. The rule is a shallow DecisionTreeClassifier fit on the
experimenter's labeled trials; tagging mirrors hmm.decode_session columns.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import pickle

import numpy as np
import pandas as pd

from visdetect.analysis.constants import (
    STATE_EASY_CHANGE_THRESH, STATE_FEATURE_COLS, STATE_LABEL_W_GRID,
    STATE_CONFIDENCE_THRESHOLD,
)


def extract_state_features(raster_df: pd.DataFrame, W: int) -> pd.DataFrame:
    """Add local-window composition features (STATE_FEATURE_COLS) per trial.

    Fractions use a symmetric, centered window of width ``W`` trials, with the
    denominator = window trials excluding 'ref'. Edges shrink (min_periods=1).
    """
    df = raster_df.reset_index(drop=True).copy()
    lv = df["lick_valence"]
    applick = (lv == "appropriate_lick").astype(int)
    inapplick = (lv == "inappropriate_lick").astype(int)
    nolick = (lv == "nolick").astype(int)
    abort = (lv == "abort").astype(int)
    ref = (lv == "ref").astype(int)
    non_ref = (1 - ref)
    is_go = df["is_go"].astype(bool)
    easy = df["change_size"].astype(float) >= STATE_EASY_CHANGE_THRESH
    # miss_easy needs is_go (a catch-trial nolick is a correct rejection, not a miss).
    # hit_hard needs no is_go guard: classify_lick_valence only ever emits
    # 'appropriate_lick' on go-trial hits, so applick already implies is_go.
    miss_easy = (nolick.astype(bool) & is_go & easy).astype(int)
    hit_hard = (applick.astype(bool) & (~easy)).astype(int)

    def roll(s):
        return s.rolling(W, center=True, min_periods=1).sum()

    denom = roll(non_ref).replace(0, np.nan)
    df["f_applick"]   = (roll(applick) / denom).fillna(0.0)
    df["f_inapplick"] = (roll(inapplick) / denom).fillna(0.0)
    df["f_nolick"]    = (roll(nolick) / denom).fillna(0.0)
    df["f_abort"]     = (roll(abort) / denom).fillna(0.0)
    df["f_miss_easy"] = (roll(miss_easy) / denom).fillna(0.0)
    df["f_hit_hard"]  = (roll(hit_hard) / denom).fillna(0.0)
    return df
