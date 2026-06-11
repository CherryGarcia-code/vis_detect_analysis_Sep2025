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


def attach_episode_labels(features_df: pd.DataFrame, episodes, session_name: str) -> pd.DataFrame:
    """Add a 'state' column from episodes (None where unlabeled), keyed by trial_idx."""
    from visdetect.analysis.state_labeling import episodes_to_trial_labels
    df = features_df.copy()
    n = int(df["trial_idx"].max()) + 1 if len(df) else 0
    lab = episodes_to_trial_labels(episodes, session_name, n)
    df["state"] = [lab[int(i)] for i in df["trial_idx"]]
    return df


def fit_state_tree(features_df: pd.DataFrame, seed: int = 42):
    """Fit a shallow, readable decision tree on labeled rows (the 'state' column)."""
    from sklearn.tree import DecisionTreeClassifier
    train = features_df[features_df["state"].notna()]
    if len(train) == 0:
        raise ValueError("fit_state_tree: no labeled rows — attach episode labels first.")
    X = train[STATE_FEATURE_COLS].values
    y = train["state"].astype(str).values
    tree = DecisionTreeClassifier(
        max_depth=3, min_samples_leaf=5, class_weight="balanced", random_state=seed,
    )
    tree.fit(X, y)
    return tree


@dataclass
class CalibrationResult:
    tree: object                 # sklearn DecisionTreeClassifier
    window: int
    state_labels: List[str]
    feature_cols: List[str]
    loso_kappa: float
    rules_text: str

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path) -> "CalibrationResult":
        with open(path, "rb") as f:
            return pickle.load(f)


def _pool_labeled(rasters: Dict[str, pd.DataFrame], episodes, W: int) -> pd.DataFrame:
    frames = []
    for sn, raster in rasters.items():
        feats = extract_state_features(raster, W)
        feats = attach_episode_labels(feats, episodes, sn)
        feats = feats[feats["state"].notna()].copy()
        feats["__session"] = sn
        frames.append(feats)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def calibrate_states(rasters, episodes, w_grid=None, seed: int = 42) -> CalibrationResult:
    """Fit the state rule: choose W by LOSO Cohen's kappa, then refit on all labels."""
    from sklearn.metrics import cohen_kappa_score
    from sklearn.tree import export_text
    if w_grid is None:
        w_grid = STATE_LABEL_W_GRID

    best = None  # (W, mean_kappa, pooled)
    for W in w_grid:
        pooled = _pool_labeled(rasters, episodes, W)
        if pooled.empty:
            continue
        sessions = pooled["__session"].unique()
        kappas = []
        for hold in sessions:
            tr = pooled[pooled["__session"] != hold]
            te = pooled[pooled["__session"] == hold]
            if te.empty or tr["state"].nunique() < 2:
                continue
            m = fit_state_tree(tr, seed=seed)
            pred = m.predict(te[STATE_FEATURE_COLS].values)
            kappas.append(cohen_kappa_score(te["state"].astype(str).values, pred))
        mean_k = float(np.mean(kappas)) if kappas else float("nan")
        if best is None or (not np.isnan(mean_k) and (np.isnan(best[1]) or mean_k > best[1])):
            best = (W, mean_k, pooled)

    if best is None:
        raise ValueError("No labeled trials found for any window in w_grid.")
    W, kappa, pooled = best
    tree = fit_state_tree(pooled, seed=seed)
    rules = export_text(tree, feature_names=list(STATE_FEATURE_COLS))
    return CalibrationResult(
        tree=tree, window=W, state_labels=list(tree.classes_),
        feature_cols=list(STATE_FEATURE_COLS), loso_kappa=kappa, rules_text=rules,
    )
