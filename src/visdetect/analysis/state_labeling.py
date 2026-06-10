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


@dataclass
class StateEpisode:
    """A contiguous span of trials the experimenter is confident about."""
    session_name: str
    start_trial: int          # inclusive index into the trial DataFrame
    end_trial: int            # inclusive
    state_label: str
    labeler: str
    timestamp: str
    notes: str = ""


_EPISODE_COLUMNS = [
    "session_name", "start_trial", "end_trial", "state_label", "labeler", "timestamp", "notes",
]


def save_episode(episode: StateEpisode, path) -> None:
    """Append one episode to the labels CSV (creates the file with a header)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([asdict(episode)])[_EPISODE_COLUMNS]
    # Write a header for a missing OR zero-byte file (a crash mid-write can leave an
    # empty file; without this guard the first append would be headerless and
    # load_episodes would misparse the data row as column names).
    header = not path.exists() or path.stat().st_size == 0
    row.to_csv(path, mode="a", header=header, index=False)


def load_episodes(path) -> List[StateEpisode]:
    """Load all episodes from the labels CSV."""
    path = Path(path)
    if not path.exists():
        return []
    df = pd.read_csv(path, dtype={"session_name": str, "notes": str})
    df["notes"] = df["notes"].fillna("")
    return [
        StateEpisode(
            session_name=str(r.session_name),
            start_trial=int(r.start_trial),
            end_trial=int(r.end_trial),
            state_label=str(r.state_label),
            labeler=str(r.labeler),
            timestamp=str(r.timestamp),
            notes=str(r.notes),
        )
        for r in df.itertuples(index=False)
    ]


def episodes_to_trial_labels(
    episodes: List[StateEpisode], session_name: str, n_trials: int
) -> np.ndarray:
    """Expand sparse episodes for one session to a per-trial label array.

    Unlabeled trials are ``None``.
    """
    labels = np.array([None] * n_trials, dtype=object)
    for ep in episodes:
        if str(ep.session_name) != str(session_name):
            continue
        lo = max(0, int(ep.start_trial))
        hi = min(n_trials - 1, int(ep.end_trial))
        labels[lo:hi + 1] = ep.state_label
    return labels


def build_outcome_raster(session) -> pd.DataFrame:
    """Per-trial raster frame: outcome, trial type, change size, lick-valence + color."""
    df = get_trial_dataframe(session)
    if df.empty:
        return df
    out = pd.DataFrame({
        "trial_idx": df["trial_idx"].astype(int),
        "outcome": df["outcome"],
        "is_go": df["is_go"].astype(bool),
        "is_catch": df["is_catch"].astype(bool),
        "change_size": df["change_size"].astype(float),
    })
    out["lick_valence"] = [
        classify_lick_valence(o, g, c)
        for o, g, c in zip(out["outcome"], out["is_go"], out["is_catch"])
    ]
    out["color"] = out["lick_valence"].map(LICK_VALENCE_COLORS)
    return out
