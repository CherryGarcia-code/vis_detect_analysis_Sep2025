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

_RASTER_COLUMNS = [
    "trial_idx", "outcome", "is_go", "is_catch", "change_size",
    "is_hit", "is_fa", "is_miss", "lick_valence", "color",
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
        return pd.DataFrame(columns=_RASTER_COLUMNS)
    out = pd.DataFrame({
        "trial_idx": df["trial_idx"].astype(int),
        "outcome": df["outcome"],
        "is_go": df["is_go"].astype(bool),
        "is_catch": df["is_catch"].astype(bool),
        "change_size": df["change_size"].astype(float),
        # SDT behavioral-label booleans, carried through so the tagged frame is
        # drop-in for hmm_downstream (compute_state_behavioral_metrics reads these).
        "is_hit": df["is_hit"].astype(bool),
        "is_fa": df["is_fa"].astype(bool),
        "is_miss": df["is_miss"].astype(bool),
    })
    out["lick_valence"] = [
        classify_lick_valence(o, g, c)
        for o, g, c in zip(out["outcome"], out["is_go"], out["is_catch"])
    ]
    out["color"] = out["lick_valence"].map(LICK_VALENCE_COLORS)
    return out


def get_labeling_queue(manifest: Optional[pd.DataFrame] = None) -> List[str]:
    """Return session names ordered Expert -> Naive (stage priority, then most-recent first).

    If ``manifest`` is None, loads the QC-filtered staging manifest.
    """
    if manifest is None:
        from visdetect.analysis.config import load_staging_manifest
        manifest = load_staging_manifest(qc_only=True)

    stage_priority = {s: i for i, s in enumerate(reversed(STAGE_ORDER))}  # Expert -> 0
    fallback = len(STAGE_ORDER)
    rows = []
    for _, r in manifest.iterrows():
        sn = str(r["session_name"])
        rank = stage_priority.get(str(r.get("stage", "")), fallback)
        ymd = parse_session_date(int(sn))                 # (yyyy, mm, dd)
        rows.append((rank, tuple(-x for x in ymd), sn))   # negate for most-recent-first
    rows.sort(key=lambda t: (t[0], t[1]))
    return [sn for _, _, sn in rows]


# change_size -> opacity for the optional difficulty shading (bigger = more opaque).
# Keys must stay in sync with constants.CHANGE_SIZES; unknown sizes fall back to 1.0.
_CS_OPACITY = {1.25: 0.30, 1.35: 0.45, 1.5: 0.60, 2.0: 0.80, 4.0: 1.0}


def _hex_to_rgb01(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def render_raster(ax, raster_df, change_size_shading: bool = False, episodes=None):
    """Draw the outcome raster on ``ax``: one colored bar per trial.

    Catch trials get a black outline. With ``change_size_shading``, go-trial hits
    and genuine (go-trial) misses are shaded by change size. ``episodes`` (list of
    StateEpisode) are drawn as translucent state spans behind the ticks.
    """
    import matplotlib.patches as mpatches  # deferred so the library imports without matplotlib

    n = len(raster_df)
    if episodes:
        state_tints = {"Impulsive": (0.84, 0.15, 0.16), "StimSens": (0.17, 0.63, 0.17),
                       "Disengaged": (0.48, 0.36, 0.72)}
        for ep in episodes:
            rgb = state_tints.get(ep.state_label, (0.5, 0.5, 0.5))
            ax.axvspan(ep.start_trial - 0.5, ep.end_trial + 0.5, color=rgb, alpha=0.18, lw=0)

    for i, row in enumerate(raster_df.itertuples(index=False)):
        lv = row.lick_valence
        base = LICK_VALENCE_COLORS.get(lv, "#999999")
        if change_size_shading and row.is_go and lv in ("appropriate_lick", "nolick"):
            rgb = _hex_to_rgb01(base)
            alpha = _CS_OPACITY.get(round(float(row.change_size), 2), 1.0)
            color = (rgb[0], rgb[1], rgb[2], alpha)
        else:
            color = base
        edge = "#111111" if row.is_catch else "none"
        ax.add_patch(mpatches.Rectangle((i - 0.5, 0), 1.0, 1.0, facecolor=color,
                                        edgecolor=edge, linewidth=0.6))
    ax.set_xlim(-0.5, max(n - 0.5, 0.5))
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("trial index")
    return ax
