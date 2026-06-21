"""B8 Phase 2 — Engine A: the generative decision-latent model.

Plain English: Phase 1 *measured* three behavioural knobs per trial (sharpness,
itchiness, timing) directly. Phase 2 instead *fits a small generative model* of
when the mouse licks, so the same three knobs become parameters we can recover
and compare across learning. The model is a closed-form cloglog hazard
accumulator (NOT pyddm): a leaky accumulator of log2-TF evidence plus a fixed
temporal-expectation bump drives a per-bin lick hazard.

This module is grown task-by-task per the Engine-A contract (§A). This first
piece is the **expert-anchor contingency gate** (Task 0.9): given the Task-0.8
session inventory, decide which sessions are trustworthy enough to anchor the
generative fit, and in which regime:

    * "expert"   — >= ``min_anchors`` sessions clear the gate; fit on those.
    * "pooled"   — too few clear it, but we can top up to ``min_anchors`` with the
                   strongest remaining (latest, best-d') sessions.
    * "fallback" — even pooling cannot reach ``min_anchors``; downstream ships the
                   Phase-1 descriptive proxies and skips the generative fit.

Behaviour-only: no session loading, no spikes, no ``ddm`` imports here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ── Engine-A constants (used by later tasks; declared here per contract §A) ──
DT_GEN = 0.05                       # generative time grid (s); one TF update
LEAK_TAU_S = 0.27                   # default leak time-constant (s)
LEAK_TAU_SWEEP = (0.15, 0.27, 0.40)  # leak sweep ("is tau learned" -> B1)


def _session_column(inventory_df: pd.DataFrame) -> str:
    """Return the session-id column name (accept `session` or `session_name`)."""
    for cand in ("session", "session_name"):
        if cand in inventory_df.columns:
            return cand
    raise KeyError(
        "inventory_df must have a 'session' or 'session_name' column; "
        f"got {list(inventory_df.columns)}"
    )


def _session_ids(col: pd.Series) -> pd.Series:
    """Normalise session ids to canonical 8-digit DDMMYYYY strings.

    pandas reads the inventory CSV's session column as int64, which silently
    strips the leading zero (``01072025`` -> ``"1072025"``). Downstream loaders
    expect the 8-digit form, so zero-pad numeric ids (MEMORY.md ``zfill(8)``
    gotcha). Already-string / non-numeric ids are passed through unchanged.
    """
    s = col.astype(str).str.strip()
    numeric = s.str.fullmatch(r"\d+")
    return s.mask(numeric.fillna(False), s.str.zfill(8))


def select_expert_anchors(inventory_df: pd.DataFrame, min_d: float = 0.7,
                          min_mood_n: int = 20, min_anchors: int = 3) -> dict:
    """Pick the generative-fit anchor sessions and the regime to fit them in.

    A session **qualifies** as an expert anchor when the mouse clearly saw the
    change *and* both engaged moods have enough trials to fit::

        dprime > min_d  AND  n_impu >= min_mood_n  AND  n_stim >= min_mood_n

    Selection (Task 0.9 / contract §A.10 item 5):

    * ``mode="expert"``   if ``len(qualifying) >= min_anchors`` — anchors are the
      qualifying session ids, in inventory order (chronological if the inventory
      is ordered; otherwise sorted by session id).
    * ``mode="pooled"``   if too few qualify *but* there are at least
      ``min_anchors`` sessions overall — the qualifiers are topped up with the
      strongest remaining sessions (``dprime`` descending, then recency =
      later-in-inventory first) until exactly ``min_anchors`` anchors.
    * ``mode="fallback"`` if even pooling cannot reach ``min_anchors`` — anchors
      are whatever qualifies (possibly empty); the caller ships Phase-1 proxies
      (``latent_trust="descriptive"``) and skips the generative fit.

    Parameters
    ----------
    inventory_df : pandas.DataFrame
        Task-0.8 inventory. Must have a session-id column (``session`` or
        ``session_name``) plus ``dprime``, ``n_impu``, ``n_stim``.
    min_d, min_mood_n, min_anchors : float, int, int
        Gate thresholds (see above).

    Returns
    -------
    dict
        ``{"anchors": list[str], "mode": "expert"|"pooled"|"fallback"}``.
    """
    df = inventory_df.reset_index(drop=True).copy()
    if len(df) == 0:
        return {"anchors": [], "mode": "fallback"}

    scol = _session_column(df)
    sess = _session_ids(df[scol])
    dprime = pd.to_numeric(df["dprime"], errors="coerce").to_numpy(float)
    n_impu = pd.to_numeric(df["n_impu"], errors="coerce").fillna(0).to_numpy(float)
    n_stim = pd.to_numeric(df["n_stim"], errors="coerce").fillna(0).to_numpy(float)

    qualifies = (dprime > min_d) & (n_impu >= min_mood_n) & (n_stim >= min_mood_n)
    qual_pos = np.flatnonzero(qualifies)            # inventory-order positions
    qual_ids = [sess.iloc[i] for i in qual_pos]     # keep inventory order

    # ── expert ──────────────────────────────────────────────────────────────
    if len(qual_pos) >= min_anchors:
        return {"anchors": qual_ids, "mode": "expert"}

    n_total = len(df)

    # ── pooled: top up qualifiers to min_anchors with the strongest others ──
    if n_total >= min_anchors:
        non_qual_pos = [i for i in range(n_total) if i not in set(qual_pos.tolist())]
        # rank remaining by dprime desc, ties broken by recency (later position first)
        d_safe = np.where(np.isfinite(dprime), dprime, -np.inf)
        non_qual_sorted = sorted(
            non_qual_pos, key=lambda i: (d_safe[i], i), reverse=True
        )
        need = min_anchors - len(qual_pos)
        topup_pos = non_qual_sorted[:need]
        anchor_pos = sorted(qual_pos.tolist() + list(topup_pos))  # inventory order
        anchors = [sess.iloc[i] for i in anchor_pos]
        return {"anchors": anchors, "mode": "pooled"}

    # ── fallback: cannot reach min_anchors even by pooling ──────────────────
    return {"anchors": qual_ids, "mode": "fallback"}


# ── Engine-A math (Task 1.1) — leaky accumulator (contract §A.3) ─────────────
def leaky_accumulate(evidence, dt=0.05, leak_tau=0.27, rectification="signed",
                     g_up=1.0, g_down=1.0):
    """A[k] = decay*A[k-1] + R(e[k])*dt, decay = exp(-dt/leak_tau)."""
    from visdetect.analysis.ddm import rectify
    kind = {"signed": "symmetric"}.get(rectification, rectification)   # ddm uses 'symmetric'
    r = rectify(np.asarray(evidence, float), kind, g_up=g_up, g_down=g_down)
    decay = np.exp(-dt / float(leak_tau))
    A = np.empty(len(r), float)
    acc = 0.0
    for k in range(len(r)):
        acc = decay * acc + r[k] * dt
        A[k] = acc
    return A
