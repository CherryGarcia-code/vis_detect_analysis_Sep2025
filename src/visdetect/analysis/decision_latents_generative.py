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

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ── Engine-A constants (used by later tasks; declared here per contract §A) ──
DT_GEN = 0.05                       # generative time grid (s); one TF update
LEAK_TAU_S = 0.27                   # default leak time-constant (s)
LEAK_TAU_SWEEP = (0.15, 0.27, 0.40)  # leak sweep ("is tau learned" -> B1)


# ── Engine-A math (Task 1.2) — cloglog hazard link (contract §A.1) ───────────
def hazard_from_lp(lp):
    """Inverse cloglog link: h = 1 - exp(-exp(lp)), numerically stable, h in (0,1)."""
    exp_lp = np.exp(np.clip(lp, -30.0, 30.0))
    return -np.expm1(-exp_lp)                      # 1 - exp(-exp(lp))


def lp_from_hazard(h):
    """Forward cloglog link: lp = log(-log(1-h))."""
    h = np.clip(np.asarray(h, float), 1e-12, 1 - 1e-12)
    return np.log(-np.log1p(-h))


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


# ── Engine-A math (Task 1.2) — temporal-expectation bump (contract §A.3) ─────
def expectation_bump(t_grid, mu, sigma):
    """Gaussian temporal-expectation profile, peak 1.0 at mu (sigma FIXED, not fitted)."""
    t_grid = np.asarray(t_grid, float)
    return np.exp(-0.5 * ((t_grid - mu) / float(sigma)) ** 2)


# ── Engine-A precompute (Task 1.3) — ragged Design (contract §A.5) ───────────
@dataclass
class Design:
    A: list            # list[np.ndarray]  leaky-accumulated evidence per trial (len == n_bins_i)
    phi: list          # list[np.ndarray]  urgency bump per trial (same lengths)
    event_bin: np.ndarray   # int   index of the decision bin per trial (== n_bins_i - 1)
    lick: np.ndarray        # int 0/1
    censored: np.ndarray    # bool
    mood_code: np.ndarray   # int   index into ParamSpec.moods
    trial_idx: np.ndarray   # int
    dt: float

    def __len__(self):
        return len(self.A)

    def subset(self, idx):
        idx = np.asarray(idx, int)
        return Design(A=[self.A[i] for i in idx], phi=[self.phi[i] for i in idx],
                      event_bin=self.event_bin[idx], lick=self.lick[idx],
                      censored=self.censored[idx], mood_code=self.mood_code[idx],
                      trial_idx=self.trial_idx[idx], dt=self.dt)


def build_design(trial_evidence_df, state_labels, mu, sigma, dt=0.05,
                 leak_tau=0.27, rectification="signed"):
    """Precompute A (leaky_accumulate) and phi (expectation_bump on the trial's t-grid)
    per trial; map mood to code (drop EXCLUDED_MOODS, keep MAIN_MOODS for fitting).
    event_bin = n_bins-1 (decision in the last bin). Returns a Design.

    Parameters
    ----------
    trial_evidence_df : pandas.DataFrame
        Output of ``decision_latents.build_trial_evidence_corrected`` — columns
        ``trial_idx, outcome, change_size, change_time, decision_time, lick,
        censored, evidence(np.ndarray), n_bins``.
    state_labels : pandas.DataFrame
        Indexed by ``trial_idx`` with a ``state_label`` column (from
        ``decision_latents.load_state_labels``). Trials whose mood is not in
        ``MAIN_MOODS`` (Impulsive / StimSens) are dropped — Disengaged is handled
        in reporting, Abort/untagged trials are dropped (Phase-1 rule).
    mu, sigma : float
        Temporal-expectation bump anchor (``mu`` = per-session empirical
        change-time anchor) and FIXED width. ``phi`` is therefore fully
        data-determined here and never depends on the fitted parameters.
    dt, leak_tau, rectification :
        Generative time-grid and leaky-accumulator settings (contract §A.3).
    """
    # MAIN_MOODS imported lazily to avoid a heavy import at module load.
    from visdetect.analysis.decision_latents import MAIN_MOODS

    A_list, phi_list = [], []
    event_bin, lick, censored, mood_code, trial_idx = [], [], [], [], []

    for row in trial_evidence_df.itertuples(index=False):
        tidx = int(row.trial_idx)
        # Look up the trial's mood; keep ONLY MAIN_MOODS (drop untagged / others).
        if tidx not in state_labels.index:
            continue
        mood = state_labels.loc[tidx, "state_label"]
        if mood not in MAIN_MOODS:
            continue

        A_i = leaky_accumulate(row.evidence, dt=dt, leak_tau=leak_tau,
                               rectification=rectification)
        n_bins_i = len(A_i)
        phi_i = expectation_bump(np.arange(n_bins_i) * dt, mu, sigma)

        A_list.append(A_i)
        phi_list.append(phi_i)
        event_bin.append(n_bins_i - 1)
        lick.append(int(row.lick))
        censored.append(bool(row.censored))
        mood_code.append(MAIN_MOODS.index(mood))
        trial_idx.append(tidx)

    return Design(
        A=A_list,
        phi=phi_list,
        event_bin=np.asarray(event_bin, int),
        lick=np.asarray(lick, int),
        censored=np.asarray(censored, bool),
        mood_code=np.asarray(mood_code, int),
        trial_idx=np.asarray(trial_idx, int),
        dt=float(dt),
    )


# ── Engine-A parameter layout (Task 1.4) — ParamSpec (contract §A.4) ──────────
@dataclass(frozen=True)
class ParamSpec:
    """Declarative ``theta`` <-> dial/mood mapping (contract §A.4).

    Owns the layout so no downstream task hardcodes parameter indices. Three
    dials -- ``v`` (sharpness), ``z`` (itchiness/caution), ``u`` (timing
    amplitude) -- each carries a per-mood term when listed in ``state_terms``,
    otherwise it is shared across moods (one slot). ``leak_tau`` and
    ``urgency_sigma`` are FIXED (not fitted) and live here so the Design and the
    likelihood agree on them.
    """
    moods: tuple = ("Impulsive", "StimSens")
    dials: tuple = ("v", "z", "u")
    state_terms: tuple = ("v", "z", "u")     # which dials carry a per-mood term
    rectification: str = "signed"
    leak_tau: float = 0.27
    urgency_sigma: float = 0.8               # FIXED seconds

    def _len(self, dial):
        return len(self.moods) if dial in self.state_terms else 1

    def n_params(self):
        return sum(self._len(d) for d in self.dials)

    def _offset(self, dial):
        off = 0
        for d in self.dials:
            if d == dial:
                return off
            off += self._len(d)
        raise KeyError(dial)

    def value(self, theta, dial, mood):
        off = self._offset(dial)
        return theta[off + self.moods.index(mood)] if dial in self.state_terms else theta[off]

    def per_trial(self, theta, mood_code):
        """mood_code: int array indexing self.moods. Returns (v, z, u) per-trial arrays."""
        out = {}
        for dial in ("v", "z", "u"):
            off = self._offset(dial)
            if dial in self.state_terms:
                vals = np.asarray([theta[off + m] for m in mood_code])
            else:
                vals = np.full(len(mood_code), theta[off])
            out[dial] = vals
        return out["v"], out["z"], out["u"]


# ── Engine-A likelihood (Task 1.4) — closed-form censored NLL (contract §A.6) ─
def hazard_nll(theta, design, param_spec, l2=0.0, seed_theta=None):
    """Closed-form censored negative log-likelihood (contract §A.6).

    Per trial the linear predictor is ``lp = z + v*A + u*phi`` and the per-bin
    lick hazard is ``h = inv_cloglog(lp)``. A lick in bin ``K`` contributes
    ``h_K * prod_{k<K}(1-h_k)``; a censored (no-lick / Miss) trial, right-censored
    at ``K``, contributes ``prod_{k<=K}(1-h_k)``. ``l2>0`` with a ``seed_theta``
    adds a ridge penalty toward the seed (used for L2-seeded backward fits).
    """
    v, z, u = param_spec.per_trial(theta, design.mood_code)
    nll = 0.0
    for i in range(len(design)):
        A, phi = design.A[i], design.phi[i]
        K = int(design.event_bin[i])                     # == len(A) - 1
        lp = z[i] + v[i] * A + u[i] * phi                # linear predictor, len == K+1
        h = np.clip(hazard_from_lp(lp), 1e-12, 1 - 1e-12)
        log_surv = np.sum(np.log1p(-h[:K]))              # log Prod_{k<K}(1-h_k)
        if design.lick[i] == 1 and not design.censored[i]:
            nll -= log_surv + np.log(h[K])               # event in bin K
        else:
            nll -= log_surv + np.log1p(-h[K])            # survived through bin K (censored)
    if l2 > 0 and seed_theta is not None:
        nll += float(l2) * np.sum((np.asarray(theta) - np.asarray(seed_theta)) ** 2)
    return float(nll)
