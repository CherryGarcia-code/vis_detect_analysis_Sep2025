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


# ── Engine-A simulator (Task 3.1) — draw through the per-bin hazard (§A.8) ────
def simulate_licks(design, true_theta, param_spec, seed=0):
    """Generate (event_bin, lick, censored) by walking each trial's per-bin hazard.
    Uses the SAME A/phi/mood as `design` (so a refit Design reuses them)."""
    assert len(true_theta) == param_spec.n_params()
    rng = np.random.default_rng(seed)
    v, z, u = param_spec.per_trial(true_theta, design.mood_code)
    n = len(design)
    event_bin = np.empty(n, int); lick = np.zeros(n, int); censored = np.zeros(n, bool)
    for i in range(n):
        A, phi = design.A[i], design.phi[i]
        h = np.clip(hazard_from_lp(z[i] + v[i] * A + u[i] * phi), 1e-12, 1 - 1e-12)
        fired = -1
        draws = rng.random(len(h))
        for k in range(len(h)):
            if draws[k] < h[k]:
                fired = k; break
        if fired >= 0:
            event_bin[i] = fired; lick[i] = 1
        else:
            event_bin[i] = len(h) - 1; censored[i] = True
    return event_bin, lick, censored


def design_with_outcomes(design, event_bin, lick, censored):
    """Return a copy of `design` with simulated outcomes (A/phi/mood unchanged), so it
    can be refit. NOTE: trials are truncated at event_bin for the likelihood via
    event_bin only; A/phi keep full length (hazard_nll reads only [:K+1])."""
    import copy
    d = copy.copy(design)
    d.event_bin = np.asarray(event_bin, int); d.lick = np.asarray(lick, int)
    d.censored = np.asarray(censored, bool)
    return d


# ── Engine-A fitter (Task 1.5) — penalized MLE + FitResult (contract §A.7) ────
@dataclass
class FitResult:
    """Result of a single-anchor penalized-MLE fit (contract §A.7).

    The ``dials`` structure is LOCKED here and consumed unchanged by Tasks
    2.x/3.x/4.1: ``{mood: {"sharpness": v, "itchiness": z, "timing": u}}``.

    Attributes
    ----------
    theta : np.ndarray
        Best-fit parameter vector (length ``param_spec.n_params()``).
    dials : dict
        ``{mood: {"sharpness": v, "itchiness": z, "timing": u}}`` read out of
        ``theta`` via ``param_spec.value``.
    ll : float
        PURE data log-likelihood at the optimum (``-hazard_nll(theta, ..., l2=0)``;
        the L2 penalty, if any, is excluded).
    n_params : int
        ``param_spec.n_params()``.
    cov : np.ndarray | None
        Inverse Hessian (parameter covariance); ``None`` if the Hessian is
        singular / non-invertible.
    hessian : np.ndarray
        Finite-difference (central-second-difference) Hessian of the unpenalized
        ``hazard_nll`` at the optimum.
    hessian_cond : float
        ``np.linalg.cond(hessian)`` (``np.inf`` if it is singular / raises).
    """
    theta: np.ndarray
    dials: dict
    ll: float
    n_params: int
    cov: np.ndarray | None
    hessian: np.ndarray
    hessian_cond: float


def _numerical_hessian(f, x, eps=1e-4):
    """Central-difference Hessian of a scalar function ``f`` at ``x``.

    Self-contained (no numdifftools / scipy dependency) so the fitter has no new
    third-party requirement. Uses the standard central second-difference stencils::

        H[i,i] = (f(x+e_i) - 2 f(x) + f(x-e_i)) / eps**2
        H[i,j] = (f(x+e_i+e_j) - f(x+e_i-e_j)
                  - f(x-e_i+e_j) + f(x-e_i-e_j)) / (4 eps**2)   (i != j)

    The Hessian is symmetrised (``0.5 (H + H.T)``) to wash out tiny asymmetries
    from finite-precision evaluation.
    """
    x = np.asarray(x, float)
    n = x.size
    H = np.zeros((n, n), float)
    f0 = float(f(x))
    e = eps
    # diagonal: standard central second difference
    for i in range(n):
        xi = x.copy()
        xi[i] += e
        f_plus = float(f(xi))
        xi[i] = x[i] - e
        f_minus = float(f(xi))
        H[i, i] = (f_plus - 2.0 * f0 + f_minus) / (e * e)
    # off-diagonals: 4-point central difference
    for i in range(n):
        for j in range(i + 1, n):
            xpp = x.copy(); xpp[i] += e; xpp[j] += e
            xpm = x.copy(); xpm[i] += e; xpm[j] -= e
            xmp = x.copy(); xmp[i] -= e; xmp[j] += e
            xmm = x.copy(); xmm[i] -= e; xmm[j] -= e
            val = (float(f(xpp)) - float(f(xpm))
                   - float(f(xmp)) + float(f(xmm))) / (4.0 * e * e)
            H[i, j] = val
            H[j, i] = val
    return 0.5 * (H + H.T)


def fit_anchor(design, param_spec, seed_theta=None, l2=0.0, n_restarts=4, seed=0):
    """Fit one anchor's generative decision-latents by penalized MLE (contract §A.7).

    Plain English: find the three behavioural knobs per mood (sharpness ``v``,
    itchiness ``z``, timing ``u``) that best explain *when* the mouse licked on
    this anchor, by minimising the closed-form censored negative log-likelihood
    (``hazard_nll``). Optimisation is L-BFGS-B from several inits (the
    ``seed_theta`` if given, plus ``n_restarts`` random restarts); the lowest-NLL
    fit wins. A finite-difference Hessian at the optimum gives the parameter
    covariance and a conditioning diagnostic.

    Parameters
    ----------
    design : Design
        The anchor's ragged precompute (with the outcomes to fit).
    param_spec : ParamSpec
        Parameter layout (``theta`` <-> dial/mood mapping).
    seed_theta : np.ndarray | None
        Optional warm-start init (e.g. an L2-seeded backward fit's prior). When
        given it is BOTH an optimisation init AND the ridge reference if ``l2>0``.
    l2 : float
        Ridge penalty strength toward ``seed_theta`` (only active when both
        ``l2>0`` and ``seed_theta`` is not None). The reported ``ll`` always
        excludes this penalty.
    n_restarts : int
        Number of random restarts (Normal(0, 1) around 0), drawn from
        ``np.random.default_rng(seed)``.
    seed : int
        RNG seed for the random restarts (reproducible).

    Returns
    -------
    FitResult
        With the LOCKED ``dials`` structure
        ``{mood: {"sharpness": v, "itchiness": z, "timing": u}}``.
    """
    from scipy.optimize import minimize

    n_params = param_spec.n_params()

    def objective(theta):
        return hazard_nll(theta, design, param_spec, l2=l2, seed_theta=seed_theta)

    # ── assemble inits: the seed (if any) + n_restarts random restarts ──
    rng = np.random.default_rng(seed)
    inits = []
    if seed_theta is not None:
        inits.append(np.asarray(seed_theta, float).copy())
    for _ in range(int(n_restarts)):
        inits.append(rng.normal(loc=0.0, scale=1.0, size=n_params))
    if not inits:                       # n_restarts==0 and no seed -> one zero init
        inits.append(np.zeros(n_params))

    best = None
    for x0 in inits:
        try:
            res = minimize(objective, x0, method="L-BFGS-B")
        except Exception:
            continue
        if not np.all(np.isfinite(res.x)):
            continue
        nll = float(res.fun)
        if not np.isfinite(nll):
            continue
        if best is None or nll < best[0]:
            best = (nll, np.asarray(res.x, float))

    if best is None:
        # Pathological: every restart failed. Fall back to the first init's value.
        theta = np.asarray(inits[0], float)
    else:
        theta = best[1]

    # ── pure data log-likelihood (exclude the L2 penalty) ──
    ll = -hazard_nll(theta, design, param_spec, l2=0.0)

    # ── finite-difference Hessian of the UNPENALIZED nll at the optimum ──
    def nll_data(theta_):
        return hazard_nll(theta_, design, param_spec, l2=0.0)

    hessian = _numerical_hessian(nll_data, theta, eps=1e-4)

    try:
        hessian_cond = float(np.linalg.cond(hessian))
    except Exception:
        hessian_cond = np.inf
    if not np.isfinite(hessian_cond):
        hessian_cond = np.inf

    try:
        cov = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        cov = None

    # ── locked dials structure ──
    dials = {
        mood: {
            "sharpness": float(param_spec.value(theta, "v", mood)),
            "itchiness": float(param_spec.value(theta, "z", mood)),
            "timing": float(param_spec.value(theta, "u", mood)),
        }
        for mood in param_spec.moods
    }

    return FitResult(
        theta=theta,
        dials=dials,
        ll=float(ll),
        n_params=int(n_params),
        cov=cov,
        hessian=hessian,
        hessian_cond=hessian_cond,
    )


# ── Engine-A model selection (Task 1.6) — rectification by CV-LL (§A.3) ───────
def select_rectification(design_builder, expert_trial_evidence, state_labels,
                         mu, sigma, candidates=("signed", "halfwave", "asym"),
                         k=5, seed=0):
    """Pick the evidence rectification by k-fold cross-validated log-likelihood.

    Plain English: should the accumulator integrate the *full* signed evidence
    (``signed`` — TF dips below base count as negative evidence), only the
    *positive* deflections (``halfwave`` — slow/below-base pulses ignored), or an
    asymmetrically-gained version (``asym``)? We answer it empirically: for each
    candidate we rebuild the trial Design with that rectification (which changes
    the accumulated ``A``), then score it by **held-out** log-likelihood under a
    refit hazard model. The candidate whose generative model best predicts
    *when* the mouse licks on data it was not fitted to wins. The winner is
    frozen for the downstream sweep (the caller's job).

    Cross-validation (FAIR comparison): the fold split is computed **once**, from
    a single ``np.random.default_rng(seed)`` shuffle of ``[0..n)`` via
    ``np.array_split``, and the SAME folds are reused for every candidate. The
    per-fold ``fit_anchor`` is given a SAME fixed seed for every candidate too.
    This is load-bearing: candidates that build identical Designs (e.g. ``asym``
    with default unit gains reduces to ``signed``) must score identically — any
    per-candidate reshuffle of the folds or per-candidate fit seed would make the
    comparison apples-to-oranges and the winner CV-noise. For each fold we
    ``fit_anchor`` on the train subset (``Design.subset(train_idx)``) and evaluate
    the held-out **data log-likelihood**
    ``= -hazard_nll(fit.theta, Design.subset(test_idx), param_spec, l2=0.0)``,
    then sum across folds. The score reported per candidate is this summed
    held-out log-likelihood (higher = better). Each candidate uses a default
    ``ParamSpec(rectification=cand)`` so the fit's parameter layout matches; the
    Design's ``A`` is already built with that rectification (so the rectification
    field on the spec is metadata here — ``A`` is precomputed).

    Parameters
    ----------
    design_builder : callable
        The ``build_design`` callable (passed in for testability). Invoked as
        ``design_builder(expert_trial_evidence, state_labels, mu, sigma,
        rectification=cand)`` for each candidate.
    expert_trial_evidence : pandas.DataFrame
        Per-trial evidence frame (``build_trial_evidence_corrected`` form) for the
        expert anchor(s).
    state_labels : pandas.DataFrame
        Trial-indexed mood labels (``build_design`` form).
    mu, sigma : float
        Temporal-expectation bump anchor and FIXED width (passed through).
    candidates : tuple[str]
        Rectifications to compare (default the three contract rectifications).
    k : int
        Number of CV folds.
    seed : int
        RNG seed for the fold split (reproducible).

    Returns
    -------
    dict
        ``{"scores": {cand: cv_loglik}, "winner": argmax_candidate}``.
    """
    # ── build every candidate's Design ONCE (rectification only changes A) ──
    designs = {
        cand: design_builder(expert_trial_evidence, state_labels, mu, sigma,
                             rectification=cand)
        for cand in candidates
    }

    # All candidates are rebuilt from the SAME evidence/labels, so they keep the
    # SAME trials in the SAME order (rectification changes A values, not which
    # trials survive). Verify, then compute ONE fold split shared by all.
    n_set = {len(d) for d in designs.values()}
    if len(n_set) != 1:
        raise ValueError(
            "candidate Designs differ in trial count "
            f"{ {c: len(d) for c, d in designs.items()} }; cannot CV-compare "
            "them on a shared fold split")
    n = n_set.pop()

    if n < k or n == 0:
        # too few trials to split into k folds -> every candidate unusable
        scores = {cand: -np.inf for cand in candidates}
        return {"scores": scores, "winner": max(scores, key=scores.get)}

    # ── ONE shuffled fold split, reused for EVERY candidate (fairness) ──
    idx = np.arange(n)
    np.random.default_rng(seed).shuffle(idx)
    folds = np.array_split(idx, k)

    scores = {}
    for cand in candidates:
        design = designs[cand]
        param_spec = ParamSpec(rectification=cand)

        cv_loglik = 0.0
        ok = True
        for f in range(k):
            test_idx = folds[f]
            if len(test_idx) == 0:
                continue
            train_idx = np.concatenate([folds[j] for j in range(k) if j != f])
            if len(train_idx) == 0:
                ok = False
                break
            train_design = design.subset(train_idx)
            test_design = design.subset(test_idx)
            # SAME fixed fit seed for every candidate & fold (apples-to-apples)
            fit = fit_anchor(train_design, param_spec, seed_theta=None,
                             l2=0.0, n_restarts=2, seed=seed)
            held_out_ll = -hazard_nll(fit.theta, test_design, param_spec, l2=0.0)
            cv_loglik += float(held_out_ll)

        scores[cand] = cv_loglik if ok else -np.inf

    winner = max(scores, key=scores.get)
    return {"scores": scores, "winner": winner}


# ── Engine-A anchor-design dict (Task 1.7) — the Phase-1->Phase-2 bridge ──────
def build_anchor_designs(sessions, param_spec, mu_by_session, sigma, dt=0.05,
                         leak_tau=0.27, rectification="signed"):
    """Assemble the per-session ``Design`` dict that the Phase-2 sweep/ladders consume.

    Plain English: for each anchor session this loads the session, rebuilds the
    Phase-1 per-trial table (which mood each trial is in) and the corrected
    per-trial log2-TF evidence, keeps ONLY the MAIN_MOOD cells the generative
    model can actually identify its dials on (``compute_cell_qc(...)
    ['usable_generative']``), builds a ragged :class:`Design` on just those cells'
    trials, and stores it keyed by session name. A session with no usable cell is
    OMITTED from the returned dict (the caller ships Phase-1 proxies for it).

    This resolves the contract §A.10-3 interface gap: the anchor-design dict is
    built here EXPLICITLY rather than hidden inside ``backward_sweep`` /
    ``learning_ladder`` (which consume ``dict[str, Design]``).

    Parameters
    ----------
    sessions : iterable[str]
        Anchor session names to build Designs for (e.g. the
        ``select_expert_anchors`` output's ``anchors``).
    param_spec : ParamSpec
        Parameter layout — accepted for interface symmetry with the downstream
        sweep/ladders (the Design itself is layout-agnostic; ``mood_code`` indexes
        ``MAIN_MOODS`` exactly as :func:`build_design` produces).
    mu_by_session : Mapping[str, float]
        Per-session temporal-expectation anchor μ (Task 0.4
        ``change_time_anchor`` applied per session, on that session's *reached*
        trials). Looked up by session name to seed each Design's urgency bump.
    sigma : float
        FIXED urgency-bump width (seconds; a ``ParamSpec`` field, not fitted).
    dt, leak_tau, rectification :
        Generative time-grid + leaky-accumulator settings, passed straight to
        :func:`build_design` (contract §A.3).

    Returns
    -------
    dict[str, Design]
        ``{session_name: Design}`` for every session with at least one
        ``usable_generative`` MAIN_MOOD cell and a non-empty Design. Sessions
        failing the QC gate (or producing an empty Design) are omitted.

    Notes
    -----
    * ``del sess; gc.collect()`` after each session (sessions are large).
    * Loaders are referenced via their home modules (``visdetect.suite.loader``
      and ``visdetect.analysis.decision_latents``) so they import lazily and stay
      cleanly monkeypatchable in tests.
    * Off the buggy ``ddm.build_trial_evidence`` evidence sampler: evidence comes
      from ``decision_latents.build_trial_evidence_corrected`` (the 60 Hz /
      runs-of-3 builder); the Phase-1 ``build_trial_table`` is used ONLY for the
      per-mood QC counts.
    """
    import gc

    from visdetect.suite import loader as _suite_loader
    from visdetect.analysis import decision_latents as dl

    out: dict[str, Design] = {}
    for sname in sessions:
        sess = _suite_loader.load_session(sname)
        try:
            labels = dl.load_state_labels(sname)
            trial_table = dl.build_trial_table(sess, labels, sname)
            ev_df = dl.build_trial_evidence_corrected(sess, dt=dt)

            # Keep ONLY MAIN_MOOD cells whose per-mood QC clears usable_generative.
            usable_moods = []
            for m in dl.MAIN_MOODS:
                cell = trial_table[trial_table["state_label"] == m]
                if len(cell) > 0 and dl.compute_cell_qc(cell)["usable_generative"]:
                    usable_moods.append(m)

            if usable_moods:
                labels_usable = labels[labels["state_label"].isin(usable_moods)]
                design = build_design(
                    ev_df, labels_usable, mu_by_session[sname], sigma, dt=dt,
                    leak_tau=leak_tau, rectification=rectification)
                if len(design) > 0:
                    out[sname] = design
        finally:
            del sess
            gc.collect()
    return out
