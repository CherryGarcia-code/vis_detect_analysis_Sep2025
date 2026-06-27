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

# Single source of truth for the zfill8 session-id key (config is import-light:
# constants + pandas only, no ddm/pyddm) -- keeps this module pyddm-free.
from visdetect.analysis.config import canonical_session_id  # noqa: F401  (re-exported)

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
def _rectify(e, kind, g_up=1.0, g_down=1.0):
    """Local copy of ddm.rectify (pure numpy), so this synthetic module stays
    ddm-free: importing ddm pulls `pyddm` at module load, but the cluster
    recovery harness runs on a numpy/scipy-only env. Behaviour-identical to
    ddm.rectify (verified by the Task-1.1 rectification tests)."""
    e = np.asarray(e, dtype=float)
    if kind == "symmetric":
        return e
    if kind == "halfwave":
        return np.clip(e, 0.0, None)
    if kind == "asym":
        return np.where(e >= 0, g_up * e, g_down * e)
    raise ValueError(kind)


def leaky_accumulate(evidence, dt=0.05, leak_tau=0.27, rectification="signed",
                     g_up=1.0, g_down=1.0):
    """A[k] = decay*A[k-1] + R(e[k])*dt, decay = exp(-dt/leak_tau)."""
    kind = {"signed": "symmetric"}.get(rectification, rectification)   # 'signed' -> 'symmetric'
    r = _rectify(np.asarray(evidence, float), kind, g_up=g_up, g_down=g_down)
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


# ── Engine-A recovery (Task 3.2) — per-dial point recovery (contract §A.9) ────
# Maps the internal theta dial keys (v/z/u) to the public, FitResult-aligned dial
# names (sharpness/itchiness/timing) used in every recovery report.
_DIAL_PUBLIC_NAME = {"v": "sharpness", "z": "itchiness", "u": "timing"}

# Default per-dial jitter SDs for perturbing the GROUND TRUTH across reps. The
# spread must be large relative to the fit's recovery noise so the Pearson r is a
# genuine across-rep signal (a constant true value would make r undefined). Tuned
# so the expert regime stays in its identifiable band (v ~1.0-1.6, z very negative
# so trials survive to the post-change excursion, u modest). NOT fitted; the
# caller may override.
_RECOVER_JITTER_SD = {"v": 0.30, "z": 0.45, "u": 0.20}


def recover_point(design, true_theta, param_spec, n_rep=100, seed=0,
                  jitter_sd=None, n_restarts=2):
    """Per-dial point-recovery of the generative decision-latents (contract §A.9).

    Plain English: the make-or-break question for the generative model is "if the
    mouse REALLY had these three behavioural knobs (sharpness ``v``, itchiness
    ``z``, timing ``u``), would our fitter get them back?" We answer it as a
    GENUINE ground-truth measurement, NOT a tautology: over ``n_rep`` replicates we
    JITTER the true dial values around ``true_theta`` (so the truth genuinely
    *varies* across reps), simulate licks through that perturbed truth, refit with
    :func:`fit_anchor`, and read the recovered dials. Per dial we then report how
    well the recovered values TRACK the (varied) truth.

    Why jitter the truth (not just resimulate one fixed truth): the Pearson ``r``
    between recovered and true is only meaningful when the truth SPANS a range — a
    fixed truth has zero variance and ``r`` is undefined. Sweeping the true value
    across reps makes ``r`` a real identifiability signal (does the estimate move
    WITH the truth?), exactly the §A.9 rigor requirement.

    For each rep ``j`` and each dial ``d``, BOTH moods contribute one
    ``(true, recovered)`` pair (the per-dial arrays POOL across moods), so with two
    moods each dial accumulates ``2 * n_rep`` pairs. Per dial we report:

    * ``r``           — Pearson correlation between recovered and true across all
      pairs (NaN if the truth has ~zero spread, which should not happen with a
      positive ``jitter_sd``);
    * ``bias``        — ``mean(recovered - true)`` across all pairs;
    * ``ci_coverage`` — fraction of pairs whose per-fit 95% CI
      (``recovered ± 1.96 * sqrt(diag(cov))`` from :attr:`FitResult.cov`) contains
      the true value. Reps whose ``cov`` is ``None`` (singular Hessian) are
      EXCLUDED from coverage (and counted in ``n_cov_excluded``); if every rep is
      excluded ``ci_coverage`` is ``NaN``.

    Parameters
    ----------
    design : Design
        The recovery ground-truth Design (A/phi/mood fixed; outcomes are RESIMULATED
        each rep). Typically ``make_recovery_design(regime)[0]``.
    true_theta : np.ndarray
        The regime's ground-truth parameter vector (length ``param_spec.n_params()``)
        — the CENTRE of the per-rep jitter.
    param_spec : ParamSpec
        Parameter layout (``theta`` <-> dial/mood mapping).
    n_rep : int
        Number of replicate simulate -> refit cycles (default 100; the test uses a
        reduced but still-genuine count for tractability).
    seed : int
        Master RNG seed. Each rep gets a deterministic child seed for both the
        truth jitter and the lick simulation, so the whole sweep is reproducible.
    jitter_sd : Mapping[str, float] | None
        Per-dial Gaussian jitter SD for perturbing the truth (defaults to
        :data:`_RECOVER_JITTER_SD`). The SAME jitter is applied to BOTH moods'
        slots of a dial each rep (so the dial moves coherently).
    n_restarts : int
        Random restarts for each per-rep :func:`fit_anchor` (default 2 — a
        tractability lever; the default ``fit_anchor`` value is 4).

    Returns
    -------
    dict
        ``{"sharpness": {...}, "itchiness": {...}, "timing": {...}}`` where each
        value is ``{"r": float, "bias": float, "sd_true": float,
        "ci_coverage": float, "n_pairs": int, "n_cov_excluded": int}``. ``sd_true``
        is the SD of the (jittered) true values across pairs — the natural scale
        for the ``|bias| <= 0.1 * SD(true)`` recovery tolerance (contract §A.9).
    """
    true_theta = np.asarray(true_theta, float)
    n_params = param_spec.n_params()
    assert len(true_theta) == n_params, (
        f"len(true_theta)={len(true_theta)} != n_params={n_params}")

    jitter = dict(_RECOVER_JITTER_SD)
    if jitter_sd is not None:
        jitter.update(jitter_sd)

    moods = list(param_spec.moods)
    dials = ("v", "z", "u")

    # Per (dial, mood) collectors of true / recovered / in-CI flags across reps.
    true_vals = {d: {m: [] for m in moods} for d in dials}
    rec_vals = {d: {m: [] for m in moods} for d in dials}
    in_ci = {d: {m: [] for m in moods} for d in dials}  # only reps with a cov

    master = np.random.default_rng(seed)
    # one independent child seed per rep (for truth jitter AND simulation)
    rep_seeds = master.integers(0, 2**31 - 1, size=int(n_rep))

    for j in range(int(n_rep)):
        rep_rng = np.random.default_rng(int(rep_seeds[j]))
        # ── jitter the TRUTH for this rep (same perturbation to both moods of a
        # dial, so the dial moves coherently across moods) ──
        theta_j = true_theta.copy()
        for d in dials:
            off = param_spec._offset(d)
            delta = float(rep_rng.normal(0.0, jitter[d]))
            for mi in range(len(moods)):
                theta_j[off + mi] = true_theta[off + mi] + delta

        # ── simulate -> refit ──
        sim_seed = int(rep_rng.integers(0, 2**31 - 1))
        eb, lk, cs = simulate_licks(design, theta_j, param_spec, seed=sim_seed)
        sim_design = design_with_outcomes(design, eb, lk, cs)
        fit = fit_anchor(sim_design, param_spec, seed_theta=None, l2=0.0,
                         n_restarts=int(n_restarts), seed=int(rep_seeds[j]))

        cov = fit.cov
        for d in dials:
            off = param_spec._offset(d)
            for mi, m in enumerate(moods):
                idx = off + mi
                t_val = float(theta_j[idx])
                r_val = float(fit.theta[idx])
                true_vals[d][m].append(t_val)
                rec_vals[d][m].append(r_val)
                if cov is not None:
                    var = float(cov[idx, idx])
                    if np.isfinite(var) and var >= 0.0:
                        se = np.sqrt(var)
                        lo, hi = r_val - 1.96 * se, r_val + 1.96 * se
                        in_ci[d][m].append(bool(lo <= t_val <= hi))

    # ── reduce to the per-dial summary (pooling across moods) ──
    out = {}
    for d in dials:
        t_pool = np.concatenate([np.asarray(true_vals[d][m], float) for m in moods])
        r_pool = np.concatenate([np.asarray(rec_vals[d][m], float) for m in moods])
        ci_pool = []
        for m in moods:
            ci_pool.extend(in_ci[d][m])

        # Pearson r (NaN-safe: undefined if the truth has ~zero spread)
        if t_pool.size >= 2 and np.std(t_pool) > 1e-12 and np.std(r_pool) > 1e-12:
            r = float(np.corrcoef(t_pool, r_pool)[0, 1])
        else:
            r = float("nan")
        bias = float(np.mean(r_pool - t_pool)) if t_pool.size else float("nan")
        sd_true = float(np.std(t_pool)) if t_pool.size else float("nan")
        ci_coverage = (float(np.mean(ci_pool)) if len(ci_pool) > 0
                       else float("nan"))
        n_cov_excluded = int(t_pool.size - len(ci_pool))

        out[_DIAL_PUBLIC_NAME[d]] = {
            "r": r,
            "bias": bias,
            "sd_true": sd_true,
            "ci_coverage": ci_coverage,
            "n_pairs": int(t_pool.size),
            "n_cov_excluded": n_cov_excluded,
        }
    return out


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


# ── Engine-A anchored sweep (Task 2.1) — expert-first, backward-seeded ────────
def backward_sweep(anchor_designs, anchors_chrono, param_spec, l2=1.0, seed=0):
    """Fit every anchor's generative decision-latents, expert-first and backward-seeded.

    Plain English: learning is a *ramp*, so we fit the mouse at its BEST first —
    the most-expert anchor — where the three behavioural knobs (sharpness ``v``,
    itchiness ``z``, timing ``u``) are most identifiable, and use that fit as a
    template to anchor the earlier, noisier sessions. Concretely:

    1. The MOST-EXPERT anchor (the LAST element of ``anchors_chrono``, i.e. the
       newest / most-expert session) is fit FIRST and FREE — ``seed_theta=None``
       and ``l2=0`` — giving the identifiable reference template.
    2. We then walk BACKWARD in reverse-chronological order (newest-1, ..., oldest).
       Each anchor is fit with ``seed_theta`` set to its more-expert (newer)
       neighbour's just-fitted ``theta`` and the passed ``l2`` ridge strength, so
       the expert template informs — but does not erase — the earlier fits
       (an L2-seeded backward fit; contract §A.6 ridge-toward-seed).

    A session listed in ``anchors_chrono`` but ABSENT from ``anchor_designs`` (e.g.
    QC-omitted by :func:`build_anchor_designs`) is SKIPPED: it is not fit and not
    returned, and the next (earlier) present anchor is seeded from the last
    successfully-fit ``theta`` — never from a missing one.

    Parameters
    ----------
    anchor_designs : dict[str, Design]
        Per-session ragged Designs to fit (from :func:`build_anchor_designs`).
        Sessions missing here are skipped.
    anchors_chrono : list[str]
        Session ids in CHRONOLOGICAL order (oldest -> newest). The last element is
        the most-expert anchor (fit first); the sweep walks this list in reverse.
    param_spec : ParamSpec
        Parameter layout (``theta`` <-> dial/mood mapping), passed to each
        :func:`fit_anchor`.
    l2 : float
        Ridge strength toward the more-expert neighbour's theta for every
        backward (non-expert) fit. The expert anchor is always fit with ``l2=0``.
    seed : int
        RNG seed for each ``fit_anchor`` random restarts (reproducible).

    Returns
    -------
    dict[str, FitResult]
        ``{session_id: FitResult}`` for every anchor present in
        ``anchor_designs`` (missing anchors are omitted).
    """
    results: dict[str, FitResult] = {}

    # Walk most-expert -> oldest: reverse of the chronological list. Iteration is
    # over ``anchors_chrono`` ONLY: any EXTRA key in ``anchor_designs`` that is not
    # a chronological anchor is never visited (so it cannot be fit out of order),
    # and a missing one is fetched with ``.get`` -> ``None`` (no KeyError).
    prev_theta = None                       # last successfully-fit theta (the seed)
    expert_done = False                     # has the free expert reference been fit?
    for a in reversed(list(anchors_chrono)):
        design = anchor_designs.get(a)
        if design is None:                  # QC-omitted: skip, keep prev_theta
            continue

        if not expert_done:
            # most-expert present anchor: free reference fit (no seed, no ridge)
            res = fit_anchor(design, param_spec, seed_theta=None, l2=0.0, seed=seed)
            expert_done = True
        else:
            # earlier anchor: L2-seeded from the more-expert neighbour's theta
            res = fit_anchor(design, param_spec, seed_theta=prev_theta, l2=l2,
                             seed=seed)

        results[a] = res
        prev_theta = res.theta

    return results


# ── Engine-A learning ladder (Task 2.2) — which dial moves across anchors ──────
# The five rungs and the dial each lets vary ACROSS anchors (the rest are SHARED
# across anchors). "which-dial-moves" is an ANCHOR-level partition, orthogonal to
# the per-mood split that ParamSpec already owns (each dial stays per-mood).
_LADDER_VARYING = {
    "M_shared": (),                 # all dials shared across anchors
    "M_sharpness": ("v",),          # only sharpness v varies per anchor
    "M_caution": ("z",),            # only itchiness z varies per anchor
    "M_timing": ("u",),             # only timing u varies per anchor
    "M_full": ("v", "z", "u"),      # all three vary per anchor
}
_LADDER_RUNGS = tuple(_LADDER_VARYING)


def _ladder_k_params(rung, param_spec, n_anchors):
    """GLM degrees of freedom for a ladder rung (contract §A.10-3).

    Plain English: count the free coefficients of the combined GLM for this rung.
    A dial that is SHARED across anchors contributes ONE per-mood block (counted
    once); a dial that VARIES across anchors contributes one per-mood block PER
    anchor. This is the genuine GLM dof — **NOT** pyddm's ``4 + len(keys)*(n-1)``
    formula (which is rejected by the contract).

    With ``n_mood = len(param_spec.moods)`` per-mood slots per dial::

        k = (n_shared_dials * n_mood) + (n_varying_dials * n_mood * n_anchors)

    For the default 2 moods / 3 dials / 2 anchors this gives M_shared=6,
    M_sharpness=M_caution=M_timing=8, M_full=12.
    """
    n_mood = len(param_spec.moods)
    varying = _LADDER_VARYING[rung]
    n_vary = len(varying)
    n_dials = len(param_spec.dials)
    n_shared = n_dials - n_vary
    return int(n_shared * n_mood + n_vary * n_mood * n_anchors)


def _ladder_layout(rung, param_spec, n_anchors):
    """Index map for a rung's COMBINED parameter vector.

    The combined theta is laid out as one SHARED block followed by one block PER
    anchor::

        [ shared_dials (per-mood) | anchor_0 varying_dials (per-mood) | anchor_1 ... ]

    Returns ``(total_len, shared_slices, anchor_slices)`` where

    * ``shared_slices``  : ``{dial: slice}`` into the combined theta for each
      SHARED dial (one block, per-mood, used by every anchor);
    * ``anchor_slices``  : ``list[{dial: slice}]`` — for each anchor, the slices
      of its VARYING dials' per-mood blocks.

    These slices let :func:`_ladder_effective_theta` reassemble, for any anchor,
    the standard 6-vector ``theta`` that the existing :class:`ParamSpec` /
    :func:`hazard_nll` consume unchanged (so no new likelihood code is needed).
    """
    n_mood = len(param_spec.moods)
    varying = set(_LADDER_VARYING[rung])
    shared_dials = [d for d in param_spec.dials if d not in varying]
    vary_dials = [d for d in param_spec.dials if d in varying]

    off = 0
    shared_slices = {}
    for d in shared_dials:
        shared_slices[d] = slice(off, off + n_mood)
        off += n_mood

    anchor_slices = []
    for _a in range(n_anchors):
        slc = {}
        for d in vary_dials:
            slc[d] = slice(off, off + n_mood)
            off += n_mood
        anchor_slices.append(slc)

    return off, shared_slices, anchor_slices


def _ladder_effective_theta(combined, param_spec, shared_slices, anchor_slice):
    """Reassemble one anchor's standard 6-vector ``theta`` from the combined vector.

    For each dial, the per-mood block comes from this anchor's block if the dial
    VARIES, else from the single SHARED block. The output ordering matches
    ``param_spec`` exactly, so the existing :func:`hazard_nll` reads it correctly.
    """
    n_mood = len(param_spec.moods)
    theta = np.empty(param_spec.n_params(), float)
    for d in param_spec.dials:
        off = param_spec._offset(d)
        if d in anchor_slice:                       # dial varies -> anchor block
            theta[off:off + n_mood] = combined[anchor_slice[d]]
        else:                                       # dial shared -> shared block
            theta[off:off + n_mood] = combined[shared_slices[d]]
    return theta


def _fit_ladder_rung(rung, designs, param_spec, n_restarts=4, seed=0):
    """Fit one ladder rung's COMBINED model over the pooled anchor trials.

    Builds the combined parameter vector (one shared block + one varying block per
    anchor) and minimises the SUMMED censored NLL across anchors (each anchor's
    contribution is the standard :func:`hazard_nll` on its effective theta). The
    summed data log-likelihood and the GLM k_params are therefore exactly
    consistent for AIC/BIC.

    Returns ``(ll, combined_theta)`` where ``ll`` is the pooled data
    log-likelihood (``-summed_nll``).
    """
    from scipy.optimize import minimize

    n_anchors = len(designs)
    total_len, shared_slices, anchor_slices = _ladder_layout(
        rung, param_spec, n_anchors)

    def objective(combined):
        nll = 0.0
        for a_idx, design in enumerate(designs):
            theta_a = _ladder_effective_theta(
                combined, param_spec, shared_slices, anchor_slices[a_idx])
            nll += hazard_nll(theta_a, design, param_spec, l2=0.0)
        return nll

    rng = np.random.default_rng(seed)
    inits = [np.zeros(total_len)]
    for _ in range(int(n_restarts)):
        inits.append(rng.normal(loc=0.0, scale=1.0, size=total_len))

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
        combined = np.asarray(inits[0], float)
        ll = -objective(combined)
    else:
        ll = -best[0]
        combined = best[1]
    return float(ll), combined


def _ladder_rung_cvll(rung, designs, param_spec, k=5, seed=0, n_restarts=2):
    """Held-out k-fold CV log-likelihood for a ladder rung.

    Plain English: the only honest way to ask "does letting THIS dial vary across
    anchors genuinely help?" is to score on data the model was NOT fit to. We split
    EACH anchor's trials into the SAME ``k`` folds (one shared fold split per
    anchor, RNG-seeded), refit the rung's combined model on the train folds, and
    evaluate the summed held-out data log-likelihood on the test folds. Summed
    across folds and anchors, higher = better. A rung that spends a per-anchor
    block on a dial that does not truly differ buys no held-out LL (and is
    penalised by AIC/BIC in-sample) — so the minimal correct rung wins.

    The fold split is computed ONCE per anchor from ``np.random.default_rng(seed)``
    and reused for every rung (fairness): all rungs see identical train/test
    partitions, so CV-LL differences reflect the model, not fold noise.
    """
    n_anchors = len(designs)

    # ── per-anchor fold indices, fixed once (shared across rungs by the caller
    # passing the same seed). Each anchor shuffled by its own offset stream so a
    # single tiny anchor cannot collapse a fold. ──
    fold_idx = []
    usable = True
    for a_idx, design in enumerate(designs):
        n = len(design)
        if n < k or n == 0:
            usable = False
            break
        idx = np.arange(n)
        np.random.default_rng(seed + a_idx).shuffle(idx)
        fold_idx.append(np.array_split(idx, k))
    if not usable:
        return -np.inf

    total_len, shared_slices, anchor_slices = _ladder_layout(
        rung, param_spec, n_anchors)

    from scipy.optimize import minimize

    cv_ll = 0.0
    for f in range(k):
        train_designs, test_designs = [], []
        ok = True
        for a_idx, design in enumerate(designs):
            test_i = fold_idx[a_idx][f]
            train_i = np.concatenate(
                [fold_idx[a_idx][j] for j in range(k) if j != f])
            if len(test_i) == 0 or len(train_i) == 0:
                ok = False
                break
            train_designs.append(design.subset(train_i))
            test_designs.append(design.subset(test_i))
        if not ok:
            return -np.inf

        # fit the rung's combined model on the TRAIN folds
        def objective(combined, _trains=train_designs):
            nll = 0.0
            for a_idx, d in enumerate(_trains):
                theta_a = _ladder_effective_theta(
                    combined, param_spec, shared_slices, anchor_slices[a_idx])
                nll += hazard_nll(theta_a, d, param_spec, l2=0.0)
            return nll

        rng = np.random.default_rng(seed)
        inits = [np.zeros(total_len)]
        for _ in range(int(n_restarts)):
            inits.append(rng.normal(loc=0.0, scale=1.0, size=total_len))
        best = None
        for x0 in inits:
            try:
                res = minimize(objective, x0, method="L-BFGS-B")
            except Exception:
                continue
            if not np.all(np.isfinite(res.x)) or not np.isfinite(res.fun):
                continue
            if best is None or float(res.fun) < best[0]:
                best = (float(res.fun), np.asarray(res.x, float))
        combined = best[1] if best is not None else np.asarray(inits[0], float)

        # held-out data log-likelihood on the TEST folds
        for a_idx, d in enumerate(test_designs):
            theta_a = _ladder_effective_theta(
                combined, param_spec, shared_slices, anchor_slices[a_idx])
            cv_ll += -hazard_nll(theta_a, d, param_spec, l2=0.0)

    return float(cv_ll)


def learning_ladder(anchor_designs, param_spec, dt=0.05, k=5, seed=0,
                    n_restarts=4, return_ll=False, compute_cvll=True):
    """Which dial moves across anchors? Model-comparison ladder (Task 2.2).

    Plain English: the science question is *which* behavioural knob learning turns
    — does the mouse get sharper (``v``), less itchy (``z``), or better-timed
    (``u``) across anchors? We answer it by a nested model comparison. Each rung
    lets a SUBSET of the three dials VARY across anchors while the rest are SHARED
    across anchors:

      * ``M_shared``    — all dials shared (one combined fit on the pooled trials);
      * ``M_sharpness`` — only ``v`` varies per anchor (``z``, ``u`` shared);
      * ``M_caution``   — only ``z`` varies per anchor;
      * ``M_timing``    — only ``u`` varies per anchor;
      * ``M_full``      — all three vary per anchor.

    Each rung is fit as a single COMBINED GLM over all anchors' pooled trials: a
    shared per-mood block for the shared dials plus one per-mood block per anchor
    for the varying dials. The pooled data log-likelihood is the SUM across anchors
    of the standard censored :func:`hazard_nll` (so no new likelihood is needed;
    the existing :class:`ParamSpec` layout is reused per anchor).

    Scoring (contract §A.10-3 — **GLM dof, NOT** pyddm's ``4 + len(keys)*(n-1)``):

      * ``AIC = 2*k_params - 2*LL``
      * ``BIC = k_params*ln(N) - 2*LL``        (``N`` = total trials across anchors)

    where ``k_params`` is the rung's GLM degrees of freedom
    (:func:`_ladder_k_params`): shared dials counted once (per-mood), varying dials
    counted per anchor (per-mood). The held-out **CV-LL** is a ``k``-fold
    cross-validated data log-likelihood (per-anchor folds via :meth:`Design.subset`,
    refit per fold; :func:`_ladder_rung_cvll`). ``winner`` = ``argmin AIC``.

    dof accounting (the load-bearing bookkeeping, default 2 moods / 3 dials /
    2 anchors)::

        M_shared    : 3 dials shared * 2 moods                       = 6
        M_sharpness : (z,u shared: 2*2) + (v per anchor: 1*2*2)      = 8
        M_caution   : (v,u shared: 2*2) + (z per anchor: 1*2*2)      = 8
        M_timing    : (v,z shared: 2*2) + (u per anchor: 1*2*2)      = 8
        M_full      : 3 dials per anchor * 2 moods * 2 anchors       = 12

    Parameters
    ----------
    anchor_designs : dict[str, Design]
        Per-anchor ragged Designs (from :func:`build_anchor_designs`). At least two
        anchors are required for a meaningful comparison; order is irrelevant (the
        ladder is symmetric in the anchors).
    param_spec : ParamSpec
        Parameter layout (``theta`` <-> dial/mood mapping); each dial stays
        per-mood. The ladder partitions ACROSS anchors orthogonally to the moods.
    dt : float
        Generative time grid (accepted for interface symmetry; the Designs already
        encode their own ``dt``).
    k : int
        Number of CV folds (per anchor) for the held-out CV-LL.
    seed : int
        RNG seed for the per-fold refits and the fold split (reproducible; the same
        split is reused across rungs for a fair comparison).
    n_restarts : int
        Random restarts for each in-sample combined fit.
    return_ll : bool
        If True, also return the per-rung pooled data log-likelihood under an
        ``"ll"`` key (lets callers/tests reconstruct AIC/BIC).
    compute_cvll : bool
        If True (default) compute the held-out k-fold CV-LL per rung. This is the
        EXPENSIVE part (k refits per rung). The ``winner`` is ``argmin AIC`` and
        does NOT depend on CV-LL, so callers that only need the winner (e.g.
        :func:`recover_confusion`) pass ``compute_cvll=False`` for a large speedup;
        in that case ``out["cvll"]`` maps every rung to ``np.nan``.

    Returns
    -------
    dict
        ``{"winner": str, "aic": {rung: float}, "bic": {rung: float},
        "cvll": {rung: float}}`` (plus ``"ll"`` when ``return_ll``). When
        ``compute_cvll=False`` the ``cvll`` values are ``np.nan`` placeholders.
    """
    designs = list(anchor_designs.values())
    n_anchors = len(designs)
    N = int(sum(len(d) for d in designs))
    log_N = np.log(N) if N > 0 else 0.0

    aic, bic, cvll, ll_by_rung = {}, {}, {}, {}
    for rung in _LADDER_RUNGS:
        ll, _theta = _fit_ladder_rung(
            rung, designs, param_spec, n_restarts=n_restarts, seed=seed)
        k_params = _ladder_k_params(rung, param_spec, n_anchors)
        aic[rung] = 2.0 * k_params - 2.0 * ll
        bic[rung] = k_params * log_N - 2.0 * ll
        # CV-LL is the costly bit (k refits/rung) and is NOT used by argmin-AIC;
        # skip it when the caller only needs the winner (the confusion measurement).
        cvll[rung] = (_ladder_rung_cvll(rung, designs, param_spec, k=k, seed=seed)
                      if compute_cvll else float("nan"))
        ll_by_rung[rung] = ll

    winner = min(aic, key=aic.get)
    out = {"winner": winner, "aic": aic, "bic": bic, "cvll": cvll}
    if return_ll:
        out["ll"] = ll_by_rung
    return out


# ── Engine-A state ladder (Task 2.3) — which dial loads on MOOD (within anchor) ─
# The five rungs and the dials each lets vary by MOOD (Impulsive vs StimSens)
# WITHIN one anchor. Unlike `learning_ladder` (which partitions ACROSS anchors),
# this ladder partitions BY MOOD — exactly what `ParamSpec.state_terms` already
# owns. So each rung IS a `ParamSpec` whose `state_terms` = the rung's varying
# dials; a non-varying dial is shared (a single value across moods). It tests the
# thesis "states load on caution/timing, not sharpness."
_STATE_LADDER_TERMS = {
    "M_none": (),                  # all dials shared across moods
    "M_v": ("v",),                 # only sharpness v varies by mood
    "M_z": ("z",),                 # only itchiness/caution z varies by mood
    "M_u": ("u",),                 # only timing u varies by mood
    "M_all": ("v", "z", "u"),      # all three vary by mood
}
_STATE_LADDER_RUNGS = tuple(_STATE_LADDER_TERMS)


def _state_ladder_spec(rung, param_spec):
    """The rung's :class:`ParamSpec`: same dials/moods/fixed settings as
    ``param_spec`` but with ``state_terms`` set to this rung's per-mood dials.

    The other (fixed, non-fitted) fields — moods, dials, rectification, leak_tau,
    urgency_sigma — are carried over from ``param_spec`` unchanged, so the only
    thing the rung changes is WHICH dials carry a per-mood term.
    """
    return ParamSpec(
        moods=param_spec.moods,
        dials=param_spec.dials,
        state_terms=_STATE_LADDER_TERMS[rung],
        rectification=param_spec.rectification,
        leak_tau=param_spec.leak_tau,
        urgency_sigma=param_spec.urgency_sigma,
    )


def _state_ladder_k_params(rung, param_spec):
    """GLM degrees of freedom for a state-ladder rung == ``ParamSpec.n_params()``.

    Plain English: count the free coefficients of this rung's GLM. A dial that is
    SHARED across moods contributes ONE slot; a dial that VARIES by mood
    contributes one slot PER mood. That is exactly what ``ParamSpec.n_params()``
    computes for the rung's ``state_terms``, so there is no separate dof formula
    here (and certainly NOT pyddm's ``4 + len(keys)*(n-1)``).

    For the default 2 moods / 3 dials this gives M_none=3, M_v=M_z=M_u=4, M_all=6.
    """
    return _state_ladder_spec(rung, param_spec).n_params()


def _state_ladder_rung_cvll(rung, design, param_spec, folds, k, seed):
    """Held-out k-fold CV log-likelihood for a state-ladder rung on ONE anchor.

    Plain English: the only honest way to ask "does letting THIS dial vary by mood
    genuinely help?" is to score on data the model was NOT fit to. The ``folds``
    (a list of k index arrays) are computed ONCE by the caller and reused for
    every rung (fairness): all rungs see identical train/test partitions, so CV-LL
    differences reflect the model, not fold noise. For each fold we ``fit_anchor``
    on the train subset (``Design.subset(train_idx)``) with the rung's ParamSpec
    and a SAME fixed fit seed for every rung, then evaluate the held-out data
    log-likelihood ``-hazard_nll(fit.theta, Design.subset(test_idx), spec, l2=0)``
    and sum across folds. A rung that spends a per-mood block on a dial that does
    not truly differ buys no held-out LL (and AIC penalises it in-sample), so the
    minimal correct rung wins.
    """
    spec = _state_ladder_spec(rung, param_spec)
    cv_ll = 0.0
    for f in range(k):
        test_idx = folds[f]
        if len(test_idx) == 0:
            continue
        train_idx = np.concatenate([folds[j] for j in range(k) if j != f])
        if len(train_idx) == 0:
            return -np.inf
        train_design = design.subset(train_idx)
        test_design = design.subset(test_idx)
        # SAME fixed fit seed for every rung & fold (apples-to-apples fairness)
        fit = fit_anchor(train_design, spec, seed_theta=None, l2=0.0,
                         n_restarts=2, seed=seed)
        cv_ll += -hazard_nll(fit.theta, test_design, spec, l2=0.0)
    return float(cv_ll)


def state_ladder(anchor_design, param_spec, k=5, seed=0, n_restarts=4,
                 return_ll=False):
    """Which dial loads on MOOD within one anchor? Model-comparison ladder (Task 2.3).

    Plain English: within a single anchor, does the difference between the mouse's
    behavioural moods (Impulsive vs StimSens) live in *sharpness* (``v``),
    *itchiness/caution* (``z``), or *timing* (``u``)? This is the project thesis
    test — "states load on caution/timing, NOT sharpness." We answer it by a nested
    model comparison on ONE anchor's Design. Each rung lets a SUBSET of the three
    dials carry a per-MOOD term while the rest are SHARED across moods — exactly
    what :class:`ParamSpec`'s ``state_terms`` already owns:

      * ``M_none`` — all dials shared across moods (``state_terms=()``);
      * ``M_v``    — only ``v`` varies by mood (``state_terms=("v",)``);
      * ``M_z``    — only ``z`` varies by mood;
      * ``M_u``    — only ``u`` varies by mood;
      * ``M_all``  — all three vary by mood (``state_terms=("v","z","u")``).

    Each rung is a single :func:`fit_anchor` on the anchor with that rung's
    ParamSpec; the data log-likelihood is the standard censored :func:`hazard_nll`
    (so no new likelihood is needed — the per-mood machinery IS the ladder).

    Scoring (GLM dof, NOT pyddm's ``4 + len(keys)*(n-1)``):

      * ``AIC = 2*k_params - 2*LL`` with ``k_params = ParamSpec.n_params()`` for the
        rung (:func:`_state_ladder_k_params`);
      * held-out **CV-LL** is a ``k``-fold cross-validated data log-likelihood via
        :meth:`Design.subset`, refit per fold (:func:`_state_ladder_rung_cvll`).

    ``winner`` = ``argmin AIC``.

    Fairness (mirrors :func:`learning_ladder`): the fold split is computed ONCE
    from a single ``np.random.default_rng(seed)`` shuffle and reused for EVERY
    rung, and every per-fold :func:`fit_anchor` is given the SAME fixed seed — so
    CV-LL differences reflect the model, not fold/optimizer noise. This avoided a
    fairness bug in the learning ladder.

    dof accounting (default 2 moods / 3 dials)::

        M_none : 3 dials shared                       = 3
        M_v    : (z,u shared: 2) + (v per mood: 1*2)  = 4
        M_z    : (v,u shared: 2) + (z per mood: 1*2)  = 4
        M_u    : (v,z shared: 2) + (u per mood: 1*2)  = 4
        M_all  : 3 dials per mood * 2 moods           = 6

    Parameters
    ----------
    anchor_design : Design
        ONE anchor's ragged Design (with the outcomes to fit). ``mood_code`` must
        index ``param_spec.moods`` (as :func:`build_design` produces).
    param_spec : ParamSpec
        Reference layout — its ``moods``/``dials``/fixed settings are reused for
        every rung; only ``state_terms`` is varied per rung. (Its own
        ``state_terms`` is irrelevant; the ladder overrides it.)
    k : int
        Number of CV folds for the held-out CV-LL.
    seed : int
        RNG seed for the fold split AND the per-fold refits (reproducible; the same
        split + fit seed are reused across rungs for a fair comparison).
    n_restarts : int
        Random restarts for each in-sample rung fit.
    return_ll : bool
        If True, also return the per-rung pooled data log-likelihood under an
        ``"ll"`` key (lets callers/tests reconstruct AIC).

    Returns
    -------
    dict
        ``{"winner": str, "aic": {rung: float}, "cvll": {rung: float}}`` (plus
        ``"ll"`` when ``return_ll``).
    """
    n = len(anchor_design)

    # ── ONE shuffled fold split, reused for EVERY rung (fairness) ──
    if n >= k and n > 0:
        idx = np.arange(n)
        np.random.default_rng(seed).shuffle(idx)
        folds = np.array_split(idx, k)
        can_cv = True
    else:
        folds = None
        can_cv = False

    aic, cvll, ll_by_rung = {}, {}, {}
    for rung in _STATE_LADDER_RUNGS:
        spec = _state_ladder_spec(rung, param_spec)
        # in-sample fit on the whole anchor with this rung's ParamSpec
        fit = fit_anchor(anchor_design, spec, seed_theta=None, l2=0.0,
                         n_restarts=n_restarts, seed=seed)
        k_params = _state_ladder_k_params(rung, param_spec)
        aic[rung] = 2.0 * k_params - 2.0 * fit.ll
        ll_by_rung[rung] = fit.ll
        if can_cv:
            cvll[rung] = _state_ladder_rung_cvll(
                rung, anchor_design, param_spec, folds, k, seed)
        else:
            cvll[rung] = -np.inf

    winner = min(aic, key=aic.get)
    out = {"winner": winner, "aic": aic, "cvll": cvll}
    if return_ll:
        out["ll"] = ll_by_rung
    return out


# ── Engine-A backward-seeding guardrails (Task 2.4) — conditioning + L2 sens ──
# Two checks that the regularization is INFORMING, not MANUFACTURING, the learning
# trajectory:
#   1. hessian_conditioning — is the fitted optimum well-curved (identifiable), or
#      is its curvature degenerate (a flat ridge / a duplicated direction)? A
#      degenerate Hessian means the dials are not jointly identifiable there, so an
#      L2 prior could move them freely without changing the fit quality.
#   2. l2_weight_sensitivity — does the SCIENTIFIC CONCLUSION (which dial moves
#      across anchors; the recovered v span) survive a sweep of ridge strengths? If
#      the conclusion only holds at one L2 weight, it is a regularization artifact.
def hessian_conditioning(fit):
    """Conditioning diagnostic for a single :class:`FitResult` (contract §A.7).

    Plain English: the finite-difference Hessian ``fit.hessian`` is the curvature of
    the negative log-likelihood at the fitted optimum. If that curvature is sharp in
    every parameter direction the dials are well-identified there; if it is (nearly)
    flat along some direction the fit sits on a ridge and an L2 prior can slide the
    parameters along it for free — so a "learning trajectory" recovered there could
    be a regularization artifact. We flag two failure modes:

    * **rank deficiency** — ``np.linalg.matrix_rank(fit.hessian) < fit.n_params``
      (an exactly-flat direction: a zero eigenvalue / a duplicated parameter), and
    * **ill conditioning** — ``cond_number > 1e8`` (a near-flat direction: the
      ratio of the largest to smallest singular value is enormous).

    ``cond_number`` is taken from ``fit.hessian_cond`` when finite, else recomputed
    via ``np.linalg.cond(fit.hessian)`` (``np.inf`` if singular / it raises).

    Parameters
    ----------
    fit : FitResult
        A fitted anchor (uses ``fit.hessian``, ``fit.hessian_cond``, ``fit.n_params``).

    Returns
    -------
    dict
        ``{"cond_number": float, "rank": int, "deficient": bool}`` where
        ``deficient == (cond_number > 1e8) or (rank < fit.n_params)``.
    """
    H = np.asarray(fit.hessian, float)
    n_params = int(fit.n_params)

    # cond_number: prefer the value stored on the fit; recompute if it is missing
    # / non-finite (a singular Hessian -> np.inf, which IS the right flag value).
    cond_number = getattr(fit, "hessian_cond", None)
    if cond_number is None or not np.isfinite(cond_number):
        try:
            cond_number = float(np.linalg.cond(H))
        except Exception:
            cond_number = np.inf
    cond_number = float(cond_number)
    if not np.isfinite(cond_number):
        cond_number = np.inf

    try:
        rank = int(np.linalg.matrix_rank(H))
    except Exception:
        rank = 0

    deficient = bool((cond_number > 1e8) or (rank < n_params))
    return {"cond_number": cond_number, "rank": rank, "deficient": deficient}


def l2_weight_sensitivity(anchor_designs, anchors_chrono, param_spec,
                          weights=(0.0, 0.01, 0.1, 1.0, 10.0), seed=0):
    """Is the learning conclusion STABLE across ridge strengths? (Task 2.4 guardrail).

    Plain English: the backward sweep L2-seeds each earlier (less-expert) anchor
    toward its more-expert neighbour. A natural worry is that this prior — not the
    data — is what produces the recovered learning trajectory. We stress-test that
    by re-running the whole pipeline across a grid of L2 weights and reporting, per
    weight, the SCIENTIFIC CONCLUSION:

    * ``ladder_winner`` — which dial :func:`learning_ladder` says moves across
      anchors. (The ladder fits each rung freely and is NOT seeded by the sweep's
      L2, so its winner does not depend on the sweep ``l2``; it is recomputed once
      and reported on every row as the stable reference conclusion.)
    * ``v_span`` — the recovered sharpness span ``v_expert - v_old`` from
      :func:`backward_sweep` **at that l2** (mean over moods; expert = last of
      ``anchors_chrono``, old = first). This is the dial delta the L2 ridge could
      plausibly shrink, so it is the load-bearing per-weight quantity.

    The point of the table is the guardrail: if ``ladder_winner`` and the sign of
    ``v_span`` are STABLE across weights >= 0.01, the recovered trajectory is a
    property of the data, not of the regularization. If they are not, that is a real
    signal (report it; do NOT loosen).

    Parameters
    ----------
    anchor_designs : dict[str, Design]
        Per-anchor ragged Designs (from :func:`build_anchor_designs`).
    anchors_chrono : list[str]
        Session ids in CHRONOLOGICAL order (oldest -> newest); the last is the
        most-expert anchor. Used to identify the OLD and EXPERT anchors for
        ``v_span`` and to drive the backward sweep.
    param_spec : ParamSpec
        Parameter layout, passed to :func:`backward_sweep` / :func:`learning_ladder`.
    weights : tuple[float]
        L2 ridge strengths to sweep (default ``(0, 0.01, 0.1, 1, 10)``).
    seed : int
        RNG seed for the per-weight sweep and the (single) ladder (reproducible).

    Returns
    -------
    pandas.DataFrame
        One row per weight with columns ``l2``, ``ladder_winner``, ``v_span``,
        ``v_old``, ``v_expert`` (the last two are the per-anchor recovered v that
        ``v_span`` is built from, kept for transparency).
    """
    # The ladder winner is independent of the sweep's l2 (the ladder refits each
    # rung free), so compute it ONCE and report it on every row as the reference.
    ladder = learning_ladder(anchor_designs, param_spec, seed=seed)
    ladder_winner = ladder["winner"]

    # Identify the OLD (oldest) and EXPERT (newest) anchors that PRESENT in the
    # design dict (skip QC-omitted ids exactly as backward_sweep does).
    present = [a for a in anchors_chrono if a in anchor_designs]
    old_anchor = present[0] if present else None
    expert_anchor = present[-1] if present else None

    def _rec_v(fit):
        """Recovered sharpness = mean of the two moods' v (None if no moods)."""
        if fit is None:
            return np.nan
        vals = [d["sharpness"] for d in fit.dials.values()]
        return float(np.mean(vals)) if vals else np.nan

    rows = []
    for l2 in weights:
        results = backward_sweep(anchor_designs, anchors_chrono, param_spec,
                                 l2=float(l2), seed=seed)
        v_old = _rec_v(results.get(old_anchor)) if old_anchor is not None else np.nan
        v_expert = (_rec_v(results.get(expert_anchor))
                    if expert_anchor is not None else np.nan)
        v_span = v_expert - v_old
        rows.append({
            "l2": float(l2),
            "ladder_winner": ladder_winner,
            "v_span": float(v_span),
            "v_old": float(v_old),
            "v_expert": float(v_expert),
        })

    return pd.DataFrame(rows, columns=["l2", "ladder_winner", "v_span",
                                       "v_old", "v_expert"])


# ── Engine-A recovery (Task 3.3) — which-dial-varies confusion matrix (§A.9) ──
# The DECISIVE recovery test. Maps each ladder rung that names a SINGLE varying
# dial to a confusion-matrix column. M_shared (no dial varies) and M_full (all
# three vary) name NO single dial, so they cannot be charged to any specific
# column: they are "no single dial identified" misses that LOWER the true dial's
# diagonal without inflating a particular off-diagonal (a confusion is a wrong
# *specific* dial, not an under/over-fit). They are still recorded per scenario in
# the returned ``no_single`` diagnostic so the honest failure mode is visible.
_LADDER_WINNER_TO_DIAL = {
    "M_sharpness": "sharpness",   # only v varies  -> column 0
    "M_caution": "caution",       # only z varies  -> column 1
    "M_timing": "timing",         # only u varies  -> column 2
    # "M_shared" / "M_full" -> no single dial (handled explicitly below)
}

# The three confusion scenarios: which dial TRULY varies across the two anchors,
# the dial's internal ParamSpec key, and the matrix-row label.
_CONFUSION_SCENARIOS = (
    ("v", "sharpness"),
    ("z", "caution"),
    ("u", "timing"),
)
_CONFUSION_LABELS = ("sharpness", "caution", "timing")
_CONFUSION_COL = {"sharpness": 0, "caution": 1, "timing": 2}

# Per-dial ACROSS-ANCHOR delta applied to BOTH moods of the varying dial in the
# second anchor (the first anchor uses ``base_theta`` unchanged). These are the
# ground-truth gaps that make each scenario a FAIR, adequately-powered
# discriminability test (contract §A.9): large enough that the varying dial's
# signal is decisive against the v<->z and u<->z confounds, while keeping every
# dial in its identifiable range. ``v`` multiplies the post-change accumulator,
# ``z`` is the cloglog intercept, ``u`` scales the timing bump; their natural
# scales differ, so the deltas differ. NOT fitted; the caller may override.
_CONFUSION_DELTA = {"v": 1.0, "z": 1.5, "u": 2.5}


def recover_confusion(design_template, base_theta, param_spec, n_rep=50, seed=0,
                      deltas=None, k=3, n_restarts=2):
    """Which-dial-varies confusion matrix — the DECISIVE recovery test (contract §A.9).

    Plain English: the make-or-break question for the learning ladder is "when the
    mouse REALLY changed ONE behavioural knob across two anchors, does the ladder
    name the RIGHT knob — or do the sharpness<->caution / timing<->caution
    trade-offs fool it?" We answer it as a genuine discriminability measurement.
    For each of the three dials in turn we build a two-anchor dataset in which ONLY
    that dial truly differs across anchors (the other two are byte-identical, and
    BOTH anchors share the SAME evidence realisation — ``design_template`` — so the
    only thing that can drive the ladder is the one dial we moved), simulate licks,
    run :func:`learning_ladder`, and record which dial its winner names. Over
    ``n_rep`` reps this builds a 3x3 confusion matrix whose ``matrix[i, j]`` is the
    fraction of reps in which true-varying dial ``i`` was identified as dial ``j``.
    The diagonal is correct identification.

    The mapping from the ladder winner to a matrix column (contract §A.9):

    * ``M_sharpness`` -> ``sharpness`` (the ladder says ``v`` varies),
    * ``M_caution``   -> ``caution``   (``z`` varies),
    * ``M_timing``    -> ``timing``    (``u`` varies),
    * ``M_shared`` / ``M_full`` -> **no single dial** — the ladder named either no
      varying dial or all three, so it cannot be charged to a specific WRONG dial.
      These reps LOWER the true dial's diagonal (a miss) but DO NOT inflate any
      off-diagonal (a confusion means a wrong *specific* dial). They are counted in
      the per-scenario ``no_single`` diagnostic so the honest failure mode is
      visible, and they make a row sum to < 1.0 when present.

    Each rep uses the SAME ``design_template`` (shared evidence/A/phi across both
    anchors and across reps) and varies only the lick-simulation seeds per anchor,
    so the matrix reflects the ladder's discriminability, not evidence-realisation
    noise. ``base_theta`` provides the SHARED dial values; the varying dial's
    second-anchor value is ``base_theta[dial] + deltas[dial]`` (applied to both
    moods so the dial moves coherently).

    Parameters
    ----------
    design_template : Design
        The shared two-mood Design (A/phi/mood fixed; outcomes RESIMULATED per
        anchor per rep). Both anchors in every scenario reuse this exact Design, so
        ONLY the moved dial differs across the two anchors (contract §A.9 "shared
        design seed"). Must carry both moods and enough trials per anchor (>= ~800
        recommended) for adequate power.
    base_theta : np.ndarray
        The SHARED ground-truth parameter vector (length ``param_spec.n_params()``)
        — anchor A's theta, and the base anchor B perturbs in exactly one dial.
    param_spec : ParamSpec
        Parameter layout (``theta`` <-> dial/mood mapping).
    n_rep : int
        Reps per scenario (default 50; the test uses a reduced count for
        tractability — the whole matrix is 3 scenarios x n_rep full ladders).
    seed : int
        Master RNG seed. Each (scenario, rep) gets a deterministic pair of child
        seeds for the two anchors' lick simulations, and each ladder is run at a
        fixed seed, so the whole matrix is reproducible.
    deltas : Mapping[str, float] | None
        Per-dial across-anchor delta for the varying dial (defaults to
        :data:`_CONFUSION_DELTA`). Applied to BOTH moods of the dial in anchor B.
    k : int
        Accepted for interface symmetry only. ``recover_confusion`` runs each
        ladder with ``compute_cvll=False`` (the winner is ``argmin AIC``, which
        never touches CV-LL), so NO cross-validation is performed and ``k`` is
        forwarded but unused — this AIC-only fast path is the main tractability
        lever (no k-fold refits).
    n_restarts : int
        Random restarts for each ladder rung's single in-sample combined fit
        (default 2 — kept low for tractability; the AIC margins here are large).

    Returns
    -------
    dict
        ``{"matrix": np.ndarray (3, 3), "labels": ("sharpness", "caution",
        "timing"), "no_single": {label: int}, "winners": {label: list[str]},
        "n_rep": int}``. ``matrix[i, j]`` = fraction of reps where true dial ``i``
        was identified as dial ``j``; the diagonal is correct identification.
        ``no_single`` counts the M_shared/M_full ("no single dial") reps per true
        dial; ``winners`` records the raw ladder winner string per rep per scenario
        for full transparency.
    """
    base_theta = np.asarray(base_theta, float)
    n_params = param_spec.n_params()
    assert len(base_theta) == n_params, (
        f"len(base_theta)={len(base_theta)} != n_params={n_params}")

    delta = dict(_CONFUSION_DELTA)
    if deltas is not None:
        delta.update(deltas)

    n_mood = len(param_spec.moods)
    matrix = np.zeros((3, 3), float)
    no_single = {lab: 0 for lab in _CONFUSION_LABELS}
    winners = {lab: [] for lab in _CONFUSION_LABELS}

    master = np.random.default_rng(seed)
    # one independent child-seed pair (anchor A, anchor B) per (scenario, rep)
    n_sc = len(_CONFUSION_SCENARIOS)
    sim_seeds = master.integers(0, 2**31 - 1, size=(n_sc, int(n_rep), 2))
    ladder_seeds = master.integers(0, 2**31 - 1, size=(n_sc, int(n_rep)))

    for si, (dial, row_label) in enumerate(_CONFUSION_SCENARIOS):
        row_i = _CONFUSION_COL[row_label]
        off = param_spec._offset(dial)

        # ── anchor thetas: A == base; B perturbs ONLY this dial (both moods) ──
        theta_a = base_theta.copy()
        theta_b = base_theta.copy()
        for mi in range(n_mood):
            theta_b[off + mi] = base_theta[off + mi] + float(delta[dial])

        for rep in range(int(n_rep)):
            sa = int(sim_seeds[si, rep, 0])
            sb = int(sim_seeds[si, rep, 1])

            eb_a, lk_a, cs_a = simulate_licks(design_template, theta_a,
                                              param_spec, seed=sa)
            eb_b, lk_b, cs_b = simulate_licks(design_template, theta_b,
                                              param_spec, seed=sb)
            design_a = design_with_outcomes(design_template, eb_a, lk_a, cs_a)
            design_b = design_with_outcomes(design_template, eb_b, lk_b, cs_b)

            # AIC-only fast path: winner = argmin AIC, which never uses CV-LL, so
            # we skip the k-fold refits entirely (compute_cvll=False) — the single
            # biggest speedup that makes the whole 3x3 matrix tractable (contract
            # §A.9). The ``k`` argument is forwarded only for interface symmetry;
            # no CV is run.
            out = learning_ladder({"A": design_a, "B": design_b}, param_spec,
                                  dt=design_template.dt, k=k,
                                  seed=int(ladder_seeds[si, rep]),
                                  n_restarts=n_restarts, compute_cvll=False)
            winner = out["winner"]
            winners[row_label].append(winner)

            named = _LADDER_WINNER_TO_DIAL.get(winner)
            if named is None:                       # M_shared / M_full: no single dial
                no_single[row_label] += 1
            else:
                matrix[row_i, _CONFUSION_COL[named]] += 1.0

    matrix /= float(n_rep)
    return {
        "matrix": matrix,
        "labels": _CONFUSION_LABELS,
        "no_single": no_single,
        "winners": winners,
        "n_rep": int(n_rep),
    }


# ── Engine-A recovery (Task 3.4) — seeding INFORMS, does not ERASE (§A.9) ──────
def recover_true_difference(design_naive, design_expert, param_spec, true_delta,
                            l2=1.0, seed=0):
    """Does the L2-seeded backward fit RECOVER a genuine across-stage difference?

    Plain English: the backward sweep fits the expert anchor FREE and then seeds
    the earlier (naive) anchor toward that expert fit with an L2 ridge. The worry
    (the mirror of Task 2.4's guardrail) is that this prior could CRUSH a
    difference that is genuinely there — making naive look like expert by fiat. This
    function proves the opposite on ground truth: given two anchors that differ by a
    KNOWN ``true_delta`` on an IDENTIFIABLE dial, it runs that exact expert-first
    L2-seeded fit and reads back the recovered dial difference. If the seeding
    *informed* without *erasing*, the recovered difference matches the true one and
    is NOT shrunk toward zero.

    The fit is the same :func:`backward_sweep` the science pipeline uses, with the
    two anchors arranged chronologically as ``[naive, expert]`` (expert last == most
    expert): the EXPERT is fit FREE (``seed_theta=None, l2=0``) as the identifiable
    reference, then the NAIVE anchor is fit L2-seeded toward the expert's fitted
    theta at the passed ``l2``. Per dial the recovered difference is

        ``recovered_delta[d] = recovered_expert[d] - recovered_naive[d]``

    where each anchor's recovered dial value is the MEAN over moods of that anchor's
    fitted dial (so the comparison is at the dial level, matching ``true_delta``
    which is keyed by dial). Each dial is judged **crushed** (PER DIAL) when its
    recovered magnitude falls below HALF the true magnitude::

        shrunk[d] = (|recovered_delta[d]| < 0.5 * |true_delta[d]|)   for each d

    so ``shrunk[d] == True`` means the prior erased THAT dial's genuine difference.
    The veto is PER-DIAL on purpose (gate_criteria.md line 11: "shrunk == True ->
    *that dial* 'descriptive'"): a scalar ``any()`` would let one crushed dial (e.g.
    the weakly-identified sharpness ``v``) veto every dial in the sweep, wrongly
    downgrading well-recovered dials (``z``/``u``) whose own across-stage difference
    was NOT crushed. :func:`recovery_gate` accepts this per-dial mapping directly.

    Parameters
    ----------
    design_naive : Design
        The naive (earlier / less-expert) anchor's ragged Design with the outcomes
        to fit. Fit L2-seeded toward the expert fit.
    design_expert : Design
        The expert (later / most-expert) anchor's ragged Design. Fit FREE first as
        the identifiable reference template.
    param_spec : ParamSpec
        Parameter layout (``theta`` <-> dial/mood mapping), passed to the fits.
    true_delta : Mapping[str, float]
        The KNOWN ground-truth difference per dial key (``"v"``/``"z"``/``"u"``),
        defined as ``expert - naive``. Only the listed dials are reported in
        ``recovered_delta`` and judged for ``shrunk``.
    l2 : float
        Ridge strength seeding the naive fit toward the expert theta (the operating
        point is ``1.0``). The expert anchor is always fit with ``l2=0``.
    seed : int
        RNG seed for the per-anchor :func:`fit_anchor` random restarts (reproducible).

    Returns
    -------
    dict
        ``{"recovered_delta": {dial: float}, "shrunk": {dial: bool}}`` where
        ``recovered_delta[dial] == recovered_expert - recovered_naive`` (mean over
        moods) for each dial in ``true_delta``, and ``shrunk[dial]`` is that dial's
        crushed-flag (PER DIAL — see above).

    Notes
    -----
    * Calls :func:`backward_sweep` so the seeding is IDENTICAL to the rest of the
      module (expert-first, free expert, naive L2-seeded toward it). The two anchors
      are keyed ``"naive"``/``"expert"`` internally; chronological order is
      ``["naive", "expert"]``.
    * The public dial names on a :class:`FitResult` are ``sharpness``/``itchiness``/
      ``timing``; this maps the ``true_delta`` keys (``v``/``z``/``u``) to them via
      :data:`_DIAL_PUBLIC_NAME`.
    """
    anchor_designs = {"naive": design_naive, "expert": design_expert}
    anchors_chrono = ["naive", "expert"]        # expert last == most expert

    # Expert fit FREE (l2=0), naive L2-seeded toward the expert theta — the exact
    # backward sweep the pipeline uses (Task 2.1 / contract §A.6 ridge-toward-seed).
    results = backward_sweep(anchor_designs, anchors_chrono, param_spec,
                             l2=float(l2), seed=seed)
    fit_naive = results["naive"]
    fit_expert = results["expert"]

    def _rec_dial(fit, dial):
        """Recovered dial value = mean of the per-mood fitted values (dial level)."""
        pub = _DIAL_PUBLIC_NAME[dial]
        vals = [md[pub] for md in fit.dials.values()]
        return float(np.mean(vals)) if vals else float("nan")

    recovered_delta = {}
    shrunk = {}
    for dial, td in true_delta.items():
        rd = _rec_dial(fit_expert, dial) - _rec_dial(fit_naive, dial)
        recovered_delta[dial] = rd
        # crushed PER DIAL: recovered magnitude below HALF the true magnitude.
        # A per-dial mapping (NOT a scalar any()) so one crushed dial cannot veto
        # the others — gate_criteria.md L11 "shrunk -> THAT dial 'descriptive'".
        shrunk[dial] = bool(abs(rd) < 0.5 * abs(float(td)))

    return {"recovered_delta": recovered_delta, "shrunk": shrunk}


# ── Recovery gate (Task 3.5) — per-dial generative/descriptive trust ─────────
# The gate's public dial keys are the CONFUSION labels (sharpness / caution /
# timing). recover_point names the z dial 'itchiness'; this map bridges it.
_GATE_DIALS = ("sharpness", "caution", "timing")
_GATE_DIAL_TO_RAW = {"sharpness": "v", "caution": "z", "timing": "u"}
_GATE_DIAL_TO_POINT = {"sharpness": "sharpness", "caution": "itchiness",
                       "timing": "timing"}


def recovery_gate(point_res, confusion_res, truediff_res, cond_res, regime,
                  r_min=0.8, bias_max_frac=0.1, coverage_min=0.90, ccc_min=0.70,
                  confusion_min_diag=0.8, confusion_max_offdiag=0.2,
                  naive_relax=0.0):
    """Per-dial generative/descriptive trust verdict (contract §A.9; gate_criteria.md).

    Plain English: each behavioural dial (sharpness ``v``, caution/itchiness ``z``,
    timing ``u``) only EARNS a 'generative' (mechanistic) interpretation for this
    (dial x regime) cell if our recovery battery proves we can actually get that
    dial back from data. This gate applies the RATIFIED thresholds
    (``.superpowers/sdd/gate_criteria.md``) as a conservative AND rule: a dial is
    ``'generative'`` IFF it passes EVERY applicable diagnostic; otherwise it falls
    back to the Phase-1 ``'descriptive'`` proxy. A failing dial does NOT contaminate
    a passing one — the verdict is PER-DIAL, not binary.

    The four diagnostics and how this gate consumes their real return shapes:

    * **POINT RECOVERY** (``recover_point``) — per-dial
      ``{r, bias, sd_true, ci_coverage, [ccc]}`` keyed by the PUBLIC point names
      (``sharpness``/``itchiness``/``timing``; this gate bridges ``itchiness`` ->
      ``caution``). The dial passes the point block iff ALL hold:

      - ``r >= r_min`` (Pearson recovered-vs-true across the jittered-true grid);
      - ``|bias| <= bias_max_frac * sd_true`` — ``bias`` is the RAW
        ``mean(recovered - true)`` and ``sd_true`` is reported alongside it, so the
        tolerance is the SD-scaled ``0.1 * SD(true)`` of gate_criteria.md (we do NOT
        assume ``bias`` is pre-normalized);
      - ``ci_coverage >= coverage_min`` — a STRICT LOWER bound (under-coverage =
        false confidence = a fail). OVER-coverage (up to ~0.99) is conservative and
        ACCEPTABLE — it never fails the gate (gate_criteria.md §1);
      - if a ``ccc`` field is PRESENT (Lin's concordance — the cluster run provides
        it; absent in the local Wald smoke), ``ccc >= ccc_min``. When ABSENT the CCC
        sub-check is SKIPPED and recorded as ``None`` (not a failure).

    * **CONFUSION** (``recover_confusion``) — ``matrix`` (3x3) + ``labels``
      (``sharpness``/``caution``/``timing``). The dial passes the confusion block iff
      its DIAGONAL ``>= confusion_min_diag`` AND every SPECIFIC off-diagonal in its
      true-row ``<= confusion_max_offdiag`` (the ``no_single`` residual is NOT an
      off-diagonal and is not charged here).

    * **shrunk veto** (``recover_true_difference['shrunk']``) — if the L2 prior
      CRUSHED a genuine across-stage difference for that dial, it is ``'descriptive'``
      (sweep-level veto, where applicable). ``shrunk`` may be a per-dial mapping
      (keyed by raw ``v``/``z``/``u`` or by gate label) OR a scalar bool; a scalar
      ``True`` applies to every dial.

    * **Hessian veto** (``hessian_conditioning['deficient']``) — a rank-deficient /
      ill-conditioned Hessian is an ANCHOR-LEVEL veto: if ``cond_res['deficient']``
      is ``True``, ALL dials are ``'descriptive'`` regardless of the other dicts (the
      fit sits on a ridge; no dial is trustworthy there).

    Regime relaxation (``naive_relax``): per the statistician's ratified default of
    **0.0** the thresholds are UNIFORM across regimes (the honest naive-``v`` ->
    ``'descriptive'`` IS the finding, not something to relax away). The parameter is
    KEPT for flexibility: when ``regime == 'naive'`` it subtracts ``naive_relax`` from
    BOTH ``r_min`` and ``confusion_min_diag``. With the default ``0.0`` this is a
    no-op; it is ignored entirely outside the naive regime.

    .. warning::
       The published ``'generative'`` VERDICT is only valid at FULL config on the
       cluster (R1): ``n_rep >= 100`` for point recovery, ``n_rep >= 50`` for
       confusion, full-size designs, ``n_restarts >= 2`` (``>= 4`` for point). A
       local reduced-config run PROVES THE MACHINERY, not the verdict. R3: the gate
       reads whatever ``ci_coverage`` the upstream produced — local smoke uses
       asymptotic Wald (inverse-Hessian) coverage as a PROXY (flagged), while the
       cluster gate uses PARAMETRIC BOOTSTRAP coverage (the v<->z ridge makes the
       Hessian unreliable exactly where sharpness lives).

    Parameters
    ----------
    point_res : dict
        ``recover_point`` output: per-dial ``{r, bias, sd_true, ci_coverage,
        [ccc], ...}`` keyed by public point names.
    confusion_res : dict
        ``recover_confusion`` output: ``{matrix, labels, ...}``.
    truediff_res : dict
        ``recover_true_difference`` output: ``{recovered_delta, shrunk}``;
        ``shrunk`` may be a bool or a per-dial mapping.
    cond_res : dict
        ``hessian_conditioning`` output: ``{cond_number, rank, deficient}``.
    regime : str
        The regime label (e.g. ``'expert'`` / ``'naive'``); echoed in the result and
        gates the ``naive_relax`` path.
    r_min, bias_max_frac, coverage_min, ccc_min : float
        Point-recovery thresholds (gate_criteria.md §1).
    confusion_min_diag, confusion_max_offdiag : float
        Confusion thresholds (gate_criteria.md §2).
    naive_relax : float
        Threshold relaxation applied to ``r_min``/``confusion_min_diag`` ONLY when
        ``regime == 'naive'`` (default 0.0 = uniform thresholds; see above).

    Returns
    -------
    dict
        ``{"per_dial_trust": {dial: "generative"|"descriptive"}, "regime": str,
        "passed": {dial: {sub_check: bool|None}}}`` for ``dial`` in
        ``("sharpness", "caution", "timing")``. ``passed`` is the auditable record
        of every sub-check (``point_r``, ``bias``, ``coverage``, ``ccc``,
        ``confusion_diag``, ``confusion_offdiag``, ``not_shrunk``, ``hessian_ok``);
        ``ccc`` is ``None`` when CCC was unavailable (skipped, not failed).
    """
    # ── regime-gated relaxation (no-op at the ratified default 0.0) ──
    relax = float(naive_relax) if str(regime).lower() == "naive" else 0.0
    r_thr = float(r_min) - relax
    diag_thr = float(confusion_min_diag) - relax

    # ── anchor-level Hessian veto (reads once; applies to every dial) ──
    hessian_ok = not bool(cond_res.get("deficient", False))

    # ── confusion matrix + labels ──
    M = np.asarray(confusion_res.get("matrix"), float)
    labels = list(confusion_res.get("labels", _GATE_DIALS))

    # ── normalize the shrunk veto into a per-dial lookup ──
    shrunk_raw = truediff_res.get("shrunk", False)

    def _is_shrunk(gate_dial):
        """True if this dial's genuine difference was crushed by the L2 prior."""
        if isinstance(shrunk_raw, dict):
            # accept either raw (v/z/u) or gate-label keys; default False if absent
            raw = _GATE_DIAL_TO_RAW[gate_dial]
            if raw in shrunk_raw:
                return bool(shrunk_raw[raw])
            if gate_dial in shrunk_raw:
                return bool(shrunk_raw[gate_dial])
            return False
        return bool(shrunk_raw)   # scalar bool -> applies to every dial

    per_dial_trust = {}
    passed = {}

    for gate_dial in _GATE_DIALS:
        checks = {}

        # ── POINT RECOVERY block ──
        pkey = _GATE_DIAL_TO_POINT[gate_dial]
        pr = point_res.get(pkey, {}) or {}

        r_val = pr.get("r", float("nan"))
        checks["point_r"] = bool(np.isfinite(r_val) and r_val >= r_thr)

        bias = pr.get("bias", float("nan"))
        sd_true = pr.get("sd_true", float("nan"))
        if np.isfinite(bias) and np.isfinite(sd_true):
            checks["bias"] = bool(abs(bias) <= float(bias_max_frac) * sd_true)
        else:
            checks["bias"] = False

        cov = pr.get("ci_coverage", float("nan"))
        # STRICT lower bound; over-coverage is acceptable (never fails)
        checks["coverage"] = bool(np.isfinite(cov) and cov >= float(coverage_min))

        # CCC: checked WHEN AVAILABLE, else skipped (None = not a failure)
        if "ccc" in pr and pr["ccc"] is not None:
            ccc = pr["ccc"]
            checks["ccc"] = bool(np.isfinite(ccc) and ccc >= float(ccc_min))
        else:
            checks["ccc"] = None

        # ── CONFUSION block ──
        if gate_dial in labels and M.size:
            i = labels.index(gate_dial)
            checks["confusion_diag"] = bool(M[i, i] >= diag_thr)
            # every SPECIFIC off-diagonal in this true-row <= max
            offdiag_ok = True
            for j in range(M.shape[1]):
                if j != i and M[i, j] > float(confusion_max_offdiag):
                    offdiag_ok = False
                    break
            checks["confusion_offdiag"] = offdiag_ok
        else:
            # no confusion info for this dial -> cannot certify it
            checks["confusion_diag"] = False
            checks["confusion_offdiag"] = False

        # ── vetoes ──
        checks["not_shrunk"] = not _is_shrunk(gate_dial)
        checks["hessian_ok"] = hessian_ok

        passed[gate_dial] = checks

        # ── AND rule across applicable sub-checks (None CCC = skipped) ──
        applicable = [v for v in checks.values() if v is not None]
        is_generative = all(applicable)
        per_dial_trust[gate_dial] = "generative" if is_generative else "descriptive"

    return {"per_dial_trust": per_dial_trust, "regime": str(regime),
            "passed": passed}


# ── Engine-A deliverable (Task 4.1) — append generative latents to Phase-1 ────
# Maps the gate's per-dial trust keys -> the deliverable's trust_* columns. The
# recovery gate names the caution dial 'caution' (= itchiness/z); the deliverable
# keeps that name in `trust_caution` while the per-trial latent itself is named
# `itchiness_caution` (Phase-1 vocabulary is "itchiness", Phase-2 dial is
# "caution"; both refer to the z/start-point knob).
_TRUST_GATE_KEY = {"trust_sharpness": "sharpness",
                   "trust_caution": "caution",
                   "trust_timing": "timing"}

# The exact set of columns appended by `append_generative_latents` (documented so
# downstream consumers + the test agree on the contract). 14 columns total.
APPENDED_LATENT_COLUMNS = (
    "sharpness_drift",                  # the trial mood's v   (FitResult.dials[mood]['sharpness'])
    "itchiness_caution",               # the trial mood's z   (...['itchiness'])
    "timing_urgency_at_decision",      # REALIZED: u * phi[event_bin] (NOT the coef u)
    "evidence_integral_at_decision",   # REALIZED: A[event_bin]
    "expected_change_time",            # mu_by_session[session]
    "lick_minus_expected",             # decision_time - mu_by_session[session]
    "anchor_id",                       # the session's anchor id (== session_name if fitted)
    "rectification_kind",              # provenance: rectification used
    "leak_tau",                        # provenance: leak time-constant
    "recovery_regime",                 # the trial session's regime
    "trust_sharpness",                 # per-dial trust: 'generative' | 'descriptive'
    "trust_caution",
    "trust_timing",
    "generative_omitted",              # bool: True for QC-omitted (no-anchor) sessions
)


def _trial_evidence_lookup(ev_obj):
    """Return a ``{trial_idx: evidence_array}`` map from a session's evidence object.

    Accepts the canonical ``build_trial_evidence_corrected`` DataFrame (columns
    ``trial_idx`` + ``evidence``) OR an already-built mapping
    ``{trial_idx: np.ndarray}``. Anything else raises a clear error.
    """
    if ev_obj is None:
        return {}
    if isinstance(ev_obj, pd.DataFrame):
        if "trial_idx" not in ev_obj.columns or "evidence" not in ev_obj.columns:
            raise KeyError(
                "trial-evidence DataFrame must have 'trial_idx' and 'evidence' "
                f"columns; got {list(ev_obj.columns)}")
        return {int(r.trial_idx): np.asarray(r.evidence, float)
                for r in ev_obj.itertuples(index=False)}
    if isinstance(ev_obj, dict):
        return {int(k): np.asarray(v, float) for k, v in ev_obj.items()}
    raise TypeError(
        "trial_evidence_by_session values must be a DataFrame "
        "(build_trial_evidence_corrected form) or a {trial_idx: array} dict; "
        f"got {type(ev_obj)!r}")


def append_generative_latents(per_trial_csv, anchor_fits, recovery_by_regime,
                              param_spec, mu_by_session, trial_evidence_by_session,
                              regime_by_session, sigma, dt=0.05, leak_tau=0.27,
                              rectification="signed"):
    """Append the Engine-A generative decision-latents to the Phase-1 deliverable.

    Plain English: Phase 1 shipped a 25-column per-trial table that *measured* three
    behavioural knobs per cell. This function reads that table and bolts on, per
    trial, the GENERATIVE model's view of the same trial — the fitted dials for the
    trial's mood, the genuinely **trial-specific realized** urgency and accumulated
    evidence *at the decision bin*, the timing-expectation bookkeeping, and a full
    provenance trail (which anchor, which rectification/leak, which regime, and —
    per dial — whether the recovery battery earned that dial a 'generative'
    interpretation or it falls back to the Phase-1 'descriptive' proxy). The 25
    Phase-1 columns are **never** overwritten; this only APPENDS.

    Appended columns (see :data:`APPENDED_LATENT_COLUMNS`, 14 total):

    * ``sharpness_drift`` — the trial mood's ``v`` from its session's
      ``FitResult.dials[mood]['sharpness']`` (a regression-varying coefficient).
    * ``itchiness_caution`` — the trial mood's ``z`` (``...['itchiness']``).
    * ``timing_urgency_at_decision`` — the **realized** urgency at the decision bin
      ``= u * phi[event_bin]`` where ``u`` is the mood's timing coefficient,
      ``phi = expectation_bump(arange(n_bins)*dt, mu_session, sigma)`` and
      ``event_bin = n_bins - 1``. This is a genuinely TRIAL-SPECIFIC value (it
      depends on the trial's length and ``mu``), **NOT** the coefficient ``u``.
    * ``evidence_integral_at_decision`` — ``A[event_bin]`` for the trial, where
      ``A = leaky_accumulate(trial_evidence, dt, leak_tau, rectification)``.
    * ``expected_change_time`` — ``mu_by_session[session]``.
    * ``lick_minus_expected`` — ``decision_time - mu_by_session[session]``.
    * ``anchor_id`` — the session's anchor id (the session name when it was fitted;
      empty/NaN for a QC-omitted session).
    * ``rectification_kind``, ``leak_tau`` — accumulator provenance.
    * ``recovery_regime`` — the trial's session regime (``regime_by_session``).
    * ``trust_sharpness`` / ``trust_caution`` / ``trust_timing`` — per-dial
      ``'generative' | 'descriptive'`` from
      ``recovery_by_regime[regime]['per_dial_trust']`` (the gate names the z dial
      ``'caution'``). A dial that failed recovery in the trial's regime is
      ``'descriptive'`` here.
    * ``generative_omitted`` — ``True`` for trials whose session has no fitted
      anchor (QC-omitted): those rows get NaN latents and ``trust_*='descriptive'``
      but are NEVER dropped.

    Parameters
    ----------
    per_trial_csv : str | path-like
        Path to the Phase-1 ``decision_latents_by_state.csv`` (the 25-column
        per-trial deliverable). Read-only — never written.
    anchor_fits : Mapping[str, FitResult]
        ``{session_name: FitResult}`` for every fitted anchor. A session ABSENT
        here is treated as QC-omitted (NaN latents, ``generative_omitted=True``).
    recovery_by_regime : Mapping[str, dict]
        ``{regime: recovery_gate_output}`` — only ``['per_dial_trust']`` is read
        (keys ``'sharpness'``/``'caution'``/``'timing'``).
    param_spec : ParamSpec
        Accepted for interface symmetry / future-proofing (the dials are read from
        ``FitResult.dials``, which already encodes the layout). Not indexed here.
    mu_by_session : Mapping[str, float]
        Per-session temporal-expectation anchor μ (Task 0.4).
    trial_evidence_by_session : Mapping[str, DataFrame|dict]
        Per-session evidence: the ``build_trial_evidence_corrected`` DataFrame
        (``trial_idx`` + ``evidence`` columns) or a ``{trial_idx: array}`` map.
    regime_by_session : Mapping[str, str]
        ``{session_name: regime}`` (e.g. ``'expert'`` / ``'naive'``) — selects the
        ``recovery_by_regime`` trust row for the session's trials.
    sigma : float
        FIXED urgency-bump width (seconds; a ``ParamSpec`` field, not fitted).
    dt, leak_tau, rectification :
        Generative time-grid + leaky-accumulator settings (contract §A.3); recorded
        as provenance and used to compute ``A`` (so they MATCH the fit).

    Returns
    -------
    pandas.DataFrame
        The Phase-1 table with :data:`APPENDED_LATENT_COLUMNS` appended (one row
        per input trial; nothing dropped).
    """
    df = pd.read_csv(per_trial_csv)
    n = len(df)

    # Canonicalize session ids to zfill8 on BOTH sides (the CSV session_name column,
    # below, and the per-session dicts here), so a fitted anchor matches its trials
    # regardless of representation: the deliverable stores session_name as int64
    # (a leading-zero DAY dropped, '01072025' -> 1072025) while the fit / geometry
    # dicts are keyed by the canonical zfill8 form. Without this, every
    # leading-zero-day anchor (1-9 of a month) would silently miss -> NaN latents +
    # generative_omitted despite being fitted. recovery_by_regime is keyed by REGIME
    # (not session), so it is left untouched.
    anchor_fits = {canonical_session_id(k): v for k, v in dict(anchor_fits).items()}
    mu_by_session = {canonical_session_id(k): v for k, v in dict(mu_by_session).items()}
    regime_by_session = {canonical_session_id(k): v
                         for k, v in dict(regime_by_session).items()}
    trial_evidence_by_session = {canonical_session_id(k): v
                                 for k, v in dict(trial_evidence_by_session).items()}

    # Pre-resolve the per-dial trust row for each regime (gate keys ->
    # trust_* columns). QC-omitted trials override these to 'descriptive'.
    def _trust_row(regime):
        rec = recovery_by_regime.get(regime, {}) or {}
        pdt = rec.get("per_dial_trust", {}) or {}
        return {col: str(pdt.get(gate_key, "descriptive"))
                for col, gate_key in _TRUST_GATE_KEY.items()}

    # Per-session evidence lookups built lazily (so absent sessions cost nothing).
    _ev_cache: dict = {}

    def _ev_for(session):
        if session not in _ev_cache:
            _ev_cache[session] = _trial_evidence_lookup(
                trial_evidence_by_session.get(session))
        return _ev_cache[session]

    # Output accumulators (preallocated so omitted/failed rows stay aligned).
    sharpness_drift = np.full(n, np.nan)
    itchiness_caution = np.full(n, np.nan)
    timing_urgency = np.full(n, np.nan)
    evidence_integral = np.full(n, np.nan)
    expected_change_time = np.full(n, np.nan)
    lick_minus_expected = np.full(n, np.nan)
    anchor_id = np.empty(n, dtype=object)
    recovery_regime = np.empty(n, dtype=object)
    trust_sharpness = np.empty(n, dtype=object)
    trust_caution = np.empty(n, dtype=object)
    trust_timing = np.empty(n, dtype=object)
    generative_omitted = np.zeros(n, dtype=bool)

    sess_arr = np.array([canonical_session_id(s)
                         for s in df["session_name"].to_numpy()], dtype=object)
    tidx_arr = df["trial_idx"].to_numpy()
    mood_arr = df["state_label"].astype(object).to_numpy()
    dtime_arr = pd.to_numeric(df["decision_time"], errors="coerce").to_numpy(float)

    for i in range(n):
        session = sess_arr[i]
        regime = regime_by_session.get(session)
        recovery_regime[i] = regime if regime is not None else None

        fit = anchor_fits.get(session)
        mu = mu_by_session.get(session, np.nan)
        expected_change_time[i] = float(mu) if mu is not None else np.nan

        # ── QC-omitted session (no fitted anchor): NaN latents, flag, descriptive ─
        if fit is None:
            generative_omitted[i] = True
            anchor_id[i] = None
            trust_sharpness[i] = "descriptive"
            trust_caution[i] = "descriptive"
            trust_timing[i] = "descriptive"
            # lick_minus_expected is a behavioural quantity; still well-defined if
            # we have decision_time + mu, but the latent is "omitted" so keep NaN
            # unless mu is known (it is informative bookkeeping, not a fitted latent).
            if np.isfinite(dtime_arr[i]) and mu is not None and np.isfinite(mu):
                lick_minus_expected[i] = float(dtime_arr[i]) - float(mu)
            continue

        anchor_id[i] = session

        # per-dial trust for the trial's regime (gate -> trust_* columns)
        trust = _trust_row(regime)
        trust_sharpness[i] = trust["trust_sharpness"]
        trust_caution[i] = trust["trust_caution"]
        trust_timing[i] = trust["trust_timing"]

        # behavioural bookkeeping (defined regardless of mood-dial availability)
        if np.isfinite(dtime_arr[i]) and mu is not None and np.isfinite(mu):
            lick_minus_expected[i] = float(dtime_arr[i]) - float(mu)

        mood = mood_arr[i]
        dials = fit.dials.get(mood) if hasattr(fit, "dials") else None
        if dials is None:
            # Fitted session but this trial's mood is not in the fit (e.g. a
            # Disengaged trial in a MAIN_MOODS-only fit): latents stay NaN, but the
            # row is kept with its provenance + trust (NOT flagged omitted).
            continue

        v = float(dials["sharpness"])
        z = float(dials["itchiness"])
        u = float(dials["timing"])
        sharpness_drift[i] = v
        itchiness_caution[i] = z

        # ── realized quantities at the decision bin ──
        ev_map = _ev_for(session)
        tkey = int(tidx_arr[i]) if np.isfinite(tidx_arr[i]) else None
        evidence = ev_map.get(tkey) if tkey is not None else None
        if evidence is None or len(evidence) == 0:
            # no evidence for this trial -> cannot realize A/phi; leave NaN
            continue
        n_bins = len(evidence)
        event_bin = n_bins - 1

        A = leaky_accumulate(evidence, dt=dt, leak_tau=leak_tau,
                             rectification=rectification)
        phi = expectation_bump(np.arange(n_bins) * dt, float(mu), float(sigma))

        # REALIZED urgency = coefficient u * the bump value at the decision bin
        # (genuinely trial-specific; NOT the coefficient u itself).
        timing_urgency[i] = u * float(phi[event_bin])
        evidence_integral[i] = float(A[event_bin])

    # ── append (never overwrite the originals) ──
    df["sharpness_drift"] = sharpness_drift
    df["itchiness_caution"] = itchiness_caution
    df["timing_urgency_at_decision"] = timing_urgency
    df["evidence_integral_at_decision"] = evidence_integral
    df["expected_change_time"] = expected_change_time
    df["lick_minus_expected"] = lick_minus_expected
    df["anchor_id"] = anchor_id
    df["rectification_kind"] = str(rectification)
    df["leak_tau"] = float(leak_tau)
    df["recovery_regime"] = recovery_regime
    df["trust_sharpness"] = trust_sharpness
    df["trust_caution"] = trust_caution
    df["trust_timing"] = trust_timing
    df["generative_omitted"] = generative_omitted

    return df
