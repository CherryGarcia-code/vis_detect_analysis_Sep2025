"""B8 Phase 2 — CLUSTER recovery harness: the FULL-POWER recovery verdict.

Plain English: the local unit tests only PROVE the recovery machinery at a
reduced config (small n_rep / small designs). The published "can we trust each
behavioural dial as a real mechanism?" verdict is only valid at FULL config on a
cluster (gate_criteria.md R1). This script is that full-power run: for BOTH
regimes (expert-like / naive-like) it

  1. POINT RECOVERY at full config — simulate->refit over many jittered-truth
     replicates, per dial reporting Pearson r, Lin's CCC, bias + SD(true), and
     PARAMETRIC-BOOTSTRAP CI coverage (NOT Wald — gate_criteria.md R3);
  2. CONFUSION matrix at full power — the 3x3 which-dial-varies discriminability
     test (learning ladder, AIC-only fast path);
  3. recover_true_difference (shrunk veto) + hessian_conditioning (anchor veto);
  4. applies ``recovery_gate`` per (dial x regime) -> the latent_trust verdict.

It is process-parallel over replicates (``multiprocessing.Pool``); BLAS is kept
single-threaded inside the workers so the process-level parallelism scales (set
OMP/OPENBLAS/MKL/NUMEXPR/VECLIB = 1 BEFORE importing numpy).

Outputs (to ``--out-dir``):
  * ``recovery_results.json`` — every metric + CI + the per-(dial x regime)
    verdict + the config used (auditable, restartable);
  * ``fig_b8_F6_recovery.png`` — F6 figure: recovery bars per dial/regime with
    the r>=0.8 line, the confusion matrices as heatmaps, and a verdict table.

Run locally with ``--quick`` (tiny config, ~1-3 min) to smoke the end-to-end
path. Run the FULL config on the cluster via ``cluster_recovery_harness.sbatch``.
"""
from __future__ import annotations

# ── BLAS single-thread BEFORE numpy import (process-parallel design) ──────────
# We parallelise over replicate processes, so each worker must keep BLAS to one
# thread or the threads oversubscribe the cores and the Pool stops scaling.
import os as _os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_v, "1")

import argparse
import json
import multiprocessing as mp
import os
import platform
import sys
import time
from datetime import datetime, timezone

import numpy as np

# cp1252-safe console (the user runs on a Windows cp1252 terminal locally).
try:  # pragma: no cover - console only
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass


# ════════════════════════════════════════════════════════════════════════════
# Pre-flight: versions + import assertions (fail loudly before burning the job)
# ════════════════════════════════════════════════════════════════════════════
def preflight():
    """Print env/versions, assert the imports, return the imported module handles.

    Fails LOUDLY (raises) if visdetect / the fixtures cannot be imported — better
    to die in the pre-flight than to silently produce a junk verdict.
    """
    print("=" * 70)
    print("B8 Phase 2 — cluster recovery harness  (pre-flight)")
    print("=" * 70)
    print(f"  python   : {platform.python_version()}  ({sys.executable})")
    try:
        import numpy
        import scipy
        print(f"  numpy    : {numpy.__version__}")
        print(f"  scipy    : {scipy.__version__}")
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"FATAL: numpy/scipy import failed: {exc}")

    # Make the fixtures (a sibling script module) importable by path.
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)

    try:
        from visdetect.analysis import decision_latents_generative as dlg
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "FATAL: cannot import visdetect.analysis.decision_latents_generative "
            f"({exc}). Set PYTHONPATH=<repo>/src.")
    try:
        from _recovery_fixtures import make_recovery_design
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            f"FATAL: cannot import _recovery_fixtures.make_recovery_design ({exc}). "
            "It must sit beside this script.")

    print(f"  visdetect: {dlg.__file__}")
    # sanity: the BLAS thread caps actually took
    print("  BLAS threads: OMP={} OPENBLAS={} MKL={}".format(
        os.environ.get("OMP_NUM_THREADS"),
        os.environ.get("OPENBLAS_NUM_THREADS"),
        os.environ.get("MKL_NUM_THREADS")))
    print("=" * 70, flush=True)
    return dlg, make_recovery_design


# ════════════════════════════════════════════════════════════════════════════
# Metric helpers
# ════════════════════════════════════════════════════════════════════════════
def lins_ccc(true_vals, rec_vals):
    """Lin's concordance correlation coefficient between paired arrays.

    CCC = 2 * cov(x,y) / (var(x) + var(y) + (mean(x)-mean(y))**2). Unlike Pearson
    r it penalises systematic bias / scale shift (gate_criteria.md §1). NaN-safe:
    returns NaN if either side has ~zero spread.
    """
    x = np.asarray(true_vals, float)
    y = np.asarray(rec_vals, float)
    if x.size < 2 or y.size < 2:
        return float("nan")
    mx, my = x.mean(), y.mean()
    # population variance / covariance (ddof=0), standard for CCC
    vx = x.var(ddof=0)
    vy = y.var(ddof=0)
    cov = np.mean((x - mx) * (y - my))
    denom = vx + vy + (mx - my) ** 2
    if denom <= 1e-12:
        return float("nan")
    return float(2.0 * cov / denom)


# ════════════════════════════════════════════════════════════════════════════
# Point recovery — ONE replicate (top-level so it is picklable for Pool)
# ════════════════════════════════════════════════════════════════════════════
# Each worker rebuilds the regime's recovery Design from the fixtures (the Design
# is cheap to build and not reliably picklable across processes), jitters the
# truth EXACTLY as recover_point does, simulates->refits, and ALSO runs an inner
# parametric-bootstrap loop to get per-fit, Hessian-free CI coverage (R3).
#
# A small per-process cache avoids rebuilding the Design for every rep handled by
# the same worker.
_DESIGN_CACHE: dict = {}


def _get_design(regime, n_trials, design_seed, dlg, make_recovery_design):
    key = (regime, n_trials, design_seed)
    if key not in _DESIGN_CACHE:
        _DESIGN_CACHE[key] = make_recovery_design(
            regime, n_trials=n_trials, seed=design_seed)
    return _DESIGN_CACHE[key]


def _point_rep(args):
    """Run ONE point-recovery replicate; return per-(dial,mood) true/rec pairs and
    per-pair bootstrap-CI in/out flags.

    args = (regime, n_trials, design_seed, true_theta, jitter, rep_seed,
            n_restarts, n_bootstrap, boot_seed)
    """
    (regime, n_trials, design_seed, true_theta, jitter, rep_seed,
     n_restarts, n_bootstrap, boot_seed) = args

    # late import inside the worker (after BLAS env is set at module import)
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    from visdetect.analysis import decision_latents_generative as dlg
    from _recovery_fixtures import make_recovery_design

    design, _true_theta0, ps = _get_design(
        regime, n_trials, design_seed, dlg, make_recovery_design)
    true_theta = np.asarray(true_theta, float)
    moods = list(ps.moods)
    dials = ("v", "z", "u")

    rep_rng = np.random.default_rng(int(rep_seed))

    # ── jitter the TRUTH for this rep — SAME perturbation to both moods of a dial
    #    (coherent move), MIRRORING recover_point exactly ──
    theta_j = true_theta.copy()
    for d in dials:
        off = ps._offset(d)
        delta = float(rep_rng.normal(0.0, jitter[d]))
        for mi in range(len(moods)):
            theta_j[off + mi] = true_theta[off + mi] + delta

    # ── simulate -> refit ──
    sim_seed = int(rep_rng.integers(0, 2**31 - 1))
    eb, lk, cs = dlg.simulate_licks(design, theta_j, ps, seed=sim_seed)
    sim_design = dlg.design_with_outcomes(design, eb, lk, cs)
    fit = dlg.fit_anchor(sim_design, ps, seed_theta=None, l2=0.0,
                         n_restarts=int(n_restarts), seed=int(rep_seed))
    theta_hat = np.asarray(fit.theta, float)

    # ── PARAMETRIC BOOTSTRAP CI per parameter (R3: NOT Wald) ──
    #    simulate from the FIT, refit, collect theta*, take a percentile CI, and
    #    record whether the JITTERED TRUTH value falls inside it (coverage of the
    #    truth — the gate-relevant quantity).
    n_params = ps.n_params()
    boot_thetas = np.empty((int(n_bootstrap), n_params), float)
    boot_rng = np.random.default_rng(int(boot_seed))
    n_ok = 0
    for b in range(int(n_bootstrap)):
        bs = int(boot_rng.integers(0, 2**31 - 1))
        eb_b, lk_b, cs_b = dlg.simulate_licks(design, theta_hat, ps, seed=bs)
        d_b = dlg.design_with_outcomes(design, eb_b, lk_b, cs_b)
        # 1 restart inside the bootstrap (seeded at theta_hat) — fast and stable;
        # the outer fit already found the basin.
        fit_b = dlg.fit_anchor(d_b, ps, seed_theta=theta_hat, l2=0.0,
                               n_restarts=1, seed=bs)
        tb = np.asarray(fit_b.theta, float)
        if np.all(np.isfinite(tb)):
            boot_thetas[n_ok] = tb
            n_ok += 1
    boot_thetas = boot_thetas[:n_ok]

    # per-(dial,mood) records for this rep
    rec = {d: {m: {} for m in moods} for d in dials}
    have_ci = n_ok >= 2
    if have_ci:
        lo = np.percentile(boot_thetas, 2.5, axis=0)
        hi = np.percentile(boot_thetas, 97.5, axis=0)
    for d in dials:
        off = ps._offset(d)
        for mi, m in enumerate(moods):
            idx = off + mi
            t_val = float(theta_j[idx])
            r_val = float(theta_hat[idx])
            in_ci = None
            if have_ci:
                in_ci = bool(lo[idx] <= t_val <= hi[idx])
            rec[d][m] = {"true": t_val, "rec": r_val, "in_ci": in_ci}
    return rec


_DIAL_PUBLIC = {"v": "sharpness", "z": "itchiness", "u": "timing"}
# matches the validated recovery test jitter
# (test_decision_latents_generative.py:2393); narrower values tighten the bias/r
# tolerances below what was validated.
_RECOVER_JITTER_SD = {"v": 0.60, "z": 0.55, "u": 0.55}


def run_point_recovery(regime, dlg, make_recovery_design, *, n_rep, n_trials,
                       n_restarts, n_bootstrap, seed, pool):
    """Full-config point recovery for one regime, parallel over replicates.

    Produces a per-dial dict shaped EXACTLY like ``recover_point`` output (keys
    ``sharpness``/``itchiness``/``timing`` with ``r, bias, sd_true, ci_coverage,
    n_pairs, n_cov_excluded``) PLUS a ``ccc`` field — i.e. precisely what
    ``recovery_gate`` consumes.
    """
    design, true_theta, ps = make_recovery_design(regime, n_trials=n_trials,
                                                  seed=seed)
    jitter = dict(_RECOVER_JITTER_SD)
    moods = list(ps.moods)
    dials = ("v", "z", "u")

    master = np.random.default_rng(seed)
    rep_seeds = master.integers(0, 2**31 - 1, size=int(n_rep))
    boot_seeds = master.integers(0, 2**31 - 1, size=int(n_rep))

    tasks = [
        (regime, n_trials, seed, true_theta, jitter, int(rep_seeds[j]),
         n_restarts, n_bootstrap, int(boot_seeds[j]))
        for j in range(int(n_rep))
    ]

    t0 = time.time()
    print(f"[point/{regime}] {n_rep} reps x {n_trials} trials, "
          f"{n_bootstrap} bootstrap each ... ", flush=True)

    if pool is not None:
        results = []
        for k, r in enumerate(pool.imap_unordered(_point_rep, tasks), 1):
            results.append(r)
            if k % max(1, int(n_rep) // 10) == 0 or k == int(n_rep):
                print(f"    [point/{regime}] {k}/{n_rep} reps "
                      f"({time.time()-t0:.0f}s)", flush=True)
    else:
        results = [_point_rep(t) for t in tasks]

    # ── pool the per-(dial,mood) pairs across reps, reduce to the per-dial dict ──
    out = {}
    for d in dials:
        t_all, r_all, ci_all = [], [], []
        for rep in results:
            for m in moods:
                cell = rep[d][m]
                t_all.append(cell["true"])
                r_all.append(cell["rec"])
                if cell["in_ci"] is not None:
                    ci_all.append(cell["in_ci"])
        t_pool = np.asarray(t_all, float)
        r_pool = np.asarray(r_all, float)

        if t_pool.size >= 2 and np.std(t_pool) > 1e-12 and np.std(r_pool) > 1e-12:
            r = float(np.corrcoef(t_pool, r_pool)[0, 1])
        else:
            r = float("nan")
        bias = float(np.mean(r_pool - t_pool)) if t_pool.size else float("nan")
        sd_true = float(np.std(t_pool)) if t_pool.size else float("nan")
        ccc = lins_ccc(t_pool, r_pool)
        ci_coverage = float(np.mean(ci_all)) if len(ci_all) > 0 else float("nan")
        n_cov_excluded = int(t_pool.size - len(ci_all))

        out[_DIAL_PUBLIC[d]] = {
            "r": r, "bias": bias, "sd_true": sd_true, "ccc": ccc,
            "ci_coverage": ci_coverage, "n_pairs": int(t_pool.size),
            "n_cov_excluded": n_cov_excluded,
        }
    print(f"[point/{regime}] done ({time.time()-t0:.0f}s)", flush=True)
    return out, design, true_theta, ps


# ════════════════════════════════════════════════════════════════════════════
# Confusion — ONE (scenario, rep) (top-level for Pool)
# ════════════════════════════════════════════════════════════════════════════
_CONFUSION_SCENARIOS = (("v", "sharpness"), ("z", "caution"), ("u", "timing"))
_CONFUSION_LABELS = ("sharpness", "caution", "timing")
_CONFUSION_COL = {"sharpness": 0, "caution": 1, "timing": 2}
_CONFUSION_DELTA = {"v": 1.0, "z": 1.5, "u": 2.5}
_LADDER_WINNER_TO_DIAL = {"M_sharpness": "sharpness", "M_caution": "caution",
                          "M_timing": "timing"}


def _confusion_rep(args):
    """One (scenario, rep): build the two anchors (only one dial differs), simulate
    licks, run the learning ladder (AIC-only fast path), return the winner string.

    args = (regime, n_trials, design_seed, base_theta, dial, delta, sa, sb,
            ladder_seed, n_restarts)
    """
    (regime, n_trials, design_seed, base_theta, dial, delta, sa, sb,
     ladder_seed, n_restarts) = args

    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    from visdetect.analysis import decision_latents_generative as dlg
    from _recovery_fixtures import make_recovery_design

    design, _tt, ps = _get_design(
        regime, n_trials, design_seed, dlg, make_recovery_design)
    base_theta = np.asarray(base_theta, float)
    n_mood = len(ps.moods)
    off = ps._offset(dial)

    theta_a = base_theta.copy()
    theta_b = base_theta.copy()
    for mi in range(n_mood):
        theta_b[off + mi] = base_theta[off + mi] + float(delta)

    eb_a, lk_a, cs_a = dlg.simulate_licks(design, theta_a, ps, seed=int(sa))
    eb_b, lk_b, cs_b = dlg.simulate_licks(design, theta_b, ps, seed=int(sb))
    da = dlg.design_with_outcomes(design, eb_a, lk_a, cs_a)
    db = dlg.design_with_outcomes(design, eb_b, lk_b, cs_b)

    out = dlg.learning_ladder({"A": da, "B": db}, ps, dt=design.dt,
                              seed=int(ladder_seed), n_restarts=int(n_restarts),
                              compute_cvll=False)
    return dial, out["winner"]


def run_confusion(regime, dlg, make_recovery_design, *, n_rep, n_trials,
                  n_restarts, seed, pool):
    """Full-power 3x3 confusion matrix for one regime, parallel over (scenario, rep).

    Shaped EXACTLY like ``recover_confusion`` output (``matrix``, ``labels``,
    ``no_single``, ``winners``, ``n_rep``) so ``recovery_gate`` consumes it.
    """
    design, base_theta, ps = make_recovery_design(regime, n_trials=n_trials,
                                                  seed=seed)
    delta = dict(_CONFUSION_DELTA)

    master = np.random.default_rng(seed)
    n_sc = len(_CONFUSION_SCENARIOS)
    sim_seeds = master.integers(0, 2**31 - 1, size=(n_sc, int(n_rep), 2))
    ladder_seeds = master.integers(0, 2**31 - 1, size=(n_sc, int(n_rep)))

    tasks = []
    for si, (dial, _row) in enumerate(_CONFUSION_SCENARIOS):
        for rep in range(int(n_rep)):
            tasks.append((
                regime, n_trials, seed, base_theta, dial, float(delta[dial]),
                int(sim_seeds[si, rep, 0]), int(sim_seeds[si, rep, 1]),
                int(ladder_seeds[si, rep]), n_restarts,
            ))

    t0 = time.time()
    print(f"[confusion/{regime}] {n_sc} scenarios x {n_rep} reps "
          f"x {n_trials} trials (AIC-only ladder) ... ", flush=True)

    if pool is not None:
        raw = []
        total = len(tasks)
        for k, r in enumerate(pool.imap_unordered(_confusion_rep, tasks), 1):
            raw.append(r)
            if k % max(1, total // 10) == 0 or k == total:
                print(f"    [confusion/{regime}] {k}/{total} fits "
                      f"({time.time()-t0:.0f}s)", flush=True)
    else:
        raw = [_confusion_rep(t) for t in tasks]

    # ── reduce winners into the 3x3 matrix ──
    matrix = np.zeros((3, 3), float)
    no_single = {lab: 0 for lab in _CONFUSION_LABELS}
    winners = {lab: [] for lab in _CONFUSION_LABELS}
    dial_to_label = {"v": "sharpness", "z": "caution", "u": "timing"}
    for dial, winner in raw:
        row_label = dial_to_label[dial]
        winners[row_label].append(winner)
        named = _LADDER_WINNER_TO_DIAL.get(winner)
        if named is None:
            no_single[row_label] += 1
        else:
            matrix[_CONFUSION_COL[row_label], _CONFUSION_COL[named]] += 1.0
    matrix /= float(n_rep)

    print(f"[confusion/{regime}] done ({time.time()-t0:.0f}s)", flush=True)
    return {"matrix": matrix, "labels": _CONFUSION_LABELS,
            "no_single": no_single, "winners": winners, "n_rep": int(n_rep)}


# ════════════════════════════════════════════════════════════════════════════
# Vetoes: hessian conditioning (on a free expert/naive fit) + shrunk
# ════════════════════════════════════════════════════════════════════════════
def run_vetoes(regime, design, true_theta, ps, dlg, make_recovery_design, *,
               n_trials, n_restarts, seed):
    """Compute the two veto inputs for a regime.

    * ``cond_res`` from ``hessian_conditioning`` on a FREE fit of one simulated
      dataset at the regime's true theta (the identifiability of the operating
      point);
    * ``truediff_res`` from ``recover_true_difference`` between a naive-like and
      expert-like anchor with a KNOWN identifiable difference on ``v`` (the
      shrunk veto). This is regime-independent (it tests the seeding), so we
      compute it ONCE (for 'expert') and reuse it; for 'naive' we pass the same.
    """
    # ── hessian conditioning on a free fit at the regime's truth ──
    eb, lk, cs = dlg.simulate_licks(design, true_theta, ps, seed=seed)
    sim_design = dlg.design_with_outcomes(design, eb, lk, cs)
    fit = dlg.fit_anchor(sim_design, ps, seed_theta=None, l2=0.0,
                         n_restarts=int(n_restarts), seed=seed)
    cond_res = dlg.hessian_conditioning(fit)
    return cond_res


def run_truediff(dlg, make_recovery_design, *, n_trials, seed):
    """Shrunk veto: do the L2-seeded backward fits RECOVER a genuine across-stage
    difference (rather than crush it)? Uses the naive + expert recovery designs at
    their true thetas with a known v difference.
    """
    d_naive, th_naive, ps = make_recovery_design("naive", n_trials=n_trials,
                                                 seed=seed)
    d_expert, th_expert, _ps = make_recovery_design("expert", n_trials=n_trials,
                                                    seed=seed + 1)
    # simulate outcomes at each anchor's true theta so the fit has real data
    eb_n, lk_n, cs_n = dlg.simulate_licks(d_naive, th_naive, ps, seed=seed)
    eb_e, lk_e, cs_e = dlg.simulate_licks(d_expert, th_expert, ps, seed=seed + 7)
    dn = dlg.design_with_outcomes(d_naive, eb_n, lk_n, cs_n)
    de = dlg.design_with_outcomes(d_expert, eb_e, lk_e, cs_e)

    # KNOWN true delta (expert - naive) per dial, from the fixture thetas.
    def _dial_mean(theta, dial):
        off = ps._offset(dial)
        return float(np.mean([theta[off + mi] for mi in range(len(ps.moods))]))

    true_delta = {d: _dial_mean(th_expert, d) - _dial_mean(th_naive, d)
                  for d in ("v", "z", "u")}
    res = dlg.recover_true_difference(dn, de, ps, true_delta, l2=1.0, seed=seed)
    res = dict(res)
    res["true_delta"] = true_delta
    return res


# ════════════════════════════════════════════════════════════════════════════
# F6 figure
# ════════════════════════════════════════════════════════════════════════════
def make_figure(results, out_path):
    """F6 recovery figure: recovery bars (per dial x regime) + confusion heatmaps +
    a verdict table. Plain-language title/caption. Uses Agg."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})

    regimes = ["expert", "naive"]
    dial_pub = ["sharpness", "itchiness", "timing"]
    dial_show = ["Sharpness\n(v)", "Itchiness/caution\n(z)", "Timing\n(u)"]
    C = {"expert": "#1b7837", "naive": "#d6604d"}

    fig = plt.figure(figsize=(13, 8.5))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1.05, 0.95],
                           hspace=0.45, wspace=0.32,
                           left=0.07, right=0.97, top=0.90, bottom=0.10)

    # ── Panel A: recovery r bars per dial x regime ──
    axA = fig.add_subplot(gs[0, 0])
    x = np.arange(3)
    w = 0.38
    for k, reg in enumerate(regimes):
        rs = [results["point"][reg][d]["r"] for d in dial_pub]
        bars = axA.bar(x + (k - 0.5) * w, rs, w, color=C[reg],
                       label=("Expert-like" if reg == "expert"
                              else "Naive-like"))
        for rect, v in zip(bars, rs):
            if np.isfinite(v):
                axA.text(rect.get_x() + rect.get_width() / 2, v + 0.02,
                         f"{v:.2f}", ha="center", fontsize=8, fontweight="bold")
    axA.axhline(0.8, ls="--", lw=1.4, color="#333333")
    axA.text(2.45, 0.815, "trust line r>=0.8", ha="right", fontsize=8,
             color="#333333")
    axA.set_xticks(x)
    axA.set_xticklabels(dial_show, fontsize=8.5)
    axA.set_ylim(0, 1.12)
    axA.set_ylabel("recovery  (recovered-vs-true r)")
    axA.set_title("A. Can we recover each dial?", fontsize=11, fontweight="bold")
    axA.legend(frameon=False, fontsize=8, loc="lower left", ncol=1)

    # ── Panel B: CCC bars per dial x regime ──
    axB = fig.add_subplot(gs[0, 1])
    for k, reg in enumerate(regimes):
        cs = [results["point"][reg][d].get("ccc", np.nan) for d in dial_pub]
        bars = axB.bar(x + (k - 0.5) * w, cs, w, color=C[reg])
        for rect, v in zip(bars, cs):
            if np.isfinite(v):
                axB.text(rect.get_x() + rect.get_width() / 2, v + 0.02,
                         f"{v:.2f}", ha="center", fontsize=8, fontweight="bold")
    axB.axhline(0.70, ls="--", lw=1.4, color="#333333")
    axB.text(2.45, 0.715, "CCC>=0.70", ha="right", fontsize=8, color="#333333")
    axB.set_xticks(x)
    axB.set_xticklabels(dial_show, fontsize=8.5)
    axB.set_ylim(0, 1.12)
    axB.set_ylabel("Lin's concordance (CCC)")
    axB.set_title("B. Concordance (bias/scale)", fontsize=11, fontweight="bold")

    # ── Panel C: coverage bars per dial x regime ──
    axC = fig.add_subplot(gs[0, 2])
    for k, reg in enumerate(regimes):
        cv = [results["point"][reg][d].get("ci_coverage", np.nan)
              for d in dial_pub]
        bars = axC.bar(x + (k - 0.5) * w, cv, w, color=C[reg])
        for rect, v in zip(bars, cv):
            if np.isfinite(v):
                axC.text(rect.get_x() + rect.get_width() / 2, v + 0.02,
                         f"{v:.2f}", ha="center", fontsize=8, fontweight="bold")
    axC.axhline(0.90, ls="--", lw=1.4, color="#333333")
    axC.text(2.45, 0.915, "coverage>=0.90", ha="right", fontsize=8,
             color="#333333")
    axC.set_xticks(x)
    axC.set_xticklabels(dial_show, fontsize=8.5)
    axC.set_ylim(0, 1.12)
    axC.set_ylabel("parametric-bootstrap CI coverage")
    axC.set_title("C. CI coverage (honesty)", fontsize=11, fontweight="bold")

    # ── Panels D/E: confusion heatmaps per regime ──
    conf_labels = ["sharp.", "caution", "timing"]
    for k, reg in enumerate(regimes):
        ax = fig.add_subplot(gs[1, k])
        M = np.asarray(results["confusion"][reg]["matrix"], float)
        im = ax.imshow(M, cmap="Greens", vmin=0, vmax=1, aspect="equal")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        fontsize=9,
                        color="white" if M[i, j] > 0.5 else "#222222")
        ax.set_xticks(range(3))
        ax.set_xticklabels(conf_labels, fontsize=8)
        ax.set_yticks(range(3))
        ax.set_yticklabels(conf_labels, fontsize=8)
        ax.set_xlabel("identified as")
        ax.set_ylabel("truly varied")
        ax.set_title(f"D{k}. Confusion — {reg}", fontsize=11, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ── Panel F: verdict table ──
    axT = fig.add_subplot(gs[1, 2])
    axT.axis("off")
    axT.set_title("F. Trust verdict (per dial x regime)", fontsize=11,
                  fontweight="bold")
    rows = []
    cell_colors = []
    for reg in regimes:
        verdict = results["gate"][reg]["per_dial_trust"]
        for gd in ("sharpness", "caution", "timing"):
            v = verdict.get(gd, "?")
            rows.append([reg, gd, v])
            col = "#c7e9c0" if v == "generative" else "#fdd0a2"
            cell_colors.append(["#f0f0f0", "#f0f0f0", col])
    tbl = axT.table(cellText=rows,
                    colLabels=["regime", "dial", "trust"],
                    cellColours=cell_colors, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.0, 1.35)

    fig.suptitle(
        "B8 F6 — Can we trust each behavioural dial as a real mechanism?  "
        "(full-power recovery)",
        fontsize=14, fontweight="bold")
    fig.text(0.5, 0.015,
             "A dial is trusted ('generative') only if it recovers (r>=0.8), is "
             "concordant (CCC>=0.70), is honestly covered (>=0.90), and is not "
             "confused with another dial (diag>=0.80, off-diag<=0.20). "
             "Otherwise it falls back to the descriptive Phase-1 proxy.",
             ha="center", fontsize=8.2, color="#555555")

    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[fig] wrote {out_path}", flush=True)


# ════════════════════════════════════════════════════════════════════════════
# JSON helper (numpy -> native)
# ════════════════════════════════════════════════════════════════════════════
def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None  # JSON has no NaN/inf; record as null (still auditable)
    return obj


# ════════════════════════════════════════════════════════════════════════════
# Incremental checkpointing + resume
# ════════════════════════════════════════════════════════════════════════════
# The stages are expensive and the cluster job can time out / be preempted. We
# write recovery_results.json after EVERY completed stage so a kill is never a
# total loss, and resume from it on restart. Only the MAIN process ever writes.
def _write_checkpoint(results, out_dir):
    """Atomically write ``_jsonable(results)`` to ``recovery_results.json``.

    Dump to a ``.tmp`` sibling then ``os.replace`` it onto the final path — the
    replace is atomic on Linux + Windows, so a kill mid-write can never leave a
    half-written / corrupt JSON behind.
    """
    final = os.path.join(out_dir, "recovery_results.json")
    tmp = final + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(_jsonable(results), fh, indent=2)
    os.replace(tmp, final)
    print(f"[checkpoint] wrote {final}", flush=True)


# The SCIENTIFIC-knob signature: the tuple of args that actually change the
# numbers. ``cpus`` and ``out_dir`` are EXCLUDED — seeds are drawn in the master
# process and the imap_unordered reduction is order-independent, so resuming on a
# different core count is bit-identical.
def _config_signature(args):
    return (args.n_rep_point, args.n_rep_confusion, args.n_trials,
            args.n_restarts, args.bootstrap, args.seed, args.quick)


def _config_signature_from_dict(cfg_dict):
    """Same tuple as ``_config_signature`` but from a loaded JSON ``config`` dict."""
    return (cfg_dict.get("n_rep_point"), cfg_dict.get("n_rep_confusion"),
            cfg_dict.get("n_trials"), cfg_dict.get("n_restarts"),
            cfg_dict.get("bootstrap"), cfg_dict.get("seed"),
            cfg_dict.get("quick"))


# Numeric per-dial point fields that the gate/figure consume as floats. After a
# JSON round-trip a NaN was serialised as ``null`` -> Python ``None``; coerce it
# back to ``float('nan')`` on resume so ``np.isfinite`` (gate) and the figure get
# floats, NOT None (which would raise / silently change a sub-check).
_POINT_NUMERIC_FIELDS = ("r", "bias", "sd_true", "ccc", "ci_coverage")


def _coerce_point_nans(point_results):
    """In-place: turn JSON-null point metrics back into float('nan')."""
    for reg, dials in (point_results or {}).items():
        if not isinstance(dials, dict):
            continue
        for dial, metrics in dials.items():
            if not isinstance(metrics, dict):
                continue
            for field in _POINT_NUMERIC_FIELDS:
                if field in metrics and metrics[field] is None:
                    metrics[field] = float("nan")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="B8 Phase 2 full-power recovery harness (cluster).")
    p.add_argument("--out-dir", required=True,
                   help="output directory for recovery_results.json + F6 figure")
    p.add_argument("--n-rep-point", type=int, default=100,
                   help="point-recovery replicates per regime (gate: >=100)")
    p.add_argument("--n-rep-confusion", type=int, default=50,
                   help="confusion replicates per scenario per regime (gate: >=50)")
    p.add_argument("--n-trials", type=int, default=800,
                   help="trials per synthetic anchor (default 800 = the validated "
                        "point-recovery config and above the 600-trial confusion "
                        "floor; matches the real per-anchor operating point closely).")
    p.add_argument("--n-restarts", type=int, default=4,
                   help="fit_anchor random restarts (point: >=4)")
    p.add_argument("--cpus", type=int, default=os.cpu_count(),
                   help="worker processes (default = os.cpu_count())")
    p.add_argument("--seed", type=int, default=0, help="master seed")
    p.add_argument("--bootstrap", type=int, default=500,
                   help="parametric-bootstrap resamples per point fit (gate: >=500)")
    p.add_argument("--quick", action="store_true",
                   help="SMOKE: tiny n_rep/trials/bootstrap, fast (NOT the verdict)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # ── quick smoke overrides (machinery proof only — NOT the published verdict) ──
    if args.quick:
        args.n_rep_point = min(args.n_rep_point, 6)
        args.n_rep_confusion = min(args.n_rep_confusion, 3)
        args.n_trials = min(args.n_trials, 250)
        args.n_restarts = min(args.n_restarts, 2)
        args.bootstrap = min(args.bootstrap, 8)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    dlg, make_recovery_design = preflight()

    print(f"[config] {'QUICK SMOKE' if args.quick else 'FULL'}  "
          f"n_rep_point={args.n_rep_point} n_rep_confusion={args.n_rep_confusion} "
          f"n_trials={args.n_trials} n_restarts={args.n_restarts} "
          f"bootstrap={args.bootstrap} cpus={args.cpus} seed={args.seed}",
          flush=True)
    if args.quick:
        print("[config] *** QUICK is a smoke test: it PROVES THE MACHINERY, "
              "NOT the gate verdict (gate_criteria.md R1). ***", flush=True)

    t_start = time.time()
    regimes = ["expert", "naive"]

    n_workers = max(1, int(args.cpus))

    # ── RESUME: load a prior checkpoint if one exists in this --out-dir ──
    #    Only the MAIN process writes/reads it. The config signature (SCIENTIFIC
    #    knobs only; cpus/out_dir excluded) must MATCH or we refuse — resuming a
    #    different config into the same file would silently mix incompatible runs.
    json_path = os.path.join(out_dir, "recovery_results.json")
    resumed = False
    if os.path.exists(json_path):
        with open(json_path, encoding="utf-8") as fh:
            prior = json.load(fh)
        if _config_signature_from_dict(prior.get("config", {})) != _config_signature(args):
            raise SystemExit(
                f"REFUSING to resume: config in {json_path} differs from requested "
                "args. Use a fresh --out-dir or delete the file.")
        results = prior
        resumed = True
        for k in ("point", "confusion", "veto", "gate"):
            results.setdefault(k, {})
        results["veto"].setdefault("cond", {})
        # coerce any JSON-null (NaN round-trip) back to nan in loaded point metrics
        # so the gate/figure consume floats, not None.
        _coerce_point_nans(results.get("point", {}))
        # refresh metadata only (NOT any scientific sub-tree); keep loaded config.
        results["meta"]["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        results["meta"]["n_workers"] = n_workers
        print(f"[resume] loaded {json_path}; skipping completed stages", flush=True)
    else:
        results = {"config": vars(args),
                   "meta": {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            "visdetect_file": dlg.__file__,
                            "python": platform.python_version(),
                            "numpy": np.__version__,
                            "n_workers": n_workers},
                   "point": {}, "confusion": {}, "veto": {}, "gate": {}}

    # ── process pool (one shared pool for both point + confusion) ──
    pool = None
    if n_workers > 1:
        ctx = mp.get_context("spawn")  # safe on Windows + Linux
        pool = ctx.Pool(processes=n_workers)
        print(f"[pool] {n_workers} worker processes (spawn)", flush=True)
    else:
        print("[pool] serial (cpus=1)", flush=True)

    try:
        # ── shrunk veto: regime-independent (tests the seeding) — compute ONCE ──
        #    Skip if already present (resumed); checkpoint after computing.
        if "truediff" not in results["veto"]:
            results["veto"]["truediff"] = run_truediff(
                dlg, make_recovery_design, n_trials=args.n_trials, seed=args.seed)
            _write_checkpoint(results, out_dir)
        # the gate loop needs it whether fresh or loaded.
        truediff_res = results["veto"]["truediff"]

        designs = {}
        for reg in regimes:
            if reg not in results["point"]:
                point_res, design, true_theta, ps = run_point_recovery(
                    reg, dlg, make_recovery_design,
                    n_rep=args.n_rep_point, n_trials=args.n_trials,
                    n_restarts=args.n_restarts, n_bootstrap=args.bootstrap,
                    seed=args.seed, pool=pool)
                results["point"][reg] = point_res
                designs[reg] = (design, true_theta, ps)
                _write_checkpoint(results, out_dir)
            else:
                # resumed: cheaply rebuild the deterministic (design, true_theta,
                # ps) triple the veto loop needs (identical by construction).
                designs[reg] = tuple(make_recovery_design(
                    reg, n_trials=args.n_trials, seed=args.seed))

        for reg in regimes:
            if reg not in results["confusion"]:
                conf_res = run_confusion(
                    reg, dlg, make_recovery_design,
                    n_rep=args.n_rep_confusion, n_trials=args.n_trials,
                    n_restarts=max(2, args.n_restarts), seed=args.seed, pool=pool)
                results["confusion"][reg] = conf_res
                _write_checkpoint(results, out_dir)

        results["veto"].setdefault("cond", {})
        for reg in regimes:
            # recompute only if BOTH the gate and the cond veto are missing — write
            # cond + gate together so a half-done regime is never persisted.
            if not (reg in results["gate"]
                    and reg in results["veto"].get("cond", {})):
                design, true_theta, ps = designs[reg]
                cond_res = run_vetoes(
                    reg, design, true_theta, ps, dlg, make_recovery_design,
                    n_trials=args.n_trials, n_restarts=args.n_restarts,
                    seed=args.seed)
                results["veto"]["cond"][reg] = cond_res
                results["gate"][reg] = dlg.recovery_gate(
                    results["point"][reg], results["confusion"][reg],
                    truediff_res, cond_res, regime=reg)
                _write_checkpoint(results, out_dir)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    results["meta"]["elapsed_s"] = round(time.time() - t_start, 1)

    # ── write JSON ──
    json_path = os.path.join(args.out_dir, "recovery_results.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(_jsonable(results), fh, indent=2)
    print(f"[json] wrote {json_path}", flush=True)

    # ── write F6 figure ──
    fig_path = os.path.join(args.out_dir, "fig_b8_F6_recovery.png")
    try:
        make_figure(results, fig_path)
    except Exception as exc:  # pragma: no cover - figure is best-effort
        print(f"[fig] WARNING: figure failed ({exc})", flush=True)

    # ── print the verdict summary ──
    print("\n" + "=" * 70)
    print("VERDICT — per (dial x regime) latent_trust")
    print("=" * 70)
    for reg in regimes:
        verdict = results["gate"][reg]["per_dial_trust"]
        line = "  ".join(f"{d}={verdict.get(d, '?')}"
                         for d in ("sharpness", "caution", "timing"))
        print(f"  {reg:7s}: {line}")
    print("=" * 70)
    print(f"elapsed {results['meta']['elapsed_s']}s   "
          f"({'QUICK SMOKE — machinery only' if args.quick else 'FULL — verdict'})")
    print(f"results: {json_path}")
    print(f"figure : {fig_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
