"""Parallel-determinism regression for ``learning_ladder`` (B8 Phase-2 follow-up #1).

WHY THIS EXISTS
---------------
``learning_ladder(..., n_workers>1)`` fans its expensive per-rung restart fits and
per-rung k-fold CV-LL refits across a spawn-context ``ProcessPoolExecutor``. The
function is *built* so its returned numbers do NOT depend on the worker count:

  * every random restart init is pre-generated PARENT-SIDE from one seeded RNG
    stream per rung (workers never touch an RNG);
  * the restart reduction tiebreaks on init INDEX, not arrival order
    (``_reduce_restart_results``);
  * ``pool.map`` preserves task order; and
  * the module self-pins BLAS to one thread at import, so the elementwise hazard
    math is bit-stable in freshly-spawned workers.

The B8 Phase-2 NEURAL phase reuses these fitters at ``n_workers>1`` (~12 locally,
~128 on the SLURM cluster). If a future refactor silently broke determinism —
moved an RNG draw into a worker, dropped the index tiebreak, or introduced an
order-dependent float reduction — the fitted latents would change with the
machine's core count and the neural regression would stop reproducing. This test
LOCKS the guarantee in CI: byte-identical ``aic``/``bic``/``cvll``/``ll`` and the
same ``winner`` across ``n_workers`` in {1, 2, 3}.

``state_ladder`` has NO internal worker fan-out (it is sequential; the
orchestration's parallelism over ``state_ladder`` is per-anchor and keyed, hence
order-independent), so its determinism is already covered by
``test_state_ladder_is_seed_reproducible`` in
``test_decision_latents_generative.py``.
"""
import numpy as np
import pandas as pd
import pytest

from visdetect.analysis import decision_latents_generative as dlg
from visdetect.analysis.decision_latents import MAIN_MOODS


def _anchor_design(ps, v_level, *, n_trials=120, seed=7, sim_seed=201,
                   step=1.2, noise=0.2, dt=0.05, go_p=0.75):
    """A compact identifiable two-mood simulated Design (mirrors the generative
    tests' ``_ramp_anchor_design``): a fluctuating baseline log2-TF plus a
    post-change positive excursion on go trials so all three dials leave a
    survival signature; ``z`` very negative so trials survive to the excursion;
    moods alternating by trial. Returned with simulated outcomes so the ladder can
    fit it."""
    rng = np.random.default_rng(seed)
    rows, change_times = [], []
    for tidx in range(n_trials):
        n_bins = int(rng.integers(30, 61))          # 1.5-3.0 s on the dt grid
        ct = float(rng.uniform(0.5, 1.2))
        change_times.append(ct)
        go = bool(rng.random() < go_p)
        ev = rng.normal(0.0, noise, size=n_bins)
        if go:
            t = np.arange(n_bins) * dt
            ev = ev + np.where(t >= ct, step, 0.0)
        rows.append(dict(trial_idx=tidx, outcome="hit" if go else "miss",
                         change_size=2.0 if go else 1.0, change_time=ct,
                         decision_time=n_bins * dt, lick=1, censored=False,
                         evidence=ev, n_bins=n_bins))
    ev_df = pd.DataFrame(rows)
    labels = pd.DataFrame({
        "trial_idx": np.arange(n_trials),
        "state_label": [MAIN_MOODS[i % len(MAIN_MOODS)] for i in range(n_trials)],
    }).set_index("trial_idx")
    design = dlg.build_design(ev_df, labels, mu=float(np.median(change_times)),
                             sigma=0.8, dt=dt)
    # true_theta = [v_Imp, v_Stim, z_Imp, z_Stim, u_Imp, u_Stim]
    true_theta = np.array([v_level, v_level, -4.0, -4.0, 0.4, 0.3])
    eb, lk, cs = dlg.simulate_licks(design, true_theta, ps, seed=sim_seed)
    return dlg.design_with_outcomes(design, eb, lk, cs)


@pytest.fixture(scope="module")
def two_anchors():
    """(param_spec, anchor_designs) — two anchors with a true ``v`` ramp. Built
    once per module (the designs are deterministic)."""
    ps = dlg.ParamSpec(moods=("Impulsive", "StimSens"),
                       dials=("v", "z", "u"), state_terms=("v", "z", "u"))
    designs = {
        "A_low":  _anchor_design(ps, 0.5, sim_seed=201),
        "A_high": _anchor_design(ps, 1.3, sim_seed=202),
    }
    return ps, designs


def _assert_byte_identical(par, base, keys, n_workers):
    for key in keys:
        assert par[key] == base[key], (
            f"learning_ladder['{key}'] NOT byte-identical at n_workers={n_workers} "
            f"vs sequential:\n  base (n_workers=1): {base[key]}\n"
            f"  parallel (n_workers={n_workers}): {par[key]}")


def test_learning_ladder_byte_identical_with_cvll(two_anchors):
    """Full path (restart fan-out AND per-rung CV-LL fan-out): n_workers=2 must
    return byte-identical aic/bic/cvll/ll and the same winner as sequential."""
    ps, designs = two_anchors
    kw = dict(dt=0.05, k=2, seed=0, n_restarts=2, return_ll=True, compute_cvll=True)
    base = dlg.learning_ladder(designs, ps, n_workers=1, **kw)
    par = dlg.learning_ladder(designs, ps, n_workers=2, **kw)
    assert par["winner"] == base["winner"]
    _assert_byte_identical(par, base, ("aic", "bic", "cvll", "ll"), n_workers=2)


@pytest.mark.parametrize("n_workers", [2, 3])
def test_learning_ladder_byte_identical_restart_fanout(two_anchors, n_workers):
    """Restart fan-out only (compute_cvll=False — the AIC fast path the
    orchestration uses inside ``recover_confusion``): byte-identical aic/bic/ll and
    the same winner across worker counts; cvll stays NaN placeholders."""
    ps, designs = two_anchors
    kw = dict(dt=0.05, k=2, seed=0, n_restarts=2, return_ll=True, compute_cvll=False)
    base = dlg.learning_ladder(designs, ps, n_workers=1, **kw)
    par = dlg.learning_ladder(designs, ps, n_workers=n_workers, **kw)
    assert par["winner"] == base["winner"]
    _assert_byte_identical(par, base, ("aic", "bic", "ll"), n_workers=n_workers)
    assert all(np.isnan(v) for v in par["cvll"].values())
