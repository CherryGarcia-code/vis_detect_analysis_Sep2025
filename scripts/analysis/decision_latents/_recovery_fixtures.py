"""B8 Phase 2 — Task 3.0: shared recovery ground-truth fixtures (Engine A).

Plain English: every recovery test (3.2-3.5) — the make-or-break gate for the
generative decision-latent model — needs a *known answer* to recover. This
module builds that known answer: two synthetic but realistic anchors, an
**expert-like** one and a **naive-like** one, each as a ragged
:class:`~visdetect.analysis.decision_latents_generative.Design` on per-trial
log2-TF evidence at ``dt=0.05`` PLUS the ``true_theta`` that generated it.

The two regimes are deliberately different (contract §A.9):

* **expert-like** — *change-driven* licks. Higher sharpness ``v`` (in the
  IDENTIFIABLE band ~1.2-1.4) so the post-change evidence excursion shapes lick
  timing, and a LOWER baseline itchiness ``z`` (very negative -> low baseline
  hazard) so trials SURVIVE long enough to reach that excursion. This is the
  regime where ``v`` is well-identified.

* **naive-like** — *flat-evidence hair-trigger* licks. LOW sharpness ``v``
  (~0.3, the accumulator barely matters) and HIGH itchiness ``z`` (less
  negative -> high baseline hazard, early licks). This is deliberately the HARD
  regime where ``v`` is weakly identified — that is the point: recovery must
  still be TESTED here (and is expected to be weaker; the per-dial gate in Task
  3.5 acts on it).

Both anchors carry BOTH moods (Impulsive / StimSens) so the per-mood dials are
exercised. The evidence realisation is controlled by a SHARED per-regime design
seed, and the returned ``true_theta`` matches the returned ``ParamSpec`` layout
exactly (``len(true_theta) == param_spec.n_params()``).

Evidence construction mirrors the existing decisive recovery tests' synthesis
(``tests/analysis/test_decision_latents_generative.py::
_identifiable_recovery_design``) but with the contract's ``change_time >= 6 s``:
each trial is a fluctuating zero-mean log2-TF baseline with, on go trials, a
sustained positive excursion (a TF *increase*) after a sampled change time. We
do NOT use the buggy ``ddm.build_trial_evidence`` sampler.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from visdetect.analysis import decision_latents_generative as dlg
from visdetect.analysis.decision_latents import MAIN_MOODS

# ── Per-regime ground-truth dials (theta laid out per ParamSpec as
#    [v_Imp, v_Stim, z_Imp, z_Stim, u_Imp, u_Stim]). ──
#
# expert: high v (identifiable band), very negative z (low baseline hazard so
#         trials survive to the post-change excursion -> v identifiable).
# naive : low v (accumulator barely matters), less-negative z (hair-trigger,
#         early licks on flat evidence). u kept modest in both.
#
# z is tuned so BOTH regimes simulate a non-degenerate lick/censor mix
# (lick rate in [0.30, 0.75]); naive sits higher on lick rate than expert.
_REGIME_THETA = {
    "expert": np.array([1.3, 1.2, -6.0, -6.2, 0.4, 0.3], float),
    "naive":  np.array([0.3, 0.3, -5.2, -5.3, 0.3, 0.3], float),
}

# Evidence-synthesis settings per regime. ``go_p`` = fraction of go trials (with
# a post-change excursion); ``step`` = log2-TF excursion height; ``noise`` =
# baseline log2-TF fluctuation SD.
_REGIME_EVIDENCE = {
    # expert sees a strong, reliable change excursion (change-driven licking)
    "expert": {"go_p": 0.8, "step": 1.5, "noise": 0.25},
    # naive sees a weaker, less frequent excursion -> licking is mostly flat-
    # evidence / hair-trigger, not change-driven
    "naive":  {"go_p": 0.5, "step": 0.6, "noise": 0.25},
}

# change_time >= this (seconds) per contract §A.9 / Task 3.0 brief.
_CHANGE_TIME_MIN_S = 6.0
_CHANGE_TIME_MAX_S = 8.0
# trial length sampled to comfortably outlast the change so go trials can reach
# the excursion (expert) and there is room for the urgency bump to bite.
_TRIAL_LEN_MIN_S = 8.0
_TRIAL_LEN_MAX_S = 11.0


def _build_evidence_frame(regime, n_trials, dt, rng):
    """Synthesize the per-trial evidence frame + mood labels for a regime.

    Returns ``(ev_df, labels, mu)`` where ``ev_df`` is the
    ``build_trial_evidence_corrected``-shaped frame, ``labels`` is the
    trial-indexed mood frame (alternating MAIN_MOODS), and ``mu`` is the median
    change time (the temporal-expectation anchor for the urgency bump).
    """
    cfg = _REGIME_EVIDENCE[regime]
    go_p, step, noise = cfg["go_p"], cfg["step"], cfg["noise"]

    rows = []
    change_times = []
    for tidx in range(n_trials):
        # trial length in bins (>= change_time so go trials reach the excursion)
        dur_s = float(rng.uniform(_TRIAL_LEN_MIN_S, _TRIAL_LEN_MAX_S))
        n_bins = int(round(dur_s / dt))
        # change_time >= 6 s (contract §A.9), kept below the trial length
        ct = float(rng.uniform(_CHANGE_TIME_MIN_S, _CHANGE_TIME_MAX_S))
        change_times.append(ct)

        go = bool(rng.random() < go_p)
        # fluctuating zero-mean baseline log2-TF evidence
        ev = rng.normal(0.0, noise, size=n_bins)
        if go:
            t_grid = np.arange(n_bins) * dt
            # sustained positive excursion (a TF increase) after change_time
            ev = ev + np.where(t_grid >= ct, step, 0.0)
        rows.append({
            "trial_idx": tidx,
            "outcome": "hit" if go else "miss",
            "change_size": 2.0 if go else 1.0,
            "change_time": ct,
            "decision_time": n_bins * dt,
            "lick": 1,                 # placeholder (simulate_licks overwrites)
            "censored": False,         # placeholder
            "evidence": ev,
            "n_bins": n_bins,
        })

    ev_df = pd.DataFrame(rows)
    labels = pd.DataFrame({
        "trial_idx": np.arange(n_trials),
        "state_label": [MAIN_MOODS[i % len(MAIN_MOODS)] for i in range(n_trials)],
    }).set_index("trial_idx")
    mu = float(np.median(change_times))
    return ev_df, labels, mu


def make_recovery_design(regime, n_trials=2000, seed=0):
    """Build a recovery ground-truth ``(Design, true_theta, ParamSpec)`` triple.

    Parameters
    ----------
    regime : {"expert", "naive"}
        ``"expert"`` -> change-driven licks (higher ``v``, lower ``z``);
        ``"naive"``  -> flat-evidence hair-trigger licks (low ``v``, high ``z``).
    n_trials : int
        Number of synthetic trials (alternating Impulsive / StimSens moods).
    seed : int
        Seed for the SHARED per-regime evidence realisation (so the Design is
        reproducible and a regime's evidence is controlled).

    Returns
    -------
    (Design, np.ndarray, ParamSpec)
        The ragged Design (on real per-trial evidence at ``dt=0.05`` with a
        change excursion at ``change_time >= 6 s``), the ``true_theta`` that
        generated it (length == ``param_spec.n_params()``), and the matching
        ParamSpec.

    Raises
    ------
    ValueError
        If ``regime`` is not one of ``"expert"`` / ``"naive"``.
    """
    if regime not in _REGIME_THETA:
        raise ValueError(
            f"unknown regime {regime!r}; expected one of "
            f"{sorted(_REGIME_THETA)}")

    dt = dlg.DT_GEN
    param_spec = dlg.ParamSpec(
        moods=("Impulsive", "StimSens"),
        dials=("v", "z", "u"),
        state_terms=("v", "z", "u"),
    )
    true_theta = _REGIME_THETA[regime].copy()
    assert len(true_theta) == param_spec.n_params()

    rng = np.random.default_rng(seed)
    ev_df, labels, mu = _build_evidence_frame(regime, n_trials, dt, rng)

    design = dlg.build_design(
        ev_df, labels, mu=mu, sigma=param_spec.urgency_sigma, dt=dt,
        leak_tau=param_spec.leak_tau, rectification=param_spec.rectification,
    )
    return design, true_theta, param_spec
