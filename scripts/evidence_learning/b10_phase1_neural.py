"""B10 Phase 1 (neural) — the neural impulsivity kernel on TF-responsive cells.

Fig B10.2: (A) signed-population TF signal, FA vs withhold + CI; (B) Naive vs
Expert; (C) sensory-vs-gain decomposition (stimulus-matched control). DMS pool
(BG_046+039) + VMS (BG_031) separate. Stimulus-referenced (motor-safe); per
session then aggregated over sessions.

Run: py scripts/evidence_learning/b10_phase1_neural.py
Out: FIGURES/evidence_learning/neural/b10_neural_kernel.png,
     data/cache/evidence_learning/b10_neural_kernel_stats.csv
"""
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if os.path.join(_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visdetect.analysis import psychophysical_kernel as pk
from visdetect.suite.plotting import setup_style
from visdetect.analysis.evidence_learning_io import (
    CACHE_DIR, FIG_DIR, subject_sessions, tf_responsive_units)

setup_style()
HEADLINE = ("Naive", "Expert")
REGION_POOLS = {"DMS": ("BG_046", "BG_039"), "VMS": ("BG_031",)}
MIN_POOL_PAIRS = 20


def _window_from_signal(S, lick_t, dt=pk.DT):
    """Slice the len-L pre-lick window from a per-trial signed signal S."""
    L = len(pk.kernel_lags(dt))
    j1 = int(round((lick_t - pk.KERNEL_REFRACTORY_S) / dt))
    j0 = j1 - L
    if j0 < 0 or j1 > S.size:
        return None
    return S[j0:j1].copy()


def neural_fa_withhold(session, unit_signs, rng):
    """Paired FA vs matched-withhold NEURAL windows (+ their stimulus windows).

    Returns (fa_pop, wh_pop, fa_stim, wh_stim) — equal-length lists of len-L
    arrays. Neural signal = signed_population_signal on TF-responsive cells."""
    if not unit_signs:
        return [], [], [], []
    sig = pk.signed_population_signal(session, unit_signs)
    eps = pk.fa_kernel_epochs(session)
    wh = pk.withhold_epochs(session, eps, rng=rng)
    # (widx, change_time) for hit/miss trials — indices via enumerate, NOT
    # list.index (Trial.__eq__ compares numpy fields -> ambiguous truth value).
    widx_ct = []
    for widx, tr in enumerate(session.trials):
        oc = (getattr(tr, "trialoutcome", "") or "").lower()
        ct = float(getattr(tr, "change_time", np.nan) or np.nan)
        if oc in ("hit", "miss") and np.isfinite(ct):
            widx_ct.append((widx, ct))
    fa_pop, wh_pop, fa_stim, wh_stim = [], [], [], []
    for e, w in zip(eps, wh):
        if w is None or e["trial_idx"] not in sig:
            continue
        _, S = sig[e["trial_idx"]]
        fa_win = _window_from_signal(S, e["lick_t"])
        wh_win = None
        for widx, ct in widx_ct:
            if widx in sig and ct - pk.KERNEL_REFRACTORY_S >= e["lick_t"]:
                wh_win = _window_from_signal(sig[widx][1], e["lick_t"])
                if wh_win is not None:
                    break
        if fa_win is not None and wh_win is not None:
            fa_pop.append(fa_win)
            wh_pop.append(wh_win)
            fa_stim.append(e["window"])
            wh_stim.append(w)
    return fa_pop, wh_pop, fa_stim, wh_stim


def main():
    lags = pk.kernel_lags()
    stats = []
    fig, axes = plt.subplots(len(REGION_POOLS), 3, figsize=(15, 8), squeeze=False)
    for ri, (region, subs) in enumerate(REGION_POOLS.items()):
        pooled = {s: ([], [], [], []) for s in HEADLINE}
        for subject in subs:
            tf = tf_responsive_units(subject)
            rng = np.random.default_rng(pk.BOOT_SEED)
            for skey, sname, stage, sess in subject_sessions(subject, HEADLINE):
                if stage not in pooled:
                    continue
                fp, wp, fs, ws = neural_fa_withhold(sess, tf.get(skey, {}), rng)
                for tgt, src in zip(pooled[stage], (fp, wp, fs, ws)):
                    tgt.extend(src)
        for s in HEADLINE:
            fp, wp, fs, ws = pooled[s]
            if len(fp) < MIN_POOL_PAIRS:
                continue
            k, lo, hi = pk.bootstrap_kernel_ci(fp, wp)
            dec = pk.stimulus_matched_control(fs, ws, fp, wp)
            axes[ri][0].plot(lags, k, label=s)
            axes[ri][0].fill_between(lags, lo, hi, alpha=0.2)
            axes[ri][1].plot(lags, k, label=s)
            axes[ri][2].plot(lags, dec["sensory"], label=f"{s} sensory")
            axes[ri][2].plot(lags, dec["gain"], "--", label=f"{s} gain")
            stats.append({"region": region, "stage": s, "n_pairs": len(fp),
                          **pk.kernel_shape_metrics(k)})
        axes[ri][0].axhline(0, color="k", lw=0.5)
        axes[ri][0].set_title(f"{region}: neural FA vs withhold")
        axes[ri][0].set_ylabel("signed pop. TF signal (z)")
        axes[ri][0].legend()
        axes[ri][1].axhline(0, color="k", lw=0.5)
        axes[ri][1].set_title(f"{region}: Naive vs Expert")
        axes[ri][1].legend()
        axes[ri][2].axhline(0, color="k", lw=0.5)
        axes[ri][2].set_title(f"{region}: sensory vs gain")
        axes[ri][2].legend(fontsize=7)
        for c in range(3):
            axes[ri][c].set_xlabel("time before recorded lick (s)")
    fig.text(0.5, -0.01, "n=1 region for VMS; per-session-then-aggregate; region "
             "labels provisional (region_bank_confirmed pending).",
             ha="center", fontsize=8, style="italic")
    fig.tight_layout()
    outdir = os.path.join(FIG_DIR, "neural")
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, "b10_neural_kernel.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    os.makedirs(CACHE_DIR, exist_ok=True)
    pd.DataFrame(stats).to_csv(
        os.path.join(CACHE_DIR, "b10_neural_kernel_stats.csv"), index=False)
    print(pd.DataFrame(stats))


if __name__ == "__main__":
    main()
