"""B10 (positive) — the striatum carries a real-time readout of the grating speed.

Does the signed TF-responsive population signal S(t) track the MOMENTARY baseline
log2-TF fluctuation, in real time, at the neural integration lag (~200-250 ms)?
Controls: (i) a trial-SHUFFLE null (S from trial i vs stimulus from a random other
trial) — kills any non-time-specific correlation; (ii) NON-responsive cells (should
sit near 0). Upside: is tracking fidelity higher when the animal is engaged
(StimSens vs Disengaged)?

Leakage-safe by construction: baseline-period only, stimulus-referenced (no lick/
motor alignment) -> immune to the N1 leakage trap and B10's motor-gain confound.

Run: py scripts/evidence_learning/b10_tf_tracking.py
Out: FIGURES/evidence_learning/tracking/b10_tf_tracking.png,
     data/cache/evidence_learning/b10_tf_tracking_stats.csv
"""
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if os.path.join(_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visdetect.analysis import psychophysical_kernel as pk
from visdetect.suite.plotting import setup_style
from visdetect.analysis.evidence_learning_io import (
    CACHE_DIR, FIG_DIR, subject_sessions, tf_responsive_units,
    load_state_labels_by_key)

setup_style()
REGION_POOLS = {"DMS": ("BG_046", "BG_039"), "VMS": ("BG_031",)}
MAX_LAG_S = 0.5
# The neural response INTEGRATES the stimulus over ~250 ms (Khilkevich-Lohse), so
# correlate at that timescale, not raw 50 ms bins (Poisson-noise dominated).
SMOOTH_SIGMA_S = 0.15
CONF = 0.8


def session_tracking(session, signs, rng, max_lag_s=MAX_LAG_S, dt=pk.DT):
    """Per session: (real_curve, shuffle_curve, {trial_idx: real_lag_curve}).

    S(t) and y(t) are smoothed to the integration timescale (SMOOTH_SIGMA_S)
    before correlating. real = mean over trials of corr(S_i, y_i(t-lag));
    shuffle = corr(S_i, y_j(t-lag)) with j a random OTHER trial (time-broken).
    The per-trial dict holds each trial's full lag curve (for the state split)."""
    max_lag = int(round(max_lag_s / dt))
    sig_bins = SMOOTH_SIGMA_S / dt
    sig = pk.signed_population_signal(session, signs)
    if not sig:
        return None, None, {}
    smooth = lambda v: gaussian_filter1d(v, sig_bins) if v.size > 1 else v
    Ss = {idx: smooth(S) for idx, (_, S) in sig.items()}
    ys = {idx: smooth(pk.baseline_log2tf(session.trials[idx], dt=dt)[1]) for idx in sig}
    idxs = list(sig.keys())
    real, shuf, per_trial = [], [], {}
    for idx in idxs:
        S = Ss[idx]
        y = ys[idx]
        n = min(len(S), len(y))
        if n < max_lag + 5:
            continue
        rc = pk.stimulus_tracking_xcorr(S[:n], y[:n], max_lag)
        real.append(rc)
        per_trial[idx] = rc
        j = idxs[int(rng.integers(len(idxs)))]
        yj = ys[j]
        nj = min(len(S), len(yj))
        if nj >= max_lag + 5:
            shuf.append(pk.stimulus_tracking_xcorr(S[:nj], yj[:nj], max_lag))
    real = np.nanmean(real, axis=0) if real else None
    shuf = np.nanmean(shuf, axis=0) if shuf else None
    return real, shuf, per_trial


def _boot_band(curves, n_boot=1000, seed=pk.BOOT_SEED):
    """mean + 95% band, bootstrapping over the per-session curves (sessions=unit)."""
    a = np.array([c for c in curves if c is not None], float)
    if a.shape[0] == 0:
        return None, None, None
    mean = np.nanmean(a, axis=0)
    if a.shape[0] == 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    boots = np.array([np.nanmean(a[rng.integers(0, a.shape[0], a.shape[0])], axis=0)
                      for _ in range(n_boot)])
    lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
    return mean, lo, hi


def main():
    dt = pk.DT
    max_lag = int(round(MAX_LAG_S / dt))
    lags = np.arange(max_lag + 1) * dt
    stats = []
    fig, axes = plt.subplots(len(REGION_POOLS), 2,
                             figsize=(12, 4 * len(REGION_POOLS)), squeeze=False)
    for ri, (region, subs) in enumerate(REGION_POOLS.items()):
        resp_real, resp_shuf, nonresp_real = [], [], []
        state_curves = {"StimSens": [], "Disengaged": []}
        for subject in subs:
            resp = tf_responsive_units(subject, responsive=True)
            nonr = tf_responsive_units(subject, responsive=False)
            rng = np.random.default_rng(pk.BOOT_SEED)
            for skey, sname, stage, sess in subject_sessions(subject):
                rsigns = resp.get(skey, {})
                if not rsigns:
                    continue
                r_real, r_shuf, per_trial = session_tracking(sess, rsigns, rng)
                if r_real is not None:
                    resp_real.append(r_real)
                if r_shuf is not None:
                    resp_shuf.append(r_shuf)
                nall = nonr.get(skey, {})
                if nall:
                    keys = list(nall)
                    k = min(len(rsigns), len(keys))
                    pick = rng.choice(len(keys), k, replace=False)
                    nsigns = {keys[i]: nall[keys[i]] for i in pick}
                    n_real, _, _ = session_tracking(sess, nsigns, rng)
                    if n_real is not None:
                        nonresp_real.append(n_real)
                # state split: accumulate each trial's full lag curve by state
                labels = load_state_labels_by_key(subject, skey)
                if labels is not None:
                    for idx, rc in per_trial.items():
                        if idx in labels.index:
                            row = labels.loc[idx]
                            if (float(row["state_confidence"]) >= CONF
                                    and row["state_label"] in state_curves):
                                state_curves[row["state_label"]].append(rc)
        # Panel 1: lag-r curves
        ax = axes[ri][0]
        m, lo, hi = _boot_band(resp_real)
        if m is not None:
            ax.plot(lags, m, color="C1", lw=2, label="TF-responsive (real)")
            ax.fill_between(lags, lo, hi, color="C1", alpha=0.2)
            pk_i = int(np.nanargmax(m))
            stats.append({"region": region, "signal": "responsive_real",
                          "n_sessions": len(resp_real), "peak_r": float(m[pk_i]),
                          "peak_lag_s": float(lags[pk_i])})
        ms, _, _ = _boot_band(resp_shuf)
        if ms is not None:
            ax.plot(lags, ms, color="0.5", ls="--", label="trial-shuffle null")
            stats.append({"region": region, "signal": "responsive_shuffle",
                          "n_sessions": len(resp_shuf),
                          "peak_r": float(np.nanmax(ms)), "peak_lag_s": np.nan})
        mn, _, _ = _boot_band(nonresp_real)
        if mn is not None:
            ax.plot(lags, mn, color="C0", label="non-responsive (control)")
            stats.append({"region": region, "signal": "nonresponsive_real",
                          "n_sessions": len(nonresp_real),
                          "peak_r": float(np.nanmax(mn)), "peak_lag_s": np.nan})
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"{region}: population tracks momentary TF")
        ax.set_xlabel("neural lag behind stimulus (s)")
        ax.set_ylabel("Pearson r (S(t) vs log2-TF)")
        ax.legend(fontsize=8)
        # Panel 2: engagement — mean lag curve for engaged (StimSens) vs Disengaged
        # trials (proper per-condition curves, NOT a per-trial max).
        ax2 = axes[ri][1]
        for st, col in (("StimSens", "#6baed6"), ("Disengaged", "#3474ae")):
            curves = state_curves[st]
            if curves:
                m = np.nanmean(np.asarray(curves, float), axis=0)
                ax2.plot(lags, m, color=col, label=f"{st} (n={len(curves)})")
                pk_i = int(np.nanargmax(m))
                stats.append({"region": region, "signal": f"track_{st}",
                              "n_sessions": len(curves), "peak_r": float(m[pk_i]),
                              "peak_lag_s": float(lags[pk_i])})
        ax2.axhline(0, color="k", lw=0.5)
        ax2.set_title(f"{region}: tracking by engagement state")
        ax2.set_xlabel("neural lag behind stimulus (s)")
        ax2.set_ylabel("Pearson r (S(t) vs log2-TF)")
        ax2.legend(fontsize=8)
    fig.text(0.5, -0.01, "Baseline-only, stimulus-referenced (leakage-safe). Real vs "
             "trial-shuffle null vs non-responsive control. Region labels provisional.",
             ha="center", fontsize=8, style="italic")
    fig.tight_layout()
    outdir = os.path.join(FIG_DIR, "tracking")
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, "b10_tf_tracking.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    os.makedirs(CACHE_DIR, exist_ok=True)
    pd.DataFrame(stats).to_csv(
        os.path.join(CACHE_DIR, "b10_tf_tracking_stats.csv"), index=False)
    print(pd.DataFrame(stats))


if __name__ == "__main__":
    main()
