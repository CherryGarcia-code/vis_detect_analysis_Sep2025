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
from visdetect.analysis.config import session_date_key
from visdetect.suite.plotting import setup_style
from visdetect.analysis.evidence_learning_io import (
    CACHE_DIR, FIG_DIR, subject_sessions, tf_responsive_units,
    load_state_labels_by_key)

# Transient/sustained split of TF-responsive cells (method from the vd_tf_bg046
# chat): a hard threshold on the GLM TF-kernel width `kernel_fwhm` (already a
# registry column). transient <= 0.05 s (one 50 ms bin), sustained >= 0.15 s.
# ~60% of cells sit at the 50 ms floor -> "transient" is a coarse, resolution-
# limited bucket; treat the split as a heuristic.
FWHM_TRANSIENT_S = 0.05
FWHM_SUSTAINED_S = 0.15

setup_style()
REGION_POOLS = {"DMS": ("BG_046", "BG_039"), "VMS": ("BG_031",)}
MAX_LAG_S = 1.0      # extended past 0.5 so the DMS peak (near the old edge) is interior
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


def region_kernel_peak_t(subs):
    """(median, q25, q75) of the registry pulse-response kernel_peak_t for the
    responsive cells of a region — the INDEPENDENT pulse-based estimate of these
    cells' TF-response latency, to anchor the continuous-tracking peak lag."""
    vals = []
    for subject in subs:
        fn = subject.replace("_", "").lower()
        reg = pd.read_csv(os.path.join("data", "cache", "tf_responsive",
                                       f"{fn}_tf_responsive.csv"))
        r = reg[reg["resp_log2"] == True]
        if "kernel_peak_t" in r:
            vals.append(np.asarray(r["kernel_peak_t"].dropna(), float))
    if not vals:
        return None
    v = np.concatenate(vals)
    return float(np.median(v)), float(np.percentile(v, 25)), float(np.percentile(v, 75))


def tf_responsive_classes(subject):
    """{session_date_key: {cluster_id: 'transient'|'sustained'|'intermediate'}}
    for responsive cells, from the registry `kernel_fwhm` (vd_tf_bg046 method)."""
    fn = subject.replace("_", "").lower()
    reg = pd.read_csv(os.path.join("data", "cache", "tf_responsive",
                                   f"{fn}_tf_responsive.csv"))
    reg = reg[reg["resp_log2"] == True].copy()
    reg["skey"] = reg["session"].map(session_date_key)

    def _cls(w):
        w = float(w)
        if w <= FWHM_TRANSIENT_S:
            return "transient"
        if w >= FWHM_SUSTAINED_S:
            return "sustained"
        return "intermediate"

    out = {}
    for skey, g in reg.groupby("skey"):
        out[skey] = {int(u): _cls(w) for u, w in zip(g["unit"], g["kernel_fwhm"])}
    return out


def _overlay_kernel_peak_t(ax, kpt):
    """Shade the registry pulse-kernel peak-time IQR + median on a lag axis."""
    if kpt is None:
        return
    med, q1, q3 = kpt
    ax.axvspan(q1, q3, color="green", alpha=0.07, zorder=0)
    ax.axvline(med, color="green", ls=":", lw=1.2,
               label=f"pulse-kernel peak (med {med:.2f}s)")


def main():
    dt = pk.DT
    max_lag = int(round(MAX_LAG_S / dt))
    lags = np.arange(max_lag + 1) * dt
    stats = []
    fig, axes = plt.subplots(len(REGION_POOLS), 3,
                             figsize=(17, 4 * len(REGION_POOLS)), squeeze=False)
    for ri, (region, subs) in enumerate(REGION_POOLS.items()):
        resp_real, resp_shuf, nonresp_real = [], [], []
        transient_real, sustained_real = [], []
        state_curves = {"StimSens": [], "Disengaged": []}   # per-SESSION mean curves
        state_ntr = {"StimSens": 0, "Disengaged": 0}
        for subject in subs:
            resp = tf_responsive_units(subject, responsive=True)
            nonr = tf_responsive_units(subject, responsive=False)
            cls_by = tf_responsive_classes(subject)
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
                # transient vs sustained (registry kernel_fwhm class)
                cmap = cls_by.get(skey, {})
                t_signs = {c: s for c, s in rsigns.items() if cmap.get(c) == "transient"}
                s_signs = {c: s for c, s in rsigns.items() if cmap.get(c) == "sustained"}
                if t_signs:
                    t_real, _, _ = session_tracking(sess, t_signs, rng)
                    if t_real is not None:
                        transient_real.append(t_real)
                if s_signs:
                    s_real, _, _ = session_tracking(sess, s_signs, rng)
                    if s_real is not None:
                        sustained_real.append(s_real)
                # state split: per-session MEAN lag-curve per state (session = the
                # bootstrap unit, so the state panels get session-level CIs)
                labels = load_state_labels_by_key(subject, skey)
                if labels is None:
                    continue
                sess_state = {"StimSens": [], "Disengaged": []}
                for idx, rc in per_trial.items():
                    if idx in labels.index:
                        row = labels.loc[idx]
                        if (float(row["state_confidence"]) >= CONF
                                and row["state_label"] in sess_state):
                            sess_state[row["state_label"]].append(rc)
                for st, cs in sess_state.items():
                    if cs:
                        state_curves[st].append(np.nanmean(np.asarray(cs, float), axis=0))
                        state_ntr[st] += len(cs)
        kpt = region_kernel_peak_t(subs)
        # Panel 1: lag-r curves (real + both nulls), all with bootstrap-over-session CIs
        ax = axes[ri][0]
        _overlay_kernel_peak_t(ax, kpt)
        for curves, col, lab, sig in (
                (resp_real, "C1", "TF-responsive (real)", "responsive_real"),
                (resp_shuf, "0.5", "trial-shuffle null", "responsive_shuffle"),
                (nonresp_real, "C0", "non-responsive (control)", "nonresponsive_real")):
            m, lo, hi = _boot_band(curves)
            if m is None:
                continue
            ls = "--" if sig == "responsive_shuffle" else "-"
            lw = 2 if sig == "responsive_real" else 1.3
            ax.plot(lags, m, color=col, ls=ls, lw=lw, label=lab)
            ax.fill_between(lags, lo, hi, color=col, alpha=0.15)
            pk_i = int(np.nanargmax(m))
            stats.append({"region": region, "signal": sig, "n_sessions": len(curves),
                          "peak_r": float(m[pk_i]), "peak_lag_s": float(lags[pk_i])})
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"{region}: population tracks momentary TF")
        ax.set_xlabel("neural lag behind stimulus (s)")
        ax.set_ylabel("Pearson r (S(t) vs log2-TF)")
        ax.legend(fontsize=7)
        # Panel 2: engagement — per-session state curves with session-level CIs
        ax2 = axes[ri][1]
        _overlay_kernel_peak_t(ax2, kpt)
        for st, col in (("StimSens", "#6baed6"), ("Disengaged", "#3474ae")):
            m, lo, hi = _boot_band(state_curves[st])
            if m is None:
                continue
            ax2.plot(lags, m, color=col, lw=1.6,
                     label=f"{st} ({len(state_curves[st])} sess, {state_ntr[st]} tr)")
            ax2.fill_between(lags, lo, hi, color=col, alpha=0.18)
            pk_i = int(np.nanargmax(m))
            stats.append({"region": region, "signal": f"track_{st}",
                          "n_sessions": len(state_curves[st]),
                          "peak_r": float(m[pk_i]), "peak_lag_s": float(lags[pk_i])})
        ax2.axhline(0, color="k", lw=0.5)
        ax2.set_title(f"{region}: tracking by engagement state")
        ax2.set_xlabel("neural lag behind stimulus (s)")
        ax2.set_ylabel("Pearson r (S(t) vs log2-TF)")
        ax2.legend(fontsize=7)
        # Panel 3: tracking by TF-kernel class (transient vs sustained responders)
        ax3 = axes[ri][2]
        _overlay_kernel_peak_t(ax3, kpt)
        for curves, col, lab in (
                (transient_real, "#e6550d", "transient (fwhm<=50ms)"),
                (sustained_real, "#31a354", "sustained (fwhm>=150ms)")):
            m, lo, hi = _boot_band(curves)
            if m is None:
                continue
            ax3.plot(lags, m, color=col, lw=1.6, label=f"{lab} ({len(curves)} sess)")
            ax3.fill_between(lags, lo, hi, color=col, alpha=0.18)
            pk_i = int(np.nanargmax(m))
            stats.append({"region": region, "signal": f"track_{lab.split()[0]}",
                          "n_sessions": len(curves), "peak_r": float(m[pk_i]),
                          "peak_lag_s": float(lags[pk_i])})
        ax3.axhline(0, color="k", lw=0.5)
        ax3.set_title(f"{region}: tracking by TF-kernel class")
        ax3.set_xlabel("neural lag behind stimulus (s)")
        ax3.set_ylabel("Pearson r (S(t) vs log2-TF)")
        ax3.legend(fontsize=7)
    fig.text(0.5, -0.01, "Baseline-only, stimulus-referenced (leakage-safe). Bands = 95% "
             "bootstrap CI over sessions. Green = registry pulse-kernel peak time (median "
             "+ IQR) — the independent pulse-based estimate of these cells' TF latency.",
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
