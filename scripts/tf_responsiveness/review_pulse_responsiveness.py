"""Careful review: COMMON-MODE TF pulse responsiveness (not selectivity).

The selectivity gate (fast-minus-slow) cancels any response common to fast and
slow pulses by design. This script asks the *other* question -- does a unit
respond to a TF pulse AT ALL -- which is what the old tf_pulse classifier (and
its "Omni" tier) measured, and what survives the historical off-by-one (that
bug scrambled fast/slow *labels*, not pulse *times*).

For each good-and-stable unit it builds the all-pulse-triggered mean rate over a
wide window, z-scores it to a far pre-pulse baseline, and reports the post-pulse
deflection. It also keeps fast and slow separately so we can see, for the
strongest responders, (a) the response SHAPE -- a sharp transient after t=0
(sensory) vs a slow monotonic ramp (temporal expectation) -- and (b) whether the
common-mode responders show any fast-vs-slow selectivity.

Also runs two audit checks: that "fast" pulses really sit on high-TF samples
(validates the corrected indexing), and the population-mean pulse response.

Usage:
    cd /e/python_analysis/git_repos/vd_tf_phase0
    PYTHONPATH=src py scripts/tf_responsiveness/review_pulse_responsiveness.py --session 16092025
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visdetect.core.session import load_session as core_load
from visdetect.suite.loader import load_session
from visdetect.analysis.utils import get_good_cluster_ids
from visdetect.analysis.tf_pulse import (
    TFRespPulseConfig, _collect_pulses, _smooth_binned_activity, _safe_log2,
)
from visdetect.analysis.align import get_event_times_by_trial
from visdetect.analysis.constants import LOHSE_SENSORY_CD_WINDOW

_ROOT = Path(__file__).resolve().parents[2]
_FIGS = _ROOT / "figures" / "tf_responsiveness"
_CACHE = _ROOT / "data" / "cache" / "tf_selectivity"

# Wide window so we can see ramp-vs-transient shape; z baseline is far pre-pulse.
W0, W1, DT, SIG = -0.5, 0.5, 0.004, 17.0
BASE = (-0.5, -0.3)          # far pre-pulse baseline for the z-score
POST = (0.02, 0.25)          # post-pulse sensory search window
RAMP_PRE = (-0.30, -0.02)    # immediately-pre-pulse window (to gauge anticipation)


def _tvec():
    return np.arange(W0, W1, DT)


def _trig_mean_hz(st, pulses, tv):
    """All-pulse-triggered mean rate (Hz)."""
    i0 = np.searchsorted(st, pulses + float(tv[0]))
    i1 = np.searchsorted(st, pulses + float(tv[-1] + DT))
    acc = np.zeros(tv.size)
    sb = (SIG / 1000.0) / DT
    for k in range(pulses.size):
        acc += _smooth_binned_activity(st[i0[k]:i1[k]] - pulses[k], tv, sb)
    return acc / max(pulses.size, 1) / DT


def _zscore(trace, tv, base):
    m = (tv >= base[0]) & (tv < base[1])
    mu = float(np.mean(trace[m])); sd = float(np.std(trace[m]))
    if not np.isfinite(sd) or sd <= 1e-6:
        sd = 1.0
    return (trace - mu) / sd, mu, sd


def audit_fast_pulses(sess, fast, slow, cfg, n=400):
    """Confirm fast pulses really land on high-TF samples (corrected indexing)."""
    base = np.array(get_event_times_by_trial(sess, "Baseline_ON"), float)
    # Build a global (time -> log2 TF) lookup by replaying _collect_pulses' math.
    times, l2s = [], []
    for i, t in enumerate(sess.trials):
        bv = getattr(t, "baseline_values", None)
        if bv is None or i >= base.size or not np.isfinite(base[i]):
            continue
        arr = np.asarray(bv).ravel()
        if cfg.baseline_stride > 1:
            arr = arr[::cfg.baseline_stride]
        l2 = _safe_log2(arr)
        t0 = float(base[i])
        for b, v in enumerate(l2):
            if np.isfinite(v):
                times.append(t0 + b * cfg.sample_period); l2s.append(v)
    times = np.array(times); l2s = np.array(l2s)
    order = np.argsort(times); times = times[order]; l2s = l2s[order]

    def lookup(p):
        j = np.searchsorted(times, p)
        j = np.clip(j, 0, times.size - 1)
        return l2s[j]

    fz = lookup(fast[:n]); sz = lookup(slow[:n])
    print(f"[audit] fast-pulse log2TF: median={np.median(fz):+.3f} "
          f"frac>=+0.25={np.mean(fz >= 0.25):.2f}  (should be ~1.0)")
    print(f"[audit] slow-pulse log2TF: median={np.median(sz):+.3f} "
          f"frac<=-0.25={np.mean(sz <= -0.25):.2f}  (should be ~1.0)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--ntop", type=int, default=16)
    args = ap.parse_args()

    try:
        sess = load_session(args.session)
    except Exception:
        sess = core_load(f"data/pkls/BG_046/BG_046_{args.session}.pkl")
    cids = get_good_cluster_ids(sess)
    cfg = TFRespPulseConfig(trace_pre=W0, dt=DT, post_window=(0.0, W1))
    fast, slow = _collect_pulses(sess, cfg)
    allp = np.concatenate([fast, slow])
    print(f"[review] {args.session}: {len(cids)} units, {fast.size} fast / "
          f"{slow.size} slow / {allp.size} all pulses")
    audit_fast_pulses(sess, fast, slow, cfg)

    by = {int(c.cluster_id): np.sort(np.asarray(c.spike_times, float).ravel())
          for c in sess.clusters}
    tv = _tvec()
    rows, cm_traces, fast_traces, slow_traces = [], {}, {}, {}
    pop = np.zeros(tv.size)
    for cid in cids:
        st = by.get(int(cid))
        if st is None or st.size == 0:
            continue
        cm = _trig_mean_hz(st, allp, tv)
        cmz, mu, sd = _zscore(cm, tv, BASE)
        pm = (tv >= POST[0]) & (tv < POST[1])
        rm = (tv >= RAMP_PRE[0]) & (tv < RAMP_PRE[1])
        post_peak = float(cmz[pm][np.argmax(np.abs(cmz[pm]))])
        pre_ramp = float(np.mean(cmz[rm]))     # anticipation level just before pulse
        # "transient" = how much the post-pulse peak exceeds the immediately-pre ramp
        transient = post_peak - pre_ramp
        rows.append(dict(cluster_id=int(cid), baseline_hz=mu, cm_post_peak_z=post_peak,
                         cm_pre_ramp_z=pre_ramp, cm_transient_z=transient,
                         peak_latency=float(tv[pm][np.argmax(np.abs(cmz[pm]))])))
        cm_traces[int(cid)] = cmz
        pop += cmz
    df = pd.DataFrame(rows)
    pop /= max(len(cm_traces), 1)

    n3 = int((df["cm_post_peak_z"].abs() >= 3).sum())
    n5 = int((df["cm_post_peak_z"].abs() >= 5).sum())
    nt3 = int((df["cm_transient_z"].abs() >= 3).sum())
    print(f"[review] common-mode post-pulse |z|>=3: {n3}/{len(df)}  |z|>=5: {n5}")
    print(f"[review] TRANSIENT (post-peak minus pre-ramp) |z|>=3: {nt3}/{len(df)} "
          f"(these are pulse responses NOT explained by the anticipatory ramp)")

    _CACHE.mkdir(parents=True, exist_ok=True)
    df.sort_values("cm_post_peak_z", key=np.abs, ascending=False).to_csv(
        _CACHE / f"{args.session}_pulse_responsiveness.csv", index=False)

    # fast/slow traces for the top common-mode responders (for the figure)
    top = df.reindex(df["cm_post_peak_z"].abs().sort_values(ascending=False).index)
    top_ids = top["cluster_id"].head(args.ntop).tolist()
    for cid in top_ids:
        st = by[int(cid)]
        fz, _, _ = _zscore(_trig_mean_hz(st, fast, tv), tv, BASE)
        sz, _, _ = _zscore(_trig_mean_hz(st, slow, tv), tv, BASE)
        fast_traces[int(cid)] = fz; slow_traces[int(cid)] = sz

    ncol = 4; nrow = int(np.ceil(args.ntop / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.6 * nrow),
                             sharex=True, squeeze=False)
    for idx, cid in enumerate(top_ids):
        ax = axes[idx // ncol][idx % ncol]
        ax.plot(tv, cm_traces[cid], "k-", lw=1.7, label="all pulses")
        ax.plot(tv, fast_traces[cid], color="#1f77b4", lw=0.9, label="fast")
        ax.plot(tv, slow_traces[cid], color="#d62728", lw=0.9, label="slow")
        ax.axvline(0, color="r", ls=":", lw=0.8)
        ax.axvspan(*LOHSE_SENSORY_CD_WINDOW, color="orange", alpha=0.15)
        ax.axhline(0, color="0.7", lw=0.5)
        r = df[df["cluster_id"] == cid].iloc[0]
        ax.set_title(f"clu {cid}  z={r.cm_post_peak_z:.1f} trans={r.cm_transient_z:.1f}",
                     fontsize=8)
        if idx % ncol == 0:
            ax.set_ylabel("z (vs far pre)")
        if idx == 0:
            ax.legend(fontsize=6, loc="upper left")
    for j in range(len(top_ids), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(f"Common-mode TF pulse responsiveness (corrected pulses) — "
                 f"{args.session}: top {args.ntop} by post-pulse |z|", y=1.0)
    fig.tight_layout()
    _FIGS.mkdir(parents=True, exist_ok=True)
    out = _FIGS / f"{args.session}_pulse_responsiveness.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"[review] wrote {out}")

    # population-mean common-mode response (is there ANY pulse response on average?)
    fig2, ax = plt.subplots(figsize=(6, 4))
    ax.plot(tv, pop, "k-", lw=2)
    ax.axvline(0, color="r", ls=":", lw=0.8); ax.axhline(0, color="0.7", lw=0.5)
    ax.axvspan(*LOHSE_SENSORY_CD_WINDOW, color="orange", alpha=0.15)
    ax.set_xlabel("time from pulse (s)"); ax.set_ylabel("mean z (n=%d units)" % len(cm_traces))
    ax.set_title(f"Population-mean common-mode pulse response — {args.session}")
    fig2.tight_layout()
    out2 = _FIGS / f"{args.session}_pulse_response_population.png"
    fig2.savefig(out2, dpi=140); plt.close(fig2)
    print(f"[review] wrote {out2}")


if __name__ == "__main__":
    main()
