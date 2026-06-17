"""Phase 0: validate the drift correction on a few exemplar units.

Eyeball gate before full-session scaling (Plan 2). Renders raw vs detrended
fast/slow pulse traces with the circular-shift null envelope, and reports the
pre-pulse slope before/after per unit.

Usage:
  py scripts/tf_responsiveness/validate_drift_phase0.py \
     --session 7072025 --clusters 42 108 211 --kernel-s 20 --out phase0.png
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visdetect.analysis.constants import (
    TF_PULSE_PRE_WINDOW, TF_PULSE_POST_WINDOW, TF_PULSE_TRACE_PRE,
)
from visdetect.analysis.tf_pulse import (
    _collect_pulses, _zscore_trace, TFRespPulseConfig,
)
from visdetect.analysis.tf_drift import (
    estimate_drift, detrended_pulse_average, prepulse_slope, circular_shift_null,
)

PRE = TF_PULSE_PRE_WINDOW
POST = TF_PULSE_POST_WINDOW
DT = 0.005
SIGMA_MS = 20.0
BIN_S = 1.0
# Extend the KDE support window well before the measurement window so the
# Gaussian smoothing has full support at -0.4 s; the pre-pulse SLOPE is then
# measured on the clean interior (PRE), not a boundary artifact at the edge.
TRACE_START = TF_PULSE_TRACE_PRE  # -1.0 s


def _raw_pulse_average_hz(spike_times, pulses):
    from visdetect.analysis.tf_pulse import _mean_activity_per_unit
    mean_fine, sem, t_vec = _mean_activity_per_unit(
        spike_times, pulses, PRE, POST, DT, SIGMA_MS, trace_start=TRACE_START)
    if mean_fine.size == 0:
        return mean_fine, t_vec
    return mean_fine / DT, t_vec


def run_units(session, cluster_ids, kernel_s=20.0, n_shuffles=100, out_png="phase0.png",
              use_constraints=True):
    """Validate drift correction for the given clusters; returns a list of dict rows.

    ``use_constraints`` defaults to True (the scientifically-correct pulse-selection
    guards used on real sessions and by the manual Phase-0 eyeball gate). The smoke
    test sets it False because the toy synthetic session's short baseline geometry
    cannot satisfy the real guards; the drift-correction pipeline downstream of
    pulse selection is identical either way.
    """
    cfg = TFRespPulseConfig(use_constraints=use_constraints)
    fast_pulses, slow_pulses = _collect_pulses(session, cfg)
    spikes_by_cid = {int(c.cluster_id): np.asarray(c.spike_times, float).ravel()
                     for c in session.clusters}

    rows = []
    n = len(cluster_ids)
    fig, axes = plt.subplots(n, 2, figsize=(11, 2.6 * max(n, 1)), squeeze=False)
    for r, cid in enumerate(cluster_ids):
        st = np.sort(spikes_by_cid.get(int(cid), np.array([])))
        # Last-spike proxy for session duration — fine for the eyeball gate; Plan 2 should use the true recording length.
        sess_dur = float(st.max()) if st.size else 1.0
        gt, dr, mr = estimate_drift(st, 0.0, sess_dur, bin_s=BIN_S, kernel_s=kernel_s)
        for col, (pulses, label) in enumerate(
                [(fast_pulses, "fast ▲"), (slow_pulses, "slow ▼")]):
            ax = axes[r][col]
            raw_hz, t_vec = _raw_pulse_average_hz(st, pulses)
            det_hz, _, t_det = detrended_pulse_average(
                st, pulses, PRE, POST, DT, SIGMA_MS, gt, dr, mr,
                trace_start=TRACE_START)
            if raw_hz.size == 0 or det_hz.size == 0:
                ax.text(0.5, 0.5, "no pulses", ha="center", transform=ax.transAxes)
                continue
            null_z, t_null = circular_shift_null(
                st, pulses, PRE, POST, DT, SIGMA_MS, BIN_S, kernel_s,
                session_dur=sess_dur, n_shuffles=n_shuffles, seed=0,
                trace_start=TRACE_START)
            raw_z = _zscore_trace(raw_hz, t_vec, PRE)
            det_z = _zscore_trace(det_hz, t_det, PRE)
            lo = np.percentile(null_z, 5, axis=0)
            hi = np.percentile(null_z, 95, axis=0)
            ax.fill_between(t_null, lo, hi, color="0.75", alpha=0.5, lw=0,
                            label="null 5-95%")
            ax.plot(t_det, raw_z, "k--", lw=1.0, label="raw")
            ax.plot(t_det, det_z, "k-", lw=1.6, label="detrended")
            ax.axvline(0, color="0.5", lw=0.7, ls=":")
            ax.axhline(0, color="0.6", lw=0.4)
            ax.set_xlim(-0.55, POST[1])  # hide the extended KDE-support region
            ax.set_title(f"clu{cid} {label}", fontsize=9)
            if r == 0 and col == 0:
                ax.legend(fontsize=6, loc="upper left")
            post_mask = (t_det >= 0.0) & (t_det < POST[1])
            rows.append({
                "cluster_id": int(cid), "direction": label.split()[0],
                "slope_raw": prepulse_slope(raw_z, t_vec, PRE),
                "slope_detrended": prepulse_slope(det_z, t_det, PRE),
                "post_peak_raw": float(np.nanmax(np.abs(raw_z[post_mask]))) if post_mask.any() else float("nan"),
                "post_peak_det": float(np.nanmax(np.abs(det_z[post_mask]))) if post_mask.any() else float("nan"),
            })
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return rows


def main():
    from visdetect.suite.loader import load_session
    ap = argparse.ArgumentParser(description="Phase 0 drift-correction validation")
    ap.add_argument("--session", required=True)
    ap.add_argument("--clusters", type=int, nargs="+", required=True)
    ap.add_argument("--kernel-s", type=float, default=20.0)
    ap.add_argument("--n-shuffles", type=int, default=100)
    ap.add_argument("--out", default="figures/tf_responsiveness/phase0_drift.png")
    args = ap.parse_args()

    sess = load_session(args.session)
    rows = run_units(sess, args.clusters, kernel_s=args.kernel_s,
                     n_shuffles=args.n_shuffles, out_png=args.out)
    print(f"\n  {'cluster':>8s} {'dir':>5s} {'slope_raw':>12s} {'slope_detr':>12s}")
    for r in rows:
        print(f"  {r['cluster_id']:8d} {r['direction']:>5s} "
              f"{r['slope_raw']:12.3f} {r['slope_detrended']:12.3f}")
    print(f"\n  Figure: {args.out}")


if __name__ == "__main__":
    main()
