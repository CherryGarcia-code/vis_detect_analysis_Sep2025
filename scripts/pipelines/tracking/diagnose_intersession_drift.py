#!/usr/bin/env python3
"""Estimate inter-session probe drift for BG_046, from average waveforms.

Why this exists
---------------
Cross-session unit tracking (UnitMatch / DeepUnitMatch) tolerates probe drift
only up to ~100 um (UnitMatch ``max_dist``). If the probe shifts more than that
between sessions, a unit's footprint leaves its expected channels and the match
is lost. Both tools *do* correct drift internally, but the correction is
bootstrapped from putative matches -- large drift starves that bootstrap.

This script measures the drift directly, with no matching, so we know whether
the failed tracking is drift-limited (=> seed the correction) or not
(=> the failure is sort consistency / features instead).

Method
------
For each session, every unit's average waveform (UnitMatch input format,
shape (82, n_chan, 2)) gives a peak channel and a peak-to-peak amplitude.
Each unit is dropped into a depth (probe-y) histogram weighted by its
amplitude -> a per-session "amplitude-depth fingerprint" of the probe.
Consecutive fingerprints are cross-correlated over a range of rigid y-shifts;
the argmax shift is the estimated inter-session drift. Done whole-probe and
per-shank (NP2.0 4-shank: shanks separated by probe-x).

Pooling every unit's amplitude makes the fingerprint robust to which
individual units are present -- the failure mode that breaks match-based
drift estimates on low-yield early sessions.

Usage
-----
    py scripts/pipelines/tracking/diagnose_intersession_drift.py
    py scripts/pipelines/tracking/diagnose_intersession_drift.py --input <dir>
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = REPO_ROOT / "data" / "unit_match" / "input" / "BG_046"
OUT_DIR = REPO_ROOT / "FIGURES" / "tracking_qc"

DEPTH_BIN_UM = 10.0          # depth histogram bin
MAX_SHIFT_UM = 300.0         # search +/- this rigid shift between sessions
SMOOTH_BINS = 2              # gaussian-ish smoothing of the fingerprint (in bins)


def parse_session_date(name: str) -> datetime:
    """Session dir names are DDMMYYYY (leading zero on day may be missing)."""
    s = str(name)
    if len(s) == 7:
        s = "0" + s
    return datetime.strptime(s, "%d%m%Y")


def session_fingerprint(sess_dir: Path, y_edges: np.ndarray,
                        shank_of_chan: np.ndarray):
    """Amplitude-depth fingerprint for one session, whole-probe and per-shank.

    Every channel of every unit contributes its peak-to-peak amplitude to the
    depth bin of that channel -- i.e. each unit's *full spatial footprint* is
    used, not just its peak channel. Pooling the whole footprint of every unit
    makes the fingerprint dense and far less sensitive to which units the sort
    happened to find.

    Returns (whole_profile, {shank: profile}, n_units).
    """
    chan_pos = np.load(sess_dir / "channel_positions.npy")   # (n_chan, 2) = x, y
    y = chan_pos[:, 1]
    wf_files = sorted((sess_dir / "RawWaveforms").glob("*.npy"))

    n_bins = len(y_edges) - 1
    chan_bin = np.clip(np.searchsorted(y_edges, y) - 1, 0, n_bins - 1)
    whole = np.zeros(n_bins)
    shanks = sorted(set(shank_of_chan.tolist()))
    per_shank = {sh: np.zeros(n_bins) for sh in shanks}

    for f in wf_files:
        wave = np.load(f)
        if wave.ndim != 3 or wave.shape[1] != len(y):
            continue
        w = wave.mean(axis=2)                       # average 2 CV halves -> (T, n_chan)
        ptp = w.max(axis=0) - w.min(axis=0)         # per-channel peak-to-peak
        np.add.at(whole, chan_bin, ptp)
        for sh in shanks:
            m = shank_of_chan == sh
            np.add.at(per_shank[sh], chan_bin[m], ptp[m])

    return whole, per_shank, len(wf_files)


def smooth(profile: np.ndarray, n: int) -> np.ndarray:
    if n <= 0:
        return profile
    k = np.exp(-0.5 * (np.arange(-3 * n, 3 * n + 1) / n) ** 2)
    k /= k.sum()
    return np.convolve(profile, k, mode="same")


def estimate_shift(ref: np.ndarray, mov: np.ndarray, max_lag_bins: int):
    """Rigid y-shift (in bins) that best aligns ``mov`` onto ``ref``.

    Positive shift => ``mov`` is deeper (higher y) than ``ref``.
    Returns (best_shift_bins, peak_normalised_correlation).
    """
    ref = ref - ref.mean()
    mov = mov - mov.mean()
    denom = np.sqrt((ref ** 2).sum() * (mov ** 2).sum())
    if denom < 1e-9:
        return 0, 0.0
    lags = np.arange(-max_lag_bins, max_lag_bins + 1)
    best_lag, best_c = 0, -np.inf
    for lag in lags:
        shifted = np.roll(mov, lag)
        if lag > 0:
            shifted[:lag] = 0
        elif lag < 0:
            shifted[lag:] = 0
        c = float((ref * shifted).sum() / denom)
        if c > best_c:
            best_c, best_lag = c, lag
    return best_lag, best_c


def assign_shanks(chan_x: np.ndarray) -> np.ndarray:
    """Label each channel 0..3 by which NP2.0 shank its x-position belongs to."""
    xs = np.sort(np.unique(chan_x))
    # shanks are well separated; split on the largest x-gaps
    gaps = np.diff(xs)
    split_x = xs[np.argsort(gaps)[-3:]] if len(xs) > 4 else xs[:0]
    bounds = np.sort(split_x) + 1e-6
    return np.searchsorted(bounds, chan_x)


def main():
    ap = argparse.ArgumentParser(description="Inter-session probe drift diagnostic")
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                    help="UnitMatch input dir with per-session RawWaveforms")
    ap.add_argument("--no-fig", action="store_true")
    args = ap.parse_args()

    sess_dirs = sorted([d for d in args.input.iterdir() if d.is_dir()],
                       key=lambda d: parse_session_date(d.name))
    if not sess_dirs:
        raise SystemExit(f"No session dirs under {args.input}")
    print(f"{len(sess_dirs)} sessions: {sess_dirs[0].name} ... {sess_dirs[-1].name}")

    # Common depth grid + shank assignment from the first session's geometry
    chan_pos0 = np.load(sess_dirs[0] / "channel_positions.npy")
    y_lo, y_hi = chan_pos0[:, 1].min(), chan_pos0[:, 1].max()
    y_edges = np.arange(y_lo - DEPTH_BIN_UM, y_hi + 2 * DEPTH_BIN_UM, DEPTH_BIN_UM)
    shank_of_chan = assign_shanks(chan_pos0[:, 0])
    shanks = sorted(set(shank_of_chan.tolist()))
    max_lag = int(MAX_SHIFT_UM / DEPTH_BIN_UM)
    print(f"depth grid {y_lo:.0f}-{y_hi:.0f} um, {len(y_edges)-1} bins, "
          f"{len(shanks)} shanks, search +/-{MAX_SHIFT_UM:.0f} um")

    whole_profiles, shank_profiles, n_units = [], [], []
    for d in sess_dirs:
        w, ps, nu = session_fingerprint(d, y_edges, shank_of_chan)
        whole_profiles.append(smooth(w, SMOOTH_BINS))
        shank_profiles.append({k: smooth(v, SMOOTH_BINS) for k, v in ps.items()})
        n_units.append(nu)
        print(f"  {d.name:<10} {nu:4d} units")

    # Consecutive-session drift, whole probe
    rows = []
    cum = 0.0
    for i, d in enumerate(sess_dirs):
        if i == 0:
            rows.append(dict(session=d.name, n_units=n_units[i],
                             step_um=0.0, cum_um=0.0, corr=1.0))
            continue
        lag, corr = estimate_shift(whole_profiles[i - 1], whole_profiles[i], max_lag)
        step = lag * DEPTH_BIN_UM
        cum += step
        rows.append(dict(session=d.name, n_units=n_units[i],
                         step_um=step, cum_um=cum, corr=corr))
    df = pd.DataFrame(rows)

    # Per-shank consecutive drift
    for sh in shanks:
        col = []
        for i in range(len(sess_dirs)):
            if i == 0:
                col.append(0.0)
                continue
            lag, _ = estimate_shift(shank_profiles[i - 1][sh],
                                    shank_profiles[i][sh], max_lag)
            col.append(lag * DEPTH_BIN_UM)
        df[f"step_shank{sh}_um"] = col

    # Drift vs a fixed reference (first session) -- absolute displacement
    ref_lag = []
    for i in range(len(sess_dirs)):
        lag, _ = estimate_shift(whole_profiles[0], whole_profiles[i], max_lag)
        ref_lag.append(lag * DEPTH_BIN_UM)
    df["drift_vs_ref0_um"] = ref_lag

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "intersession_drift.csv"
    df.to_csv(csv_path, index=False)

    # ---- summary ----
    steps = df["step_um"].abs().values[1:]
    print("\n" + "=" * 64)
    print("INTER-SESSION DRIFT SUMMARY")
    print("=" * 64)
    print(f"  consecutive-session |step|: median {np.median(steps):.0f} um, "
          f"mean {steps.mean():.0f} um, max {steps.max():.0f} um")
    print(f"  steps > 50 um:  {(steps > 50).sum()}/{len(steps)}")
    print(f"  steps > 100 um: {(steps > 100).sum()}/{len(steps)}   "
          f"(UnitMatch max_dist budget)")
    print(f"  total drift range vs session 0: "
          f"[{min(ref_lag):.0f}, {max(ref_lag):.0f}] um  "
          f"(span {max(ref_lag) - min(ref_lag):.0f} um)")
    med_corr = df['corr'].values[1:]
    print(f"  fingerprint alignment corr: median {np.median(med_corr):.2f}")
    print("\n  interpretation:")
    if np.median(steps) > 50 or (steps > 100).mean() > 0.15:
        print("  -> LARGE inter-session drift. Likely a primary cause of poor")
        print("     matching. Seed UnitMatch/DeepUM with these shift estimates.")
    else:
        print("  -> Modest inter-session drift. Drift is NOT the main cause;")
        print("     look to per-session sort consistency / waveform features.")
    print(f"\nCSV -> {csv_path}")

    if not args.no_fig:
        _make_figure(df, sess_dirs, whole_profiles, y_edges, shanks)


def _make_figure(df, sess_dirs, whole_profiles, y_edges, shanks):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle("BG_046 inter-session probe drift (amplitude-depth fingerprint)",
                 fontsize=13)
    x = np.arange(len(sess_dirs))
    labels = [d.name for d in sess_dirs]

    ax = axes[0, 0]
    ax.bar(x, df["step_um"], color="steelblue")
    ax.axhline(0, color="k", lw=0.6)
    ax.axhline(100, color="r", ls="--", lw=0.8, label="+/-100 um (UM budget)")
    ax.axhline(-100, color="r", ls="--", lw=0.8)
    ax.set_ylabel("step drift (um)"); ax.set_title("Consecutive-session drift")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=90, fontsize=5)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(x, df["drift_vs_ref0_um"], "o-", color="darkorange", ms=4)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("drift vs session 0 (um)")
    ax.set_title("Cumulative drift vs first session")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=90, fontsize=5)

    ax = axes[1, 0]
    P = np.array(whole_profiles).T
    yc = 0.5 * (y_edges[:-1] + y_edges[1:])
    im = ax.imshow(P, aspect="auto", origin="lower", cmap="magma",
                   extent=[0, len(sess_dirs), yc[0], yc[-1]])
    ax.set_xlabel("session"); ax.set_ylabel("probe depth (um)")
    ax.set_title("Amplitude-depth fingerprint per session")
    plt.colorbar(im, ax=ax, label="summed amplitude", shrink=0.8)

    ax = axes[1, 1]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]
    for sh in shanks:
        ax.plot(x, np.cumsum(df[f"step_shank{sh}_um"]), "o-", ms=3,
                color=colors[sh % 4], label=f"shank {sh}", alpha=0.8)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("cumulative drift (um)"); ax.set_xlabel("session")
    ax.set_title("Per-shank cumulative drift")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=90, fontsize=5)
    ax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUT_DIR / "intersession_drift.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure -> {out}")


if __name__ == "__main__":
    main()
