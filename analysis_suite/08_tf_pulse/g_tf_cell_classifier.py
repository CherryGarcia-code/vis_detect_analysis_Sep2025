"""Fig41: TF-responsive cell classification — permutation-based tiered system.

Tiered classification of units by their response to baseline TF pulses:
  Tier 1 (Splitter):       Significant to BOTH fast & slow with opposite sign
  Tier 2 (Unilateral):     Significant to fast XOR slow only
  Tier 3 (Omni):           Significant to BOTH fast & slow with same sign
  Non-responsive:           Neither significant

Significance is assessed by circular-shift permutation test (default N=500,
alpha=0.01).  Pre-screening from cached NPZ traces skips units with
|z| < 1.5 to save time.

Metrics per unit: signed peak z, signed AUC, half-width, mirror score.
Quality filtering uses AUC to favour prolonged over transient responses.

Produces:
  cache/tf_cell_classification.csv  (per-unit classification + metrics)
  figures/08_tf_pulse/fig41_tf_cell_classification.png  (6-panel summary)

Usage:
  .venv\\Scripts\\python.exe analysis_suite/08_tf_pulse/g_tf_cell_classifier.py
  .venv\\Scripts\\python.exe analysis_suite/08_tf_pulse/g_tf_cell_classifier.py --no-perm
  .venv\\Scripts\\python.exe analysis_suite/08_tf_pulse/g_tf_cell_classifier.py --n-perms 1000 --alpha 0.005
"""

import argparse
import gc
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.ndimage import gaussian_filter1d

# numpy compat: trapezoid was added in numpy 2.0, older versions have trapz
_trapezoid = getattr(np, "trapezoid", np.trapz)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import (
    STAGE_ORDER, STAGE_COLORS, CELLTYPE_COLORS, CACHE_DIR,
    DEFAULT_Z_THRESH_TF,
)
from visdetect.analysis.constants import (
    TF_PULSE_PRE_WINDOW, TF_PULSE_POST_WINDOW,
    TF_DETREND_BASELINE, TF_DETREND_POST_WINDOW,
)
from visdetect.analysis.tf_pulse import detrend_tf_traces
from loader import (
    load_staging_manifest, load_session, load_waveform_labels,
    load_tf_traces_npz,
)
from plotting import setup_style, save_figure

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

setup_style()

# ── Constants ─────────────────────────────────────────────────────────
DT = 0.001
SIGMA_MS = 17.0
PRE_WIN = TF_PULSE_PRE_WINDOW
POST_WIN = TF_PULSE_POST_WINDOW
PRE_SCREEN_Z = 1.5
MIN_PULSES = 20
DEFAULT_N_PERMS = 500
DEFAULT_ALPHA = 0.01
MIN_SHIFT_S = 30.0

# Tier labels
TIER_SPLITTER = "Tier 1 (Splitter)"
TIER_UNILATERAL = "Tier 2 (Unilateral)"
TIER_OMNI = "Tier 3 (Omni)"
TIER_NONE = "Non-responsive"

TIER_COLORS = {
    TIER_SPLITTER: "#8E24AA",
    TIER_UNILATERAL: "#FB8C00",
    TIER_OMNI: "#43A047",
    TIER_NONE: "#BDBDBD",
}


# ── Helper functions (module-level for ProcessPoolExecutor on Windows) ─

def _vectorized_psth(spike_times, pulses, t_vec, sigma_bins):
    """Fast pulse-triggered histogram with Gaussian smoothing."""
    if pulses.size == 0:
        return np.zeros_like(t_vec)
    dt = t_vec[1] - t_vec[0]
    full0, full1 = t_vec[0], t_vec[-1] + dt
    all_rel = []
    for tp in pulses:
        lo = np.searchsorted(spike_times, tp + full0)
        hi = np.searchsorted(spike_times, tp + full1)
        if hi > lo:
            all_rel.append(spike_times[lo:hi] - tp)
    if not all_rel:
        return np.zeros_like(t_vec)
    all_rel = np.concatenate(all_rel)
    counts, _ = np.histogram(all_rel, bins=np.append(t_vec, t_vec[-1] + dt))
    rate = counts.astype(float) / pulses.size
    return gaussian_filter1d(rate, sigma=sigma_bins)


def _zscore_simple(trace, t_vec, pre_win):
    """Z-score trace using pre-window mean and std."""
    pre_mask = (t_vec >= pre_win[0]) & (t_vec < pre_win[1])
    mu = float(np.nanmean(trace[pre_mask])) if np.any(pre_mask) else 0.0
    sd = float(np.nanstd(trace[pre_mask])) if np.any(pre_mask) else 0.0
    if not np.isfinite(sd) or sd <= 0:
        return trace * 0.0
    return (trace - mu) / sd


def _signed_peak(z_post):
    """Value at the index of maximum absolute z-score."""
    if z_post.size == 0 or np.all(np.isnan(z_post)):
        return 0.0
    idx = np.nanargmax(np.abs(z_post))
    return float(z_post[idx])


def _peak_latency_ms(z_post, t_post):
    """Time (ms) of peak absolute z-score in the post window."""
    if z_post.size == 0 or np.all(np.isnan(z_post)):
        return np.nan
    idx = np.nanargmax(np.abs(z_post))
    return float(t_post[idx]) * 1000.0


def _half_width_ms(z_trace, t_vec, post_win):
    """FWHM of the dominant response peak in ms."""
    post_mask = (t_vec >= post_win[0]) & (t_vec < post_win[1])
    z_post = z_trace[post_mask]
    if z_post.size == 0 or np.all(np.isnan(z_post)):
        return np.nan
    pk_idx = np.nanargmax(np.abs(z_post))
    half_amp = np.abs(z_post[pk_idx]) / 2.0
    if half_amp == 0:
        return np.nan
    above = np.abs(z_post) >= half_amp
    if not np.any(above):
        return np.nan
    first = int(np.argmax(above))
    last = z_post.size - 1 - int(np.argmax(above[::-1]))
    dt_ms = (t_vec[1] - t_vec[0]) * 1000
    return max(float((last - first) * dt_ms), dt_ms)


def _assign_tier(sig_fast, sig_slow, sig_fast_conj, sig_slow_conj,
                 sign_fast, sign_slow):
    """Assign (tier, sub_type) from significance flags and response sign.

    Splitter / Omni require BOTH directions significant at the conjunction
    alpha (relaxed), while Unilateral requires a single direction at the
    strict alpha.  This avoids the over-conservative alpha^2 penalty on the
    conjunction test.
    """
    if sig_fast_conj and sig_slow_conj:
        if sign_fast != sign_slow:
            sub = "Fast+/Slow-" if sign_fast > 0 else "Slow+/Fast-"
            return TIER_SPLITTER, sub
        else:
            sub = "Both+" if sign_fast > 0 else "Both-"
            return TIER_OMNI, sub
    elif sig_fast:
        sub = "Fast+" if sign_fast > 0 else "Fast-"
        return TIER_UNILATERAL, sub
    elif sig_slow:
        sub = "Slow+" if sign_slow > 0 else "Slow-"
        return TIER_UNILATERAL, sub
    return TIER_NONE, "None"


def _classify_single_unit(args):
    """Worker: compute real metrics + circular-shift permutation test.

    Returns dict of scalar metrics for one unit.
    """
    (cid, spike_times, fast_pulses, slow_pulses, t_vec, sigma_bins,
     pre_win, post_win, n_perms, rec_duration, seed, do_detrend,
     detrend_bl, detrend_pw) = args

    post_mask = (t_vec >= post_win[0]) & (t_vec < post_win[1])
    t_post = t_vec[post_mask]
    spike_times = np.sort(spike_times)

    def _maybe_detrend(z_trace):
        """Apply linear detrend if active, via library wrapper."""
        if not do_detrend:
            return z_trace
        dt, _, _ = detrend_tf_traces(
            t_vec, z_trace[np.newaxis, :],
            baseline_window=detrend_bl, post_window=detrend_pw)
        return dt[0]

    # ── Real PSTHs ───────────────────────────────────────────────
    real_fast_z = _maybe_detrend(_zscore_simple(
        _vectorized_psth(spike_times, fast_pulses, t_vec, sigma_bins),
        t_vec, pre_win))
    real_slow_z = _maybe_detrend(_zscore_simple(
        _vectorized_psth(spike_times, slow_pulses, t_vec, sigma_bins),
        t_vec, pre_win))

    fz_post = real_fast_z[post_mask]
    sz_post = real_slow_z[post_mask]

    # Signed peak z-scores
    signed_peak_fast = _signed_peak(fz_post)
    signed_peak_slow = _signed_peak(sz_post)
    abs_peak_fast = abs(signed_peak_fast)
    abs_peak_slow = abs(signed_peak_slow)

    # Signed AUC (integral of z-score over post-window)
    signed_auc_fast = float(_trapezoid(fz_post, t_post)) if fz_post.size else 0.0
    signed_auc_slow = float(_trapezoid(sz_post, t_post)) if sz_post.size else 0.0

    # Mirror score: corr(fast, -slow) in post-window
    if fz_post.size > 5 and np.nanvar(fz_post) > 0 and np.nanvar(sz_post) > 0:
        mirror = float(np.corrcoef(fz_post, -sz_post)[0, 1])
    else:
        mirror = np.nan

    # Half-widths
    hw_fast = _half_width_ms(real_fast_z, t_vec, post_win)
    hw_slow = _half_width_ms(real_slow_z, t_vec, post_win)

    # ── Permutation test ─────────────────────────────────────────
    if n_perms > 0 and rec_duration > 0:
        rng = np.random.default_rng(seed)
        null_peaks_fast = np.zeros(n_perms)
        null_peaks_slow = np.zeros(n_perms)
        null_aucs_fast = np.zeros(n_perms)
        null_aucs_slow = np.zeros(n_perms)

        min_shift = max(MIN_SHIFT_S, rec_duration * 0.05)
        max_shift = rec_duration - min_shift
        if max_shift <= min_shift:
            max_shift = rec_duration * 0.95
        shifts = rng.uniform(min_shift, max_shift, size=n_perms)

        for pi in range(n_perms):
            shifted = np.sort((spike_times + shifts[pi]) % rec_duration)

            sf_z = _maybe_detrend(_zscore_simple(
                _vectorized_psth(shifted, fast_pulses, t_vec, sigma_bins),
                t_vec, pre_win))
            ss_z = _maybe_detrend(_zscore_simple(
                _vectorized_psth(shifted, slow_pulses, t_vec, sigma_bins),
                t_vec, pre_win))

            sf_post = sf_z[post_mask]
            ss_post = ss_z[post_mask]

            null_peaks_fast[pi] = float(np.nanmax(np.abs(sf_post))) if sf_post.size else 0.0
            null_peaks_slow[pi] = float(np.nanmax(np.abs(ss_post))) if ss_post.size else 0.0
            null_aucs_fast[pi] = abs(float(_trapezoid(sf_post, t_post))) if sf_post.size else 0.0
            null_aucs_slow[pi] = abs(float(_trapezoid(ss_post, t_post))) if ss_post.size else 0.0

        # p-values with +1 correction (avoids p=0)
        p_peak_fast = float((np.sum(null_peaks_fast >= abs_peak_fast) + 1) / (n_perms + 1))
        p_peak_slow = float((np.sum(null_peaks_slow >= abs_peak_slow) + 1) / (n_perms + 1))
        p_auc_fast = float((np.sum(null_aucs_fast >= abs(signed_auc_fast)) + 1) / (n_perms + 1))
        p_auc_slow = float((np.sum(null_aucs_slow >= abs(signed_auc_slow)) + 1) / (n_perms + 1))
    else:
        p_peak_fast = p_peak_slow = np.nan
        p_auc_fast = p_auc_slow = np.nan

    # Peak latencies
    lat_fast = _peak_latency_ms(fz_post, t_post)
    lat_slow = _peak_latency_ms(sz_post, t_post)

    return {
        "cluster_id": int(cid),
        "peak_fast": signed_peak_fast,
        "peak_slow": signed_peak_slow,
        "peak_latency_fast_ms": lat_fast,
        "peak_latency_slow_ms": lat_slow,
        "auc_fast": signed_auc_fast,
        "auc_slow": signed_auc_slow,
        "half_width_fast_ms": hw_fast,
        "half_width_slow_ms": hw_slow,
        "mirror_score": mirror,
        "p_peak_fast": p_peak_fast,
        "p_peak_slow": p_peak_slow,
        "p_auc_fast": p_auc_fast,
        "p_auc_slow": p_auc_slow,
    }


# ── Reclassify helper (reads existing CSV, re-applies Phase 3+4) ─────

def _reclassify(df, alpha, alpha_conj, z_thresh, has_perm,
                skip_trend_filter=False):
    """Re-classify from existing CSV with new alpha/alpha_conj, regenerate figure."""
    # Phase 3: Classification
    print("\n-- Phase 3: Classification --")
    if has_perm:
        # Union significance: a direction is significant if EITHER
        # the peak OR the AUC permutation p-value passes threshold.
        df["sig_fast"] = (
            (df["p_peak_fast"].fillna(1.0) < alpha)
            | (df["p_auc_fast"].fillna(1.0) < alpha)
        )
        df["sig_slow"] = (
            (df["p_peak_slow"].fillna(1.0) < alpha)
            | (df["p_auc_slow"].fillna(1.0) < alpha)
        )
        df["sig_fast_conj"] = (
            (df["p_peak_fast"].fillna(1.0) < alpha_conj)
            | (df["p_auc_fast"].fillna(1.0) < alpha_conj)
        )
        df["sig_slow_conj"] = (
            (df["p_peak_slow"].fillna(1.0) < alpha_conj)
            | (df["p_auc_slow"].fillna(1.0) < alpha_conj)
        )
    else:
        df["sig_fast"] = df["peak_fast"].abs() >= z_thresh
        df["sig_slow"] = df["peak_slow"].abs() >= z_thresh
        df["sig_fast_conj"] = df["sig_fast"]
        df["sig_slow_conj"] = df["sig_slow"]

    tiers, sub_types = [], []
    for _, row in df.iterrows():
        sign_f = 1 if row["peak_fast"] >= 0 else -1
        sign_s = 1 if row["peak_slow"] >= 0 else -1
        tier, sub = _assign_tier(
            row["sig_fast"], row["sig_slow"],
            row["sig_fast_conj"], row["sig_slow_conj"],
            sign_f, sign_s,
        )
        tiers.append(tier)
        sub_types.append(sub)
    df["tier"] = tiers
    df["sub_type"] = sub_types

    # ── Splitter rescue: upgrade unilateral units that have clear
    #    opposite-sign responses but failed the conjunction test.
    #    Criteria: opposite sign peaks, |non-sig peak| > 2.0, mirror > 0.3
    RESCUE_MIN_PEAK = 2.0
    RESCUE_MIN_MIRROR = 0.3
    uni_mask = df["tier"] == TIER_UNILATERAL
    opp_sign = np.sign(df["peak_fast"]) != np.sign(df["peak_slow"])
    both_peaks = (df["peak_fast"].abs() > RESCUE_MIN_PEAK) & (df["peak_slow"].abs() > RESCUE_MIN_PEAK)
    good_mirror = df["mirror_score"] > RESCUE_MIN_MIRROR
    rescue_mask = uni_mask & opp_sign & both_peaks & good_mirror

    n_rescued = int(rescue_mask.sum())
    if n_rescued > 0:
        for idx in df.index[rescue_mask]:
            sign_f = 1 if df.at[idx, "peak_fast"] >= 0 else -1
            sub = "Fast+/Slow-" if sign_f > 0 else "Slow+/Fast-"
            df.at[idx, "tier"] = TIER_SPLITTER
            df.at[idx, "sub_type"] = sub
        print(f"\n  Splitter rescue: {n_rescued} unilateral → splitter "
              f"(opp_sign, |peak|>{RESCUE_MIN_PEAK}, mirror>{RESCUE_MIN_MIRROR})")

    # ── Pre-trend filter: flag units whose post-pulse "response" is
    #    mostly explained by a pre-existing firing-rate trend.
    #    trend_ratio = |extrapolated_trend_at_250ms| / |actual_peak|
    TREND_RATIO_THRESH = 0.5
    manifest_for_npz = load_staging_manifest(qc_only=True)
    _npz_cache = {}
    for _, r in manifest_for_npz.iterrows():
        sn = int(r["session_name"])
        npz = load_tf_traces_npz(sn)
        if npz is not None:
            _npz_cache[sn] = npz

    trend_ratios = np.full(len(df), np.nan)
    for i, row in df.iterrows():
        if row["tier"] == TIER_NONE:
            continue
        sn, cid = int(row["session_name"]), int(row["cluster_id"])
        npz = _npz_cache.get(sn)
        if npz is None:
            continue
        cids = list(npz["cluster_ids"].astype(int))
        if cid not in cids:
            continue
        idx_npz = cids.index(cid)
        tv = npz["t_vec"]
        fz = gaussian_filter1d(npz["fast_z"][idx_npz], sigma=5, mode="nearest")
        sz = gaussian_filter1d(npz["slow_z"][idx_npz], sigma=5, mode="nearest")
        dom = fz if abs(row["peak_fast"]) >= abs(row["peak_slow"]) else sz

        pre_mask = (tv >= -0.3) & (tv < 0.0)
        t_pre = tv[pre_mask]
        d_pre = dom[pre_mask]
        if len(t_pre) < 10:
            continue
        slope = np.polyfit(t_pre, d_pre, 1)[0]
        post_mask = (tv >= TF_PULSE_POST_WINDOW[0]) & (tv < TF_PULSE_POST_WINDOW[1])
        actual_peak = float(np.nanmax(np.abs(dom[post_mask])))
        trend_at_250ms = slope * 0.25
        trend_ratios[i] = abs(trend_at_250ms) / max(actual_peak, 0.01)

    df["trend_ratio"] = trend_ratios
    if skip_trend_filter:
        print("  Trend filter: SKIPPED (detrend mode — linear drift already removed)")
    else:
        trend_flagged = (df["tier"] != TIER_NONE) & (df["trend_ratio"] > TREND_RATIO_THRESH)
        n_trend = int(trend_flagged.sum())
        if n_trend > 0:
            df.loc[trend_flagged, "tier"] = TIER_NONE
            df.loc[trend_flagged, "sub_type"] = "Trend-excluded"
            print(f"  Trend filter: {n_trend} units excluded (trend_ratio > {TREND_RATIO_THRESH})")

    # Summary
    print(f"\n  {'Tier':<28s}  {'N':>5s}  {'%':>6s}")
    print("  " + "-" * 43)
    for tier_name in [TIER_SPLITTER, TIER_UNILATERAL, TIER_OMNI, TIER_NONE]:
        n = int((df["tier"] == tier_name).sum())
        pct = 100 * n / len(df) if len(df) else 0
        print(f"  {tier_name:<28s}  {n:5d}  {pct:5.1f}%")
        if tier_name != TIER_NONE:
            for st_name in sorted(df.loc[df["tier"] == tier_name, "sub_type"].unique()):
                ns = int(((df["tier"] == tier_name) & (df["sub_type"] == st_name)).sum())
                print(f"    {st_name:<24s}  {ns:5d}")

    n_resp = int((df["tier"] != TIER_NONE).sum())
    print(f"\n  Total responsive: {n_resp}/{len(df)} ({100*n_resp/len(df):.1f}%)")

    for stg in STAGE_ORDER:
        sub = df[df["stage"] == stg]
        nr = int((sub["tier"] != TIER_NONE).sum())
        print(f"    {stg}: {nr}/{len(sub)} ({100*nr/len(sub):.1f}%)" if len(sub) else f"    {stg}: 0/0")

    suffix = "_detrended" if skip_trend_filter else ""
    csv_path = os.path.join(CACHE_DIR, f"tf_cell_classification{suffix}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    # Phase 4: Figure
    print("\n-- Phase 4: Figure --")
    npz_traces = _npz_cache  # reuse from trend-filter step above

    n_perms_label = int(df["permutation_tested"].sum()) if has_perm else 0
    fig = plt.figure(figsize=(26, 22))
    outer = gridspec.GridSpec(4, 1, height_ratios=[1.0, 1.0, 1.0, 1.0],
                              hspace=0.40, top=0.93, bottom=0.05)

    def _populate_tier_row(row_idx, tier_name, title_prefix):
        gs_row = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[row_idx], wspace=0.30)
        examples = _select_example_units(df, tier_name, npz_traces, 3)
        for j, (sn, cid, ui) in enumerate(examples):
            ax = fig.add_subplot(gs_row[0, j])
            _plot_example_neuron(ax, npz_traces[sn], ui, sn, cid)
            if j == 0:
                ax.set_ylabel("Z-score", fontsize=9)
                ax.text(-0.1, 1.15, f"{tier_name} examples", fontsize=11,
                        fontweight="bold", transform=ax.transAxes)
            if row_idx == 0 and j == 0:
                ax.legend(fontsize=7, loc="best")
        ax_m = fig.add_subplot(gs_row[0, 3])
        _plot_mean_traces(ax_m, df, tier_name, npz_traces, f"{title_prefix} - {tier_name} Mean")

    _populate_tier_row(0, TIER_SPLITTER, "A")
    _populate_tier_row(1, TIER_UNILATERAL, "B")
    _populate_tier_row(2, TIER_OMNI, "C")

    gs_sum = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[3], wspace=0.32)
    ax_null = fig.add_subplot(gs_sum[0, 0])
    if has_perm:
        _plot_null_distribution(ax_null, df, {}, n_perms_label, alpha)
    else:
        ax_null.text(0.5, 0.5, "Permutation not run\n(--no-perm mode)",
                     ha="center", va="center", fontsize=12, transform=ax_null.transAxes)
        ax_null.set_title("D - Permutation null distribution")
    ax_stage = fig.add_subplot(gs_sum[0, 1])
    _plot_tier_by_stage(ax_stage, df)
    ax_stage.set_title("E - " + ax_stage.get_title())
    ax_auc = fig.add_subplot(gs_sum[0, 2])
    _plot_auc_scatter(ax_auc, df)
    ax_auc.set_title("F - " + ax_auc.get_title())
    ax_cell = fig.add_subplot(gs_sum[0, 3])
    _plot_tier_by_celltype(ax_cell, df)
    ax_cell.set_title("G - " + ax_cell.get_title())
    method_label = (
        f"Permutation (perm-tested={n_perms_label}), alpha={alpha}, alpha_conj={alpha_conj}"
        if has_perm else f"|z| >= {z_thresh}"
    )
    detrend_tag = "  [DETRENDED]" if skip_trend_filter else ""
    fig.suptitle(
        f"TF Cell Classification - Medial Striatum{detrend_tag}  ({method_label})\n"
        "Baseline TF pulses  |  Khilkevich & Lohse, Nature 2024",
        fontsize=13, fontweight="bold", y=0.99,
    )
    fig_suffix = "_detrended" if skip_trend_filter else ""
    save_figure(fig, f"fig41_tf_cell_classification{fig_suffix}", "08_tf_pulse")
    print("  Done.")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    from visdetect.analysis.tf_pulse import _collect_pulses, TFRespPulseConfig

    parser = argparse.ArgumentParser(description="TF cell classification with permutation testing")
    parser.add_argument("--n-perms", type=int, default=DEFAULT_N_PERMS,
                        help="Circular-shift permutations (default: 500)")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                        help="Significance threshold (default: 0.01)")
    parser.add_argument("--n-workers", type=int, default=None,
                        help="Parallel workers (default: min(cpu_count, 8))")
    parser.add_argument("--no-perm", action="store_true",
                        help="Skip permutation, use z-threshold only (fast)")
    parser.add_argument("--pre-screen-z", type=float, default=PRE_SCREEN_Z,
                        help="Pre-screening |z| threshold (default: 1.5)")
    parser.add_argument("--alpha-conj", type=float, default=None,
                        help="Per-component alpha for conjunction tests "
                             "(Splitter/Omni). Default: 5*alpha.")
    parser.add_argument("--reclassify", action="store_true",
                        help="Re-classify from existing CSV (skip Phase 1+2)")
    parser.add_argument("--detrend", action="store_true",
                        help="Apply linear detrending to NPZ traces before "
                             "classification. Output: tf_cell_classification_detrended.csv")
    args = parser.parse_args()

    n_perms = 0 if args.no_perm else args.n_perms
    alpha = args.alpha
    alpha_conj = args.alpha_conj if args.alpha_conj is not None else min(10 * alpha, 0.10)
    z_thresh = DEFAULT_Z_THRESH_TF
    n_workers = args.n_workers or min(os.cpu_count() or 1, 8)
    do_detrend = args.detrend

    # When detrending, narrow peak measurement window to match detrend extrapolation
    if do_detrend:
        post_win_eff = TF_DETREND_POST_WINDOW
        print("  [DETREND] POST_WIN narrowed to "
              f"{TF_DETREND_POST_WINDOW} (from {TF_PULSE_POST_WINDOW})")
    else:
        post_win_eff = TF_PULSE_POST_WINDOW

    # ── Reclassify mode: re-apply Phase 3+4 from existing CSV ────
    if args.reclassify:
        suffix = "_detrended" if do_detrend else ""
        csv_path = os.path.join(CACHE_DIR, f"tf_cell_classification{suffix}.csv")
        if not os.path.exists(csv_path):
            print(f"ERROR: {csv_path} not found. Run without --reclassify first.")
            sys.exit(1)
        df = pd.read_csv(csv_path)
        has_perm = df["permutation_tested"].any()
        n_perms_label = int(df["permutation_tested"].sum()) if has_perm else 0
        print("=" * 70)
        print(f"[08g] TF Cell Reclassification  [from existing CSV]")
        print(f"       alpha={alpha}  alpha_conj={alpha_conj}  perm_tested={n_perms_label}")
        print("=" * 70)
        _reclassify(df, alpha, alpha_conj, z_thresh, has_perm,
                    skip_trend_filter=do_detrend)
        return

    detrend_label = " +DETREND" if do_detrend else ""
    mode = "permutation" if n_perms > 0 else "threshold"
    print("=" * 70)
    print(f"[08g] TF Cell Classification  [{mode}{detrend_label}]")
    print(f"       N_perms={n_perms}  alpha={alpha}  alpha_conj={alpha_conj}  workers={n_workers}")
    print("=" * 70)

    manifest = load_staging_manifest(qc_only=True)
    print(f"  Sessions: {len(manifest)}")

    # Cell-type lookup
    ct_lookup = {}
    try:
        wf = load_waveform_labels()
        for _, row in wf.iterrows():
            ct_lookup[(int(row["session_name"]), int(row["cluster_id"]))] = row["cell_type"]
    except (FileNotFoundError, KeyError):
        print("  Warning: cell-type labels not found")

    # ── Phase 1: Pre-screen from NPZ cache ────────────────────────
    print("\n-- Phase 1: Pre-screening from NPZ cache --")
    all_units = []
    candidates = {}  # sname -> [cid, ...]
    npz_traces = {}  # sname -> npz dict (kept for figure examples)

    session_list = [
        (int(row["session_name"]), row["stage"], row["session_idx"])
        for _, row in manifest.iterrows()
    ]

    for sname, stage, sidx in tqdm(session_list, desc="Loading NPZ"):
        npz = load_tf_traces_npz(sname)
        if npz is None:
            continue

        t_vec_npz = npz["t_vec"]
        post_mask_npz = (t_vec_npz >= post_win_eff[0]) & (t_vec_npz < post_win_eff[1])
        t_post_npz = t_vec_npz[post_mask_npz]

        # Optionally detrend full session traces (2D)
        if do_detrend:
            fast_z_all, _, _ = detrend_tf_traces(
                t_vec_npz, npz["fast_z"],
                baseline_window=TF_DETREND_BASELINE,
                post_window=TF_DETREND_POST_WINDOW)
            slow_z_all, _, _ = detrend_tf_traces(
                t_vec_npz, npz["slow_z"],
                baseline_window=TF_DETREND_BASELINE,
                post_window=TF_DETREND_POST_WINDOW)
        else:
            fast_z_all = npz["fast_z"]
            slow_z_all = npz["slow_z"]

        for i, cid in enumerate(npz["cluster_ids"]):
            cid = int(cid)

            fz = fast_z_all[i]
            sz = slow_z_all[i]
            fz_post = fz[post_mask_npz] if fz.size else np.array([])
            sz_post = sz[post_mask_npz] if sz.size else np.array([])

            # z_abs_max: from detrended traces when active, else from NPZ scalars
            if do_detrend:
                z_abs = max(
                    np.nanmax(np.abs(fz_post)) if fz_post.size else 0.0,
                    np.nanmax(np.abs(sz_post)) if sz_post.size else 0.0,
                )
            else:
                z_abs = max(
                    abs(npz["z_max_fast"][i]), abs(npz["z_min_fast"][i]),
                    abs(npz["z_max_slow"][i]), abs(npz["z_min_slow"][i]),
                )

            pk_fast = _signed_peak(fz_post)
            pk_slow = _signed_peak(sz_post)
            lat_fast = _peak_latency_ms(fz_post, t_post_npz)
            lat_slow = _peak_latency_ms(sz_post, t_post_npz)
            # Dominant-direction latency (whichever peak is larger)
            lat_dom = lat_fast if abs(pk_fast) >= abs(pk_slow) else lat_slow

            unit = {
                "session_name": sname,
                "cluster_id": cid,
                "stage": stage,
                "session_idx": sidx,
                "cell_type": ct_lookup.get((sname, cid), "Unknown"),
                "z_abs_max_npz": z_abs,
                "peak_z_abs": z_abs,
                "peak_fast": pk_fast,
                "peak_slow": pk_slow,
                "peak_latency_fast_ms": lat_fast,
                "peak_latency_slow_ms": lat_slow,
                "peak_latency_ms": lat_dom,
                "auc_fast": float(_trapezoid(fz_post, t_post_npz)) if fz_post.size else 0.0,
                "auc_slow": float(_trapezoid(sz_post, t_post_npz)) if sz_post.size else 0.0,
                "half_width_fast_ms": _half_width_ms(fz, t_vec_npz, post_win_eff),
                "half_width_slow_ms": _half_width_ms(sz, t_vec_npz, post_win_eff),
                "mirror_score": (
                    float(np.corrcoef(fz_post, -sz_post)[0, 1])
                    if fz_post.size > 5 and np.nanvar(fz_post) > 0 and np.nanvar(sz_post) > 0
                    else np.nan
                ),
                "p_peak_fast": np.nan,
                "p_peak_slow": np.nan,
                "p_auc_fast": np.nan,
                "p_auc_slow": np.nan,
                "permutation_tested": False,
                "n_fast_pulses": 0,
                "n_slow_pulses": 0,
            }
            all_units.append(unit)

            if z_abs >= args.pre_screen_z:
                candidates.setdefault(sname, []).append(cid)

        # Keep NPZ for figure plotting later
        npz_traces[sname] = npz

    n_total = len(all_units)
    n_candidates = sum(len(v) for v in candidates.values())
    print(f"  Total units: {n_total}")
    print(f"  Pre-screen candidates (|z| >= {args.pre_screen_z}): {n_candidates}")

    # Build fast update lookup
    unit_lookup = {}
    for idx, u in enumerate(all_units):
        unit_lookup[(u["session_name"], u["cluster_id"])] = idx

    # ── Phase 2: Permutation testing ──────────────────────────────
    if n_perms > 0 and n_candidates > 0:
        print(f"\n-- Phase 2: Permutation testing ({n_perms} permutations) --")
        cfg = TFRespPulseConfig()
        t_vec = np.arange(PRE_WIN[0], post_win_eff[1], DT, dtype=float)
        sigma_bins = (SIGMA_MS / 1000.0) / DT

        pbar = tqdm(total=n_candidates, desc="Permutation tests")

        for sname, cid_list in candidates.items():
            try:
                sess = load_session(sname)
            except FileNotFoundError:
                print(f"    Warning: session {sname} pkl not found, skipping")
                pbar.update(len(cid_list))
                continue

            fast_times, slow_times = _collect_pulses(sess, cfg, show_progress=False)
            n_fast = fast_times.size
            n_slow = slow_times.size

            # Update pulse counts for all units in this session
            for cid in cid_list:
                ui = unit_lookup.get((sname, cid))
                if ui is not None:
                    all_units[ui]["n_fast_pulses"] = int(n_fast)
                    all_units[ui]["n_slow_pulses"] = int(n_slow)

            if n_fast < MIN_PULSES or n_slow < MIN_PULSES:
                print(f"    {sname}: {n_fast} fast / {n_slow} slow pulses - skip")
                pbar.update(len(cid_list))
                del sess; gc.collect()
                continue

            # Extract spike times for candidate units
            spike_dict = {}
            for c in sess.clusters:
                ci = int(c.cluster_id)
                if ci in cid_list:
                    spike_dict[ci] = np.asarray(c.spike_times, dtype=float).flatten()

            rec_duration = max(
                (float(st.max()) for st in spike_dict.values() if st.size > 0),
                default=0.0,
            )

            worker_args = []
            for cid in cid_list:
                if cid not in spike_dict or spike_dict[cid].size == 0:
                    continue
                seed = hash((sname, cid)) % (2**32)
                worker_args.append((
                    cid, spike_dict[cid], fast_times, slow_times,
                    t_vec, sigma_bins, PRE_WIN, post_win_eff,
                    n_perms, rec_duration, seed,
                    do_detrend, TF_DETREND_BASELINE, TF_DETREND_POST_WINDOW,
                ))

            if worker_args:
                actual_workers = min(n_workers, len(worker_args))
                with ProcessPoolExecutor(max_workers=actual_workers) as executor:
                    futures = {
                        executor.submit(_classify_single_unit, a): a[0]
                        for a in worker_args
                    }
                    for future in as_completed(futures):
                        cid = futures[future]
                        try:
                            result = future.result()
                            ui = unit_lookup.get((sname, cid))
                            if ui is not None:
                                for key in result:
                                    if key != "cluster_id":
                                        all_units[ui][key] = result[key]
                                all_units[ui]["permutation_tested"] = True
                                # Update dominant-direction latency
                                u = all_units[ui]
                                u["peak_latency_ms"] = (
                                    u["peak_latency_fast_ms"]
                                    if abs(u["peak_fast"]) >= abs(u["peak_slow"])
                                    else u["peak_latency_slow_ms"]
                                )
                        except Exception as e:
                            print(f"    Warning: {sname}/{cid} failed: {e}")
                        pbar.update(1)
            else:
                pbar.update(len(cid_list))

            del sess, spike_dict
            gc.collect()

        pbar.close()

    # ── Phase 3: Classification ───────────────────────────────────
    print("\n-- Phase 3: Classification --")
    df = pd.DataFrame(all_units)

    # Significance flags — union of peak and AUC p-values.
    # A direction is significant if EITHER the peak OR the AUC
    # permutation p-value passes the threshold.
    if n_perms > 0:
        df["sig_fast"] = (
            (df["p_peak_fast"].fillna(1.0) < alpha)
            | (df["p_auc_fast"].fillna(1.0) < alpha)
        )
        df["sig_slow"] = (
            (df["p_peak_slow"].fillna(1.0) < alpha)
            | (df["p_auc_slow"].fillna(1.0) < alpha)
        )
        # Relaxed alpha for conjunction tests (Splitter / Omni require BOTH).
        df["sig_fast_conj"] = (
            (df["p_peak_fast"].fillna(1.0) < alpha_conj)
            | (df["p_auc_fast"].fillna(1.0) < alpha_conj)
        )
        df["sig_slow_conj"] = (
            (df["p_peak_slow"].fillna(1.0) < alpha_conj)
            | (df["p_auc_slow"].fillna(1.0) < alpha_conj)
        )
    else:
        df["sig_fast"] = df["peak_fast"].abs() >= z_thresh
        df["sig_slow"] = df["peak_slow"].abs() >= z_thresh
        df["sig_fast_conj"] = df["sig_fast"]
        df["sig_slow_conj"] = df["sig_slow"]

    # Tier assignment
    tiers, sub_types = [], []
    for _, row in df.iterrows():
        sign_f = 1 if row["peak_fast"] >= 0 else -1
        sign_s = 1 if row["peak_slow"] >= 0 else -1
        tier, sub = _assign_tier(
            row["sig_fast"], row["sig_slow"],
            row["sig_fast_conj"], row["sig_slow_conj"],
            sign_f, sign_s,
        )
        tiers.append(tier)
        sub_types.append(sub)
    df["tier"] = tiers
    df["sub_type"] = sub_types

    # ── Summary ──
    print(f"\n  {'Tier':<28s}  {'N':>5s}  {'%':>6s}")
    print("  " + "-" * 43)
    for tier_name in [TIER_SPLITTER, TIER_UNILATERAL, TIER_OMNI, TIER_NONE]:
        n = int((df["tier"] == tier_name).sum())
        pct = 100 * n / len(df) if len(df) else 0
        print(f"  {tier_name:<28s}  {n:5d}  {pct:5.1f}%")
        if tier_name != TIER_NONE:
            for st_name in sorted(df.loc[df["tier"] == tier_name, "sub_type"].unique()):
                ns = int(((df["tier"] == tier_name) & (df["sub_type"] == st_name)).sum())
                print(f"    {st_name:<24s}  {ns:5d}")

    n_resp = int((df["tier"] != TIER_NONE).sum())
    print(f"\n  Total responsive: {n_resp}/{len(df)} ({100*n_resp/len(df):.1f}%)")

    # By stage
    for stg in STAGE_ORDER:
        sub = df[df["stage"] == stg]
        nr = int((sub["tier"] != TIER_NONE).sum())
        print(f"    {stg}: {nr}/{len(sub)} ({100*nr/len(sub):.1f}%)" if len(sub) else f"    {stg}: 0/0")

    # Save CSV
    suffix = "_detrended" if do_detrend else ""
    csv_path = os.path.join(CACHE_DIR, f"tf_cell_classification{suffix}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    # ── Phase 4: Figure ───────────────────────────────────────────
    print("\n-- Phase 4: Figure --")
    fig = plt.figure(figsize=(26, 22))
    outer = gridspec.GridSpec(4, 1, height_ratios=[1.0, 1.0, 1.0, 1.0],
                              hspace=0.40, top=0.93, bottom=0.05)

    def _populate_tier_row(row_idx, tier_name, title_prefix):
        gs_row = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[row_idx], wspace=0.30)
        examples = _select_example_units(df, tier_name, npz_traces, 3)
        for j, (sn, cid, ui) in enumerate(examples):
            ax = fig.add_subplot(gs_row[0, j])
            _plot_example_neuron(ax, npz_traces[sn], ui, sn, cid)
            if j == 0:
                ax.set_ylabel("Z-score", fontsize=9)
                ax.text(-0.1, 1.15, f"{tier_name} examples", fontsize=11,
                        fontweight="bold", transform=ax.transAxes)
            if row_idx == 0 and j == 0:
                ax.legend(fontsize=7, loc="best")
                
        ax_m = fig.add_subplot(gs_row[0, 3])
        _plot_mean_traces(ax_m, df, tier_name, npz_traces, f"{title_prefix} - {tier_name} Mean")

    # ── Row 0: Tier 1 (Splitter) ──────────────────────────────────
    _populate_tier_row(0, TIER_SPLITTER, "A")

    # ── Row 1: Tier 2 (Unilateral) ────────────────────────────────
    _populate_tier_row(1, TIER_UNILATERAL, "B")
    
    # ── Row 2: Tier 3 (Omni) ──────────────────────────────────────
    _populate_tier_row(2, TIER_OMNI, "C")

    # ── Row 3: Summary panels ─────────────────────────────────────
    gs_sum = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[3], wspace=0.32)
    
    ax_null = fig.add_subplot(gs_sum[0, 0])
    if n_perms > 0:
        _plot_null_distribution(ax_null, df, candidates, n_perms, alpha)
    else:
        ax_null.text(0.5, 0.5, "Permutation not run\n(--no-perm mode)",
                     ha="center", va="center", fontsize=12, transform=ax_null.transAxes)
        ax_null.set_title("D - Permutation null distribution")

    ax_stage = fig.add_subplot(gs_sum[0, 1])
    _plot_tier_by_stage(ax_stage, df)
    ax_stage.set_title("E - " + ax_stage.get_title())

    ax_auc = fig.add_subplot(gs_sum[0, 2])
    _plot_auc_scatter(ax_auc, df)
    ax_auc.set_title("F - " + ax_auc.get_title())

    ax_cell = fig.add_subplot(gs_sum[0, 3])
    _plot_tier_by_celltype(ax_cell, df)
    ax_cell.set_title("G - " + ax_cell.get_title())

    method_label = (
        f"Permutation N={n_perms}, alpha={alpha}, alpha_conj={alpha_conj}"
        if n_perms > 0 else f"|z| >= {z_thresh}"
    )
    detrend_tag = "  [DETRENDED]" if do_detrend else ""
    fig.suptitle(
        f"TF Cell Classification – Medial Striatum{detrend_tag}  ({method_label})\n"
        "Baseline TF pulses  |  Khilkevich & Lohse, Nature 2024",
        fontsize=14, fontweight="bold", y=0.98,
    )
    fig_suffix = "_detrended" if do_detrend else ""
    save_figure(fig, f"fig41_tf_cell_classification{fig_suffix}", "08_tf_pulse")
    print("  Done.")


# ── Plotting helpers ──────────────────────────────────────────────────

def _plot_mean_traces(ax, df, tier_name, npz_traces, title):
    """Population mean +/- SEM of fast/slow z-traces for a tier."""
    tier_df = df[df["tier"] == tier_name].copy()
    if tier_df.empty:
        ax.text(0.5, 0.5, "No units in this tier",
                ha="center", va="center", fontsize=11, transform=ax.transAxes)
        ax.set_title(title)
        return

    fast_stack, slow_stack = [], []
    tv_ms = None
    for _, row in tier_df.iterrows():
        sname = int(row["session_name"])
        cid = int(row["cluster_id"])
        npz = npz_traces.get(sname)
        if npz is None:
            continue
        cids = list(npz["cluster_ids"].astype(int))
        if cid not in cids:
            continue
        idx = cids.index(cid)
        fz = npz["fast_z"][idx].copy()
        sz = npz["slow_z"][idx].copy()
        if tv_ms is None:
            tv_ms = npz["t_vec"] * 1000
        # Sign-align before population averaging.
        # Splitters: flip so fast response is always positive (slow then negative).
        # Others: flip so the dominant response is positive.
        if tier_name == TIER_SPLITTER:
            if row["peak_fast"] < 0:
                fz, sz = -fz, -sz
        else:
            if row.get("sub_type", "").startswith("Fast-") or row.get("sub_type", "").startswith("Both-"):
                if row["peak_fast"] < 0:
                    fz, sz = -fz, -sz
            elif row.get("sub_type", "").startswith("Slow-"):
                if row["peak_slow"] < 0:
                    fz, sz = -fz, -sz
        fast_stack.append(fz)
        slow_stack.append(sz)

    if not fast_stack or tv_ms is None:
        ax.text(0.5, 0.5, "No NPZ data", ha="center", va="center",
                fontsize=11, transform=ax.transAxes)
        ax.set_title(title)
        return

    fast_arr = np.vstack(fast_stack)
    slow_arr = np.vstack(slow_stack)
    n_units = fast_arr.shape[0]

    # Re-baseline each unit using a safe interior pre-window (-300 to -50 ms)
    # to remove KDE boundary artifacts at the trace edges.
    safe_mask = (tv_ms >= -300) & (tv_ms < -50)
    for i in range(n_units):
        fast_arr[i] -= np.nanmean(fast_arr[i][safe_mask])
        slow_arr[i] -= np.nanmean(slow_arr[i][safe_mask])

    sigma_smooth = 5
    for i in range(n_units):
        fast_arr[i] = gaussian_filter1d(fast_arr[i], sigma=sigma_smooth, mode='nearest')
        slow_arr[i] = gaussian_filter1d(slow_arr[i], sigma=sigma_smooth, mode='nearest')

    fast_mean = np.nanmean(fast_arr, axis=0)
    slow_mean = np.nanmean(slow_arr, axis=0)
    fast_sem = np.nanstd(fast_arr, axis=0) / max(np.sqrt(n_units), 1)
    slow_sem = np.nanstd(slow_arr, axis=0) / max(np.sqrt(n_units), 1)

    ax.fill_between(tv_ms, fast_mean - fast_sem, fast_mean + fast_sem,
                    color="#1565C0", alpha=0.15)
    ax.fill_between(tv_ms, slow_mean - slow_sem, slow_mean + slow_sem,
                    color="#E53935", alpha=0.15)
    ax.plot(tv_ms, fast_mean, color="#1565C0", linewidth=2.0, label="Fast")
    ax.plot(tv_ms, slow_mean, color="#E53935", linewidth=2.0, label="Slow")

    ax.axvline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axhline(0, color="grey", linewidth=0.4, alpha=0.3)
    # Clip display to hide KDE boundary artifact in first ~50 ms of window
    ax.set_xlim(-300, tv_ms[-1])
    ax.set_xlabel("Time from TF pulse (ms)")
    ax.set_ylabel("Z-score")
    ax.set_title(f"{title}  (n={n_units})")
    ax.legend(fontsize=8, loc="best")


def _select_example_units(df, tier_name, npz_traces, n_examples=3):
    """Pick top-N units by post-pulse effect size for a tier.

    Returns list of (session_name, cluster_id, idx_in_npz) tuples.
    """
    tier_df = df[df["tier"] == tier_name].copy()
    scored = []
    for _, row in tier_df.iterrows():
        sname = int(row["session_name"])
        cid = int(row["cluster_id"])
        npz = npz_traces.get(sname)
        if npz is None:
            continue
        cids = list(npz["cluster_ids"].astype(int))
        if cid not in cids:
            continue
        idx = cids.index(cid)
        fz = npz["fast_z"][idx]
        sz = npz["slow_z"][idx]
        tv = npz["t_vec"]
        post = tv >= 0
        eff = max(np.max(np.abs(fz[post])), np.max(np.abs(sz[post])))
        scored.append((eff, sname, cid, idx))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [(s, c, i) for _, s, c, i in scored[:n_examples]]


def _plot_example_neuron(ax, npz, unit_idx, sname, cid, sigma_smooth=5):
    """Plot a single neuron's fast/slow mean +/- SEM from NPZ cache."""
    tv_ms = npz["t_vec"] * 1000
    fz = gaussian_filter1d(npz["fast_z"][unit_idx], sigma=sigma_smooth)
    sz = gaussian_filter1d(npz["slow_z"][unit_idx], sigma=sigma_smooth)
    f_sem = gaussian_filter1d(npz["fast_z_sem"][unit_idx], sigma=sigma_smooth)
    s_sem = gaussian_filter1d(npz["slow_z_sem"][unit_idx], sigma=sigma_smooth)

    ax.fill_between(tv_ms, fz - f_sem, fz + f_sem,
                    color="#1565C0", alpha=0.18, linewidth=0)
    ax.fill_between(tv_ms, sz - s_sem, sz + s_sem,
                    color="#E53935", alpha=0.18, linewidth=0)
    ax.plot(tv_ms, fz, color="#1565C0", linewidth=1.4, label="Fast")
    ax.plot(tv_ms, sz, color="#E53935", linewidth=1.4, label="Slow")
    ax.axvline(0, color="k", linewidth=0.7, linestyle="--", alpha=0.5)
    ax.axhline(0, color="grey", linewidth=0.4, alpha=0.3)
    ax.set_title(f"s{sname} clu{cid}", fontsize=8, pad=3)
    ax.tick_params(labelsize=7)


def _plot_null_distribution(ax, df, candidates, n_perms, alpha):
    """Show permutation null histogram for the best Tier 1 unit."""
    # Pick the best example that was permutation-tested
    tested = df[(df["tier"] == TIER_SPLITTER) & df["permutation_tested"]].copy()
    if tested.empty:
        tested = df[df["permutation_tested"]].copy()
    if tested.empty:
        ax.text(0.5, 0.5, "No permutation-tested units",
                ha="center", va="center", fontsize=11, transform=ax.transAxes)
        ax.set_title("C – Permutation null distribution")
        return

    tested["_rank"] = tested["auc_fast"].abs() + tested["auc_slow"].abs()
    best = tested.nlargest(1, "_rank").iloc[0]

    # We don't have the null arrays saved, so show a schematic
    # with the observed p-value annotated
    obs_peak = max(abs(best["peak_fast"]), abs(best["peak_slow"]))
    p_val = min(best["p_peak_fast"], best["p_peak_slow"])
    cid = int(best["cluster_id"])
    sname = int(best["session_name"])

    # Generate a synthetic null illustration based on the p-value
    rng = np.random.default_rng(42)
    # Approximate null as chi-distributed (reasonable for |z| peak)
    null_approx = rng.standard_normal((n_perms, 500))
    null_peaks = np.max(np.abs(null_approx), axis=1)
    # Scale to roughly match observed magnitude range
    pctile = min(100.0 * (1 - p_val + 0.01), 99.9)
    null_peaks = null_peaks * (obs_peak / np.percentile(null_peaks, pctile))

    ax.hist(null_peaks, bins=40, color="#78909C", alpha=0.7, edgecolor="white",
            linewidth=0.5, density=True, label="Null distribution\n(schematic)")
    ax.axvline(obs_peak, color="#E53935", linewidth=2.0,
               label=f"Observed |z|={obs_peak:.1f}")
    ax.axvline(np.percentile(null_peaks, 100 * (1 - alpha)), color="grey",
               linewidth=1.2, linestyle="--",
               label=f"α={alpha} threshold")
    ax.set_xlabel("Peak |z-score|")
    ax.set_ylabel("Density")
    ax.set_title(f"C – Permutation test (unit #{cid}, p={p_val:.3f})")
    ax.legend(fontsize=7, loc="upper right")


def _plot_tier_by_stage(ax, df):
    """Stacked bar chart of tier distribution by stage."""
    stages = [s for s in STAGE_ORDER if s in df["stage"].values]
    tier_order = [TIER_SPLITTER, TIER_UNILATERAL, TIER_OMNI, TIER_NONE]
    x = np.arange(len(stages))
    width = 0.55

    bottoms = np.zeros(len(stages))
    for tier_name in tier_order:
        fracs = []
        for s in stages:
            sub = df[df["stage"] == s]
            fracs.append(100 * (sub["tier"] == tier_name).sum() / len(sub) if len(sub) else 0)
        ax.bar(x, fracs, width, bottom=bottoms,
               color=TIER_COLORS[tier_name], edgecolor="white", linewidth=0.3,
               label=tier_name)
        bottoms += np.array(fracs)

    ax.set_xticks(x)
    n_per_stage = [len(df[df["stage"] == s]) for s in stages]
    ax.set_xticklabels([f"{s}\n(n={n})" for s, n in zip(stages, n_per_stage)])
    ax.set_ylabel("% of units")
    ax.set_title("D – Tier distribution by stage")
    ax.legend(fontsize=7, loc="upper right", ncol=1)
    ax.set_ylim(0, 105)


def _plot_auc_scatter(ax, df):
    """Scatter of signed AUC_fast vs AUC_slow, colored by tier."""
    for tier_name in [TIER_NONE, TIER_OMNI, TIER_UNILATERAL, TIER_SPLITTER]:
        sub = df[df["tier"] == tier_name]
        if sub.empty:
            continue
        ax.scatter(
            sub["auc_fast"], sub["auc_slow"],
            c=TIER_COLORS[tier_name], s=12, alpha=0.5,
            edgecolors="none", label=tier_name,
            zorder=2 if tier_name == TIER_NONE else 3,
        )
    ax.axhline(0, color="grey", linewidth=0.4, alpha=0.3)
    ax.axvline(0, color="grey", linewidth=0.4, alpha=0.3)
    # Quadrant labels
    lim = max(abs(df["auc_fast"].quantile(0.01)), abs(df["auc_fast"].quantile(0.99)),
              abs(df["auc_slow"].quantile(0.01)), abs(df["auc_slow"].quantile(0.99)), 0.1)
    ax.set_xlim(-lim * 1.1, lim * 1.1)
    ax.set_ylim(-lim * 1.1, lim * 1.1)
    ax.set_xlabel("Signed AUC (fast)")
    ax.set_ylabel("Signed AUC (slow)")
    ax.set_title("E – Response AUC by tier")
    ax.legend(fontsize=6, loc="upper left", markerscale=2)


def _plot_tier_by_celltype(ax, df):
    """Grouped bar chart: % responsive per tier × cell type."""
    cell_types = sorted([c for c in df["cell_type"].unique() if c != "Unknown"])
    tier_order = [TIER_SPLITTER, TIER_UNILATERAL, TIER_OMNI]

    if not cell_types:
        ax.text(0.5, 0.5, "No cell-type labels", ha="center", va="center",
                fontsize=11, transform=ax.transAxes)
        ax.set_title("F – Tier by cell type")
        return

    x = np.arange(len(cell_types))
    width = 0.22
    offsets = np.linspace(-width, width, len(tier_order))

    for k, tier_name in enumerate(tier_order):
        fracs = []
        for ct in cell_types:
            sub = df[df["cell_type"] == ct]
            fracs.append(100 * (sub["tier"] == tier_name).sum() / len(sub) if len(sub) else 0)
        ax.bar(x + offsets[k], fracs, width * 0.9,
               color=TIER_COLORS[tier_name], edgecolor="black", linewidth=0.3,
               label=tier_name)

    n_per_ct = [len(df[df["cell_type"] == ct]) for ct in cell_types]
    ax.set_xticks(x)
    ax.set_xticklabels([f"{ct}\n(n={n})" for ct, n in zip(cell_types, n_per_ct)], fontsize=8)
    ax.set_ylabel("% of units in tier")
    ax.set_title("F – Tier by cell type")
    ax.legend(fontsize=7, loc="upper right")


if __name__ == "__main__":
    main()
