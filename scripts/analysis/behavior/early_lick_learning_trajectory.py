"""Early-lick (anticipatory / `fa`-label) behaviour across learning — BG_046.

Current-era companion to plot_cross_session_behavior.py. Where that script plots
the manifest `fa_rate` column (which is the *SDT* false-alarm rate = licking on
catch trials, behavior.py:143), THIS figure plots the anticipatory **early-lick
rate** = `fraction_fa` = n_fa_label / n_trials, i.e. the `fa` behavioural label
(early/impulsive lick during baseline, before any change). The two are distinct
constructs; see CLAUDE.md ("`fa` ≠ SDT false alarm").

Per-session summaries come from the canonical `compute_session_performance()`.
The individual early-lick reaction times are also collected so we can look at the
RT *distribution shape* rather than lean on the hardcoded FA_RT_SPLIT=3.0 s split:
a data-driven threshold (KDE antimode / 2-means-in-log fallback) is estimated on
the Expert early-lick RTs and used for the composition panel. FA_RT_SPLIT is kept
only as a visual reference line.

Sessions: the UNFILTERED staging manifest (qc_only=False, apply_filter=False) so
the full Naive→Expert trajectory is shown — the strict SESSION_FILTER (min_dprime
0.8, merge_naive_learning) drops exactly the early/impulsive sessions a learning
curve needs. Only truly unusable "Excluded" sessions are omitted.

Run: py scripts/analysis/behavior/early_lick_learning_trajectory.py [--force]
Out: FIGURES/behavior/BG_046/early_lick_learning_trajectory.png
     data/cache/behavior/early_lick_learning.csv        (per session)
     data/cache/behavior/early_lick_rts.csv             (per FA lick)
     data/cache/behavior/early_lick_learning_stats.csv
"""
import os
import sys
import gc
import argparse

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if os.path.join(_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kruskal, gaussian_kde
from statsmodels.stats.proportion import proportion_confint

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

from visdetect.analysis.config import (
    load_staging_manifest,
    canonical_session_id,
    _ALL_STAGE_ORDER as STAGE_ORDER_FULL,
    _ALL_STAGE_COLORS as STAGE_COLORS_FULL,
)
from visdetect.analysis.behavior import compute_session_performance, get_trial_dataframe
from visdetect.analysis.constants import FA_RT_SPLIT
from visdetect.analysis.spectrum_stats import silverman_bootstrap
from visdetect.suite.loader import resolve_session_pkl
from visdetect.core.session import load_session as load_session_from_path
from visdetect.suite.plotting import setup_style, save_figure

setup_style()

CACHE_DIR = os.path.join(_ROOT, "data", "cache", "behavior")
CACHE_FILE = os.path.join(CACHE_DIR, "early_lick_learning.csv")
RT_FILE = os.path.join(CACHE_DIR, "early_lick_rts.csv")
STATS_FILE = os.path.join(CACHE_DIR, "early_lick_learning_stats.csv")
CAVEATS_FILE = os.path.join(CACHE_DIR, "early_lick_learning_CAVEATS.txt")
CT_FILE = os.path.join(CACHE_DIR, "early_lick_changetimes.csv")   # change onset times

IMPULSIVE_COLOR = "#fb6a4a"   # HMM Impulsive orange-red  (fast/impulsive early lick)
SELFTIMED_COLOR = "#6baed6"   # HMM Engaged blue          (slower self-timed early lick)
EARLYLICK_COLOR = "#8856a7"   # early-lick rate (distinct purple)
ABORT_COLOR = "#969696"       # aborts (grey)
SDT_FA_COLOR = "#FF9800"      # SDT false-alarm rate (OUTCOME FA colour)

# RT-distribution display range (s) for the early-lick reaction times
RT_LO, RT_HI = 0.05, 8.0


# ── Data ──────────────────────────────────────────────────────────────
def compute_or_load(force=False):
    if (os.path.exists(CACHE_FILE) and os.path.exists(RT_FILE)
            and os.path.exists(CT_FILE) and not force):
        return (pd.read_csv(CACHE_FILE, dtype={"session_name": str}),
                pd.read_csv(RT_FILE, dtype={"session_name": str}),
                pd.read_csv(CT_FILE))

    os.makedirs(CACHE_DIR, exist_ok=True)
    manifest = load_staging_manifest(qc_only=False, apply_filter=False)
    manifest = manifest[manifest["stage"].isin(STAGE_ORDER_FULL)].copy()
    # chronological order is guaranteed by load_staging_manifest; the boolean
    # stage filter preserves it. session_idx is assigned as a contiguous counter
    # over successfully-loaded sessions (below), NOT here, so a missing pkl leaves
    # no gap in the x-axis.
    manifest = manifest.reset_index(drop=True)

    rows, rt_rows, ct_rows, skipped = [], [], [], []
    n = len(manifest)
    session_idx = 0
    for i, m in manifest.iterrows():
        sid = canonical_session_id(m["session_name"])
        # Prefer the canonical name resolver; fall back to the manifest's
        # authoritative `path` column, which handles re-recording suffixes the
        # resolver does not try (e.g. BG_046_05092025_b.pkl).
        pkl = resolve_session_pkl(sid)
        if pkl is None:
            raw = str(m.get("path", "") or "").replace("\\", "/")
            cand = raw if os.path.isabs(raw) else os.path.join(_ROOT, raw)
            pkl = cand if raw and os.path.exists(cand) else None
        if pkl is None:
            print(f"[{i + 1}/{n}] {sid} ({m['stage']}) -> SKIP (pkl not found)")
            skipped.append((sid, m["stage"]))
            continue
        print(f"[{i + 1}/{n}] {sid} ({m['stage']})")
        sess = load_session_from_path(pkl)
        perf = compute_session_performance(sess)
        tdf = get_trial_dataframe(sess)
        fa_rts = tdf.loc[tdf["is_fa"], "rt"].values
        for rt in fa_rts:
            if np.isfinite(rt):
                rt_rows.append({"session_name": sid, "session_idx": session_idx,
                                "stage": m["stage"], "rt": float(rt)})
        # change-onset times (trials where a change was actually presented) — these
        # censor the FA reaction times, so they explain the slow RT mode.
        for c in tdf.loc[tdf["outcome"].isin(["hit", "miss"]), "change_time"].values:
            if np.isfinite(c) and c > 0:
                ct_rows.append({"session_idx": session_idx, "stage": m["stage"],
                                "change_time": float(c)})
        rows.append({
            "session_name": sid,
            "session_idx": session_idx,
            "stage": m["stage"],
            "n_trials": perf["n_trials"],
            "early_lick_rate": perf["fraction_fa"],
            "n_fa": perf["n_fa"],
            "n_fa_early": perf["n_fa_early"],
            "n_fa_late": perf["n_fa_late"],
            "n_go": perf["n_go"],
            "n_catch": perf["n_catch"],
            "n_abort": perf["n_abort"],
            "n_sdt_fas": perf["n_sdt_fas"],
            "abort_rate": perf["abort_rate"],
            "d_prime": perf["d_prime"],
            "hit_rate": perf["hit_rate"],
            "sdt_fa_rate": perf["fa_rate_total"],
        })
        session_idx += 1
        del sess, tdf
        gc.collect()

    if skipped:
        print(f"\nSkipped {len(skipped)} session(s) with no pkl: " +
              ", ".join(f"{s}({st})" for s, st in skipped))

    df = pd.DataFrame(rows)
    rt_df = pd.DataFrame(rt_rows)
    ct_df = pd.DataFrame(ct_rows)
    df.to_csv(CACHE_FILE, index=False)
    rt_df.to_csv(RT_FILE, index=False)
    ct_df.to_csv(CT_FILE, index=False)
    return df, rt_df, ct_df


# ── Data-driven RT threshold ──────────────────────────────────────────
def data_driven_split(rts):
    """Estimate an empirical impulsive/self-timed boundary from early-lick RTs.

    Primary: antimode (deepest density trough) between the two dominant modes of
    a KDE in log10-RT space. Fallback: midpoint of a 1-D 2-means split in log-RT.
    Returns (threshold_seconds, method).
    """
    rts = np.asarray(rts, float)
    rts = rts[np.isfinite(rts) & (rts > 0)]
    if len(rts) < 40:
        return None, "insufficient"
    am = _kde_antimode(rts, np.linspace(RT_LO, float(np.percentile(rts, 99)), 800))
    if am is not None:
        return float(am), "kde_antimode_linear"
    # 2-means (Lloyd) fallback on linear RT
    xs = np.sort(rts)
    c = np.array([np.percentile(xs, 25), np.percentile(xs, 75)])
    for _ in range(100):
        g0 = xs[np.abs(xs - c[0]) <= np.abs(xs - c[1])]
        g1 = xs[np.abs(xs - c[0]) > np.abs(xs - c[1])]
        if len(g0) == 0 or len(g1) == 0:
            break
        nc = np.array([g0.mean(), g1.mean()])
        if np.allclose(nc, c):
            break
        c = nc
    return float(c.mean()), "two_means_linear"


def _kde_curve(rts, grid_log):
    rts = np.asarray(rts, float)
    rts = rts[np.isfinite(rts) & (rts > 0)]
    if len(rts) < 5:
        return None
    return gaussian_kde(np.log10(rts))(grid_log)


# ── Helpers ───────────────────────────────────────────────────────────
def shade_stages(ax, df, alpha=0.10):
    for stage in STAGE_ORDER_FULL:
        r = df[df["stage"] == stage]
        if r.empty:
            continue
        ax.axvspan(r["session_idx"].min() - 0.5, r["session_idx"].max() + 0.5,
                   color=STAGE_COLORS_FULL[stage], alpha=alpha, zorder=0)


def stage_colors(df):
    return df["stage"].map(STAGE_COLORS_FULL).values


def annotate_stat(ax, text, loc="upper left"):
    x, ha = (0.03, "left") if "left" in loc else (0.97, "right")
    y, va = (0.97, "top") if "upper" in loc else (0.03, "bottom")
    ax.text(x, y, text, transform=ax.transAxes, fontsize=8, va=va, ha=ha,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.75))


def annotate_n(ax, n, unit="sessions"):
    ax.text(0.98, 0.03, f"n = {n} {unit}", transform=ax.transAxes, fontsize=8,
            va="bottom", ha="right", color="gray")


def wilson_ci(k, n):
    """Vectorised Wilson 95% CI for a binomial proportion; NaN where n==0."""
    k = np.asarray(k, float)
    n = np.asarray(n, float)
    lo = np.full(k.shape, np.nan)
    hi = np.full(k.shape, np.nan)
    ok = n > 0
    if ok.any():
        lo[ok], hi[ok] = proportion_confint(k[ok], n[ok], method="wilson")
    return lo, hi


def kde_logboot(rts, grid_log, n_boot=300, seed=42):
    """KDE of log10(RT) on grid_log + bootstrap 95% band. Returns (base, lo, hi).

    Working in log10(RT) means equal area on the plotted axis is equal probability
    mass, so a heavy right tail cannot masquerade as a dominant mode.
    """
    rts = np.asarray(rts, float)
    rts = rts[np.isfinite(rts) & (rts > 0)]
    if len(rts) < 20:
        return None, None, None
    lr = np.log10(rts)
    base = gaussian_kde(lr)(grid_log)
    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, grid_log.size))
    for b in range(n_boot):
        boot[b] = gaussian_kde(rng.choice(lr, size=lr.size, replace=True))(grid_log)
    return base, np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)


def kde_lin_boot(rts, grid, n_boot=300, seed=42):
    """Linear-space KDE on grid (seconds) + bootstrap 95% band. (base, lo, hi)."""
    rts = np.asarray(rts, float)
    rts = rts[np.isfinite(rts) & (rts > 0)]
    if len(rts) < 20:
        return None, None, None
    base = gaussian_kde(rts)(grid)
    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, grid.size))
    for b in range(n_boot):
        boot[b] = gaussian_kde(rng.choice(rts, size=rts.size, replace=True))(grid)
    return base, np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)


def _kde_antimode(values, grid):
    """Deepest density trough between the two dominant KDE modes, or None."""
    d = gaussian_kde(values)(grid)
    maxima = [i for i in range(1, len(grid) - 1) if d[i] > d[i - 1] and d[i] > d[i + 1]]
    if len(maxima) < 2:
        return None
    a, b = sorted(sorted(maxima, key=lambda i: d[i])[-2:])
    return float(grid[a + int(np.argmin(d[a:b + 1]))])


def trough_band(rts):
    """(lo, hi) empirical trough from linear- AND log-space KDE antimodes.

    The antimode is transform-dependent, so we report the pair as a band rather
    than a single false-precise value.
    """
    rts = np.asarray(rts, float)
    rts = rts[np.isfinite(rts) & (rts > 0)]
    if len(rts) < 40:
        return None
    hi_cap = float(np.percentile(rts, 99))
    lin = _kde_antimode(rts, np.linspace(RT_LO, hi_cap, 800))
    lg = _kde_antimode(np.log10(rts), np.linspace(np.log10(RT_LO), np.log10(RT_HI), 500))
    vals = [v for v in (lin, (10 ** lg if lg is not None else None)) if v is not None]
    return (min(vals), max(vals)) if vals else None


def block_perm_p(idx, vals, n_perm=5000, seed=42):
    """Autocorrelation-preserving p for |Spearman| via circular shifts of vals."""
    idx = np.asarray(idx)
    vals = np.asarray(vals, float)
    ok = np.isfinite(vals)
    idx, vals = idx[ok], vals[ok]
    rho0, _ = spearmanr(idx, vals)
    rng = np.random.default_rng(seed)
    n = len(vals)
    cnt = sum(abs(spearmanr(idx, np.roll(vals, int(rng.integers(1, n))))[0]) >= abs(rho0)
              for _ in range(n_perm))
    return rho0, cnt / n_perm


def boot_ci_mean(vals, n_boot=2000, seed=42):
    """Mean and bootstrap 95% CI of a 1-D sample."""
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 2:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    bs = np.array([rng.choice(vals, vals.size, replace=True).mean() for _ in range(n_boot)])
    return vals.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def silverman_p(x_log):
    """Silverman critical-bandwidth bootstrap p(unimodal) on already-log10 data."""
    try:
        return float(silverman_bootstrap(np.asarray(x_log, float), n_boot=500,
                                         seed=42).get("p_unimodal", np.nan))
    except Exception:
        return np.nan


# Seconds-labelled ticks for a log10(RT) axis
_LOG_TICKS = np.log10([0.1, 0.3, 1, 3, 10])
_LOG_TICKLABELS = ["0.1", "0.3", "1", "3", "10"]


# ── Figure ────────────────────────────────────────────────────────────
def make_figure(df, rt_df, ct_df=None):
    df = df.sort_values("session_idx").reset_index(drop=True)
    x = df["session_idx"].values
    cols = stage_colors(df)
    rate = df["early_lick_rate"].values
    elo, ehi = wilson_ci(df["n_fa"].values, df["n_trials"].values)   # Wilson 95% CI on early-lick rate
    stats_rows = []

    # Data-driven threshold on Expert early-lick RTs. The antimode is transform-
    # dependent (log ~1.5 s, linear ~2.3 s), so we also carry a trough BAND for
    # display; dd_thr (log antimode) stays the operative split for composition.
    expert_rts = rt_df.loc[rt_df["stage"] == "Expert", "rt"].values
    dd_thr, dd_method = data_driven_split(expert_rts)
    band_disp = trough_band(expert_rts)
    if dd_thr is not None:
        stats_rows.append(("expert_datadriven_rt_threshold_s", dd_thr, np.nan, len(expert_rts)))
        stats_rows.append(("expert_datadriven_rt_method_" + dd_method, dd_thr, np.nan, len(expert_rts)))
    if band_disp is not None:
        stats_rows.append(("expert_rt_trough_band_lo_s", band_disp[0], np.nan, len(expert_rts)))
        stats_rows.append(("expert_rt_trough_band_hi_s", band_disp[1], np.nan, len(expert_rts)))

    # Per-session RT summary (threshold-free) + composition under dd threshold
    g = rt_df.groupby("session_idx")["rt"]
    rt_summary = pd.DataFrame({
        "median_fa_rt": g.median(),
        "q25_fa_rt": g.quantile(0.25),
        "q75_fa_rt": g.quantile(0.75),
        "n_fa_rt": g.count(),
    })
    if dd_thr is not None:
        imp = rt_df.assign(imp=rt_df["rt"] < dd_thr).groupby("session_idx")["imp"]
        rt_summary["frac_impulsive_dd"] = imp.mean()
        rt_summary["n_imp_dd"] = imp.sum()
    df = df.merge(rt_summary, on="session_idx", how="left")

    fig = plt.figure(figsize=(21, 9.5))
    gs = gridspec.GridSpec(2, 4, hspace=0.42, wspace=0.30)

    # A. Early-lick rate across learning ------------------------------------
    axA = fig.add_subplot(gs[0, 0])
    shade_stages(axA, df)
    axA.errorbar(x, rate, yerr=[rate - elo, ehi - rate], fmt="none", ecolor="0.6",
                 elinewidth=0.8, alpha=0.7, zorder=1)   # per-session Wilson 95% CI
    axA.plot(x, rate, "-", color="0.4", lw=1.2, zorder=2)
    axA.scatter(x, rate, c=cols, s=42, edgecolors="white", linewidths=0.5, zorder=3)
    rho, p = spearmanr(x, rate)
    rho_bp, p_bp = block_perm_p(x, rate)   # autocorrelation-preserving p
    stats_rows.append(("early_lick_rate_vs_session_spearman", rho, p, len(df)))
    stats_rows.append(("early_lick_rate_vs_session_blockperm_p", rho_bp, p_bp, len(df)))
    annotate_stat(axA, f"ρ = {rho:.2f}\np = {p:.2g} (iid)\np = {p_bp:.2g} (block-perm)")
    annotate_n(axA, len(df))
    axA.set_xlabel("Session (chronological)")
    axA.set_ylabel("Early-lick rate\nP(anticipatory lick)")
    axA.set_title("A. Early-lick rate across learning", fontweight="bold", loc="left", fontsize=11)
    axA.set_ylim(bottom=0)
    handles = [Patch(facecolor=STAGE_COLORS_FULL[s], label=s, alpha=0.9)
               for s in STAGE_ORDER_FULL if (df["stage"] == s).any()]
    axA.legend(handles=handles, loc="upper right", frameon=False, fontsize=8)

    # B. Early-lick RT across learning (threshold-free median + IQR) --------
    axB = fig.add_subplot(gs[0, 1])
    shade_stages(axB, df)
    axB.fill_between(x, df["q25_fa_rt"], df["q75_fa_rt"], color="0.6", alpha=0.25,
                     label="IQR (25–75%)")
    axB.plot(x, df["median_fa_rt"], "-", color="0.35", lw=1.2, zorder=2)
    axB.scatter(x, df["median_fa_rt"], c=cols, s=42, edgecolors="white",
                linewidths=0.5, zorder=3, label="median RT")
    axB.axhline(FA_RT_SPLIT, color="k", ls="--", lw=0.8, alpha=0.55)
    axB.text(x.max(), FA_RT_SPLIT, " 3 s (default split)", fontsize=7, color="0.3",
             va="bottom", ha="right")
    if band_disp is not None:
        axB.axhspan(band_disp[0], band_disp[1], color=EARLYLICK_COLOR, alpha=0.15)
        axB.text(x.max(), band_disp[1], f" trough {band_disp[0]:.1f}–{band_disp[1]:.1f} s",
                 fontsize=7, color=EARLYLICK_COLOR, va="bottom", ha="right")
    rho, p = spearmanr(x, df["median_fa_rt"], nan_policy="omit")
    stats_rows.append(("median_fa_rt_vs_session_spearman", rho, p, int(df["median_fa_rt"].notna().sum())))
    annotate_stat(axB, f"median RT vs session\nρ = {rho:.2f}, p = {p:.2g}", loc="lower right")
    axB.set_xlabel("Session (chronological)")
    axB.set_ylabel("Early-lick RT from baseline onset (s)")
    axB.set_title("B. Early-lick reaction time", fontweight="bold", loc="left", fontsize=11)
    axB.legend(loc="upper left", frameon=False, fontsize=7.5, ncol=2)

    # C. Impulsivity in context: early-lick vs abort vs SDT-FA --------------
    # Wilson 95% CI bands (shaded) on each rate; denominators differ per series.
    axC = fig.add_subplot(gs[0, 2])
    shade_stages(axC, df)
    for col, k, n, color, label in [
        ("early_lick_rate", df["n_fa"], df["n_trials"], EARLYLICK_COLOR, "Early-lick rate (fa label)"),
        ("abort_rate", df["n_abort"], df["n_trials"], ABORT_COLOR, "Abort rate"),
        ("sdt_fa_rate", df["n_sdt_fas"], df["n_catch"], SDT_FA_COLOR, "SDT FA rate (catch licks)"),
    ]:
        lo, hi = wilson_ci(k.values, n.values)
        axC.fill_between(x, lo, hi, color=color, alpha=0.15, lw=0)
        axC.plot(x, df[col], "-o", color=color, ms=3.5, lw=1.3, label=label)
    axC.set_xlabel("Session (chronological)")
    axC.set_ylabel("Rate (± Wilson 95% CI)")
    axC.set_title("C. Impulsivity signals in context", fontweight="bold", loc="left", fontsize=11)
    axC.set_ylim(bottom=0)
    axC.legend(loc="upper right", frameon=False, fontsize=7.5)

    # D. Early-lick rate vs sensitivity (d') --------------------------------
    axD = fig.add_subplot(gs[0, 3])
    axD.errorbar(df["d_prime"], rate, yerr=[rate - elo, ehi - rate], fmt="none",
                 ecolor="0.7", elinewidth=0.7, alpha=0.5, zorder=1)   # Wilson 95% CI
    axD.scatter(df["d_prime"], rate, c=cols, s=42, edgecolors="white",
                linewidths=0.5, zorder=3)
    m = df["d_prime"].notna() & df["early_lick_rate"].notna()
    rho, p = spearmanr(df.loc[m, "d_prime"], df.loc[m, "early_lick_rate"])
    stats_rows.append(("early_lick_rate_vs_dprime_spearman", rho, p, int(m.sum())))
    annotate_stat(axD, f"ρ = {rho:.2f}, p = {p:.2g}", loc="upper right")
    annotate_n(axD, int(m.sum()))
    axD.set_xlabel("d′ (sensitivity)")
    axD.set_ylabel("Early-lick rate")
    axD.set_title("D. Impulsivity vs sensitivity", fontweight="bold", loc="left", fontsize=11)
    axD.set_ylim(bottom=0)

    # E. Expert early-lick RT distribution (LINEAR seconds; area = mass) -----
    axE = fig.add_subplot(gs[1, 0])
    X_HI = 10.0   # display cap; tail fraction annotated
    er = expert_rts[np.isfinite(expert_rts) & (expert_rts > 0)]
    med = float(np.median(er))
    band = band_disp
    axE.hist(er, bins=np.linspace(0, np.ceil(er.max()), 60), density=True,
             color="0.82", edgecolor="white", linewidth=0.3)
    grid = np.linspace(RT_LO, X_HI, 400)
    base, blo, bhi = kde_lin_boot(er, grid)
    if base is not None:
        axE.fill_between(grid, blo, bhi, color="k", alpha=0.18, lw=0)
        axE.plot(grid, base, color="k", lw=1.4, label="KDE ± 95% CI")
    if band is not None:
        axE.axvspan(band[0], band[1], color=EARLYLICK_COLOR, alpha=0.20,
                    label=f"empirical trough {band[0]:.1f}–{band[1]:.1f} s")
    axE.axvline(FA_RT_SPLIT, color="k", ls="--", lw=0.9, alpha=0.6, label="3 s (default)")
    axE.axvline(med, color="0.35", ls=":", lw=1.1, label=f"median {med:.1f} s")
    # change-onset window: a lick is only an FA if it precedes the change (≥~6 s),
    # so the slow RT mode is a structural pile-up before the change, not learned timing
    ec5 = np.nan
    if ct_df is not None and len(ct_df):
        ect = ct_df.loc[ct_df["stage"] == "Expert", "change_time"].values
        ect = ect[np.isfinite(ect) & (ect > 0)]
        if len(ect):
            ec5 = float(np.percentile(ect, 5))
            axE.axvline(ec5, color="#2ca25f", lw=1.3, ls=(0, (5, 2)),
                        label=f"earliest change ~{ec5:.1f} s")
    axE2 = axE.twinx()   # ECDF makes cumulative mass unambiguous
    axE2.plot(np.sort(er), np.arange(1, er.size + 1) / er.size, color="#2166ac",
              lw=1.1, alpha=0.85)
    axE2.set_ylim(0, 1)
    axE2.set_ylabel("cumulative fraction", color="#2166ac", fontsize=9)
    axE2.tick_params(axis="y", labelcolor="#2166ac", labelsize=8)
    sb_lin, sb_log = silverman_p(er), silverman_p(np.log10(er))
    stats_rows.append(("expert_rt_silverman_p_unimodal_linear", np.nan, sb_lin, len(er)))
    stats_rows.append(("expert_rt_silverman_p_unimodal_logRT", np.nan, sb_log, len(er)))
    axE.set_xlim(0, X_HI)
    axE.set_xlabel("Early-lick RT from baseline onset (s)")
    axE.set_ylabel("Density (area = probability mass)")
    axE.set_title("E. Expert early-lick RT distribution", fontweight="bold", loc="left", fontsize=11)
    annotate_stat(axE, f"Silverman p(unimodal)\n= {sb_lin:.3f}\n{(er > X_HI).mean() * 100:.0f}% > {X_HI:.0f}s "
                       f"(max {er.max():.0f}s)", loc="upper left")
    annotate_n(axE, len(er), unit="FA licks")
    axE.legend(loc="upper right", frameon=False, fontsize=7)

    # F. Early-lick RT distribution by stage (LINEAR seconds; area = mass) ---
    axF = fig.add_subplot(gs[1, 1])
    grid = np.linspace(RT_LO, X_HI, 400)
    for s in STAGE_ORDER_FULL:
        srt = rt_df.loc[rt_df["stage"] == s, "rt"].values
        base, blo, bhi = kde_lin_boot(srt, grid)
        if base is None:
            continue
        axF.fill_between(grid, blo, bhi, color=STAGE_COLORS_FULL[s], alpha=0.20, lw=0)
        axF.plot(grid, base, color=STAGE_COLORS_FULL[s], lw=1.7,
                 label=f"{s} (n={int(np.isfinite(srt).sum())})")
    if band is not None:
        axF.axvspan(band[0], band[1], color=EARLYLICK_COLOR, alpha=0.15)
    if np.isfinite(ec5):
        axF.axvline(ec5, color="#2ca25f", lw=1.3, ls=(0, (5, 2)))
    axF.axvline(FA_RT_SPLIT, color="k", ls="--", lw=0.9, alpha=0.6)
    axF.set_xlim(0, X_HI)
    axF.set_xlabel("Early-lick RT from baseline onset (s)")
    axF.set_ylabel("Density (area = probability mass)")
    axF.set_title("F. RT distribution by stage (± 95% CI)", fontweight="bold", loc="left", fontsize=11)
    # per-stage multimodality sharpens with learning (linear Silverman)
    sil_by_stage = []
    for s in STAGE_ORDER_FULL:
        srt = rt_df.loc[rt_df["stage"] == s, "rt"].values
        srt = srt[np.isfinite(srt) & (srt > 0)]
        if len(srt) >= 40:
            sil_by_stage.append(f"{s[:3]} {silverman_p(srt):.2f}")
    if sil_by_stage:
        annotate_stat(axF, "Silverman p(unimodal):\n" + ", ".join(sil_by_stage), loc="upper left")
    axF.legend(loc="upper right", frameon=False, fontsize=7.5)

    # G. Composition (data-driven split) across sessions --------------------
    axG = fig.add_subplot(gs[1, 2])
    if dd_thr is not None and "frac_impulsive_dd" in df:
        fi = df["frac_impulsive_dd"].values
        ok = ~np.isnan(fi)
        blo, bhi = wilson_ci(df["n_imp_dd"].values, df["n_fa_rt"].values)  # boundary CI
        axG.fill_between(x[ok], 0, fi[ok], color=IMPULSIVE_COLOR, alpha=0.85,
                         label=f"Premature (<{dd_thr:.1f} s)")
        axG.fill_between(x[ok], fi[ok], 1.0, color=SELFTIMED_COLOR, alpha=0.85,
                         label=f"Late / pre-change (≥{dd_thr:.1f} s)")
        axG.fill_between(x[ok], blo[ok], bhi[ok], color="0.1", alpha=0.22, lw=0,
                         label="boundary 95% CI")
        axG.plot(x[ok], fi[ok], color="k", lw=0.7, alpha=0.8)
        rho, p = spearmanr(x[ok], fi[ok])
        _, p_bp = block_perm_p(x[ok], fi[ok])
        stats_rows.append(("frac_impulsive_dd_vs_session_spearman", rho, p, int(ok.sum())))
        annotate_stat(axG, f"ρ = {rho:.2f}\np = {p:.2g} (iid)\np = {p_bp:.2g} (block-perm)")
    axG.set_ylim(0, 1)
    axG.set_xlim(x.min() - 0.5, x.max() + 0.5)
    axG.set_xlabel("Session (chronological)")
    axG.set_ylabel("Fraction of early licks")
    axG.set_title("G. Early-lick composition: premature vs late", fontweight="bold", loc="left", fontsize=10.5)
    axG.legend(loc="lower left", frameon=True, framealpha=0.85, fontsize=7.5)

    # H. Per-stage early-lick rate distribution -----------------------------
    axH = fig.add_subplot(gs[1, 3])
    present = [s for s in STAGE_ORDER_FULL if (df["stage"] == s).any()]
    groups = [df.loc[df["stage"] == s, "early_lick_rate"].dropna().values for s in present]
    positions = np.arange(len(present))
    bp = axH.boxplot(groups, positions=positions, widths=0.55, patch_artist=True,
                     showfliers=False, medianprops=dict(color="k", lw=1.4))
    for patch, s in zip(bp["boxes"], present):
        patch.set_facecolor(STAGE_COLORS_FULL[s])
        patch.set_alpha(0.55)
    for pos, s in zip(positions, present):
        vals = df.loc[df["stage"] == s, "early_lick_rate"].dropna().values
        jit = pos - 0.12 + (np.linspace(-0.08, 0.08, len(vals)) if len(vals) > 1 else np.array([0.0]))
        axH.scatter(jit, vals, s=22, color=STAGE_COLORS_FULL[s], edgecolors="0.3",
                    linewidths=0.4, zorder=3)
        # mean ± bootstrap 95% CI (session-level), offset to the right of the box
        mean, clo, chi = boot_ci_mean(vals)
        axH.errorbar(pos + 0.30, mean, yerr=[[mean - clo], [chi - mean]], fmt="D",
                     color="k", ms=5, capsize=3, lw=1.2, zorder=5)
    axH.plot([], [], "kD", ms=5, label="mean ± 95% CI")
    if len(groups) >= 2:
        H, p = kruskal(*groups)
        stats_rows.append(("early_lick_rate_by_stage_kruskal", H, p, len(df)))
        annotate_stat(axH, f"Kruskal–Wallis\nH = {H:.2f}, p = {p:.2g}", loc="upper right")
    axH.legend(loc="upper center", frameon=False, fontsize=7.5)
    axH.set_xticks(positions)
    axH.set_xticklabels(present)
    axH.set_xlabel("Learning stage")
    axH.set_ylabel("Early-lick rate")
    axH.set_title("H. Early-lick rate by stage", fontweight="bold", loc="left", fontsize=11)
    axH.set_ylim(bottom=0)

    fig.suptitle("BG_046 — anticipatory early-lick behaviour across learning",
                 fontsize=14, fontweight="bold", y=0.99)

    stats_df = pd.DataFrame(stats_rows, columns=["metric", "statistic", "p_value", "n"])
    stats_df.to_csv(STATS_FILE, index=False)

    # ── Gate-7 caveats: self-documenting sidecar next to the numbers ────────
    sd = stats_df.set_index("metric")
    gv = lambda m, c: (sd.loc[m, c] if m in sd.index else float("nan"))
    caveats = [
        "CAVEATS — early_lick_learning_trajectory (BG_046).",
        "",
        "SINGLE SUBJECT (BG_046 = DMS). n=34 sessions is the sampling unit; adjacent",
        "  sessions are autocorrelated (lag-1 residual ~0.30), so the iid trajectory p is",
        "  OPTIMISTIC. Report the block-permutation p (autocorrelation-preserving):",
        f"  early-lick rate vs session: rho={gv('early_lick_rate_vs_session_spearman','statistic'):.2f}, "
        f"p_iid={gv('early_lick_rate_vs_session_spearman','p_value'):.3f}, "
        f"p_blockperm={gv('early_lick_rate_vs_session_blockperm_p','p_value'):.3f}.",
        "  A one-mouse trend is NOT a population claim — replicate in BG_039 (DMS) and",
        "  BG_031 (VMS, impulsive non-learner = negative control) before asserting it.",
        "",
        "METRIC PROVENANCE: 'early-lick rate' = fraction_fa = anticipatory 'fa' label /",
        "  n_trials. This is NOT sdt_fa_rate (SDT false alarm = catch-trial licking).",
        "",
        "NOT CIRCULAR: learning STAGE is assigned from d' only (stage_sessions.py), and d'",
        "  uses the SDT fa_rate, not the early-lick rate — so 'early-lick differs by stage'",
        "  is not definitional. Kruskal-Wallis by stage still treats sessions as independent.",
        "",
        "RT-DISTRIBUTION SHAPE IS PARTLY STRUCTURAL, NOT ALL LEARNED ANTICIPATION:",
        "  a lick is an FA only if it PRECEDES the scheduled change, and the change is never",
        "  presented before ~6 s (change_time 5th pct = 6.05 s, median ~6.9 s, in BOTH Naive and",
        "  Expert). FA reaction times are therefore CENSORED by change onset and pile up toward",
        "  ~6 s. The slow RT mode is a task-imposed 'late baseline lick' pile-up, NOT 'self-timed",
        "  anticipation' (panels relabelled). The fast (<~2 s) mode is genuine premature licking.",
        "",
        "BIMODALITY is transform- AND stage-dependent — do NOT overstate it:",
        f"  pooled Expert Silverman p(unimodal) = {gv('expert_rt_silverman_p_unimodal_linear','p_value'):.3f} (linear) /"
        f" {gv('expert_rt_silverman_p_unimodal_logRT','p_value'):.3f} (log10).",
        "  PER STAGE on linear RT: Naive ~0.23 (UNIMODAL), Learning ~0.06, Expert ~0.007 — the",
        "  two-mode separation SHARPENS with learning; Naive is NOT robustly bimodal. Do NOT cite",
        "  GMM delta-BIC (inflated by right-skew). The trough is a BAND "
        f"({gv('expert_rt_trough_band_lo_s','statistic'):.1f}-{gv('expert_rt_trough_band_hi_s','statistic'):.1f} s), not precise.",
        "",
        "READING PANELS E/F: LINEAR seconds, y is density so equal plotted area = equal mass.",
        "  Green line = earliest possible change (~6 s); each FA precedes its OWN trial's change,",
        "  so FA RTs are right-censored per trial. x capped at 10 s (~10% of licks fall 10-16 s, in",
        "  the ECDF). Median early-lick RT ~4.6 s; the fast premature mode is a MINORITY (~1/3).",
        "  NOTE: the raw RT density mixes lick-hazard x survival — use a HAZARD analysis to separate",
        "  genuine anticipation (rising hazard toward the change) from per-trial censoring.",
    ]
    with open(CAVEATS_FILE, "w", encoding="utf-8") as fh:
        fh.write("\n".join(caveats) + "\n")

    return fig, stats_df


# ── Main ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Early-lick behaviour across learning (BG_046).")
    ap.add_argument("--force", action="store_true", help="Recompute the per-session caches")
    args = ap.parse_args()

    df, rt_df, ct_df = compute_or_load(force=args.force)
    fig, stats_df = make_figure(df, rt_df, ct_df)
    paths = save_figure(fig, "early_lick_learning_trajectory", "behavior/BG_046")
    print("\nSaved figure:", paths[0])
    print("Saved caches:", CACHE_FILE, "|", RT_FILE)
    print("Saved stats: ", STATS_FILE)
    print("\nStats:\n", stats_df.to_string(index=False))
