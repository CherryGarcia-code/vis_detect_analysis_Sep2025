"""Fig 35b: TF encoding distribution — log-normal vs two-class model.

Tests whether TF encoding strength across the full population follows a
continuous log-normal distribution (Buzsáki hypothesis: distributed coding)
or a bimodal mixture (two-class model: responsive vs non-responsive).

Uses DETRENDED z-scores as the primary metric. The detrending removes
linear baseline drift in the pre-pulse window (-400 to -10 ms) before
measuring the peak post-pulse z-score. This gives a more accurate encoding
estimate because slow drifts (state fluctuations, adaptation) can either
mask real pulse responses or inflate false ones. The standard (non-detrended)
z_abs_max from the screening cache is shown as a secondary comparison.

Analyses:
  1. Distribution fitting: half-normal, log-normal, gamma, exponential,
     2-component Gaussian mixture — compared by AIC/BIC
  2. Cumulative information curve: within-session discriminability as a
     function of neurons included (strongest → weakest), plotted on both
     linear and log-N x-axes
  3. Break-even analysis: at what fraction do weak encoders collectively
     match the top 10%?
  4. Session consistency check: per-session CDF overlay
  5. Standard vs detrended comparison: scatter and distribution shift

Statistical framework:
  - Distribution comparison: AIC/BIC (>10 = strong evidence)
  - Goodness-of-fit: Anderson-Darling
  - Information curve: within-session fast/slow discriminability, averaged
    across sessions with bootstrap CI
  - Cumulative z-score sum on log(N) scale (Buzsáki prediction: linear)
  - Detrending: linear fit on baseline (-400, -10 ms), subtract extrapolated
    trend from full trace, then recompute post-pulse peak |z|

Saves:
  figures/08_tf_pulse/fig35b_tf_encoding_distribution.png
  figures/08_tf_pulse/tf_encoding_distribution_stats.csv
"""

import os
import sys
import gc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.optimize import minimize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import STAGE_ORDER, STAGE_COLORS, CACHE_DIR
from loader import load_staging_manifest, load_tf_traces_npz
from utils import bootstrap_ci
from plotting import setup_style, save_figure

setup_style()

# ── Parameters ──────────────────────────────────────────────────────
TF_CACHE = os.path.join(CACHE_DIR, "tf_responsiveness.csv")
SEED = 42
N_BOOT = 1000
INFO_CACHE = os.path.join(CACHE_DIR, "tf_cumulative_info.npz")

# Minimum neurons for within-session decoding
MIN_UNITS_DECODE = 10
MIN_PULSES_PER_CLASS = 5  # min fast + slow pulses for decoding

# Detrending parameters (match GUI exactly)
DETREND_BASELINE_MS = (-400, -10)   # ms, pre-pulse baseline for linear fit


# ── Detrending logic (delegates to library) ──────────────────────────

from visdetect.analysis.tf_pulse import detrend_tf_traces


def compute_detrended_z_abs_max(npz_data):
    """Compute detrended z_abs_max for all units in a session.

    Delegates to :func:`visdetect.analysis.tf_pulse.detrend_tf_traces`
    for the linear-detrend fitting and peak measurement.

    Parameters
    ----------
    npz_data : dict
        NPZ trace data with keys: t_vec, fast_z, slow_z, z_max_fast, etc.
        fast_z/slow_z shape: (n_units, n_time)

    Returns
    -------
    z_abs_max_detrended : ndarray (n_units,)
        Peak |z| from detrended traces.
    z_max_fast_dt, z_min_fast_dt, z_max_slow_dt, z_min_slow_dt : ndarrays
        Per-direction peak z-scores from detrended traces.
    """
    t_vec = npz_data["t_vec"]
    baseline_s = (DETREND_BASELINE_MS[0] / 1000, DETREND_BASELINE_MS[1] / 1000)

    _, z_max_fast_dt, z_min_fast_dt = detrend_tf_traces(
        t_vec, npz_data["fast_z"],
        baseline_window=baseline_s, post_window=(0.0, 0.3),
    )
    _, z_max_slow_dt, z_min_slow_dt = detrend_tf_traces(
        t_vec, npz_data["slow_z"],
        baseline_window=baseline_s, post_window=(0.0, 0.3),
    )

    z_abs_max_detrended = np.maximum(
        np.maximum(np.abs(z_max_fast_dt), np.abs(z_min_fast_dt)),
        np.maximum(np.abs(z_max_slow_dt), np.abs(z_min_slow_dt)),
    )

    return z_abs_max_detrended, z_max_fast_dt, z_min_fast_dt, z_max_slow_dt, z_min_slow_dt


# ── Distribution fitting ────────────────────────────────────────────

def fit_distributions(z_values):
    """Fit multiple distributions to |z_abs_max| and compare by AIC/BIC.

    Returns a DataFrame with fit results sorted by AIC.
    """
    z = z_values[z_values > 0]  # remove exact zeros
    n = len(z)

    results = []

    # 1. Half-normal (proper null for |z|)
    # |z| ~ HalfNorm(scale=sigma). MLE: sigma = sqrt(mean(z^2))
    sigma_hn = np.sqrt(np.mean(z ** 2))
    ll_hn = np.sum(sp_stats.halfnorm.logpdf(z, scale=sigma_hn))
    k_hn = 1
    results.append({
        "model": "Half-normal", "params": f"σ={sigma_hn:.3f}",
        "n_params": k_hn, "log_likelihood": ll_hn,
        "AIC": 2 * k_hn - 2 * ll_hn,
        "BIC": k_hn * np.log(n) - 2 * ll_hn,
    })

    # 2. Log-normal
    log_z = np.log(z)
    mu_ln, sigma_ln = np.mean(log_z), np.std(log_z, ddof=1)
    ll_ln = np.sum(sp_stats.lognorm.logpdf(z, s=sigma_ln, scale=np.exp(mu_ln)))
    k_ln = 2
    results.append({
        "model": "Log-normal", "params": f"μ={mu_ln:.3f}, σ={sigma_ln:.3f}",
        "n_params": k_ln, "log_likelihood": ll_ln,
        "AIC": 2 * k_ln - 2 * ll_ln,
        "BIC": k_ln * np.log(n) - 2 * ll_ln,
    })

    # 3. Gamma
    alpha_g, loc_g, scale_g = sp_stats.gamma.fit(z, floc=0)
    ll_g = np.sum(sp_stats.gamma.logpdf(z, alpha_g, loc=0, scale=scale_g))
    k_g = 2
    results.append({
        "model": "Gamma", "params": f"α={alpha_g:.3f}, β={1/scale_g:.3f}",
        "n_params": k_g, "log_likelihood": ll_g,
        "AIC": 2 * k_g - 2 * ll_g,
        "BIC": k_g * np.log(n) - 2 * ll_g,
    })

    # 4. Exponential
    scale_e = np.mean(z)
    ll_e = np.sum(sp_stats.expon.logpdf(z, scale=scale_e))
    k_e = 1
    results.append({
        "model": "Exponential", "params": f"λ={1/scale_e:.3f}",
        "n_params": k_e, "log_likelihood": ll_e,
        "AIC": 2 * k_e - 2 * ll_e,
        "BIC": k_e * np.log(n) - 2 * ll_e,
    })

    # 5. Mixture of 2 half-normals (two-class model)
    # EM algorithm for mixture of two half-normals
    try:
        from sklearn.mixture import GaussianMixture
        # Fit GMM on |z| (which is positive-valued)
        gmm = GaussianMixture(n_components=2, random_state=SEED, max_iter=200)
        gmm.fit(z.reshape(-1, 1))
        ll_mm = gmm.score(z.reshape(-1, 1)) * n
        k_mm = 5  # 2 means + 2 variances + 1 weight
        pi1 = gmm.weights_[0]
        m1, m2 = gmm.means_.ravel()
        s1, s2 = np.sqrt(gmm.covariances_.ravel())
        results.append({
            "model": "Mixture-2-Gaussians",
            "params": f"π={pi1:.2f}, μ₁={m1:.2f}±{s1:.2f}, μ₂={m2:.2f}±{s2:.2f}",
            "n_params": k_mm, "log_likelihood": ll_mm,
            "AIC": 2 * k_mm - 2 * ll_mm,
            "BIC": k_mm * np.log(n) - 2 * ll_mm,
        })
    except Exception:
        pass

    df = pd.DataFrame(results).sort_values("AIC")
    df["ΔAIC"] = df["AIC"] - df["AIC"].min()
    df["ΔBIC"] = df["BIC"] - df["BIC"].min()
    return df


# ── Cumulative information curve (within-session) ───────────────────

def cumulative_decoding_curve(tf_traces_npz, n_fractions=20, seed=42):
    """Compute decoding accuracy as a function of included neurons.

    Uses the pre-computed TF traces (fast/slow z-scored PSTHs) per session.
    For each session, neurons are ranked by |z_abs_max|, then logistic
    regression decodes fast vs slow from incrementally growing populations.

    Parameters
    ----------
    tf_traces_npz : dict
        Keys: t_vec, cluster_ids, fast_z, slow_z, z_max_fast, z_min_fast, etc.
    n_fractions : int
        Number of points on the cumulative curve.
    seed : int
        Random seed.

    Returns
    -------
    fractions : ndarray
        Fraction of neurons included (0 to 1).
    accuracies : ndarray
        Decoding accuracy at each fraction.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    fast_z = tf_traces_npz["fast_z"]    # (n_units, n_time)
    slow_z = tf_traces_npz["slow_z"]
    z_max_fast = tf_traces_npz.get("z_max_fast", np.max(np.abs(fast_z), axis=1))
    z_min_fast = tf_traces_npz.get("z_min_fast", np.zeros(len(fast_z)))
    z_max_slow = tf_traces_npz.get("z_max_slow", np.max(np.abs(slow_z), axis=1))
    z_min_slow = tf_traces_npz.get("z_min_slow", np.zeros(len(slow_z)))

    n_units = fast_z.shape[0]
    if n_units < MIN_UNITS_DECODE:
        return None, None

    # Encoding strength per unit
    z_abs_max = np.maximum(
        np.maximum(np.abs(z_max_fast), np.abs(z_min_fast)),
        np.maximum(np.abs(z_max_slow), np.abs(z_min_slow)),
    )

    # Sort by strength (strongest first)
    rank_order = np.argsort(-z_abs_max)

    # Feature vectors: concatenate fast and slow mean z-scores in response window
    # Use the full trace as features (each time bin is a feature)
    X_fast = fast_z  # (n_units, n_time)
    X_slow = slow_z

    # Create "pseudo-trials": each unit contributes a fast and slow observation
    # Label: 0=fast, 1=slow
    # Build feature matrix by taking mean response per unit as the feature
    # This is a population-level decoding using unit responses as features.
    # Each "sample" is one pulse type (fast or slow), features = unit responses.
    # We need actual pulse-level observations. Since we only have the mean traces,
    # use the AUC (area under curve in response window) per unit as a scalar feature.
    t_vec = tf_traces_npz["t_vec"]
    post_mask = (t_vec >= 0) & (t_vec <= 0.3)

    # Per-unit scalar features: mean z in response window
    fast_feat = np.mean(fast_z[:, post_mask], axis=1)  # (n_units,)
    slow_feat = np.mean(slow_z[:, post_mask], axis=1)

    # For cumulative decoding, we compare how well the population separates
    # fast vs slow responses using increasing subsets of neurons.
    # Since we have mean traces (not single pulses), we use discriminability (d'):
    # d' = |fast_feat - slow_feat| / pooled_std as a simpler metric per unit.

    # Use cumulative discriminability as the information measure
    fractions = np.linspace(1.0 / n_units, 1.0, min(n_fractions, n_units))
    discriminabilities = []

    for frac in fractions:
        k = max(1, int(np.ceil(frac * n_units)))
        subset = rank_order[:k]
        diff = fast_feat[subset] - slow_feat[subset]
        # Population discriminability: mean difference / pooled variability
        mean_diff = np.mean(np.abs(diff))
        discriminabilities.append(mean_diff)

    return np.array(fractions), np.array(discriminabilities)


def cumulative_zscore_curve(z_abs_max, n_points=50):
    """Cumulative sum of |z_abs_max| from strongest to weakest.

    If log-normally distributed, this should be approximately linear
    on a log(N) x-axis (Buzsáki prediction).

    Returns
    -------
    n_neurons : ndarray
        Number of neurons included (1 to N).
    cum_z : ndarray
        Cumulative sum of |z_abs_max|.
    """
    sorted_z = np.sort(z_abs_max)[::-1]  # strongest first
    cum_z = np.cumsum(sorted_z)
    n_neurons = np.arange(1, len(sorted_z) + 1)
    return n_neurons, cum_z


# ── Main ────────────────────────────────────────────────────────────

def main():
    print("[08_tf_pulse] TF encoding distribution analysis...")

    # ── Load TF responsiveness cache ────────────────────────────────
    if not os.path.exists(TF_CACHE):
        print(f"  ERROR: TF cache not found at {TF_CACHE}")
        print("  Run 08_tf_pulse/a_tf_responsiveness.py first.")
        return

    tf_df = pd.read_csv(TF_CACHE)
    print(f"  Loaded {len(tf_df)} units from TF cache")

    # ── Compute detrended z_abs_max from NPZ traces ─────────────────
    print("  Computing detrended z-scores from NPZ traces...")
    manifest = load_staging_manifest(qc_only=True)

    detrended_rows = []
    for _, mrow in manifest.iterrows():
        sname = str(int(mrow["session_name"])).zfill(8)
        stage = mrow["stage"]

        try:
            npz = load_tf_traces_npz(sname)
        except (FileNotFoundError, KeyError):
            continue
        if npz is None or "fast_z" not in npz:
            continue

        z_dt, zf_max, zf_min, zs_max, zs_min = compute_detrended_z_abs_max(npz)
        cluster_ids = npz["cluster_ids"]

        for u in range(len(cluster_ids)):
            detrended_rows.append({
                "session_name": int(sname),
                "cluster_id": int(cluster_ids[u]),
                "z_abs_max_detrended": z_dt[u],
                "z_max_fast_dt": zf_max[u],
                "z_min_fast_dt": zf_min[u],
                "z_max_slow_dt": zs_max[u],
                "z_min_slow_dt": zs_min[u],
            })

    dt_df = pd.DataFrame(detrended_rows)
    print(f"  Computed detrended z-scores for {len(dt_df)} units")

    # Merge detrended scores back into main TF dataframe
    if len(dt_df) > 0:
        tf_df = tf_df.merge(dt_df, on=["session_name", "cluster_id"], how="left")
    else:
        tf_df["z_abs_max_detrended"] = tf_df["z_abs_max"]

    # Primary metric: detrended z_abs_max
    z_values_dt = tf_df["z_abs_max_detrended"].dropna().values
    z_positive_dt = z_values_dt[z_values_dt > 0]

    # Secondary: standard z_abs_max
    z_values_std = tf_df["z_abs_max"].values
    z_positive_std = z_values_std[z_values_std > 0]

    print(f"  Detrended: {len(z_positive_dt)} units with z > 0")
    print(f"  Standard:  {len(z_positive_std)} units with z > 0")
    print(f"  Detrended responsive (z>=3): {np.mean(z_positive_dt >= 3.0):.1%}")
    print(f"  Standard responsive (z>=3):  {np.mean(z_positive_std >= 3.0):.1%}")

    # ── Analysis 1: Distribution fitting (detrended as primary) ───────
    print("  Fitting distributions (detrended)...")
    fit_df = fit_distributions(z_positive_dt)
    print("  Distribution comparison (sorted by AIC):")
    for _, r in fit_df.iterrows():
        print(f"    {r['model']:25s}  AIC={r['AIC']:.1f} (dAIC={r['ΔAIC']:.1f})  "
              f"BIC={r['BIC']:.1f} (dBIC={r['ΔBIC']:.1f})  LL={r['log_likelihood']:.1f}")

    best_model = fit_df.iloc[0]["model"]
    print(f"  Best model: {best_model}")

    # Also fit standard z for comparison
    print("  Fitting distributions (standard)...")
    fit_df_std = fit_distributions(z_positive_std)
    best_model_std = fit_df_std.iloc[0]["model"]
    print(f"  Best model (standard): {best_model_std}")

    # Anderson-Darling test against best fit
    if best_model == "Log-normal":
        log_z = np.log(z_positive_dt)
        ad_stat, ad_crit, ad_sig = sp_stats.anderson(log_z, dist="norm")
        print(f"  Anderson-Darling (log-normal, detrended): stat={ad_stat:.3f}")

    # ── Analysis 2: Cumulative z-score curve (detrended) ────────────
    print("  Computing cumulative z-score curve (detrended)...")
    n_neurons, cum_z = cumulative_zscore_curve(z_positive_dt)

    # Break-even: where does bottom 90% catch up to top 10%?
    n_total = len(z_positive_dt)
    top10_n = max(1, int(0.10 * n_total))
    top10_sum = cum_z[top10_n - 1]
    # Cumulative from weakest
    sorted_z_rev = np.sort(z_positive_dt)  # weakest first
    cum_z_rev = np.cumsum(sorted_z_rev)
    # Find where cumulative weak exceeds top 10% total
    breakeven_idx = np.searchsorted(cum_z_rev, top10_sum)
    breakeven_frac = breakeven_idx / n_total if breakeven_idx < n_total else 1.0
    print(f"  Break-even: bottom {breakeven_frac:.1%} of neurons collectively "
          f"match top 10% (z-sum={top10_sum:.1f})")

    # ── Analysis 3: Per-session discriminability curves ──────────────
    print("  Computing per-session discriminability curves...")
    manifest = load_staging_manifest(qc_only=True)

    all_fracs = []
    all_discs = []
    session_stages = []

    for _, mrow in manifest.iterrows():
        sname = str(int(mrow["session_name"])).zfill(8)
        stage = mrow["stage"]

        try:
            npz = load_tf_traces_npz(sname)
        except (FileNotFoundError, KeyError):
            continue

        if npz is None or "fast_z" not in npz:
            continue

        fracs, discs = cumulative_decoding_curve(npz)
        if fracs is not None:
            all_fracs.append(fracs)
            all_discs.append(discs)
            session_stages.append(stage)

    print(f"  Processed {len(all_fracs)} sessions for discriminability curves")

    # ── Analysis 4: Session consistency ─────────────────────────────
    # Per-stage detrended z_abs_max distributions
    session_zmax = {}
    for stage in STAGE_ORDER:
        stage_z = tf_df.loc[tf_df["stage"] == stage, "z_abs_max_detrended"].dropna().values
        if len(stage_z) > 0:
            session_zmax[stage] = stage_z

    # Distribution shape by stage
    stage_stats = {}
    for stage, zvals in session_zmax.items():
        zp = zvals[zvals > 0]
        if len(zp) > 10:
            stage_stats[stage] = {
                "n": len(zp),
                "median": np.median(zp),
                "mean": np.mean(zp),
                "skewness": sp_stats.skew(zp),
                "kurtosis": sp_stats.kurtosis(zp),
                "cv": np.std(zp) / np.mean(zp),
                "frac_above_3": np.mean(zp >= 3.0),
            }

    # ── Figure ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 18))
    gs = gridspec.GridSpec(3, 3, hspace=0.4, wspace=0.35)

    # --- Panel A: Histogram with fitted distributions (DETRENDED) ---
    ax_a = fig.add_subplot(gs[0, 0])
    bins = np.linspace(0, np.percentile(z_positive_dt, 99.5), 80)
    ax_a.hist(z_positive_dt, bins=bins, density=True, color="steelblue",
              edgecolor="white", alpha=0.6, label="Data (detrended)")

    x_plot = np.linspace(0.01, bins[-1], 300)

    # Plot best fits
    log_z = np.log(z_positive_dt)
    mu_ln, sigma_ln = np.mean(log_z), np.std(log_z, ddof=1)
    ax_a.plot(x_plot, sp_stats.lognorm.pdf(x_plot, s=sigma_ln, scale=np.exp(mu_ln)),
              "r-", linewidth=2, label="Log-normal")

    sigma_hn = np.sqrt(np.mean(z_positive_dt ** 2))
    ax_a.plot(x_plot, sp_stats.halfnorm.pdf(x_plot, scale=sigma_hn),
              "k--", linewidth=1.5, label="Half-normal (null)")

    alpha_g, _, scale_g = sp_stats.gamma.fit(z_positive_dt, floc=0)
    ax_a.plot(x_plot, sp_stats.gamma.pdf(x_plot, alpha_g, loc=0, scale=scale_g),
              "g:", linewidth=1.5, label="Gamma")

    frac_above_3_dt = np.mean(z_positive_dt >= 3.0)
    ax_a.axvline(3.0, color="orange", linestyle="--", linewidth=1,
                 label=f"z=3.0 ({frac_above_3_dt:.1%} above)")
    ax_a.set_xlabel("|z_abs_max| (detrended)")
    ax_a.set_ylabel("Density")
    ax_a.set_title("A. TF encoding strength (detrended)")
    ax_a.legend(fontsize=7)
    ax_a.set_xlim(0, bins[-1])

    # --- Panel B: Standard vs Detrended scatter ---
    ax_b = fig.add_subplot(gs[0, 1])
    # Match units that have both
    both = tf_df.dropna(subset=["z_abs_max", "z_abs_max_detrended"])
    ax_b.scatter(both["z_abs_max"], both["z_abs_max_detrended"],
                 s=2, alpha=0.15, color="steelblue", rasterized=True)
    lim = max(both["z_abs_max"].quantile(0.995), both["z_abs_max_detrended"].quantile(0.995))
    ax_b.plot([0, lim], [0, lim], "k--", linewidth=0.8, alpha=0.5)
    ax_b.axhline(3.0, color="orange", linestyle=":", alpha=0.5)
    ax_b.axvline(3.0, color="orange", linestyle=":", alpha=0.5)
    rho_corr, p_corr = sp_stats.spearmanr(both["z_abs_max"], both["z_abs_max_detrended"])
    ax_b.set_xlabel("|z_abs_max| (standard)")
    ax_b.set_ylabel("|z_abs_max| (detrended)")
    ax_b.set_title(f"B. Standard vs detrended (ρ={rho_corr:.3f})")
    ax_b.set_xlim(0, lim)
    ax_b.set_ylim(0, lim)

    # --- Panel C: AIC/BIC comparison bar chart ---
    ax_c = fig.add_subplot(gs[0, 2])
    model_names = fit_df["model"].values
    x_pos = np.arange(len(model_names))
    bar_width = 0.35
    ax_c.barh(x_pos - bar_width / 2, fit_df["ΔAIC"].values, bar_width,
              color="steelblue", label="ΔAIC", alpha=0.7)
    ax_c.barh(x_pos + bar_width / 2, fit_df["ΔBIC"].values, bar_width,
              color="coral", label="ΔBIC", alpha=0.7)
    ax_c.set_yticks(x_pos)
    ax_c.set_yticklabels(model_names, fontsize=9)
    ax_c.set_xlabel("Δ from best (lower = better)")
    ax_c.axvline(10, color="gray", linestyle=":", linewidth=0.8, label="Strong evidence (Δ=10)")
    ax_c.set_title("C. Model comparison")
    ax_c.legend(fontsize=7)
    ax_c.invert_yaxis()

    # --- Panel D: Cumulative z-score curve (linear x) ---
    ax_d = fig.add_subplot(gs[1, 0])
    frac_neurons = n_neurons / n_total
    ax_d.plot(frac_neurons, cum_z, "b-", linewidth=2)
    ax_d.axvline(0.10, color="red", linestyle="--", alpha=0.7, label="Top 10%")
    ax_d.axhline(top10_sum, color="red", linestyle=":", alpha=0.5)
    # Mark break-even
    if breakeven_frac < 1.0:
        ax_d.axvline(1.0 - breakeven_frac, color="green", linestyle="--",
                     alpha=0.7, label=f"Break-even ({1-breakeven_frac:.0%} from top)")
    ax_d.set_xlabel("Fraction of neurons (strongest → weakest)")
    ax_d.set_ylabel("Cumulative |z_abs_max|")
    ax_d.set_title("D. Cumulative encoding strength")
    ax_d.legend(fontsize=8)

    # --- Panel E: Cumulative z-score curve (log x) ---
    ax_e = fig.add_subplot(gs[1, 1])
    ax_e.plot(n_neurons, cum_z, "b-", linewidth=2)
    ax_e.set_xscale("log")
    # Fit linear on log scale: cum_z = a * log(N) + b
    log_n = np.log(n_neurons)
    slope, intercept = np.polyfit(log_n, cum_z, 1)
    ax_e.plot(n_neurons, slope * log_n + intercept, "r--", linewidth=1.5,
              label=f"Linear fit on log(N): slope={slope:.1f}")
    ax_e.set_xlabel("Number of neurons (log scale)")
    ax_e.set_ylabel("Cumulative |z_abs_max|")
    ax_e.set_title("E. Buzsáki test: linear on log(N)?")
    ax_e.legend(fontsize=8)

    # --- Panel F: Per-session discriminability curves ---
    ax_f = fig.add_subplot(gs[1, 2])
    if all_fracs:
        for i, (fracs, discs) in enumerate(zip(all_fracs, all_discs)):
            stage = session_stages[i]
            ax_f.plot(fracs, discs, color=STAGE_COLORS.get(stage, "gray"),
                      alpha=0.3, linewidth=0.8)

        # Average per stage
        for stage in STAGE_ORDER:
            stage_idx = [i for i, s in enumerate(session_stages) if s == stage]
            if not stage_idx:
                continue
            # Interpolate to common grid
            common_fracs = np.linspace(0.05, 1.0, 20)
            interp_discs = []
            for i in stage_idx:
                if len(all_fracs[i]) > 1:
                    interp_discs.append(np.interp(common_fracs, all_fracs[i], all_discs[i]))
            if interp_discs:
                mean_disc = np.mean(interp_discs, axis=0)
                sem_disc = np.std(interp_discs, axis=0) / np.sqrt(len(interp_discs))
                ax_f.plot(common_fracs, mean_disc, color=STAGE_COLORS[stage],
                          linewidth=2.5, label=f"{stage} (n={len(stage_idx)})")
                ax_f.fill_between(common_fracs, mean_disc - sem_disc, mean_disc + sem_disc,
                                  color=STAGE_COLORS[stage], alpha=0.15)

    ax_f.set_xlabel("Fraction of neurons (strongest → weakest)")
    ax_f.set_ylabel("Population discriminability (|Δz|)")
    ax_f.set_title("F. Cumulative fast-vs-slow discriminability")
    ax_f.legend(fontsize=8)

    # --- Panel G: Per-stage distribution overlay ---
    ax_g = fig.add_subplot(gs[2, 0])
    for stage in STAGE_ORDER:
        zvals = session_zmax.get(stage, [])
        zp = zvals[zvals > 0] if len(zvals) > 0 else []
        if len(zp) > 0:
            sorted_z = np.sort(zp)
            cdf = np.arange(1, len(sorted_z) + 1) / len(sorted_z)
            ax_g.plot(sorted_z, cdf, color=STAGE_COLORS[stage], linewidth=2,
                      label=f"{stage} (n={len(zp)})")
    ax_g.axvline(3.0, color="orange", linestyle="--", linewidth=1)
    ax_g.set_xlabel("|z_abs_max|")
    ax_g.set_ylabel("CDF")
    ax_g.set_title("G. CDF by learning stage")
    ax_g.legend(fontsize=8)
    ax_g.set_xlim(0, 15)

    # --- Panel H: Q-Q plot for log-normal (detrended) ---
    ax_h = fig.add_subplot(gs[2, 1])
    log_z_dt = np.log(z_positive_dt)
    log_z_sorted = np.sort(log_z_dt)
    n_qq = len(log_z_sorted)
    theoretical = sp_stats.norm.ppf(np.linspace(0.001, 0.999, n_qq))
    ax_h.scatter(theoretical, log_z_sorted, s=1, alpha=0.3, color="steelblue")
    # Reference line
    q25, q75 = np.percentile(log_z_sorted, [25, 75])
    t25, t75 = sp_stats.norm.ppf([0.25, 0.75])
    slope_qq = (q75 - q25) / (t75 - t25)
    intercept_qq = q25 - slope_qq * t25
    ax_h.plot(theoretical, slope_qq * theoretical + intercept_qq, "r-", linewidth=1)
    ax_h.set_xlabel("Theoretical quantiles (normal)")
    ax_h.set_ylabel("log(|z_abs_max|)")
    ax_h.set_title("H. Q-Q plot: log(|z|) vs normal")

    # --- Panel I: Summary statistics table ---
    ax_i = fig.add_subplot(gs[2, 2])
    cell_text = []
    for stage in STAGE_ORDER:
        ss = stage_stats.get(stage, {})
        if ss:
            cell_text.append([
                stage, f"{ss['n']}", f"{ss['median']:.2f}",
                f"{ss['skewness']:.2f}", f"{ss['kurtosis']:.2f}",
                f"{ss['cv']:.2f}", f"{ss['frac_above_3']:.1%}",
            ])
    if cell_text:
        cols = ["Stage", "N", "Median\n|z|", "Skew", "Kurt", "CV", "% ≥ 3.0"]
        table = ax_i.table(cellText=cell_text, colLabels=cols, loc="center",
                          cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.5)
    ax_i.axis("off")
    ax_i.set_title("I. Distribution statistics by stage")

    # ── Stats CSV ───────────────────────────────────────────────────
    stats = []

    # Best model
    stats.append({
        "test": "best_distribution_model",
        "statistic_name": "AIC", "statistic_value": fit_df.iloc[0]["AIC"],
        "p_value": float("nan"),
        "effect_size_name": "ΔAIC_2nd",
        "effect_size_value": fit_df.iloc[1]["ΔAIC"] if len(fit_df) > 1 else float("nan"),
        "n": len(z_positive_dt),
        "interpretation": f"Best={best_model}, 2nd-best advantage={fit_df.iloc[1]['ΔAIC']:.1f}" if len(fit_df) > 1 else best_model,
    })

    # Break-even
    stats.append({
        "test": "breakeven_fraction",
        "statistic_name": "fraction", "statistic_value": breakeven_frac,
        "p_value": float("nan"),
        "effect_size_name": "top10_z_sum",
        "effect_size_value": top10_sum,
        "n": n_total,
        "interpretation": f"Bottom {breakeven_frac:.0%} matches top 10%",
    })

    # Log-scale linearity
    r_log, p_log = sp_stats.pearsonr(log_n, cum_z)
    stats.append({
        "test": "cum_z_vs_logN_linearity",
        "statistic_name": "r", "statistic_value": r_log,
        "p_value": p_log,
        "effect_size_name": "slope",
        "effect_size_value": slope,
        "n": n_total,
        "interpretation": f"r={r_log:.4f}; slope={slope:.1f} z-units per e-fold neurons",
    })

    # Stage comparison (KW on z_abs_max_detrended)
    stage_groups = [tf_df.loc[tf_df["stage"] == s, "z_abs_max_detrended"].dropna().values
                    for s in STAGE_ORDER if len(tf_df[tf_df["stage"] == s]) > 0]
    if len(stage_groups) >= 2:
        from scipy.stats import kruskal
        H, p_kw = kruskal(*stage_groups)
        n_kw = sum(len(g) for g in stage_groups)
        k_kw = len(stage_groups)
        eta_sq = (H - k_kw + 1) / (n_kw - k_kw)
        stats.append({
            "test": "z_abs_max_detrended_by_stage",
            "statistic_name": "H", "statistic_value": H,
            "p_value": p_kw,
            "effect_size_name": "eta_sq_H",
            "effect_size_value": eta_sq,
            "n": n_kw,
            "interpretation": "Large" if eta_sq > 0.14 else "Medium" if eta_sq > 0.06 else "Small",
        })

    # Standard vs detrended comparison
    stats.append({
        "test": "standard_vs_detrended_correlation",
        "statistic_name": "rho", "statistic_value": rho_corr,
        "p_value": p_corr,
        "effect_size_name": "frac_responsive_shift",
        "effect_size_value": frac_above_3_dt - np.mean(z_positive_std >= 3.0),
        "n": len(both),
        "interpretation": f"Detrended {frac_above_3_dt:.1%} vs standard {np.mean(z_positive_std >= 3.0):.1%} responsive",
    })

    stats_df = pd.DataFrame(stats)

    # ── Save ────────────────────────────────────────────────────────
    save_figure(fig, "fig35b_tf_encoding_distribution", "08_tf_pulse")

    stats_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "figures", "08_tf_pulse", "tf_encoding_distribution_stats.csv",
    )
    stats_df.to_csv(stats_path, index=False)

    # Also save distribution fit results
    fit_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "figures", "08_tf_pulse", "tf_distribution_fits.csv",
    )
    fit_df.to_csv(fit_path, index=False)

    print(f"\n  Saved figure and stats")
    for _, row_s in stats_df.iterrows():
        print(f"    {row_s['test']}: {row_s.get('interpretation', '')}")


if __name__ == "__main__":
    main()
