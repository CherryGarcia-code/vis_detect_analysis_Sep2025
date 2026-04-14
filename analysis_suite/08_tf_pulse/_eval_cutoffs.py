"""Statistical evaluation of TF-responsiveness cutoff criteria."""
import pandas as pd
import numpy as np
from scipy import stats

import os
df = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "cache", "tf_cell_classification_detrended.csv"))
resp = df[df["tier"] != "Non-responsive"].copy()
nonresp = df[df["tier"] == "Non-responsive"].copy()
lat_resp = resp["peak_latency_ms"].dropna()
lat_nonresp = nonresp["peak_latency_ms"].dropna()

print("=" * 70)
print("STATISTICAL EVALUATION: TF RESPONSIVENESS CUTOFF CRITERIA")
print("=" * 70)

# Q1: Latency distribution comparison
print("\n(1) LATENCY DISTRIBUTION: RESPONSIVE vs NON-RESPONSIVE")
U, p_mw = stats.mannwhitneyu(lat_resp, lat_nonresp, alternative="two-sided")
r_rb = 1 - 2 * U / (len(lat_resp) * len(lat_nonresp))
print(f"  Mann-Whitney U = {U:.0f}, p = {p_mw:.2e}, r_rb = {r_rb:.3f}")
print(f"  Responsive: median = {lat_resp.median():.1f} ms, IQR = [{lat_resp.quantile(0.25):.1f}, {lat_resp.quantile(0.75):.1f}]")
print(f"  Non-responsive: median = {lat_nonresp.median():.1f} ms, IQR = [{lat_nonresp.quantile(0.25):.1f}, {lat_nonresp.quantile(0.75):.1f}]")

ks_stat, ks_p = stats.ks_2samp(lat_resp, lat_nonresp)
print(f"  KS test: D = {ks_stat:.3f}, p = {ks_p:.2e}")

# Q2: Optimal latency cutoff via Youden J
print("\n(2) OPTIMAL LATENCY CUTOFF (Youden J)")
thresholds = np.arange(50, 500, 5)
best_j = -1
best_t = 0
print(f"\n  {'Cutoff':>8s}  {'Sens':>6s}  {'Spec':>6s}  {'J':>6s}  {'Resp kept':>10s}")
for t in thresholds:
    sens = (lat_resp <= t).sum() / len(lat_resp)
    spec = (lat_nonresp > t).sum() / len(lat_nonresp)
    j = sens + spec - 1
    if j > best_j:
        best_j = j
        best_t = t
        best_sens = sens
        best_spec = spec

print(f"  Optimal: {best_t} ms (Sens={best_sens:.3f}, Spec={best_spec:.3f}, J={best_j:.3f})")

for t in [100, 150, 200, 250, 275, 300]:
    s = (lat_resp <= t).sum() / len(lat_resp)
    sp = (lat_nonresp > t).sum() / len(lat_nonresp)
    j = s + sp - 1
    nk = (lat_resp <= t).sum()
    print(f"  {t:>6d}ms  {s:>6.3f}  {sp:>6.3f}  {j:>6.3f}  {nk:>5d}/{len(lat_resp)}")

# Q3: Z-threshold
print("\n(3) Z-THRESHOLD EVALUATION")
print(f"  All responsive >= 3.5: {(resp['peak_z_abs'] >= 3.5).all()}")
print(f"  Min peak |z| responsive: {resp['peak_z_abs'].min():.2f}")
print(f"  5th percentile: {resp['peak_z_abs'].quantile(0.05):.2f}")
for z in [3.0, 3.5, 4.0]:
    n = (nonresp["peak_z_abs"] >= z).sum()
    pct = 100 * n / len(nonresp)
    print(f"  Non-responsive with |z| >= {z}: {n} ({pct:.1f}%)")

# Q4: Early vs late comparison
print("\n(4) EARLY vs LATE RESPONSIVE (cutoff=300ms)")
early = resp[resp["peak_latency_ms"] <= 300]
late = resp[resp["peak_latency_ms"] > 300]
print(f"  Early: n={len(early)}, Late: n={len(late)}")

# Peak z
U_z, p_z = stats.mannwhitneyu(early["peak_z_abs"], late["peak_z_abs"], alternative="two-sided")
r_z = 1 - 2 * U_z / (len(early) * len(late))
print(f"\n  Peak |z| comparison:")
print(f"    Early median={early['peak_z_abs'].median():.2f}, Late median={late['peak_z_abs'].median():.2f}")
print(f"    U={U_z:.0f}, p={p_z:.4f}, r_rb={r_z:.3f}")

# AUC magnitude
early_auc = early[["auc_fast", "auc_slow"]].abs().max(axis=1)
late_auc = late[["auc_fast", "auc_slow"]].abs().max(axis=1)
U_a, p_a = stats.mannwhitneyu(early_auc, late_auc, alternative="two-sided")
r_a = 1 - 2 * U_a / (len(early_auc) * len(late_auc))
print(f"\n  Max |AUC| comparison:")
print(f"    Early median={early_auc.median():.3f}, Late median={late_auc.median():.3f}")
print(f"    U={U_a:.0f}, p={p_a:.4f}, r_rb={r_a:.3f}")

# Mirror score
early_mirror = early["mirror_score"].dropna()
late_mirror = late["mirror_score"].dropna()
if len(early_mirror) > 5 and len(late_mirror) > 5:
    U_m, p_m = stats.mannwhitneyu(early_mirror, late_mirror, alternative="two-sided")
    r_m = 1 - 2 * U_m / (len(early_mirror) * len(late_mirror))
    print(f"\n  Mirror score comparison:")
    print(f"    Early median={early_mirror.median():.3f}, Late median={late_mirror.median():.3f}")
    print(f"    U={U_m:.0f}, p={p_m:.4f}, r_rb={r_m:.3f}")

# Tier composition chi-squared
print("\n  Tier composition:")
tier_order = ["Tier 1 (Splitter)", "Tier 2 (Unilateral)", "Tier 3 (Omni)"]
for t in tier_order:
    ne = (early["tier"] == t).sum()
    nl = (late["tier"] == t).sum()
    pe = 100 * ne / len(early)
    pl = 100 * nl / len(late)
    print(f"    {t}: Early {ne} ({pe:.1f}%), Late {nl} ({pl:.1f}%)")
contingency = np.array([
    [(early["tier"] == t).sum() for t in tier_order],
    [(late["tier"] == t).sum() for t in tier_order],
])
chi2, p_chi, dof, expected = stats.chi2_contingency(contingency)
n_tot = contingency.sum()
V = np.sqrt(chi2 / (n_tot * (min(contingency.shape) - 1)))
print(f"    Chi-squared: chi2={chi2:.2f}, p={p_chi:.4f}, V={V:.3f}")

# Q5: Bimodality test
print("\n(5) LATENCY DISTRIBUTION SHAPE (responsive)")
sw_stat, sw_p = stats.shapiro(lat_resp)
skew = stats.skew(lat_resp)
kurt = stats.kurtosis(lat_resp)
print(f"  Shapiro-Wilk: W={sw_stat:.4f}, p={sw_p:.4f}")
print(f"  Skewness: {skew:.3f}, Kurtosis: {kurt:.3f}")

try:
    from sklearn.mixture import GaussianMixture
    X = lat_resp.values.reshape(-1, 1)
    bic_1 = GaussianMixture(n_components=1, random_state=42).fit(X).bic(X)
    bic_2 = GaussianMixture(n_components=2, random_state=42).fit(X).bic(X)
    delta_bic = bic_1 - bic_2
    print(f"\n  Gaussian Mixture BIC:")
    print(f"    1-comp BIC: {bic_1:.1f}")
    print(f"    2-comp BIC: {bic_2:.1f}")
    print(f"    delta_BIC (1-2): {delta_bic:.1f} {'(2-comp preferred)' if delta_bic > 10 else '(1-comp adequate)'}")
    gm2 = GaussianMixture(n_components=2, random_state=42).fit(X)
    means = sorted(gm2.means_.flatten())
    stds = np.sqrt(gm2.covariances_.flatten())
    weights = gm2.weights_
    print(f"    2-comp means: {means[0]:.1f} ms, {means[1]:.1f} ms")
    print(f"    2-comp weights: {weights[0]:.3f}, {weights[1]:.3f}")
    # Boundary between components
    labels = gm2.predict(X)
    boundary = np.mean(means)
    print(f"    Approximate boundary: ~{boundary:.0f} ms")
except ImportError:
    print("  sklearn not available for GMM test")

# Q6: Half-width comparison
print("\n(6) HALF-WIDTH: EARLY vs LATE")
early_hw = early[["half_width_fast_ms", "half_width_slow_ms"]].max(axis=1).dropna()
late_hw = late[["half_width_fast_ms", "half_width_slow_ms"]].max(axis=1).dropna()
if len(early_hw) > 5 and len(late_hw) > 5:
    U_hw, p_hw = stats.mannwhitneyu(early_hw, late_hw, alternative="two-sided")
    r_hw = 1 - 2 * U_hw / (len(early_hw) * len(late_hw))
    print(f"  Early: median HW = {early_hw.median():.1f} ms")
    print(f"  Late: median HW = {late_hw.median():.1f} ms")
    print(f"  U = {U_hw:.0f}, p = {p_hw:.4f}, r_rb = {r_hw:.3f}")

# SUMMARY TABLE
print("\n" + "=" * 70)
print("SUMMARY RESULTS TABLE")
print("=" * 70)
print(f"{'Test':<42s} {'Stat':>6s} {'Value':>8s} {'p':>10s} {'ES':>8s} {'Interp':<20s}")
print("-" * 100)
print(f"{'Latency: resp vs nonresp':<42s} {'U':>6s} {U:>8.0f} {p_mw:>10.2e} {r_rb:>8.3f} {'Small effect':<20s}")
print(f"{'Latency KS test':<42s} {'D':>6s} {ks_stat:>8.3f} {ks_p:>10.2e} {'':>8s} {'Sig different':<20s}")
print(f"{'Peak z: early vs late':<42s} {'U':>6s} {U_z:>8.0f} {p_z:>10.4f} {r_z:>8.3f} {'Negligible-small':<20s}")
print(f"{'Max AUC: early vs late':<42s} {'U':>6s} {U_a:>8.0f} {p_a:>10.4f} {r_a:>8.3f} {'Small':<20s}")
print(f"{'Tier x timing (chi2)':<42s} {'chi2':>6s} {chi2:>8.2f} {p_chi:>10.4f} {V:>8.3f} {'Small':<20s}")
