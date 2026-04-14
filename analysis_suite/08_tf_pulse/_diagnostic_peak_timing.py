"""Diagnostic: Peak timing & z-score distributions for TF classification."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plotting import setup_style

setup_style()

df = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "cache", "tf_cell_classification_detrended.csv"))
resp = df[df["tier"] != "Non-responsive"].copy()
nonresp = df[df["tier"] == "Non-responsive"].copy()

TIER_COLORS = {
    "Tier 1 (Splitter)": "#8E24AA",
    "Tier 2 (Unilateral)": "#FB8C00",
    "Tier 3 (Omni)": "#43A047",
    "Non-responsive": "#BDBDBD",
}

fig = plt.figure(figsize=(18, 14))
gs = gridspec.GridSpec(3, 3, hspace=0.40, wspace=0.35, top=0.93, bottom=0.07)

# Panel A: Peak |z| distribution
ax = fig.add_subplot(gs[0, 0])
bins_z = np.arange(0, 15, 0.5)
ax.hist(nonresp["peak_z_abs"].dropna(), bins=bins_z, alpha=0.5, color="#BDBDBD",
        label="Non-resp (n=%d)" % len(nonresp), density=True, edgecolor="white", lw=0.3)
ax.hist(resp["peak_z_abs"].dropna(), bins=bins_z, alpha=0.7, color="#1565C0",
        label="Responsive (n=%d)" % len(resp), density=True, edgecolor="white", lw=0.3)
ax.axvline(3.0, color="grey", ls="--", lw=1, label="z=3.0")
ax.axvline(3.5, color="red", ls="--", lw=1.5, label="z=3.5")
ax.set_xlabel("Peak |z-score| (0-300ms)")
ax.set_ylabel("Density")
ax.set_title("A - Peak z-score distributions")
ax.legend(fontsize=7, loc="upper right")

# Panel B: Peak latency distribution
ax = fig.add_subplot(gs[0, 1])
bins_t = np.arange(0, 320, 20)
ax.hist(nonresp["peak_latency_ms"].dropna(), bins=bins_t, alpha=0.5, color="#BDBDBD",
        label="Non-resp", density=True, edgecolor="white", lw=0.3)
ax.hist(resp["peak_latency_ms"].dropna(), bins=bins_t, alpha=0.7, color="#1565C0",
        label="Responsive", density=True, edgecolor="white", lw=0.3)
ax.axvline(300, color="red", ls="--", lw=1.5, label="300ms cutoff")
ax.axvline(250, color="orange", ls="--", lw=1, label="250ms cutoff")
ax.set_xlabel("Peak latency (ms)")
ax.set_ylabel("Density")
ax.set_title("B - Peak latency distributions")
ax.legend(fontsize=7, loc="upper right")

# Panel C: Peak z vs latency scatter (all units)
ax = fig.add_subplot(gs[0, 2])
for tier in ["Non-responsive", "Tier 3 (Omni)", "Tier 2 (Unilateral)", "Tier 1 (Splitter)"]:
    sub = df[df["tier"] == tier]
    ax.scatter(sub["peak_latency_ms"], sub["peak_z_abs"],
              c=TIER_COLORS[tier], s=8, alpha=0.4, edgecolors="none",
              label="%s (n=%d)" % (tier, len(sub)),
              zorder=2 if tier == "Non-responsive" else 3)
ax.axvline(300, color="red", ls="--", lw=1.5, alpha=0.7)
ax.axhline(3.5, color="red", ls="--", lw=1.5, alpha=0.7)
ax.set_xlabel("Peak latency (ms)")
ax.set_ylabel("Peak |z-score|")
ax.set_title("C - z-score vs latency (all units)")
ax.legend(fontsize=6, loc="upper right", markerscale=2)
ax.set_xlim(-10, 310)
ax.set_ylim(0, 15)

# Panel D: Peak latency by tier (box + strip)
ax = fig.add_subplot(gs[1, 0])
tier_order = ["Tier 1 (Splitter)", "Tier 2 (Unilateral)", "Tier 3 (Omni)"]
tier_data = [resp[resp["tier"] == t]["peak_latency_ms"].dropna() for t in tier_order]
bp = ax.boxplot(tier_data, labels=["Splitter", "Unilateral", "Omni"], widths=0.5,
                patch_artist=True, showfliers=False,
                medianprops={"color": "black", "lw": 2})
for patch, tier in zip(bp["boxes"], tier_order):
    patch.set_facecolor(TIER_COLORS[tier])
    patch.set_alpha(0.6)
for i, (t, d) in enumerate(zip(tier_order, tier_data)):
    jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(d))
    ax.scatter(np.full(len(d), i+1)+jitter, d, c=TIER_COLORS[t], s=12, alpha=0.5,
               edgecolors="none")
ax.axhline(300, color="red", ls="--", lw=1.5, alpha=0.7, label="300ms cutoff")
ax.set_ylabel("Peak latency (ms)")
ax.set_title("D - Peak latency by tier")
ax.legend(fontsize=8)

# Panel E: CDF of peak latency
ax = fig.add_subplot(gs[1, 1])
for tier in tier_order:
    lat = resp[resp["tier"] == tier]["peak_latency_ms"].dropna().sort_values()
    cdf = np.arange(1, len(lat)+1) / len(lat)
    ax.step(lat, cdf, color=TIER_COLORS[tier], lw=2, label=tier)
lat_all = resp["peak_latency_ms"].dropna().sort_values()
cdf_all = np.arange(1, len(lat_all)+1) / len(lat_all)
ax.step(lat_all, cdf_all, color="black", lw=2, ls="--", label="All responsive")
ax.axvline(300, color="red", ls="--", lw=1.5, alpha=0.7)
ax.axvline(250, color="orange", ls="--", lw=1, alpha=0.7)
ax.set_xlabel("Peak latency (ms)")
ax.set_ylabel("Cumulative fraction")
ax.set_title("E - CDF of peak latency")
ax.legend(fontsize=7)

# Panel F: % surviving each cutoff
ax = fig.add_subplot(gs[1, 2])
cutoffs = np.arange(50, 301, 10)
for tier in tier_order:
    sub = resp[resp["tier"] == tier]
    fracs = [(sub["peak_latency_ms"] <= c).sum()/len(sub)*100 for c in cutoffs]
    ax.plot(cutoffs, fracs, color=TIER_COLORS[tier], lw=2, label=tier)
fracs_all = [(resp["peak_latency_ms"] <= c).sum()/len(resp)*100 for c in cutoffs]
ax.plot(cutoffs, fracs_all, color="black", lw=2, ls="--", label="All responsive")
ax.axvline(300, color="red", ls="--", lw=1.5, alpha=0.7)
ax.set_xlabel("Latency cutoff (ms)")
ax.set_ylabel("% surviving")
ax.set_title("F - % responsive surviving latency cutoff")
ax.legend(fontsize=7)

# Panel G: Early vs late peak scatter (responsive only)
ax = fig.add_subplot(gs[2, 0])
early = resp[resp["peak_latency_ms"] <= 300]
late = resp[resp["peak_latency_ms"] > 300]
ax.scatter(early["peak_latency_ms"], early["peak_z_abs"], c="#1565C0", s=20, alpha=0.6,
           edgecolors="none", label="Early peak (n=%d)" % len(early))
ax.scatter(late["peak_latency_ms"], late["peak_z_abs"], c="#E53935", s=25, alpha=0.7,
           edgecolors="black", lw=0.3, label="Late peak (n=%d)" % len(late))
ax.axvline(300, color="red", ls="--", lw=1.5, alpha=0.7)
ax.set_xlabel("Peak latency (ms)")
ax.set_ylabel("Peak |z-score|")
ax.set_title("G - Responsive cells: early vs late peak")
ax.legend(fontsize=8)

# Panel H: Peak z by latency bin
ax = fig.add_subplot(gs[2, 1])
lat_bins = [(0, 100), (100, 200), (200, 300)]
colors_lat = ["#1B5E20", "#388E3C", "#66BB6A"]
for (lo, hi), col in zip(lat_bins, colors_lat):
    sub = resp[(resp["peak_latency_ms"] >= lo) & (resp["peak_latency_ms"] < hi)]
    if len(sub) > 0:
        ax.hist(sub["peak_z_abs"], bins=np.arange(3, 14, 0.5), alpha=0.5, color=col,
                label="%d-%dms (n=%d)" % (lo, hi, len(sub)), density=True,
                edgecolor="white", lw=0.3)
ax.set_xlabel("Peak |z-score|")
ax.set_ylabel("Density")
ax.set_title("H - Peak z by latency bin (responsive)")
ax.legend(fontsize=6, loc="upper right")

# Panel I: Impact summary table
ax = fig.add_subplot(gs[2, 2])
ax.axis("off")
table_data = []
for z_cut in [3.0, 3.5, 4.0]:
    for t_cut in [150, 200, 250, 300]:
        mask = (resp["peak_z_abs"] >= z_cut) & (resp["peak_latency_ms"] <= t_cut)
        n_pass = int(mask.sum())
        pct = 100*n_pass/len(resp)
        spl = int(((resp["tier"] == "Tier 1 (Splitter)") & mask).sum())
        uni = int(((resp["tier"] == "Tier 2 (Unilateral)") & mask).sum())
        omn = int(((resp["tier"] == "Tier 3 (Omni)") & mask).sum())
        table_data.append(["z>%.1f" % z_cut, "<%dms" % t_cut, "%d" % n_pass,
                          "%.0f%%" % pct, "%d" % spl, "%d" % uni, "%d" % omn])

col_labels = ["z cutoff", "Lat cutoff", "N pass", "%% of %d" % len(resp), "Spl", "Uni", "Omni"]
table = ax.table(cellText=table_data, colLabels=col_labels, loc="center",
                 cellLoc="center", colColours=["#E3F2FD"]*7)
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.1, 1.4)
ax.set_title("I - Cutoff impact on responsive cells", fontsize=11,
             fontweight="bold", pad=20)

fig.suptitle("TF Classification Diagnostics: Peak Timing & Z-score Distributions\n"
             f"Based on detrended classification (N={len(df)} units, {len(resp)} responsive)",
             fontsize=13, fontweight="bold", y=0.98)

outpath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "figures", "08_tf_pulse", "fig41_diagnostic_peak_timing.png")
fig.savefig(outpath, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved: %s" % outpath)
print("Size: %.1f MB" % (os.path.getsize(outpath)/1e6))
