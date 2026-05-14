"""Quick diagnostic figure for LOSO CV results (K=3 GLM-HMM)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy import stats

from visdetect.analysis.config import load_staging_manifest, STAGE_COLORS, STAGE_ORDER

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "figure.dpi": 150,
    "axes.spines.top": False, "axes.spines.right": False,
})

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Load data ────────────────────────────────────────────────────────
cv = pd.read_csv(os.path.join(REPO, "data", "hmm", "BG_046", "cv_results_K3.csv"))
manifest = load_staging_manifest(qc_only=False)

# Normalise to 8-char zero-padded string to match cv file (manifest stores
# session_name as integer, e.g. 1072025, while cv has "01072025").
manifest["session_name"] = manifest["session_name"].astype(int).astype(str).str.zfill(8)
cv["held_out_session"] = cv["held_out_session"].astype(str).str.zfill(8)

cv = cv.merge(manifest[["session_name", "stage"]],
              left_on="held_out_session", right_on="session_name", how="left")
# merge_naive_learning=True in SESSION_FILTER: align display to that convention
cv["stage"] = cv["stage"].replace("Naive", "Learning")
cv["session_idx"] = range(len(cv))

total_trials = cv["n_trials_test"].sum()
cv["n_train_trials"] = total_trials - cv["n_trials_test"]
cv["train_ll_per_trial"] = cv["train_ll"] / cv["n_train_trials"]

stages = [s for s in STAGE_ORDER if s in cv["stage"].values]
positions = {s: i for i, s in enumerate(stages)}
stage_colors = STAGE_COLORS

# ── Figure ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 8))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.38,
                       height_ratios=[1.1, 1])

ax_a = fig.add_subplot(gs[0, :])
ax_b = fig.add_subplot(gs[1, 0])
ax_c = fig.add_subplot(gs[1, 1])
ax_d = fig.add_subplot(gs[1, 2])


# ── A: test LL/trial trajectory ──────────────────────────────────────
ax = ax_a
for stg in stages:
    idxs = cv[cv["stage"] == stg]["session_idx"].values
    if len(idxs):
        ax.axvspan(idxs.min() - 0.5, idxs.max() + 0.5,
                   alpha=0.08, color=stage_colors[stg], zorder=0)

colors_pts = [stage_colors.get(s, "#888888") for s in cv["stage"]]
ax.plot(cv["session_idx"], cv["test_ll_per_trial"],
        color="#cccccc", lw=1, zorder=1)
ax.scatter(cv["session_idx"], cv["test_ll_per_trial"],
           c=colors_pts, s=65, edgecolors="white", linewidths=0.6, zorder=3)

for _, row in cv.nsmallest(3, "test_ll_per_trial").iterrows():
    ax.annotate(str(row["held_out_session"]),
                xy=(row["session_idx"], row["test_ll_per_trial"]),
                xytext=(0, -15), textcoords="offset points",
                fontsize=7, ha="center", color="#555555",
                arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.5))

rho, pval = stats.spearmanr(cv["session_idx"], cv["test_ll_per_trial"])
pstr = f"p={pval:.3f}" if pval >= 0.001 else "p<0.001"
ax.text(0.97, 0.06, f"Spearman ρ = {rho:+.2f}, {pstr}",
        transform=ax.transAxes, fontsize=8, ha="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  alpha=0.7, edgecolor="none"))

handles = [Patch(color=stage_colors[s], alpha=0.7, label=s) for s in stages]
ax.legend(handles=handles, loc="upper right", frameon=False)
ax.axhline(0, color="k", lw=0.5, alpha=0.3)
ax.set_xlabel("Session index (chronological)")
ax.set_ylabel("Test log-likelihood / trial")
ax.set_title("A.  LOSO held-out LL per trial across sessions",
             fontweight="bold", loc="left")
ax.set_xlim(-0.7, len(cv) - 0.3)


# ── B: stage violin — test LL/trial ──────────────────────────────────
def violin_panel(ax, col, ylabel, title):
    rng = np.random.default_rng(42)
    for stg in stages:
        sub = cv[cv["stage"] == stg][col].values
        vp = ax.violinplot(sub, positions=[positions[stg]],
                           showmedians=True, widths=0.6)
        for pc in vp["bodies"]:
            pc.set_facecolor(stage_colors[stg])
            pc.set_alpha(0.45)
        for part in ["cmedians", "cbars", "cmins", "cmaxes"]:
            vp[part].set_edgecolor(stage_colors[stg])
            vp[part].set_linewidth(1.5)
        jitter = rng.uniform(-0.13, 0.13, size=len(sub))
        ax.scatter(positions[stg] + jitter, sub,
                   color=stage_colors[stg], s=30, alpha=0.85,
                   edgecolors="white", linewidths=0.4, zorder=3)

    lrn = cv[cv["stage"] == "Learning"][col].values
    exp = cv[cv["stage"] == "Expert"][col].values
    u, p = stats.mannwhitneyu(lrn, exp, alternative="two-sided")
    n1, n2 = len(lrn), len(exp)
    rb = 1 - (2 * u) / (n1 * n2)
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    ymax = cv[col].max()
    span = cv[col].max() - cv[col].min()
    yb = ymax + 0.03 * span
    yb2 = ymax + 0.12 * span
    ax.plot([0, 0, 1, 1], [yb, yb + 0.03*span, yb + 0.03*span, yb],
            lw=1, c="k")
    ax.text(0.5, yb + 0.05*span,
            f"{stars}  r={rb:.2f}\np(MW)={p:.3f}", ha="center",
            va="bottom", fontsize=7.5)

    ax.set_xticks(list(positions.values()))
    ax.set_xticklabels(
        [f"{s}\n(n={sum(cv['stage'] == s)})" for s in stages])
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold", loc="left")


violin_panel(ax_b, "test_ll_per_trial", "Test LL / trial",
             "B.  Test LL by stage")

# ── C: stage violin — accuracy ────────────────────────────────────────
violin_panel(ax_c, "test_accuracy", "Test accuracy",
             "C.  Accuracy by stage")
ax_c.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.5)
ax_c.text(1.01, 0.5, "chance", transform=ax_c.get_yaxis_transform(),
          fontsize=7, color="gray", va="center")

# ── D: train vs test LL/trial ────────────────────────────────────────
ax = ax_d
colors_d = [stage_colors.get(s, "#888") for s in cv["stage"]]
sc = ax.scatter(cv["train_ll_per_trial"], cv["test_ll_per_trial"],
                c=colors_d, s=55, edgecolors="white", linewidths=0.5, zorder=3)

lo = min(cv["train_ll_per_trial"].min(), cv["test_ll_per_trial"].min()) - 0.01
hi = max(cv["train_ll_per_trial"].max(), cv["test_ll_per_trial"].max()) + 0.01
ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.4, label="y = x  (no gap)")

rho_d, _ = stats.spearmanr(cv["train_ll_per_trial"], cv["test_ll_per_trial"])
ax.text(0.05, 0.95, f"ρ = {rho_d:+.2f}", transform=ax.transAxes,
        fontsize=8, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  alpha=0.7, edgecolor="none"))
ax.legend(fontsize=7, frameon=False)
ax.set_xlabel("Train LL / trial")
ax.set_ylabel("Test LL / trial")
ax.set_title("D.  Train vs test LL\n(overfitting check)",
             fontweight="bold", loc="left")

# ── Save ─────────────────────────────────────────────────────────────
out_dir = os.path.join(REPO, "FIGURES", "behavior", "BG_046", "hmm")
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, "loso_cv_results.png")
fig.savefig(out, dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved:", out)
