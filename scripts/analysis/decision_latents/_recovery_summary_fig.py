"""B8 — recovery-summary figure: 'can we trust each dial?' (Task 3.2 numbers).

Per-dial parameter recovery (recovered-vs-true Pearson r) by regime, from the
recover_point unit test (REDUCED config: n_trials=800, n_rep=40, n_restarts=2).
PRELIMINARY — to be re-confirmed at full config (n_rep>=100) in the gate (Task 3.5).
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

dials = ["Sharpness\n(v)", "Itchiness/caution\n(z)", "Timing\n(u)"]
expert_r = [0.825, 0.991, 0.838]
naive_r = [0.448, 0.994, 0.884]
C_EXP, C_NAI = "#1b7837", "#d6604d"   # engaged/expert (green) vs naive/hair-trigger (red)

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(8.6, 5.2))
x = np.arange(3); w = 0.38
b1 = ax.bar(x - w / 2, expert_r, w, color=C_EXP, label="Expert-like / engaged")
b2 = ax.bar(x + w / 2, naive_r, w, color=C_NAI, label="Naive-like / hair-trigger")
ax.axhline(0.8, ls="--", lw=1.6, color="#333333")
ax.text(2.46, 0.815, "trust threshold  r ≥ 0.8", ha="right", fontsize=9, color="#333333")
for b, vals in ((b1, expert_r), (b2, naive_r)):
    for rect, v in zip(b, vals):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.015, f"{v:.2f}",
                ha="center", fontsize=9, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(dials)
ax.set_ylim(0, 1.08); ax.set_ylabel("recovery  (recovered-vs-true  r)")
ax.set_title("Can we trust each dial?  —  parameter recovery", fontsize=13.5, fontweight="bold")
ax.legend(frameon=False, fontsize=9.5, loc="lower center", ncol=2)
# annotate the sharpness collapse
ax.annotate("sharpness COLLAPSES in the\nhair-trigger regime (v↔z ridge):\nflag it 'descriptive' there",
            xy=(0 + w / 2, 0.448), xytext=(0.55, 0.30), fontsize=8.6, color="#7a1f12",
            ha="left", va="center",
            arrowprops=dict(arrowstyle="->", color="#7a1f12", lw=1.3))
fig.text(0.5, -0.02,
         "Timing & itchiness recover everywhere; sharpness only in engaged/expert regimes.  "
         "Preliminary (reduced unit-test config) — re-confirmed at full config in the gate.",
         ha="center", fontsize=8.3, color="#666666")

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))), "FIGURES", "decision_latents", "BG_046")
os.makedirs(OUT, exist_ok=True)
p = os.path.join(OUT, "fig_b8_recovery_summary.png")
fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("wrote", p)
