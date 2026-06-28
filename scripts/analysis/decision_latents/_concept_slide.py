"""B8 — conceptual PRESENTATION slide: the three decision 'dials' + what success looks like.

NOT real data. This is an illustration of (1) the QUESTION and (2) the PREDICTED
result if the decomposition works — for a talk. Each dial is only shipped if it
passes parameter recovery (Phase 3 gate); this slide shows the hypothesis, not results.
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

# ── project mood palette + dial colours ──────────────────────────────────────
IMP, STIM = "#ef6548", "#6baed6"          # Impulsive / StimSens (config.STATE_LABEL_COLORS)
C_SHARP, C_ITCH, C_TIME = "#2c7fb8", "#ef6548", "#756bb1"

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12.5, "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))), "FIGURES", "decision_latents", "BG_046")
os.makedirs(OUT, exist_ok=True)

fig = plt.figure(figsize=(15.5, 6.4))
gs = fig.add_gridspec(1, 3, width_ratios=[1.06, 1.0, 1.12], wspace=0.30,
                      left=0.035, right=0.975, top=0.79, bottom=0.13)
fig.suptitle("Turning each trial's licking into three decision “dials” — and what it looks like if it works",
             fontsize=16, fontweight="bold", y=0.955)

# ── Panel 1 — THE QUESTION: three dials ──────────────────────────────────────
ax = fig.add_subplot(gs[0, 0]); ax.axis("off")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_title("The question: 3 dials, per trial", fontweight="bold", loc="left")
rows = [
    ("Sharpness", C_SHARP, "how clearly it can tell the grating changed", 0.80),
    ("Itchiness", C_ITCH, "how trigger-happy it is BEFORE real evidence", 0.45),
    ("Timing",    C_TIME, "how strongly it expects the change right now", 0.62),
]
y = 0.86
for name, col, sub, val in rows:
    ax.add_patch(plt.Rectangle((0.05, y - 0.035), 0.90, 0.07, fc="#ececec", ec="none"))
    ax.add_patch(plt.Rectangle((0.05, y - 0.035), 0.90 * val, 0.07, fc=col, ec="none"))
    ax.text(0.05, y + 0.075, name, fontsize=12.5, fontweight="bold", color=col)
    ax.text(0.95, y + 0.078, sub, fontsize=8.6, ha="right", va="center", color="#555555")
    y -= 0.265
ax.text(0.5, 0.045, "behaviour  →  3 numbers / trial",
        fontsize=11, ha="center", style="italic", color="#333333",
        bbox=dict(boxstyle="round,pad=0.4", fc="#f4f4f4", ec="#cccccc"))

# ── Panel 2 — IF IT WORKS: learning turns SHARPNESS ──────────────────────────
ax = fig.add_subplot(gs[0, 1])
x = np.array([0, 1, 2]); xl = ["Naive", "Learning", "Expert"]
ax.plot(x, [0.22, 0.55, 0.86], "-o", color=C_SHARP, lw=3.2, ms=8, label="Sharpness (sensitivity)")
ax.plot(x, [0.30, 0.58, 0.80], "-o", color=C_TIME, lw=2.4, ms=6, label="Timing precision")
ax.plot(x, [0.70, 0.58, 0.50], "--o", color=C_ITCH, lw=2.0, ms=6, label="Itchiness")
ax.set_xticks(x); ax.set_xticklabels(xl)
ax.set_ylim(0, 1); ax.set_yticks([])
ax.set_xlabel("learning  →"); ax.set_ylabel("dial value")
ax.set_title("If it works: LEARNING turns up SHARPNESS", fontweight="bold")
ax.legend(frameon=False, fontsize=8.6, loc="center left")
ax.annotate("sensitivity rises;\nlicks migrate to the\nexpected change time",
            xy=(2, 0.86), xytext=(0.75, 0.92), fontsize=8.6, color="#222222",
            ha="left", va="top")

# ── Panel 3 — IF IT WORKS: states load on ITCHINESS/TIMING, not SHARPNESS ─────
ax = fig.add_subplot(gs[0, 2])
groups = ["Sharpness", "Itchiness", "Timing"]
imp = [0.63, 0.86, 0.46]      # Impulsive
stim = [0.67, 0.34, 0.71]     # StimSens
xg = np.arange(3); w = 0.36
ax.bar(xg - w / 2, imp, w, color=IMP, label="Impulsive")
ax.bar(xg + w / 2, stim, w, color=STIM, label="StimSens")
ax.set_xticks(xg); ax.set_xticklabels(groups)
ax.set_ylim(0, 1.0); ax.set_yticks([])
ax.set_title("If it works: STATES load on ITCHINESS/TIMING,\nnot SHARPNESS  (bias, not gain)", fontweight="bold")
ax.legend(frameon=False, fontsize=9, loc="upper right")
# bracket highlighting the near-equal sharpness
ax.annotate("", xy=(-w / 2, 0.72), xytext=(w / 2, 0.72),
            arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.3))
ax.text(0, 0.78, "≈ equal", ha="center", fontsize=9, fontweight="bold", color="#333333")
ax.text(0, 0.90, "Impulsive looks eager,\nbut its d′ is NOT higher", ha="center",
        fontsize=8.4, color="#7a1f12", style="italic")

fig.text(0.5, 0.025,
         "Illustrative — the PREDICTED pattern (not results). Each dial is shipped only if it passes parameter recovery "
         "(simulate → re-fit → the dial comes back). Lean the load-bearing claims on timing / RT / neural readouts.",
         fontsize=8.2, ha="center", color="#666666")

path = os.path.join(OUT, "fig_b8_concept_slide.png")
fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("wrote", path)
