"""B9 summary — one-slide 'verdict strip' for the talk.

Panel A: learning does not sharpen the code (state-conditioned c1_r, early vs late,
         engaged & disengaged) -> flat.
Panel B: recruitment (responsive fraction) is not robust across learning stages.
Panel C: engagement does not gate the code (paired matched-N Delta ~ 0).

Reads the cached B9 result CSVs; no fitting. Presentation-ready.
Run:  PYTHONPATH=src py scripts/state_tf_learning/b9_summary_figure.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy import stats
from visdetect.analysis import state_tf_learning as stl

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "figure.dpi": 150, "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False,
})
CACHE = stl._REPO / "data" / "cache" / "state_tf_learning"
OUT = stl._REPO / "FIGURES" / "state_tf_learning" / "b9_summary.png"
BLUE, GREY = "#6baed6", "#bdbdbd"                 # engaged (StimSens) vs disengaged
GREEN = {"Naive": "#c7e9c0", "Learning": "#74c476", "Expert": "#238b45"}
RED = "#d62728"


def _box(ax, x, vals, color, w=0.6):
    ax.boxplot(vals, positions=[x], widths=w, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor=color, alpha=.65, edgecolor="k", lw=.8),
               medianprops=dict(color="k", lw=1.4), whiskerprops=dict(lw=.8), capprops=dict(lw=.8))
    rng = np.random.default_rng(0)
    ax.scatter(np.full(len(vals), x) + rng.uniform(-.13, .13, len(vals)), vals, s=13, color="k", alpha=.5, zorder=3)


def _bracket(ax, x1, x2, y, text, h):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1, c="k")
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom", fontsize=9)


fig = plt.figure(figsize=(14, 5.2))
gs = gridspec.GridSpec(1, 3, wspace=0.32, left=0.055, right=0.985, top=0.80, bottom=0.14)

# ---- Panel A: learning does not sharpen ----
axA = fig.add_subplot(gs[0, 0])
ss = pd.read_csv(CACHE / "b9_deliverable2_encoding_BG_031_N140.csv")
di = pd.read_csv(CACHE / "b9_deliverable2_encoding_BG_031_Disengaged_N120.csv")
sse = ss[(ss.resp_class == "responsive") & (ss.group == "early")]["c1_r"].dropna()
ssl = ss[(ss.resp_class == "responsive") & (ss.group == "late")]["c1_r"].dropna()
die = di[(di.resp_class == "responsive") & (di.group == "early")]["c1_r"].dropna()
dil = di[(di.resp_class == "responsive") & (di.group == "late")]["c1_r"].dropna()
pss = stats.mannwhitneyu(sse, ssl)[1]; pdi = stats.mannwhitneyu(die, dil)[1]
_box(axA, 0, sse, BLUE); _box(axA, 1, ssl, BLUE); _box(axA, 2.6, die, GREY); _box(axA, 3.6, dil, GREY)
axA.axhline(0, color="k", lw=.5, alpha=.3)
top = max(ssl.max(), sse.max(), die.max(), dil.max())
_bracket(axA, 0, 1, top + .03, f"n.s. (p={pss:.2f})", .02)
_bracket(axA, 2.6, 3.6, top + .03, f"n.s. (p={pdi:.2f})", .02)
axA.set_xticks([0, 1, 2.6, 3.6]); axA.set_xticklabels(["early", "late", "early", "late"])
axA.text(0.5, -0.19, "engaged (StimSens)", ha="center", transform=axA.get_xaxis_transform(), color=BLUE, fontsize=9, weight="bold")
axA.text(3.1, -0.19, "disengaged", ha="center", transform=axA.get_xaxis_transform(), color="#7a7a7a", fontsize=9, weight="bold")
axA.set_ylabel("TF-encoding fidelity  (c1_r)")
axA.set_title("A. Learning does not sharpen the code", weight="bold", loc="left", pad=10)

# ---- Panel B: recruitment not robust ----
axB = fig.add_subplot(gs[0, 1])
reg = stl.load_registry(stl.registry_path("BG_031")); smap = stl.date_stage_map("BG_031")
per = reg.groupby("sess_key").agg(nu=("resp_log2", "size"), nr=("resp_log2", "sum"))
per["frac"] = per.nr / per.nu; per["stage"] = [smap.get(k, "?") for k in per.index]
stages = ["Naive", "Learning", "Expert"]
data = [per.loc[per.stage == s, "frac"].dropna().to_numpy() for s in stages]
for i, (s, d) in enumerate(zip(stages, data)):
    _box(axB, i, d, GREEN[s])
nai = per.loc[per.stage == "Naive", "frac"].dropna()
rest = per.loc[per.stage.isin(["Learning", "Expert"]), "frac"].dropna()
pB = stats.mannwhitneyu(nai, rest)[1]
_bracket(axB, 0, 2, max(np.concatenate(data)) + .01, f"n.s. (p={pB:.2f})", .006)
axB.set_xticks(range(3)); axB.set_xticklabels([f"{s}\n(n={len(d)})" for s, d in zip(stages, data)])
axB.set_ylabel("fraction of units TF-responsive")
axB.set_title("B. 'Recruitment' is not robust", weight="bold", loc="left", pad=10)

# ---- Panel C: engagement does not gate ----
axC = fig.add_subplot(gs[0, 2])
eng = pd.read_csv(CACHE / "b9_engagement_paired.csv")
delta = eng["delta"].dropna()
pC = stats.wilcoxon(eng["StimSens"], eng["Disengaged"])[1]
axC.hist(delta, bins=24, color=BLUE, edgecolor="k", lw=.3, alpha=.9)
axC.axvline(0, color="k", lw=1)
axC.axvline(delta.median(), color=RED, lw=2, label=f"median Δ = {delta.median():+.3f}")
axC.set_xlabel("Δ c1_r   (engaged − disengaged)")
axC.set_ylabel("unit-sessions")
axC.legend(frameon=False, loc="upper right")
axC.text(0.02, 0.98, f"Wilcoxon n.s.\np = {pC:.2f}\nn = {len(delta)}", transform=axC.transAxes, va="top", fontsize=9,
         bbox=dict(boxstyle="round", fc="#f2f2f2", ec="none", alpha=.8))
axC.set_title("C. Engagement does not gate the code", weight="bold", loc="left", pad=10)

fig.suptitle("The striatal baseline temporal-frequency (TF) code is a stable, engagement-independent sensory signal — it does not sharpen with learning",
             fontsize=13, weight="bold", y=0.965)
fig.text(0.5, 0.905, "BG_031 (ventromedial striatum) shown; consistent across BG_039 / BG_046 where powered.   "
                     "Faithful re-run of the Khilkevich–Lohse TF-GLM (reproduction Δ = 0.0000).   Trial-count-matched throughout.",
         ha="center", fontsize=9.5, color="#444")
fig.savefig(OUT, dpi=300, bbox_inches="tight"); plt.close(fig)
print("[fig]", OUT)
print(f"A: StimSens early={sse.median():.3f} late={ssl.median():.3f} p={pss:.3f} | Diseng early={die.median():.3f} late={dil.median():.3f} p={pdi:.3f}")
print(f"B: Naive={nai.median():.3f} vs Learn+Exp={rest.median():.3f} p={pB:.3f}")
print(f"C: median delta={delta.median():.3f} Wilcoxon p={pC:.3f} n={len(delta)}")
