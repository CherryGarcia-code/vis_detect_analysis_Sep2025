"""B9 side-finding — TF-responsive RECRUITMENT across learning (registry-only, zero fitting).

B9's original readout (encoding STRENGTH, c1_r among responsive units) is null across
Learning->Expert at matched state. This asks the complementary question the registry
hinted at: does the FRACTION of striatal units that are TF-responsive change with
learning stage? Yield-controlled (units/session is ~flat across stages), so a fraction
change is recruitment, not recording drift.

Run:  PYTHONPATH=src py scripts/state_tf_learning/b9_recruitment.py
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
from scipy import stats
from visdetect.analysis import state_tf_learning as stl
from visdetect.analysis.config import canonical_session_id as csid

SUBJECTS = ["BG_031", "BG_039"]
STAGES = ["Naive", "Learning", "Expert"]
STAGE_COL = {"Naive": "#c7c7c7", "Learning": "#f0a848", "Expert": "#3474ae"}
OUT = stl._REPO / "FIGURES" / "state_tf_learning"


def per_session_fraction(subject):
    reg = stl.load_registry(stl.registry_path(subject))
    man = pd.read_csv(stl.manifest_path(subject)); man["sess_key"] = man["session_name"].map(csid)
    smap = dict(zip(man.sess_key, man.stage.astype(str)))
    per = reg.groupby("sess_key").agg(n_units=("resp_log2", "size"), n_resp=("resp_log2", "sum"))
    per["frac"] = per["n_resp"] / per["n_units"]
    per["stage"] = [smap.get(k, "?") for k in per.index]
    return per


fig, axes = plt.subplots(1, len(SUBJECTS), figsize=(10.5, 4.4), sharey=True)
for ax, subj in zip(np.atleast_1d(axes), SUBJECTS):
    per = per_session_fraction(subj)
    data = [per.loc[per.stage == s, "frac"].dropna().to_numpy() for s in STAGES]
    bp = ax.boxplot(data, tick_labels=[f"{s}\n(n={len(d)} sess)" for s, d in zip(STAGES, data)],
                    showfliers=False, widths=.55, patch_artist=True)
    for patch, s in zip(bp["boxes"], STAGES):
        patch.set_facecolor(STAGE_COL[s]); patch.set_alpha(.55)
    rng = np.random.default_rng(0)
    for i, d in enumerate(data):
        ax.scatter(np.full(len(d), i + 1) + rng.uniform(-.12, .12, len(d)), d, s=22, color="k", alpha=.6, zorder=3)
    present = [(s, d) for s, d in zip(STAGES, data) if len(d)]
    kw = stats.kruskal(*[d for _, d in present])[1] if len(present) >= 2 and all(len(d) for _, d in present) else np.nan
    nai = per.loc[per.stage == "Naive", "frac"].dropna()
    learnexp = per.loc[per.stage.isin(["Learning", "Expert"]), "frac"].dropna()
    mw = stats.mannwhitneyu(nai, learnexp, alternative="two-sided")[1] if len(nai) and len(learnexp) else np.nan
    ax.set_title(f"{subj}\nKruskal p={kw:.3f} | Naive vs Learn+Exp p={mw:.3f}", fontsize=10)
    ax.set_xlabel("learning stage")
axes[0].set_ylabel("TF-responsive fraction (per session)")
fig.suptitle("B9 side-finding: TF-responsive RECRUITMENT with learning (registry, yield-controlled)", fontsize=12)
fig.tight_layout()
out = OUT / "b9_recruitment_fraction.png"
fig.savefig(out, dpi=150); plt.close(fig)
print(f"[fig] {out}")

for subj in SUBJECTS:
    per = per_session_fraction(subj)
    print(f"\n=== {subj} ===")
    for s in STAGES:
        d = per.loc[per.stage == s, "frac"].dropna()
        if len(d):
            print(f"  {s:9s} n_sess={len(d):2d}  median_units={per.loc[per.stage==s,'n_units'].median():.0f}  "
                  f"median_frac={d.median():.3f}  IQR=[{d.quantile(.25):.3f},{d.quantile(.75):.3f}]")
    nai = per.loc[per.stage == "Naive", "frac"].dropna()
    le = per.loc[per.stage.isin(["Learning", "Expert"]), "frac"].dropna()
    if len(nai) and len(le):
        print(f"  Naive vs Learn+Exp: MWU p={stats.mannwhitneyu(nai, le, alternative='two-sided')[1]:.4f} "
              f"(Naive median {nai.median():.3f} vs {le.median():.3f})")
