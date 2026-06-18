"""B8 prereq: sanity-check state labels, esp. on newly-labeled naive sessions.

Plain English: the mood labeler was trained on good-behavior sessions. The
early/naive sessions are 'out of distribution', so before we trust their moods
we look at: how much of each session is each mood, and how confident the
labeler is. Low confidence on the new sessions = treat their moods as shaky.
"""
import os, glob, gc, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from visdetect.suite.plotting import setup_style          # styling only
from visdetect.analysis.behavior import compute_session_performance
from visdetect.suite.loader import load_session
from visdetect.analysis.config import ROOT, SUBJECT
setup_style()
FIG_DIR = os.path.join(ROOT, "FIGURES", "decision_latents", SUBJECT); os.makedirs(FIG_DIR, exist_ok=True)
CACHE_DIR = os.path.join(ROOT, "data", "cache", "decision_latents"); os.makedirs(CACHE_DIR, exist_ok=True)

rows = []
for f in sorted(glob.glob("data/cache/state_tags/BG_046/*.csv")):
    sname = os.path.splitext(os.path.basename(f))[0]
    if not sname.isdigit():
        continue
    df = pd.read_csv(f)
    sess = load_session(sname)
    perf = compute_session_performance(sess)
    props = df["state_label"].value_counts(normalize=True)
    rows.append({"session": sname, "dprime": perf.get("d_prime", np.nan),
                 "mean_conf": df["state_confidence"].mean(),
                 **{m: props.get(m, 0.0) for m in
                    ["Impulsive", "StimSens", "Disengaged", "Abort"]}})
    del sess; gc.collect()
tab = pd.DataFrame(rows).sort_values("session")

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].scatter(tab["dprime"], tab["mean_conf"])
axes[0].set_xlabel("session d′"); axes[0].set_ylabel("mean label confidence")
axes[0].set_title("Label confidence vs performance\n(low-d′ naive sessions = watch here)")
bottom = np.zeros(len(tab))
for m, c in [("Impulsive", "#d62728"), ("StimSens", "#1f77b4"),
             ("Disengaged", "#9488bf"), ("Abort", "#7f7f7f")]:
    axes[1].bar(range(len(tab)), tab[m], bottom=bottom, label=m, color=c)
    bottom += tab[m].values
axes[1].set_xlabel("session (chronological)"); axes[1].set_ylabel("mood fraction")
axes[1].set_title("Mood composition per session"); axes[1].legend(frameon=False, fontsize=7)
fig.savefig(os.path.join(FIG_DIR, "fig_b8_prereq_label_reliability.png"), dpi=300, bbox_inches="tight")
tab.to_csv(os.path.join(CACHE_DIR, "b8_label_coverage.csv"), index=False)
print(tab.to_string(index=False))
