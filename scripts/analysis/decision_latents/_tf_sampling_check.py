"""B8 prereq: verify the TF baseline-vector sample period (should be ~50 ms).

Plain English: each trial stores a vector of temporal-frequency values shown
during the baseline. We need to know how many milliseconds each value covers,
so our time grid (dt) matches reality and we don't (as an old script did)
silently sub-sample every 3rd value.
"""
import os, sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from visdetect.suite.loader import load_staging_manifest, load_session
from visdetect.suite.plotting import setup_style          # styling only
from visdetect.analysis.config import ROOT, SUBJECT
setup_style()
FIG_DIR = os.path.join(ROOT, "FIGURES", "decision_latents", SUBJECT)
os.makedirs(FIG_DIR, exist_ok=True)

rows = []
man = load_staging_manifest(qc_only=True)
for sname in man["session_name"].astype(str).head(8):
    s = load_session(sname)
    for t in s.trials:
        bv = getattr(t, "baseline_values", None)
        ct = getattr(t, "change_time", None)
        nseen = getattr(t, "n_seen", None)
        if bv is None or ct is None or not nseen:
            continue
        # period implied if n_seen samples fill the pre-change window [0, change_time]
        rows.append(ct / nseen)
    del s
periods = np.asarray(rows, float)
periods = periods[np.isfinite(periods) & (periods > 0)]
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(periods * 1000, bins=60)
ax.axvline(50, color="r", ls="--", label="50 ms (expected)")
ax.set_xlabel("implied TF sample period (ms)"); ax.set_ylabel("trials")
ax.set_title("B8 prereq — TF baseline sample period\n(should peak at 50 ms)")
ax.legend(frameon=False)
fig.savefig(os.path.join(FIG_DIR, "fig_b8_prereq_tf_sample_period.png"), dpi=300, bbox_inches="tight")
print(f"median implied period: {np.median(periods)*1000:.1f} ms  (n={periods.size})")
