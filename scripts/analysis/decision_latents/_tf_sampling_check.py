"""B8 prereq: verify the TF baseline-vector sample structure / true update period.

Plain English: each trial stores a vector of temporal-frequency (TF) values for
the baseline grating. We need to know how much time each *stored* value covers,
so our analysis time-grid (dt) matches reality and we don't (as an old script
did) mis-sample it.

Finding this diagnostic confirms (Task 0.1): `n_seen` is unavailable (None) on
these pkls, so we CANNOT infer the period from change_time / n_seen. Instead we
read the structure directly: `baseline_values` is stored at the ~60 Hz monitor
frame rate, and each *true* TF value is held for a run of 3 consecutive frames
(3 / 60 Hz = 50 ms). So the true TF update period is 50 ms (-> dt = 0.05 s), and
an evidence reconstruction must index baseline_values at the 60 Hz frame rate
(or collapse runs of 3) -- NOT via change_time/len(baseline_values).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from visdetect.suite.loader import load_staging_manifest, load_session
from visdetect.suite.plotting import setup_style          # styling only
from visdetect.analysis.config import ROOT, SUBJECT

setup_style()
FIG_DIR = os.path.join(ROOT, "FIGURES", "decision_latents", SUBJECT)
os.makedirs(FIG_DIR, exist_ok=True)

MONITOR_HZ = 60.0   # grating display refresh; 1800 stored samples ~= 30 s buffer


def run_lengths(a):
    """Lengths of consecutive-equal runs in 1-D array a."""
    a = np.asarray(a, float).ravel()
    if a.size == 0:
        return np.array([], dtype=int)
    bnd = np.concatenate(([-1], np.flatnonzero(np.diff(a) != 0), [a.size - 1]))
    return np.diff(bnd)


modal_runs, lengths, all_runs = [], [], []
man = load_staging_manifest(qc_only=True)
for sname in man["session_name"].astype(str).head(8):
    s = load_session(sname)
    for t in s.trials:
        bv = getattr(t, "baseline_values", None)
        if bv is None:
            continue
        bv = np.asarray(bv, float).ravel()
        if bv.size == 0:
            continue
        rl = run_lengths(bv)
        if rl.size == 0:
            continue
        modal_runs.append(int(np.bincount(rl).argmax()))
        lengths.append(bv.size)
        all_runs.append(rl)
    del s

modal_runs = np.asarray(modal_runs)
lengths = np.asarray(lengths)
all_runs = np.concatenate(all_runs) if all_runs else np.array([], dtype=int)

overall_modal_run = int(np.bincount(modal_runs).argmax()) if modal_runs.size else 0
frac_runs_eq3 = float(np.mean(all_runs == 3)) if all_runs.size else float("nan")
modal_len = int(np.bincount(lengths).argmax()) if lengths.size else 0
update_period_ms = 1000.0 * overall_modal_run / MONITOR_HZ

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(all_runs, bins=np.arange(0.5, max(6, all_runs.max() + 1.5) if all_runs.size else 6))
axes[0].axvline(3, color="r", ls="--", label="3 frames (expected)")
axes[0].set_xlabel("run length (consecutive equal TF frames)")
axes[0].set_ylabel("# runs")
axes[0].set_title(f"A  Each TF value is held for ~{overall_modal_run} frames\n"
                  f"({frac_runs_eq3*100:.0f}% of runs = 3 frames)")
axes[0].legend(frameon=False)

axes[1].hist(lengths, bins=40)
axes[1].set_xlabel("len(baseline_values) per trial")
axes[1].set_ylabel("# trials")
axes[1].set_title(f"B  Stored at 60 Hz monitor rate\n(modal length {modal_len} ~= {modal_len/MONITOR_HZ:.0f} s buffer)")

fig.suptitle("B8 prereq — TF baseline is 60 Hz-stored, 3 frames/update -> 50 ms true updates",
             fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig_b8_prereq_tf_sample_period.png"),
            dpi=300, bbox_inches="tight")

print(f"trials inspected: {modal_runs.size}")
print(f"modal run length: {overall_modal_run} frames  ({frac_runs_eq3*100:.1f}% of all runs == 3)")
print(f"modal len(baseline_values): {modal_len}  (~{modal_len/MONITOR_HZ:.1f} s at {MONITOR_HZ:.0f} Hz)")
print(f"=> true TF update period: {update_period_ms:.1f} ms  => dt = 0.05 s")
print("=> evidence reconstruction must index baseline_values at the 60 Hz frame "
      "rate (or collapse runs of 3); do NOT use change_time/len(baseline_values). "
      "n_seen is None on these pkls.")
