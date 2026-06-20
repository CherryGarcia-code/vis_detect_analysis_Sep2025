"""Diagnostic: why does F1D's Impulsive line spike (~session 20-25) when F1B's
does not, at the SAME 50% criterion? Reconcile the two thresholds per cell.

F1B  = decision_latents.descriptive_cell_table(min_cell_trials=20) -> psy_threshold
F1D  = explore script: refit per (session,mood) GO trials, require >=8 go trials,
       threshold at p=0.5 = 2**(-a/b) clamped [1,8].

Identical math for p=0.5 (logit(0.5)=0). So any divergence is INCLUSION
(min_cell_trials=20 vs >=8 go) and/or binned-mean over few points. Print both.
"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from visdetect.analysis.config import ROOT
from visdetect.analysis import decision_latents as dl
from visdetect.analysis.decision_latents import _logistic

CACHE_DIR = os.path.join(ROOT, "data", "cache", "decision_latents")
trials = pd.read_csv(os.path.join(CACHE_DIR, "decision_latents_trialtable.csv"))

# session_idx is already in the trial table; build the same cells F1B uses
cells = dl.descriptive_cell_table(trials)
sidx = {s.zfill(8): i for i, s in enumerate(dl.enumerate_valid_sessions())}
cells["session_idx"] = cells["session_name"].astype(str).str.zfill(8).map(sidx)


def fit_ab(g):
    if len(g) < 8 or g["change_size"].nunique() < 2:
        return (np.nan, np.nan)
    x = np.log2(g["change_size"].values); y = g["lick"].values.astype(float)
    try:
        (a, b), _ = curve_fit(_logistic, x, y, p0=[0.0, 1.0],
                              bounds=([-20.0, -20.0], [20.0, 20.0]), maxfev=5000)
        return (float(a), float(b))
    except Exception:
        return (np.nan, np.nan)


print(f"{'sess':>9} {'sidx':>4} {'nTot':>5} {'nGo':>4} {'min20?':>6} "
      f"{'a':>7} {'b':>7} {'thrF1D':>7} {'thrF1B':>7}")
rows = []
for (sname, mood), cell in trials.groupby(["session_name", "state_label"]):
    if mood != "Impulsive":
        continue
    go = cell[cell["change_size"] > 1.0]
    n_tot, n_go = len(cell), len(go)
    a, b = fit_ab(go)
    thr_f1d = np.nan
    if np.isfinite(b) and abs(b) >= 1e-3:
        thr_f1d = float(np.clip(2.0 ** (-a / b), 1.0, 8.0))
    cr = cells[(cells["session_name"].astype(str) == str(sname)) &
               (cells["state_label"] == "Impulsive")]
    thr_f1b = float(cr["psy_threshold"].iloc[0]) if (len(cr) and "psy_threshold" in cr) else np.nan
    si = sidx.get(str(sname).zfill(8))
    rows.append((si, thr_f1d, thr_f1b, n_tot, n_go))
    print(f"{str(sname):>9} {str(si):>4} {n_tot:>5} {n_go:>4} {str(n_tot>=20):>6} "
          f"{a:>7.2f} {b:>7.2f} {thr_f1d:>7.3f} {thr_f1b:>7.3f}")

# Which cells are the spike? (F1D threshold meaningfully above floor)
rows = [r for r in rows if r[0] is not None]
spike = [r for r in rows if np.isfinite(r[1]) and r[1] > 1.10]
print("\nImpulsive cells with F1D thr > 1.10 (the spike-drivers):")
for si, t5d, t5b, nt, ng in sorted(spike):
    incl = "IN F1B" if (np.isfinite(t5b)) else "DROPPED by min20" if nt < 20 else "F1B-NaN"
    print(f"  sidx={si:>3}  F1D={t5d:.3f}  F1B={t5b if np.isfinite(t5b) else float('nan'):.3f}"
          f"  nTot={nt} nGo={ng}  -> {incl}")

# Reconcile: apply min_cell_trials=20 to the F1D set and see if the spike dies
print("\n--- F1D binned trend WITH vs WITHOUT the min20 filter (Impulsive) ---")
arr = np.array([(r[0], r[1], r[3]) for r in rows if np.isfinite(r[1])], dtype=float)
for label, mask in (("F1D as-is (>=8 go)", arr[:, 2] >= 0),
                    ("F1D + min20 filter", arr[:, 2] >= 20)):
    sub = arr[mask]
    if sub.shape[0] < 2:
        print(f"  {label}: <2 cells"); continue
    xv, yv = sub[:, 0], sub[:, 1]
    edges = np.linspace(xv.min(), xv.max(), 6)
    bc = 0.5 * (edges[:-1] + edges[1:])
    bm = [np.nanmean(yv[(xv >= edges[i]) & (xv <= edges[i + 1])])
          if np.any((xv >= edges[i]) & (xv <= edges[i + 1])) else np.nan
          for i in range(len(edges) - 1)]
    print(f"  {label}: n={sub.shape[0]:>2}  peak_bin_mean={np.nanmax(bm):.3f}  "
          f"trend=" + " ".join(f"{v:.2f}" for v in bm))
