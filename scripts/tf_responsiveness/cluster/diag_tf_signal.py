"""Diagnostic: is the per-bin TF signal (tf_bins) a real fluctuating grating TF,
or a degenerate/mostly-zero signal (frame-time vs bin-edge mismatch)?"""
import sys
import numpy as np
sys.path.insert(0, "E:/python_analysis/git_repos/vd_tf_phase0/src")
from visdetect.analysis.tf_glm import TFGLMConfig, trial_bin_edges
from visdetect.analysis.tf_glm_data import (
    load_khilkevich_session, khilkevich_trial_regressors, COL_TF_COL,
    COL_STIM_FRAME_TIME,
)

SESS = "X:/public/projects/MoHa_20260212_dmdmTemporalExpectation/data/npx_converted/1116764/ML_1116764_S02_M2_V1"
cfg = TFGLMConfig(include_movement=True, include_phase=False)
ks = load_khilkevich_session(SESS)

stim = ks.stim
print("=== stim.csv ===")
print("cols:", list(stim.columns))
ft = stim[COL_STIM_FRAME_TIME].to_numpy(float)
tf = stim[COL_TF_COL].to_numpy(float)
print(f"frame_time: n={ft.size} range=[{np.nanmin(ft):.2f}, {np.nanmax(ft):.2f}] "
      f"median dt={np.nanmedian(np.diff(ft))*1000:.1f} ms")
print(f"TF column : range=[{np.nanmin(tf):.3f}, {np.nanmax(tf):.3f}] "
      f"mean={np.nanmean(tf):.3f} sd={np.nanstd(tf):.3f}")
print(f"TF unique (first 12): {np.unique(tf[np.isfinite(tf)])[:12]}")

trials, units = khilkevich_trial_regressors(ks, cfg, region="VISp")
print(f"\n=== {len(trials)} trials built ===")
# Per-trial: fraction of tf_bins that are exactly 0 (filled), and tf variance.
zero_frac, nz_std, n_bins = [], [], []
for tr in trials:
    tb = np.asarray(tr.tf_bins, float)
    if tb.size == 0:
        continue
    zero_frac.append(np.mean(tb == 0.0))
    nz = tb[tb != 0.0]
    nz_std.append(nz.std() if nz.size > 1 else 0.0)
    n_bins.append(tb.size)
zero_frac = np.array(zero_frac); nz_std = np.array(nz_std)
print(f"tf_bins zero-fraction per trial: median={np.median(zero_frac):.2f} "
      f"min={zero_frac.min():.2f} max={zero_frac.max():.2f}")
print(f"tf_bins NONZERO within-trial SD: median={np.median(nz_std):.4f} "
      f"(if ~0 -> no fluctuation; grating TF should fluctuate)")

# Show trial 0's first 40 bins of tf_bins.
t0 = trials[0]
edges = None
tb0 = np.asarray(t0.tf_bins, float)
print(f"\n=== trial 0: {tb0.size} bins; first 40 tf_bins ===")
print(np.round(tb0[:40], 3))

# Coverage: how many stim frames fall inside trial 0's bin span?
b = np.asarray(t0.bin_edges, float) if hasattr(t0, "bin_edges") else None
print(f"\ntrial0 attrs: {[a for a in vars(t0).keys()]}")
