"""Replicate BaseTFFrames_to_ms.m: TF regressor = BASELINE grating frames only
(tag=='B'), gray dropped, change excluded. Then log2/0.25 (via _tf_octaves).
Does this recover TF-responsiveness under the log2 encoding for the flagged
VISp units (matching the authors' construction)?"""
import sys, dataclasses
import numpy as np
sys.path.insert(0, "E:/python_analysis/git_repos/vd_tf_phase0/src")
from visdetect.analysis.tf_glm import (
    TFGLMConfig, assemble_design, count_vector, fit_poisson_cv,
    make_trial_folds, identify_tf_responsive, trial_bin_edges,
)
from visdetect.analysis.tf_glm_data import (
    load_khilkevich_session, khilkevich_trial_regressors, _resample_to_bins,
)
SESS = "X:/public/projects/MoHa_20260212_dmdmTemporalExpectation/data/npx_converted/1116764/ML_1116764_S02_M2_V1"
cfg = TFGLMConfig(include_movement=False, include_phase=False, fast_fit=True)
ks = load_khilkevich_session(SESS)
trials, units = khilkevich_trial_regressors(ks, cfg, region="VISp")
trdf = ks.trials
bs = cfg.bin_s
FLAGGED = [479, 492, 526, 490]

# Rebuild tf_bins from tag=='B' (baseline grating) frames ONLY, placed by their
# own frame_time onto each trial's bins; gray/change/post -> 0 (then _tf_octaves
# maps 0 -> 0 octaves).
fixed = []
for i, tr in enumerate(trials):
    edges = trial_bin_edges(tr.t_start, tr.t_end, bs)
    tf_arr = np.asarray(trdf["TF"].iloc[i], float)
    ft_arr = np.asarray(trdf["frame_time"].iloc[i], float)
    tag_arr = np.asarray(trdf["tag"].iloc[i])
    m = (tag_arr == "B") & np.isfinite(tf_arr) & (tf_arr > 0)
    tb = _resample_to_bins(ft_arr[m], tf_arr[m], edges, bs, fill=0.0)
    fixed.append(dataclasses.replace(tr, tf_bins=tb))

allz = np.concatenate([t.tf_bins for t in fixed])
nz = allz[allz > 0]
print(f"tag=B tf_bins: zero-frac={np.mean(allz==0):.2f} "
      f"nonzero mean={nz.mean():.3f} Hz sd={nz.std():.3f} "
      f"-> log2 octaves nonzero range [{np.log2(nz.min())/0.25:.1f},{np.log2(nz.max())/0.25:.1f}]")

d = assemble_design(fixed, cfg)
folds = make_trial_folds(d.trial_index, cfg.n_folds, cfg.seed)
X = d.X; tf = np.arange(*d.col_groups["tf"].indices(X.shape[1]))
Xr = X.copy(); Xr[:, tf] = 0.0
print(f"tf col-SD = {X[:, tf].std(0).mean():.3f}\n")
print(f"{'unit':>6} {'spk':>7} | {'dR_log2_baseB':>13} {'c2_p':>9} {'resp':>6}")
for u in FLAGGED:
    if u not in units:
        print(f"{u:>6}  (missing)"); continue
    y = count_vector(fixed, units[u], d)
    full = fit_poisson_cv(X, y, cfg, folds)
    red = fit_poisson_cv(Xr, y, cfg, folds)
    o = identify_tf_responsive(d, y, full, red, cfg)
    print(f"{u:>6} {int(y.sum()):>7} | {o['r_full_mean']-o['r_red_mean']:>+13.4f} "
          f"{o['c2_p']:>9.1e} {str(o['is_responsive']):>6}", flush=True)
