"""Test the fix: rebuild tf_bins from each trial's OWN per-frame TF/frame_time
arrays (trials.parquet), and check the TF kernel becomes predictive for VISp."""
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
cfg = TFGLMConfig(include_movement=True, include_phase=False, fast_fit=True)
ks = load_khilkevich_session(SESS)
trials, units = khilkevich_trial_regressors(ks, cfg, region="VISp")
trdf = ks.trials
bs = cfg.bin_s

# Rebuild tf_bins from per-trial frame arrays (THE FIX). Two variants:
#  instant  = whole-trial instantaneous TF (gray=0, baseline fluct, change high)
#  baseonly = zero TF after change onset (baseline fluctuations only, BG-path style)
fixed_inst, fixed_base = [], []
for i, tr in enumerate(trials):
    edges = trial_bin_edges(tr.t_start, tr.t_end, bs)
    tf_arr = np.asarray(trdf["TF"].iloc[i], float)
    ft_arr = np.asarray(trdf["frame_time"].iloc[i], float)
    new_tf = _resample_to_bins(ft_arr, tf_arr, edges, bs, fill=0.0)
    fixed_inst.append(dataclasses.replace(tr, tf_bins=new_tf.copy()))
    bo = new_tf.copy()
    if np.isfinite(tr.change_time):
        bo[edges >= tr.change_time] = 0.0
    fixed_base.append(dataclasses.replace(tr, tf_bins=bo))

for nm, fx in [("instant", fixed_inst), ("baseonly", fixed_base)]:
    allz = np.concatenate([t.tf_bins for t in fx])
    print(f"{nm}: tf_bins zero-frac={np.mean(allz==0):.2f} "
          f"mean={allz.mean():.3f} sd={allz.std():.3f} max={allz.max():.2f}")

spk = {u: float(np.sum(np.isfinite(units[u]))) for u in units}
uids = [u for u in sorted(units, key=lambda u: spk[u], reverse=True)][:4]

for nm, fx in [("instant", fixed_inst), ("baseonly", fixed_base)]:
    d = assemble_design(fx, cfg)
    folds = make_trial_folds(d.trial_index, cfg.n_folds, cfg.seed)
    X = d.X; tf = np.arange(*d.col_groups["tf"].indices(X.shape[1]))
    Xr = X.copy(); Xr[:, tf] = 0.0
    print(f"\n--- {nm} (tf col-SD={X[:,tf].std(0).mean():.3f}) ---")
    for u in uids:
        y = count_vector(fx, units[u], d)
        full = fit_poisson_cv(X, y, cfg, folds)
        red = fit_poisson_cv(Xr, y, cfg, folds)
        o = identify_tf_responsive(d, y, full, red, cfg)
        dR = o["r_full_mean"] - o["r_red_mean"]
        print(f"  u{u} {int(y.sum())}spk: dR={dR:+.4f} c2_p={o['c2_p']:.1e} "
              f"resp={o['is_responsive']}", flush=True)
