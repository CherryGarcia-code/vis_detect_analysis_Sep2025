"""Top suspect from the MATLAB diff: glmnet standardize=true z-scores the WHOLE
design matrix before fitting. We z-score only movement; TF + events are raw.
Test: standardize the ENTIRE design (every column to unit variance) before the
ridge-Poisson fit, log2-TF, on the linear-flagged VISp units. Does log2 recover?"""
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

# baseline-only tag-B TF (their construction)
fixed = []
for i, tr in enumerate(trials):
    edges = trial_bin_edges(tr.t_start, tr.t_end, bs)
    tf_arr = np.asarray(trdf["TF"].iloc[i], float)
    ft_arr = np.asarray(trdf["frame_time"].iloc[i], float)
    tag_arr = np.asarray(trdf["tag"].iloc[i])
    m = (tag_arr == "B") & np.isfinite(tf_arr) & (tf_arr > 0)
    fixed.append(dataclasses.replace(tr, tf_bins=_resample_to_bins(ft_arr[m], tf_arr[m], edges, bs, fill=0.0)))

d = assemble_design(fixed, cfg)
folds = make_trial_folds(d.trial_index, cfg.n_folds, cfg.seed)
X = d.X
tfcols = np.arange(*d.col_groups["tf"].indices(X.shape[1]))

def zscore_cols(M):
    M = M.copy()
    mu = M.mean(0); sd = M.std(0); sd[sd < 1e-9] = 1.0
    return (M - mu) / sd

# Full vs reduced, each standardized as glmnet would (per its own design).
Xr = X.copy(); Xr[:, tfcols] = 0.0
variants = {
    "raw (current)":        (X, Xr),
    "full-design z-scored":  (zscore_cols(X), zscore_cols(Xr)),
}
for name, (Xf, Xrd) in variants.items():
    print(f"\n--- {name} ---")
    for u in FLAGGED:
        if u not in units:
            continue
        y = count_vector(fixed, units[u], d)
        full = fit_poisson_cv(Xf, y, cfg, folds)
        red = fit_poisson_cv(Xrd, y, cfg, folds)
        o = identify_tf_responsive(d, y, full, red, cfg)
        print(f"  u{u} {int(y.sum())}spk: dR={o['r_full_mean']-o['r_red_mean']:+.4f} "
              f"c2_p={o['c2_p']:.1e} resp={o['is_responsive']}", flush=True)
