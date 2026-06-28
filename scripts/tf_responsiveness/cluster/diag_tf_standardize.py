"""Diagnostic: does standardizing the log2-TF block recover TF-responsiveness?

The cluster run gave 0% TF-responsive in VISp (biologically impossible for
visual cortex viewing a drifting grating). The `tf` design block has column-SD
~0.55 while wheel/motion/pupil are standardized to 1.0; under shared-lambda
ridge the TF kernel may be over-shrunk. This refits a few high-spike VISp units
with the TF block AS-IS vs Z-SCORED and compares the TF-ablation improvement
(r_full - r_red) and C2 p-value.
"""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, "E:/python_analysis/git_repos/vd_tf_phase0/src")
from visdetect.analysis.tf_glm import (
    TFGLMConfig, assemble_design, count_vector, fit_poisson_cv,
    make_trial_folds, identify_tf_responsive,
)
from visdetect.analysis.tf_glm_data import (
    load_khilkevich_session, khilkevich_trial_regressors,
)

SESS = "X:/public/projects/MoHa_20260212_dmdmTemporalExpectation/data/npx_converted/1116764/ML_1116764_S02_M2_V1"
cfg = TFGLMConfig(include_movement=True, include_phase=False, fast_fit=True)
ks = load_khilkevich_session(SESS)
trials, units = khilkevich_trial_regressors(ks, cfg, region="VISp")
d = assemble_design(trials, cfg)
folds = make_trial_folds(d.trial_index, cfg.n_folds, cfg.seed)
X = d.X
tf_cols = np.arange(*d.col_groups["tf"].indices(X.shape[1]))

# Standardized-TF variant: z-score each TF column to unit SD (like wheel/motion).
Xs = X.copy()
mu = Xs[:, tf_cols].mean(0)
sd = Xs[:, tf_cols].std(0)
sd[sd < 1e-9] = 1.0
Xs[:, tf_cols] = (Xs[:, tf_cols] - mu) / sd

# Reduced (TF zeroed) is identical for both variants.
Xr = X.copy(); Xr[:, tf_cols] = 0.0

# Pick the 4 highest-spike VISp units.
spk = {u: float(np.sum(np.isfinite(units[u]))) for u in units}
uids = [u for u in sorted(units, key=lambda u: spk[u], reverse=True)][:4]

print(f"{len(trials)} trials, X={X.shape}, tf_cols={tf_cols.size}")
print(f"{'unit':>6} {'spk':>7} | {'dR_asis':>8} {'p_asis':>8} | {'dR_zsc':>8} {'p_zsc':>8}")
for u in uids:
    y = count_vector(trials, units[u], d)
    if y.sum() < 500:
        continue
    red = fit_poisson_cv(Xr, y, cfg, folds)
    full_a = fit_poisson_cv(X, y, cfg, folds)
    full_z = fit_poisson_cv(Xs, y, cfg, folds)
    oa = identify_tf_responsive(d, y, full_a, red, cfg)
    oz = identify_tf_responsive(d, y, full_z, red, cfg)
    dRa = oa["r_full_mean"] - oa["r_red_mean"]
    dRz = oz["r_full_mean"] - oz["r_red_mean"]
    print(f"{u:>6} {int(y.sum()):>7} | {dRa:>+8.4f} {oa['c2_p']:>8.1e} | "
          f"{dRz:>+8.4f} {oz['c2_p']:>8.1e}", flush=True)
