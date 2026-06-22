"""Controlled head-to-head: on the VISp units the linear-TF diagnostic FLAGGED,
vary ONLY the TF transform (linear vs log2-octaves), no-movement reduced model
(matching the diagnostic). Isolates exactly why log2 -> 0% while linear -> 27%."""
import sys
import numpy as np
sys.path.insert(0, "E:/python_analysis/git_repos/vd_tf_phase0/src")
import visdetect.analysis.tf_glm as TG
from visdetect.analysis.tf_glm import (
    TFGLMConfig, assemble_design, count_vector, fit_poisson_cv,
    make_trial_folds, identify_tf_responsive,
)
from visdetect.analysis.tf_glm_data import (
    load_khilkevich_session, khilkevich_trial_regressors,
)
SESS = "X:/public/projects/MoHa_20260212_dmdmTemporalExpectation/data/npx_converted/1116764/ML_1116764_S02_M2_V1"
# no-movement reduced model == the diagnostic that flagged 27%
cfg = TFGLMConfig(include_movement=False, include_phase=False, fast_fit=True)
ks = load_khilkevich_session(SESS)
trials, units = khilkevich_trial_regressors(ks, cfg, region="VISp")
FLAGGED = [479, 492, 526, 490, 555]  # from khilkevich_diagnostic.csv (S02 mostly)

orig = TG._tf_octaves
linear = lambda tf: np.asarray(tf, float).ravel()  # identity = raw linear TF

designs = {}
for name, fn in [("log2", orig), ("linear", linear)]:
    TG._tf_octaves = fn
    designs[name] = assemble_design(trials, cfg)
TG._tf_octaves = orig

# Reduced design (TF zeroed) is identical across transforms -> fit once/unit.
d0 = designs["log2"]
folds = make_trial_folds(d0.trial_index, cfg.n_folds, cfg.seed)
tfcols = np.arange(*d0.col_groups["tf"].indices(d0.X.shape[1]))
Xr = d0.X.copy(); Xr[:, tfcols] = 0.0
for nm, d in designs.items():
    print(f"{nm}: tf col-SD = {d.X[:, tfcols].std(0).mean():.3f}")

print(f"\n{'unit':>6} {'spk':>7} | {'dR_linear':>10} {'p_linear':>9} | "
      f"{'dR_log2':>9} {'p_log2':>9}")
for u in FLAGGED:
    if u not in units:
        print(f"{u:>6}  (not in this session)")
        continue
    y = count_vector(trials, units[u], d0)
    red = fit_poisson_cv(Xr, y, cfg, folds)
    res = {}
    for nm in ("linear", "log2"):
        full = fit_poisson_cv(designs[nm].X, y, cfg, folds)
        o = identify_tf_responsive(designs[nm], y, full, red, cfg)
        res[nm] = (o["r_full_mean"] - o["r_red_mean"], o["c2_p"])
    print(f"{u:>6} {int(y.sum()):>7} | {res['linear'][0]:>+10.4f} "
          f"{res['linear'][1]:>9.1e} | {res['log2'][0]:>+9.4f} "
          f"{res['log2'][1]:>9.1e}", flush=True)
