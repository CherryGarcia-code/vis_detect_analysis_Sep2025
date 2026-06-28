"""Is the negative log2 dR a fast_fit (single-lambda) artifact? Refit a flagged
VISp unit with FULL nested CV (per-fold lambda) under the baseline-tagB log2
encoding, and compare dR to the fast_fit value (-0.0036 for u492)."""
import sys, dataclasses, time
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
ks = load_khilkevich_session(SESS)
trdf = ks.trials
UNITS = [492, 526]  # clear linear signal, moderate spike count (faster than 479)

def build(cfg):
    trials, units = khilkevich_trial_regressors(ks, cfg, region="VISp")
    fixed = []
    for i, tr in enumerate(trials):
        edges = trial_bin_edges(tr.t_start, tr.t_end, cfg.bin_s)
        tf_arr = np.asarray(trdf["TF"].iloc[i], float)
        ft_arr = np.asarray(trdf["frame_time"].iloc[i], float)
        tag_arr = np.asarray(trdf["tag"].iloc[i])
        m = (tag_arr == "B") & np.isfinite(tf_arr) & (tf_arr > 0)
        tb = _resample_to_bins(ft_arr[m], tf_arr[m], edges, cfg.bin_s, fill=0.0)
        fixed.append(dataclasses.replace(tr, tf_bins=tb))
    return fixed, units

for fast in (True, False):
    cfg = TFGLMConfig(include_movement=False, include_phase=False, fast_fit=fast)
    fixed, units = build(cfg)
    d = assemble_design(fixed, cfg)
    folds = make_trial_folds(d.trial_index, cfg.n_folds, cfg.seed)
    X = d.X; tf = np.arange(*d.col_groups["tf"].indices(X.shape[1]))
    Xr = X.copy(); Xr[:, tf] = 0.0
    print(f"=== fast_fit={fast} ===", flush=True)
    for u in UNITS:
        if u not in units:
            continue
        y = count_vector(fixed, units[u], d)
        t0 = time.time()
        full = fit_poisson_cv(X, y, cfg, folds)
        red = fit_poisson_cv(Xr, y, cfg, folds)
        o = identify_tf_responsive(d, y, full, red, cfg)
        print(f"  u{u} {int(y.sum())}spk: dR={o['r_full_mean']-o['r_red_mean']:+.4f} "
              f"c2_p={o['c2_p']:.1e} resp={o['is_responsive']} [{time.time()-t0:.0f}s]",
              flush=True)
