"""Where does nonzero tf_bins land relative to baseline vs change? Clock check."""
import sys
import numpy as np
sys.path.insert(0, "E:/python_analysis/git_repos/vd_tf_phase0/src")
from visdetect.analysis.tf_glm import TFGLMConfig
from visdetect.analysis.tf_glm_data import (
    load_khilkevich_session, khilkevich_trial_regressors, COL_TF_COL,
    COL_STIM_FRAME_TIME,
)
SESS = "X:/public/projects/MoHa_20260212_dmdmTemporalExpectation/data/npx_converted/1116764/ML_1116764_S02_M2_V1"
cfg = TFGLMConfig(include_movement=True)
ks = load_khilkevich_session(SESS)
ft = ks.stim[COL_STIM_FRAME_TIME].to_numpy(float)
tf = ks.stim[COL_TF_COL].to_numpy(float)
trials, _ = khilkevich_trial_regressors(ks, cfg, region="VISp")
bs = cfg.bin_s
print(f"stim frames span [{ft.min():.2f},{ft.max():.2f}]s; TF>0 frac={np.mean(tf>0):.2f}")
print(f"{'tr':>3} {'t_start':>9} {'t_end':>9} {'chg_t':>9} {'nbin':>5} "
      f"{'nz_first':>8} {'nz_last':>8} {'chg_bin':>7} {'frInWin':>7} {'frTF>0':>7}")
for i in [0, 1, 2, 10, 50]:
    tr = trials[i]
    tb = np.asarray(tr.tf_bins, float)
    nz = np.where(tb != 0.0)[0]
    chg = tr.change_time
    chg_bin = int((chg - tr.t_start) / bs) if np.isfinite(chg) else -1
    inwin = (ft >= tr.t_start) & (ft < tr.t_end)
    frtf = np.mean(tf[inwin] > 0) if inwin.sum() else float("nan")
    print(f"{i:>3} {tr.t_start:>9.2f} {tr.t_end:>9.2f} "
          f"{(chg if np.isfinite(chg) else -1):>9.2f} {tb.size:>5} "
          f"{(nz[0] if nz.size else -1):>8} {(nz[-1] if nz.size else -1):>8} "
          f"{chg_bin:>7} {int(inwin.sum()):>7} {frtf:>7.2f}")
# Does TF>0 occur during BASELINE (t_start..change) for trial 0?
tr = trials[0]
inbase = (ft >= tr.t_start) & (ft < (tr.change_time if np.isfinite(tr.change_time) else tr.t_end))
print(f"\ntrial0 baseline window [{tr.t_start:.2f},{tr.change_time:.2f}]: "
      f"{int(inbase.sum())} frames, TF>0 frac={np.mean(tf[inbase]>0) if inbase.sum() else float('nan'):.2f}, "
      f"TF mean={np.nanmean(tf[inbase]) if inbase.sum() else float('nan'):.3f}")
