"""Prototype the FULL faithful Khilkevich design and test if log2-TF recovers:
 - tag-B baseline-only TF (gray dropped, change excluded)
 - reconstructed grating phase (12x30deg bins) from cumulative integral of TF
 - tiled baseline: 80 x 200ms boxcars since baseline onset (drop <10-trial tiles)
 - whole-design standardization (glmnet standardize=true)
Replaces our single ramp with their tiled baseline; keeps trial_start (~baseON).
"""
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
cfg = TFGLMConfig(include_movement=False, include_phase=True, fast_fit=True)
ks = load_khilkevich_session(SESS)
trials, units = khilkevich_trial_regressors(ks, cfg, region="VISp")
trdf = ks.trials
bs = cfg.bin_s
FLAGGED = [479, 492, 526, 490]

# Per trial: tag-B baseline TF + reconstructed phase (deg) from cumulative cycles.
fixed = []
for i, tr in enumerate(trials):
    edges = trial_bin_edges(tr.t_start, tr.t_end, bs)
    tf_arr = np.asarray(trdf["TF"].iloc[i], float)
    ft_arr = np.asarray(trdf["frame_time"].iloc[i], float)
    tag_arr = np.asarray(trdf["tag"].iloc[i])
    m = (tag_arr == "B") & np.isfinite(tf_arr) & (tf_arr > 0)
    tfb = _resample_to_bins(ft_arr[m], tf_arr[m], edges, bs, fill=0.0)
    # phase: cumulative cycles * 360 mod 360, only during baseline grating
    base = edges < (tr.change_time if np.isfinite(tr.change_time) else tr.t_end)
    cyc = np.where(tfb > 0, tfb * bs, 0.0)
    phase = (np.cumsum(cyc) * 360.0) % 360.0
    ph = np.full(edges.size, np.nan)
    ok = base & (tfb > 0)
    ph[ok] = phase[ok]
    fixed.append(dataclasses.replace(tr, tf_bins=tfb, phase_bins=ph))

d = assemble_design(fixed, cfg)
ti = d.trial_index
be = d.bin_edges
tstart = np.array([tr.t_start for tr in fixed])[ti]
chg = np.array([tr.change_time for tr in fixed])[ti]
since = be - tstart
base = (since >= 0) & (~np.isfinite(chg) | (be < chg))
tile = np.floor(since / 0.200).astype(int)
tile[~base] = -1
tile[tile >= 80] = -1
TB = np.zeros((be.size, 80))
for k in range(80):
    sel = tile == k
    if sel.any() and np.unique(ti[sel]).size >= 10:   # drop <10-trial tiles
        TB[sel, k] = 1.0
TB = TB[:, TB.any(0)]

# Remove our ramp (time_in_base); append tiled baseline. tf is group 0 (before
# ramp) so its column indices are unchanged by the removal.
ramp = d.col_groups["time_in_base"]
keep = np.ones(d.X.shape[1], bool); keep[ramp] = False
tf_sl = d.col_groups["tf"]
tfcols = np.arange(tf_sl.start, tf_sl.stop)
X = np.hstack([d.X[:, keep], TB])
print(f"X={X.shape} (tiled-baseline cols kept={TB.shape[1]}, phase included, "
      f"ramp removed); tfcols={tfcols.size}")

def zc(M):
    M = M.copy(); mu = M.mean(0); sd = M.std(0); sd[sd < 1e-9] = 1.0
    return (M - mu) / sd

folds = make_trial_folds(d.trial_index, cfg.n_folds, cfg.seed)
Xr = X.copy(); Xr[:, tfcols] = 0.0
Xr_z = zc(Xr)
red_design = X  # tf cols here are LOG2 octaves (from assemble_design via _tf_octaves)

# Build a LINEAR-TF variant of the design's tf columns: re-FIR the raw linear
# tf_bins (Hz) in place of log2 octaves. We rebuild only the tf block per trial.
import visdetect.analysis.tf_glm as TG
from visdetect.analysis.tf_glm import fir_continuous, _resize as _rz
def tf_block(octave_fn):
    blocks = []
    for tr in fixed:
        e = trial_bin_edges(tr.t_start, tr.t_end, bs)
        blocks.append(fir_continuous(octave_fn(_rz(tr.tf_bins, e.size)), cfg.kern["tf"], bs))
    return np.concatenate(blocks, 0)
Xlin = X.copy(); Xlin[:, tfcols] = tf_block(lambda v: np.asarray(v, float).ravel())   # linear Hz
Xlog = X.copy(); Xlog[:, tfcols] = tf_block(TG._tf_octaves)                            # log2/0.25 (==assemble)
Xlin_z, Xlog_z = zc(Xlin), zc(Xlog)

print(f"\n{'unit':>6} {'spk':>7} | {'dR_linear':>10} {'p_lin':>8} | {'dR_log2':>9} {'p_log2':>8}")
for u in FLAGGED:
    if u not in units:
        continue
    y = count_vector(fixed, units[u], d)
    red = fit_poisson_cv(Xr_z, y, cfg, folds)
    res = {}
    for nm, Xv in (("lin", Xlin_z), ("log", Xlog_z)):
        full = fit_poisson_cv(Xv, y, cfg, folds)
        o = identify_tf_responsive(d, y, full, red, cfg)
        res[nm] = (o["r_full_mean"] - o["r_red_mean"], o["c2_p"])
    print(f"{u:>6} {int(y.sum()):>7} | {res['lin'][0]:>+10.4f} {res['lin'][1]:>8.1e} | "
          f"{res['log'][0]:>+9.4f} {res['log'][1]:>8.1e}", flush=True)
