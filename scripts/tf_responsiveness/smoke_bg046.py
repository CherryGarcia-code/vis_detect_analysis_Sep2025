"""Smoke test: corrected (pulse-criterion) GLM on ONE BG_046 Expert session.

Local only (BG_046 pkls are on E:), so no ceph. Validates: session load, the BG
regressor path (tf_bins from trial.baseline_values), the faithful design
(tiled-baseline + standardize, no movement, no phase), 0.5-s.d. pulse detection,
the per-unit fit + pulse-response identification, and the fast/slow pulse PETH
used for the visualization.
"""
import sys, time
sys.path.insert(0, "E:/python_analysis/git_repos/vd_tf_bg046/src")
import numpy as np
from visdetect.core.session import load_session
from visdetect.analysis.tf_glm import (
    TFGLMConfig, assemble_design, count_vector, fit_poisson_cv,
    make_trial_folds, identify_tf_responsive_pulse, pulse_times_from_tf,
    tf_pulse_peth,
)
from visdetect.analysis.tf_glm_data import session_trial_regressors

PKL = ("E:/python_analysis/git_repos/vis_detect_analysis_Sep2025/"
       "data/pkls/BG_046/BG_046_01092025.pkl")

sess = load_session(PKL)
gs = list(getattr(sess, "good_and_stable_ids", None) or [])
gc = list(getattr(sess, "good_cluster_ids", None) or [])
print(f"session trials={len(sess.trials)} | good_and_stable={len(gs)} good={len(gc)}")
print("ni_events keys:", list((sess.ni_events or {}).keys()))
t0 = sess.trials[0]
_bv = getattr(t0, "baseline_values", None)
bv = np.asarray(_bv, float) if _bv is not None else np.zeros(0)
print(f"trial0: baseline_values len={bv.size} (St1TrialVector) "
      f"change_size={t0.change_size} outcome={t0.trialoutcome}")

cfg = TFGLMConfig(include_movement=False, include_phase=False,
                  include_tiled_baseline=True, standardize_design=True,
                  fast_fit=True, tf_encoding="log2", min_pulses_per_label=20)
trials, units = session_trial_regressors(sess, cfg)
d = assemble_design(trials, cfg)
fast, slow = pulse_times_from_tf(d, cfg)
print(f"\ntrials={len(trials)} units={len(units)} | design X={d.X.shape} "
      f"groups={list(d.col_groups)}")
print(f"fast/slow 0.5sd pulses = {fast.size}/{slow.size}")

folds = make_trial_folds(d.trial_index, cfg.n_folds, cfg.seed)
tfcols = np.arange(*d.col_groups["tf"].indices(d.X.shape[1]))
Xr = d.X.copy(); Xr[:, tfcols] = 0.0
spk = {u: float(np.sum(np.isfinite(units[u]))) for u in units}
uids = [u for u in sorted(units, key=lambda u: spk[u], reverse=True)
        if spk[u] >= 500][:4]
print(f"\nfitting {len(uids)} top-spike units:")
ti, win, bs = d.trial_index, cfg.pulse_eval_win, cfg.bin_s
for u in uids:
    y = count_vector(trials, units[u], d)
    if y.sum() < 500:
        continue
    t = time.time()
    full = fit_poisson_cv(d.X, y, cfg, folds)
    red = fit_poisson_cv(Xr, y, cfg, folds)
    o = identify_tf_responsive_pulse(d, y, full, red, cfg)
    _, pf = tf_pulse_peth(y, d.bin_edges, fast, win, bs, trial_index=ti)
    _, ps = tf_pulse_peth(y, d.bin_edges, slow, win, bs, trial_index=ti)
    fms = pf - ps
    print(f"  u{u} {int(y.sum())}spk: C1={o['c1_r']:.3f} c2_p={o['c2_p']:.1e} "
          f"resp={o['is_responsive']} | fast-slow PETH range="
          f"{(np.nanmax(fms)-np.nanmin(fms))/bs:.2f} Hz [{time.time()-t:.0f}s]",
          flush=True)
print("\nSMOKE OK" if uids else "\n(no units >=500 spk)")
