"""Mechanical validation of the faithful-design code on SYNTHETIC data (no ceph,
no session loads). Exercises: tiled baseline, phase block, whole-design
standardization, tf_encoding toggle, and the _baseline_tf_and_phase helper, plus
an end-to-end fit on a synthetic TF-driven Poisson unit."""
import sys
import numpy as np
sys.path.insert(0, "E:/python_analysis/git_repos/vd_tf_phase0/src")
from visdetect.analysis.tf_glm import (
    TFGLMConfig, TrialRegressors, assemble_design, trial_bin_edges,
    count_vector, fit_poisson_cv, make_trial_folds, identify_tf_responsive,
    identify_tf_responsive_pulse, _tiled_baseline_block, _tf_octaves,
)
from visdetect.analysis.tf_glm_data import _baseline_tf_and_phase

rng = np.random.default_rng(0)
bs = 0.05
OCT_SD = 0.25 * np.log(2)  # 0.25 octave in natural-log units for lognormal TF

# ── 1. _baseline_tf_and_phase: tag masking, baseline-only, phase recon ──────
edges = trial_bin_edges(0.0, 6.0, bs)
# 120 frames: 60 gray (G, TF=0), 40 baseline (B, ~1Hz), 20 change (C, high TF)
ft = np.linspace(0.0, 6.0, 120, endpoint=False)
tf = np.concatenate([np.zeros(20),                       # gray
                     np.exp(rng.normal(0, OCT_SD, 60)),  # baseline grating ~1Hz
                     np.zeros(20),                       # gray
                     4.0 + rng.random(20)])              # change (high TF)
tag = np.array(["G"] * 20 + ["B"] * 60 + ["G"] * 20 + ["C"] * 20)
tfb, ph = _baseline_tf_and_phase(tf, ft, tag, edges, bs, change_time=5.0)
assert tfb.shape == edges.shape and ph.shape == edges.shape
assert np.all(tfb[edges >= 5.0] == 0.0), "TF not zeroed after change"
assert np.nanmax(tfb) < 3.0, "change frames leaked into baseline TF"
assert np.all(np.isfinite(ph[tfb > 0])) and np.all(np.isnan(ph[tfb == 0]))
assert np.nanmax(ph) <= 360 and np.nanmin(ph) >= 0
print(f"[1] baseline_tf_and_phase OK: nonzero bins={int((tfb>0).sum())}, "
      f"phase range [{np.nanmin(ph):.0f},{np.nanmax(ph):.0f}]")

# ── build synthetic trials ─────────────────────────────────────────────────
def make_trials(ntr=40):
    trials = []
    for k in range(ntr):
        t0 = k * 10.0; ct = t0 + 4.0; t1 = t0 + 8.0
        e = trial_bin_edges(t0, t1, bs); n = e.size
        base = e < ct
        tfb = np.zeros(n)
        tfb[base] = np.exp(rng.normal(0, OCT_SD, int(base.sum())))  # ~1Hz baseline
        cyc = np.where(tfb > 0, tfb * bs, 0.0)
        phase = np.full(n, np.nan); g = tfb > 0
        phase[g] = (np.cumsum(cyc) * 360.0 % 360.0)[g]
        trials.append(TrialRegressors(
            t_start=t0, t_end=t1, change_time=ct, change_size=2.0, tf_bins=tfb,
            lick_times=np.array([ct + 0.5]), reward_time=ct + 0.6, abort_time=np.nan,
            wheel_bins=np.abs(rng.normal(0, 1, n)), phase_bins=phase,
            motion_bins=np.abs(rng.normal(0, 1, n)),
            pupil_bins=np.abs(rng.normal(0, 1, n)), airpuff_time=np.nan))
    return trials

trials = make_trials()

# ── 2. faithful design assembles with tiled baseline + phase + standardize ──
cfg = TFGLMConfig(include_movement=True, include_phase=True,
                  include_tiled_baseline=True, standardize_design=True, fast_fit=True)
d = assemble_design(trials, cfg)
g = d.col_groups
assert "tiled_baseline" in g and "phase" in g and "time_in_base" not in g, list(g)
sd = d.X.std(0); nz = sd > 1e-8
assert np.allclose(sd[nz], 1.0, atol=1e-6), f"standardize failed: {sd[nz][:5]}"
nt = g["tiled_baseline"]; print(f"[2] faithful design OK: X={d.X.shape}, "
      f"tiled_baseline cols={nt.stop-nt.start}, phase cols={g['phase'].stop-g['phase'].start}, "
      f"all non-const cols unit-variance")

# ── 3. tf_encoding toggle changes the tf block (log2 vs linear) ─────────────
cfg_lin = TFGLMConfig(include_movement=True, include_phase=True,
                      include_tiled_baseline=True, standardize_design=False,
                      tf_encoding="linear", fast_fit=True)
cfg_log = TFGLMConfig(include_movement=True, include_phase=True,
                      include_tiled_baseline=True, standardize_design=False,
                      tf_encoding="log2", fast_fit=True)
dl = assemble_design(trials, cfg_lin); dlg = assemble_design(trials, cfg_log)
tfsl = dl.col_groups["tf"]
assert not np.allclose(dl.X[:, tfsl], dlg.X[:, tfsl]), "tf_encoding had no effect"
print("[3] tf_encoding toggle OK: linear != log2 tf block")

# ── 4. legacy path still works (no tiled baseline, no standardize) ──────────
cfg_legacy = TFGLMConfig(include_movement=False, include_phase=False, fast_fit=True)
dleg = assemble_design(trials, cfg_legacy)
assert "time_in_base" in dleg.col_groups and "tiled_baseline" not in dleg.col_groups
print("[4] legacy design OK: ramp present, tiled_baseline absent")

# ── 5. end-to-end fit on a TF-driven synthetic unit (faithful design) ───────
tf_oct = _tf_octaves(d.tf_bins)
rate = np.exp(-1.0 + 0.8 * tf_oct)            # firing genuinely driven by log2-TF
y = rng.poisson(rate).astype(float)
folds = make_trial_folds(d.trial_index, cfg.n_folds, cfg.seed)
tfcols = np.arange(*g["tf"].indices(d.X.shape[1]))
Xr = d.X.copy(); Xr[:, tfcols] = 0.0
full = fit_poisson_cv(d.X, y, cfg, folds)
red = fit_poisson_cv(Xr, y, cfg, folds)
o = identify_tf_responsive(d, y, full, red, cfg)
print(f"[5] end-to-end fit OK (dense metric): TF-driven unit "
      f"dR={o['r_full_mean']-o['r_red_mean']:+.4f} c2_p={o['c2_p']:.1e} "
      f"resp={o['is_responsive']}")
assert (o["r_full_mean"] - o["r_red_mean"]) > 0, "TF-driven unit not detected (dense)!"

# ── 6. AUTHORS' pulse-response criterion detects the TF-driven unit ──────────
cfgp = TFGLMConfig(include_movement=True, include_phase=True,
                   include_tiled_baseline=True, standardize_design=True,
                   fast_fit=True, min_pulses_per_label=8)  # synthetic has few pulses (strict ratio threshold)
op = identify_tf_responsive_pulse(d, y, full, red, cfgp)
print(f"[6] pulse-response criterion OK: TF-driven unit "
      f"C1(full pulse corr)={op['c1_r']:.3f} (>0.2 to pass) "
      f"c2_p={op['c2_p']:.1e} resp={op['is_responsive']} "
      f"folds={op['n_folds_used']}")
assert op["n_folds_used"] >= 2, "pulse criterion got too few usable folds"
assert np.isfinite(op["c1_r"]) and op["c1_r"] > 0, "pulse C1 not positive for TF-driven unit"
print("\nALL SYNTHETIC CHECKS PASSED")
