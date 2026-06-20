"""Exemplar TF-responsive PSTH plotter (pulse PETH + lick + GLM kernel).

Presentation-ready per-cell figure for showing WHY a unit is called
TF-responsive. One ROW per cell, four panels:

  (A) Fast-vs-slow TF-pulse PETH (firing rate in Hz aligned to baseline-TF
      pulses): FAST pulses (baseline TF >= +0.5 SD) vs SLOW pulses (<= -0.5 SD),
      two mean lines with SEM shading, x in seconds around the pulse.
  (B) Fast-minus-slow difference (the TF-selectivity trace) with a label-shuffle
      null band; the shuffle p (label-permutation test) is annotated.
  (C) Lick-triggered PETH (firing aligned to lick onsets) -- the motor control,
      so the viewer can judge whether the response is TF-driven or movement.
  (D) GLM TF kernel: the full-model TF FIR filter (weight vs lag 0-1.5 s),
      re-fit on a TRIAL SUBSET (~250 trials) for speed. If the fit is slow or
      flaky, panel D is skipped gracefully with an on-axes note (never fails).

Plumbing reuses the existing Phase-0 stack:
  - visdetect.analysis.tf_selectivity.compute_unit_selectivity -> panels A, B
  - visdetect.analysis.tf_glm.{assemble_design, pulse_times_from_tf,
        TFGLMConfig, fit_poisson_cv, make_trial_folds, count_vector} -> panel D
        + the neural-clock fast/slow pulse times for A/B
  - visdetect.analysis.tf_glm_data adapters for source='khilkevich' / 'bg'

Run (self-test):
  PYTHONPATH=src py scripts/tf_responsiveness/plot_tf_exemplars.py \
      --source khilkevich --region VISp \
      --cells "[('ML_1116764_S02_M2_V1', 492)]" --out vis_exemplar.png
  PYTHONPATH=src py scripts/tf_responsiveness/plot_tf_exemplars.py \
      --source bg --region DMS \
      --cells "[('BG_046_01072025', 596)]" --out dms_exemplar.png
"""
from __future__ import annotations
import argparse
import ast
import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from visdetect.viz.plotting import set_style, despine
from visdetect.analysis.tf_selectivity import (
    TFSelectivityConfig, compute_unit_selectivity, _per_pulse_rate_matrix,
)
from visdetect.analysis.tf_glm import (
    TFGLMConfig, assemble_design, pulse_times_from_tf, count_vector,
    fit_poisson_cv, make_trial_folds, _lag_offsets,
)
from visdetect.analysis.tf_glm_data import (
    load_khilkevich_session, khilkevich_trial_regressors,
    session_trial_regressors,
)

# Khilkevich npx_converted root + session-label -> dir map (animal 1116764,
# the two sessions that carry both a V1 probe and a CP probe). Mirrors the
# hardcoded paths in run_tf_glm_diagnostic.py.
KHIL_ROOT = Path(
    "X:/public/projects/MoHa_20260212_dmdmTemporalExpectation/data/npx_converted")
KHIL_SESSIONS = {
    "ML_1116764_S02_M2_V1": "1116764/ML_1116764_S02_M2_V1",
    "ML_1116764_S03_M2_V1": "1116764/ML_1116764_S03_M2_V1",
}
# Coarse region label the Khilkevich adapter understands ('VISp'/'CP').
BG_PKL_DIR = Path(
    "E:/python_analysis/git_repos/vis_detect_analysis_Sep2025/data/pkls/BG_046")
# Optional c2_p lookup (the GLM diagnostic checkpoint).
C2P_CSV = _REPO / "data" / "cache" / "tf_glm" / "khilkevich_diagnostic.csv"

OUT_DIR = _REPO / "figures" / "tf_responsiveness"

# Panel D speed cap: subset of trials for the GLM re-fit + a tighter fold count.
N_TRIALS_KERNEL = 250
KERNEL_FOLDS = 5
KERNEL_FIT_TIMEOUT_S = 120.0   # soft budget; panel D is skipped if exceeded


# ── Lick-triggered PETH (panel C) ──────────────────────────────────────────
def lick_peth(spike_times, lick_times, t_lo=-1.0, t_hi=1.0, dt=0.025,
              sigma_ms=25.0):
    """Mean +/- SEM firing rate (Hz) aligned to lick onsets.

    Reuses the selectivity stack's per-event rate matrix (Gaussian-smoothed,
    searchsorted-sliced) so the smoothing matches panels A/B exactly.
    """
    t_vec = np.arange(t_lo, t_hi, dt, dtype=float)
    lk = np.asarray(lick_times, float).ravel()
    lk = lk[np.isfinite(lk)]
    if lk.size == 0:
        return t_vec, np.full(t_vec.size, np.nan), np.full(t_vec.size, np.nan), 0
    mat = _per_pulse_rate_matrix(spike_times, lk, t_vec, dt, sigma_ms)
    if mat.shape[0] == 0:
        return t_vec, np.full(t_vec.size, np.nan), np.full(t_vec.size, np.nan), 0
    mean = mat.mean(axis=0)
    sem = mat.std(axis=0) / np.sqrt(max(mat.shape[0], 1))
    return t_vec, mean, sem, mat.shape[0]


# ── Data access ────────────────────────────────────────────────────────────
def build_cell_payload(source, region, session_label, unit_id, glm_cfg):
    """Assemble everything a single cell-row needs.

    Returns dict with keys: spike_times, lick_times, fast_times, slow_times,
    trials, units, design, n_spikes, c2_p (or NaN).
    """
    if source == "khilkevich":
        if session_label not in KHIL_SESSIONS:
            raise KeyError(
                f"Unknown khilkevich session {session_label!r}; "
                f"known: {sorted(KHIL_SESSIONS)}")
        sdir = KHIL_ROOT / KHIL_SESSIONS[session_label]
        ks = load_khilkevich_session(sdir)
        trials, units = khilkevich_trial_regressors(ks, glm_cfg, region=region)
        lick_times = np.asarray(ks.licks, float).ravel()
        spike_times = units.get(int(unit_id))
        if spike_times is None:
            # unit may not be in the region subset -- fall back to all units
            spike_times = ks.units.get(int(unit_id))
        del ks
    elif source == "bg":
        pkl = BG_PKL_DIR / f"{session_label}.pkl"
        if not pkl.exists():
            raise FileNotFoundError(f"BG pkl not found: {pkl}")
        from visdetect.core.session import load_session
        sess = load_session(str(pkl))
        trials, units = session_trial_regressors(sess, glm_cfg)
        ni = sess.ni_events or {}
        lick_times = np.asarray(ni.get("Piezo_1", np.zeros(0)), float).ravel()
        spike_times = units.get(int(unit_id))
        if spike_times is None:
            by_id = {int(c.cluster_id): np.asarray(c.spike_times, float).ravel()
                     for c in sess.clusters}
            spike_times = by_id.get(int(unit_id))
        del sess
    else:
        raise ValueError(f"Unknown source {source!r} (expected khilkevich|bg)")

    if spike_times is None:
        raise KeyError(
            f"unit {unit_id} not found in {source} session {session_label}")
    spike_times = np.asarray(spike_times, float).ravel()

    # Neural-clock fast/slow pulse times via the GLM design (no spikes needed).
    design = assemble_design(trials, glm_cfg)
    fast_times, slow_times = pulse_times_from_tf(design, glm_cfg)

    return {
        "spike_times": spike_times,
        "lick_times": lick_times[np.isfinite(lick_times)],
        "fast_times": fast_times,
        "slow_times": slow_times,
        "trials": trials,
        "units": units,
        "design": design,
        "n_spikes": int(spike_times.size),
        "c2_p": lookup_c2p(source, session_label, unit_id),
    }


def lookup_c2p(source, session_label, unit_id):
    """Best-effort c2_p from the GLM diagnostic CSV; NaN if absent."""
    if source != "khilkevich" or not C2P_CSV.exists():
        return np.nan
    try:
        d = pd.read_csv(C2P_CSV)
    except Exception:
        return np.nan
    hit = d[(d["session"].astype(str) == str(session_label))
            & (d["unit"].astype(int) == int(unit_id))]
    if len(hit):
        return float(hit["c2_p"].iloc[0])
    return np.nan


# ── Panel D: GLM TF kernel on a trial subset (graceful skip) ────────────────
def fit_tf_kernel(payload, glm_cfg):
    """Re-fit the FULL Poisson GLM on a ~N_TRIALS_KERNEL subset and return
    (lags, kernel) for the TF FIR filter, or (None, reason) on skip/failure.

    Kept fast on purpose so it does not contend with the heavy BG jobs:
    a contiguous trial subset, 5 folds, fast_fit lambda selection.
    """
    try:
        t0 = time.time()
        trials = payload["trials"]
        spike_times = payload["spike_times"]
        n = len(trials)
        if n == 0:
            return None, "no trials"
        sub = trials[:N_TRIALS_KERNEL] if n > N_TRIALS_KERNEL else trials
        kcfg = TFGLMConfig(include_phase=glm_cfg.include_phase, fast_fit=True,
                           n_folds=KERNEL_FOLDS, seed=glm_cfg.seed)
        design = assemble_design(sub, kcfg)
        sl = design.col_groups.get("tf")
        if sl is None:
            return None, "no TF columns"
        y = count_vector(sub, spike_times, design)
        if y.sum() < 100:
            return None, f"too few spikes in subset ({int(y.sum())})"
        folds = make_trial_folds(design.trial_index, kcfg.n_folds, kcfg.seed)
        fit = fit_poisson_cv(design.X, y, kcfg, folds)
        if not fit.coef_by_fold:
            return None, "fit produced no folds"
        if time.time() - t0 > KERNEL_FIT_TIMEOUT_S:
            return None, "fit exceeded time budget"
        K = np.vstack([c[sl] for c in fit.coef_by_fold]).mean(axis=0)
        lags = _lag_offsets(kcfg.kern["tf"], kcfg.bin_s) * kcfg.bin_s
        return lags, K
    except Exception as exc:  # never fail the whole figure on panel D
        return None, f"fit error: {type(exc).__name__}: {exc}"


# ── Per-row drawing ─────────────────────────────────────────────────────────
FAST_C = "#c0392b"   # warm = fast pulses
SLOW_C = "#2c6fb5"   # cool = slow pulses
SEL_C = "#2d2d2d"
LICK_C = "#1a8a4a"
KERN_C = "#7b3fa0"


def draw_row(axes, payload, sel, region, session_label, unit_id):
    axA, axB, axC, axD = axes

    # --- Panel A: fast vs slow PETH (Hz) with SEM shading ---
    t = sel.t_vec
    cfg = TFSelectivityConfig()
    mat_fast = _per_pulse_rate_matrix(
        payload["spike_times"], payload["fast_times"], t,
        cfg.pulse.dt, cfg.pulse.sigma_ms)
    mat_slow = _per_pulse_rate_matrix(
        payload["spike_times"], payload["slow_times"], t,
        cfg.pulse.dt, cfg.pulse.sigma_ms)
    if mat_fast.shape[0] and mat_slow.shape[0]:
        f_sem = mat_fast.std(axis=0) / np.sqrt(mat_fast.shape[0])
        s_sem = mat_slow.std(axis=0) / np.sqrt(mat_slow.shape[0])
        axA.fill_between(t, sel.fast_hz - f_sem, sel.fast_hz + f_sem,
                         color=FAST_C, alpha=0.22, linewidth=0)
        axA.fill_between(t, sel.slow_hz - s_sem, sel.slow_hz + s_sem,
                         color=SLOW_C, alpha=0.22, linewidth=0)
    axA.plot(t, sel.fast_hz, color=FAST_C, lw=2.0,
             label=f"fast TF pulse (n={sel.n_fast})")
    axA.plot(t, sel.slow_hz, color=SLOW_C, lw=2.0,
             label=f"slow TF pulse (n={sel.n_slow})")
    axA.axvline(0.0, color="0.6", ls=":", lw=1.0)
    axA.set_ylabel("firing rate (Hz)")
    axA.set_xlabel("time from TF pulse (s)")
    axA.set_title("Speed-up vs slow-down pulses", fontsize=12)
    axA.legend(loc="best", frameon=False, fontsize=8)
    despine(axA)

    # --- Panel B: fast-minus-slow selectivity + shuffle null band ---
    axB.axhline(0.0, color="0.6", ls=":", lw=1.0)
    axB.axvline(0.0, color="0.6", ls=":", lw=1.0)
    # Flat label-shuffle null band: the permutation peak null (mean +/- 2 SD).
    if np.isfinite(sel.null_peak_mean) and np.isfinite(sel.null_peak_sd):
        band = sel.null_peak_mean + 2.0 * sel.null_peak_sd
        axB.axhspan(-band, band, color="0.75", alpha=0.35, linewidth=0,
                    label="shuffle null (+/-2 SD)")
    axB.plot(t, sel.selectivity, color=SEL_C, lw=2.0)
    axB.set_ylabel("fast - slow (SD units)")
    axB.set_xlabel("time from TF pulse (s)")
    p = sel.shuffle_p
    p_txt = "n/a" if not np.isfinite(p) else (
        "<0.005" if p < 0.005 else f"{p:.3f}")
    axB.set_title(f"TF selectivity (shuffle p={p_txt})", fontsize=12)
    axB.legend(loc="best", frameon=False, fontsize=8)
    despine(axB)

    # --- Panel C: lick-triggered PETH (motor control) ---
    lt, lmean, lsem, n_licks = lick_peth(
        payload["spike_times"], payload["lick_times"])
    if np.any(np.isfinite(lmean)):
        axC.fill_between(lt, lmean - lsem, lmean + lsem,
                         color=LICK_C, alpha=0.22, linewidth=0)
        axC.plot(lt, lmean, color=LICK_C, lw=2.0)
    else:
        axC.text(0.5, 0.5, "no licks", ha="center", va="center",
                 transform=axC.transAxes, fontsize=10, color="0.5")
    axC.axvline(0.0, color="0.6", ls=":", lw=1.0)
    axC.set_ylabel("firing rate (Hz)")
    axC.set_xlabel("time from lick (s)")
    axC.set_title(f"Lick-aligned (motor; n={n_licks})", fontsize=12)
    despine(axC)

    # --- Panel D: GLM TF kernel (subset re-fit, graceful skip) ---
    lags, K = fit_tf_kernel(payload, _GLM_CFG)
    if lags is not None:
        axD.axhline(0.0, color="0.6", ls=":", lw=1.0)
        axD.plot(lags, K, color=KERN_C, lw=2.0, marker="o", ms=3)
        axD.set_ylabel("TF weight (a.u.)")
        axD.set_xlabel("lag (s)")
        axD.set_title(f"GLM TF kernel (subset re-fit)", fontsize=12)
    else:
        axD.text(0.5, 0.5, f"panel skipped:\n{K}", ha="center", va="center",
                 transform=axD.transAxes, fontsize=9, color="0.5", wrap=True)
        axD.set_title("GLM TF kernel (skipped)", fontsize=12)
        axD.set_xlabel("lag (s)")
    despine(axD)


# Module-level GLM config used by panel D (set in main()).
_GLM_CFG = TFGLMConfig(include_phase=False, fast_fit=True)


def main(argv=None):
    global _GLM_CFG
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, choices=["khilkevich", "bg"])
    ap.add_argument("--region", required=True,
                    help="region label (VISp/CP for khilkevich; cosmetic for bg)")
    ap.add_argument("--cells", required=True,
                    help="python-literal list of (session, unit), e.g. "
                         "\"[('ML_1116764_S02_M2_V1', 492)]\"")
    ap.add_argument("--out", required=True, help="output PNG name")
    a = ap.parse_args(argv)

    cells = ast.literal_eval(a.cells)
    if not isinstance(cells, (list, tuple)) or not cells:
        raise ValueError("--cells must be a non-empty list of (session, unit)")

    # Cortex (VISp) needs the phase regressors off-data anyway; keep DMS-style.
    _GLM_CFG = TFGLMConfig(include_phase=False, fast_fit=True)
    sel_cfg = TFSelectivityConfig()

    set_style("talk")
    n_rows = len(cells)
    fig, axarr = plt.subplots(n_rows, 4, figsize=(20, 4.4 * n_rows),
                              squeeze=False)

    for r, (session_label, unit_id) in enumerate(cells):
        session_label = str(session_label)
        unit_id = int(unit_id)
        print(f"[row {r+1}/{n_rows}] {a.source} | {a.region} | "
              f"{session_label} | unit {unit_id}", flush=True)
        payload = build_cell_payload(a.source, a.region, session_label,
                                     unit_id, _GLM_CFG)
        rng = np.random.default_rng(sel_cfg.seed)
        sel = compute_unit_selectivity(
            payload["spike_times"], payload["fast_times"],
            payload["slow_times"], sel_cfg, rng)
        sel.cluster_id = unit_id

        draw_row(axarr[r], payload, sel, a.region, session_label, unit_id)

        c2 = payload["c2_p"]
        c2_txt = "" if not np.isfinite(c2) else f" | c2_p={c2:.1e}"
        title = (f"{a.region} | {session_label} | unit {unit_id}"
                 f"{c2_txt} | n_spikes={payload['n_spikes']:,}")
        axarr[r, 0].annotate(
            title, xy=(0.0, 1.18), xycoords="axes fraction",
            ha="left", va="bottom", fontsize=13, fontweight="bold")

        del payload
        gc.collect()

    fig.suptitle(
        "TF-responsive exemplars: pulse PETH (A) -> selectivity vs shuffle (B) "
        "-> lick control (C) -> GLM TF kernel (D). "
        "Fast = baseline TF >= +0.5 SD, slow <= -0.5 SD.",
        y=0.998, fontsize=12, color="0.25")
    fig.tight_layout(rect=(0, 0, 1, 0.985))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / a.out
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
