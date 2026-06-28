"""Positive control: run the TF-encoding GLM on Khilkevich `npx_converted` data.

Applies the SAME reduced-regressor Poisson encoding GLM we run on BG_046
(TF FIR + trial_start + time_in_base + 6 change sizes + lick_prep/lick_exec +
reward + abort + wheel; NO phase, NO motion-energy/pupil) to the Khilkevich-
Lohse dataset and reports the per-region fraction of TF-responsive neurons. This
is the apples-to-apples pipeline validation: our reduced model vs their published
full-model 5-45% TF-responsive range.

Validation gate: a clear cortical positive (e.g. VISp) well above 0 and above
the striatal/BG fraction signals the pipeline has power. If cortex lands ~0%
with all regressors present, the pipeline has a bug -- debug before trusting any
BG result.

Outputs:
  - data/cache/tf_glm/khilkevich_posctrl_<region>.csv  (per-unit table)
  - figures/tf_responsiveness/glm_khilkevich_fraction_by_region.png
  - figures/tf_responsiveness/glm_khilkevich_exemplars.png

Example
-------
PYTHONPATH=src py scripts/tf_responsiveness/run_tf_glm_khilkevich.py \
    --session-dir "X:/.../npx_converted/1108393/AK_1108393_S10" --region VISp \
    --striatum-session-dir "X:/.../npx_converted/1119409/ML_1119409_S03" \
    --striatum-region CP --max-units 25
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

# Allow `PYTHONPATH=src` invocation; also self-bootstrap the repo src dir.
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from visdetect.analysis.tf_glm import (
    TFGLMConfig, assemble_design, count_vector, fit_poisson_cv,
    make_trial_folds, identify_tf_responsive, pulse_times_from_tf,
    tf_pulse_peth, _lag_offsets,
)
from visdetect.analysis.tf_glm_data import (
    load_khilkevich_session, khilkevich_trial_regressors,
)
from visdetect.viz.plotting import set_style, despine

CACHE_DIR = _REPO / "data" / "cache" / "tf_glm"
FIG_DIR = _REPO / "figures" / "tf_responsiveness"
PAPER_LO, PAPER_HI = 5.0, 45.0          # published TF-responsive fraction range
MIN_SPIKES = 500                        # skip low-spike-count units (paper-power floor)


# ── core: fit one region, return per-unit DataFrame + diagnostics bundle ─────
def run_region(session_dir, region, cfg, max_units=None, max_trials=None,
               verbose=True):
    """Fit full + reduced GLM per unit in `region`; return (df, design, fits).

    `fits` is a dict {unit_id: (y, full_fit, reduced_fit)} kept ONLY for the
    units that passed the spike gate, so the exemplar figure can re-plot PETHs
    without refitting.

    `max_trials` subsamples the first N trials before assembling the design.
    The full-length Khilkevich sessions are ~470-510 trials (~290 bins each ->
    ~140k design rows), which makes each nested-CV Poisson fit very slow.
    Capping trials shrinks the design proportionally; thousands of TF pulses
    remain (>> the 50-pulse minimum) so the positive control is unaffected in
    spirit. This is a runtime knob, not a scientific one.
    """
    ks = load_khilkevich_session(session_dir)
    trials, units = khilkevich_trial_regressors(ks, cfg, region=region)
    if max_trials:
        trials = trials[:max_trials]
    design = assemble_design(trials, cfg)
    folds = make_trial_folds(design.trial_index, cfg.n_folds, cfg.seed)

    # Reduced design: zero the TF FIR block.
    Xr = design.X.copy()
    Xr[:, design.col_groups["tf"]] = 0.0

    fast, slow = pulse_times_from_tf(design, cfg)
    if verbose:
        print(f"[{region}] {len(trials)} trials, {len(units)} units, "
              f"X={design.X.shape}, fast/slow pulses={fast.size}/{slow.size}",
              flush=True)

    # Rank units by total spike count (highest first) and keep only those with
    # >= MIN_SPIKES; then cap at max_units. This uses the highest-power units for
    # the positive control (per brief: ~40 highest-spike units, skip <~500 spk).
    spk_total = {uid: float(np.sum(np.isfinite(units[uid]))) for uid in units}
    uids = [u for u in sorted(units, key=lambda u: spk_total[u], reverse=True)
            if spk_total[u] >= MIN_SPIKES]
    n_qualifying = len(uids)
    if max_units:
        uids = uids[:max_units]
    if verbose:
        print(f"  [{region}] {n_qualifying} units >= {MIN_SPIKES} spk; "
              f"fitting {len(uids)}", flush=True)

    import time
    rows, fits = [], {}
    for k, uid in enumerate(uids):
        y = count_vector(trials, units[uid], design)
        if y.sum() < MIN_SPIKES:
            if verbose:
                print(f"  [{region}] unit {uid} ({k+1}/{len(uids)}): "
                      f"{int(y.sum())} spikes < {MIN_SPIKES}, skipped", flush=True)
            continue
        t0 = time.time()
        full = fit_poisson_cv(design.X, y, cfg, folds)
        red = fit_poisson_cv(Xr, y, cfg, folds)
        out = identify_tf_responsive(design, y, full, red, cfg)
        out["unit"] = int(uid)
        out["region"] = region
        out["n_spikes"] = float(y.sum())
        rows.append(out)
        fits[int(uid)] = (y, full, red)
        if verbose:
            print(f"  [{region}] unit {uid} ({k+1}/{len(uids)}): "
                  f"{int(y.sum())} spk, c1_r={out['c1_r']:.2f} "
                  f"c2_p={out['c2_p']:.1e} resp={out['is_responsive']} "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    cols = ["unit", "region", "n_spikes", "c1_r", "c2_p", "is_responsive",
            "r_full_mean", "r_red_mean", "n_folds_used",
            "kernel_peak_t", "kernel_fwhm"]
    df = pd.DataFrame(rows)
    if len(df):
        df = df[cols]
    else:
        df = pd.DataFrame(columns=cols)
    return df, design, fits


# ── figure (a): fraction-by-region bar chart ────────────────────────────────
def plot_fraction_by_region(region_dfs, out_path):
    """Bar of % TF-responsive per region, with the paper's 5-45% band shaded."""
    set_style("talk")
    labels, fracs, ns = [], [], []
    for region, df in region_dfs.items():
        n = len(df)
        frac = 100.0 * df["is_responsive"].mean() if n else 0.0
        labels.append(region)
        fracs.append(frac)
        ns.append(n)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.axhspan(PAPER_LO, PAPER_HI, color="0.85", zorder=0,
               label=f"Khilkevich-Lohse range ({PAPER_LO:.0f}-{PAPER_HI:.0f}%)")
    colors = ["#3b7dd8", "#d8743b", "#5aa469", "#9b59b6"]
    bars = ax.bar(labels, fracs, color=colors[:len(labels)], width=0.6, zorder=2)
    for bar, frac, n in zip(bars, fracs, ns):
        ax.annotate(f"{frac:.1f}%\n(n={n})",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=12,
                    xytext=(0, 3), textcoords="offset points")
    ax.set_ylabel("% of neurons TF-responsive")
    ax.set_ylim(0, max(50.0, max(fracs) * 1.25 if fracs else 50.0))
    ax.set_title("Fraction of neurons encoding stimulus speed (temporal\n"
                 "frequency), by region - pipeline validation on\n"
                 "Khilkevich-Lohse data", fontsize=15)
    ax.legend(loc="upper right", frameon=False, fontsize=11)
    despine(ax)
    fig.text(0.5, -0.02,
             "Reduced model (no motion-energy/pupil), so fractions sit at the "
             "lower edge of the paper's full-model range; the\nvalidation test "
             "is cortex (e.g. VISp) clearly above striatum (e.g. CP) and above "
             "zero, confirming the pipeline has power.",
             ha="center", va="top", fontsize=8.5, color="0.3")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ── figure (b): exemplar TF kernels + predicted-vs-actual PETHs ──────────────
def _diff_peth(values, design, fast, slow, cfg):
    ti = design.trial_index
    _, pf = tf_pulse_peth(values, design.bin_edges, fast, cfg.pulse_eval_win,
                          cfg.bin_s, trial_index=ti)
    _, ps = tf_pulse_peth(values, design.bin_edges, slow, cfg.pulse_eval_win,
                          cfg.bin_s, trial_index=ti)
    return pf - ps


def plot_exemplars(region_dfs, region_designs, region_fits, cfg, out_path,
                   n_exemplars=3):
    """Rows of (TF FIR kernel, predicted-vs-actual fast-minus-slow PETH).

    Picks the units with the highest c1_r that pass BOTH criteria, pooled
    across the analysed regions. Always emits a figure (even if none pass).
    """
    set_style("talk")

    # Pool candidate (region, unit, c1_r) passing both criteria, rank by c1_r.
    cands = []
    for region, df in region_dfs.items():
        if not len(df):
            continue
        passing = df[df["is_responsive"] == True]  # noqa: E712
        for _, r in passing.iterrows():
            cands.append((float(r["c1_r"]), region, int(r["unit"]),
                          float(r["c2_p"]), float(r.get("r_red_mean", np.nan))))
    cands.sort(reverse=True)
    cands = cands[:n_exemplars]

    if not cands:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, "No TF-responsive units passed both criteria\n"
                "(C1 r>0.2 AND C2 p<0.01) in the analysed regions.",
                ha="center", va="center", fontsize=14)
        ax.axis("off")
        fig.suptitle("TF-responsive exemplars - Khilkevich positive control",
                     fontsize=14)
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return out_path

    nrow = len(cands)
    fig = plt.figure(figsize=(10, 2.4 * nrow + 0.6))
    gs = gridspec.GridSpec(nrow, 2, width_ratios=[1.0, 1.3],
                           hspace=0.55, wspace=0.28)
    tf_lags = _lag_offsets(cfg.kern["tf"], cfg.bin_s) * cfg.bin_s

    for ri, (c1_r, region, uid, c2_p, r_red) in enumerate(cands):
        design = region_designs[region]
        y, full, red = region_fits[region][uid]

        # LEFT: TF FIR kernel (mean weight per lag across folds).
        ax0 = fig.add_subplot(gs[ri, 0])
        sl = design.col_groups["tf"]
        K = np.vstack([cf[sl] for cf in full.coef_by_fold]).mean(axis=0)
        ax0.plot(tf_lags, K, color="#3b7dd8", lw=2)
        ax0.axhline(0, color="0.6", lw=0.8)
        ax0.set_xlabel("Lag (s)")
        ax0.set_ylabel("GLM weight")
        ax0.set_title("How the neuron's firing tracks a speed pulse over time",
                      fontsize=10)
        despine(ax0)

        # RIGHT: dense held-out predicted-vs-actual trace (the series the
        # TF-responsive metric is actually computed on:
        # corr(zscore(yTest), y_hat_pred)). Show a readable ~6 s snippet of the
        # held-out bins, z-scored actual counts vs full-model predicted rate.
        ax1 = fig.add_subplot(gs[ri, 1])
        held = np.isfinite(full.pred) & np.isfinite(y)
        yz = y.copy().astype(float)
        m, s = np.nanmean(y[held]), np.nanstd(y[held])
        yz = (yz - m) / (s if s > 1e-9 else 1.0)
        idx = np.where(held)[0]
        win_bins = min(idx.size, int(6.0 / cfg.bin_s))     # ~6 s window
        seg = idx[:win_bins]
        t_seg = np.arange(seg.size) * cfg.bin_s
        ax1.plot(t_seg, yz[seg], color="0.4", lw=1.0, label="actual (z-scored)")
        ax1.plot(t_seg, full.pred[seg], color="#d8743b", lw=2,
                 label="full-model predicted")
        ax1.set_xlabel("Time within held-out segment (s)")
        ax1.set_ylabel("Firing (z / predicted rate)")
        ax1.set_title(f"{region} unit {uid} | held-out r={c1_r:.2f} "
                      f"(vs no-TF {r_red:.2f}), p={c2_p:.1e}", fontsize=10)
        ax1.legend(loc="best", frameon=False, fontsize=9)
        despine(ax1)

    fig.suptitle("Most clearly TF-responsive neurons - Khilkevich positive "
                 "control", fontsize=14, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session-dir", required=True,
                   help="cortical (positive control) session directory")
    p.add_argument("--region", required=True,
                   help="cortical region label, e.g. VISp")
    p.add_argument("--striatum-session-dir", default=None,
                   help="optional striatal/BG session directory")
    p.add_argument("--striatum-region", default=None,
                   help="optional striatal/BG region label, e.g. CP")
    p.add_argument("--max-units", type=int, default=None)
    p.add_argument("--max-trials", type=int, default=None,
                   help="subsample first N trials (runtime knob; shrinks design)")
    p.add_argument("--include-phase", action="store_true",
                   help="(phase is ABSENT in the Khilkevich export; leave off)")
    p.add_argument("--include-movement", action="store_true",
                   help="full Khilkevich-faithful model: add motion-energy + "
                        "pupil (continuous FIR) + air-puff (event) regressors")
    p.add_argument("--fast-fit", action="store_true",
                   help="select ridge lambda ONCE per unit (much faster; "
                        "preserves held-out pred structure)")
    p.add_argument("--lambdas", default=None,
                   help="comma-separated ridge lambda grid (default: full 6-value "
                        "grid; pass e.g. 0.01,0.1,1.0 to speed fast-fit further)")
    a = p.parse_args(argv)

    kw = dict(include_phase=a.include_phase, include_movement=a.include_movement,
              fast_fit=a.fast_fit)
    if a.lambdas:
        kw["lambdas"] = tuple(float(x) for x in a.lambdas.split(","))
    cfg = TFGLMConfig(**kw)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    region_dfs, region_designs, region_fits = {}, {}, {}

    # Cortex (positive control).
    df_c, design_c, fits_c = run_region(a.session_dir, a.region, cfg,
                                        a.max_units, a.max_trials)
    region_dfs[a.region] = df_c
    region_designs[a.region] = design_c
    region_fits[a.region] = fits_c
    df_c.to_csv(CACHE_DIR / f"khilkevich_posctrl_{a.region}.csv", index=False)
    frac_c = 100.0 * df_c["is_responsive"].mean() if len(df_c) else float("nan")
    print(f"\n{a.region}: {len(df_c)} units, TF-responsive {frac_c:.1f}% "
          f"(paper {PAPER_LO:.0f}-{PAPER_HI:.0f})")

    # Striatum / BG (optional second region).
    frac_s = None
    if a.striatum_session_dir and a.striatum_region:
        df_s, design_s, fits_s = run_region(
            a.striatum_session_dir, a.striatum_region, cfg, a.max_units,
            a.max_trials)
        region_dfs[a.striatum_region] = df_s
        region_designs[a.striatum_region] = design_s
        region_fits[a.striatum_region] = fits_s
        df_s.to_csv(CACHE_DIR / f"khilkevich_posctrl_{a.striatum_region}.csv",
                    index=False)
        frac_s = 100.0 * df_s["is_responsive"].mean() if len(df_s) else float("nan")
        print(f"{a.striatum_region}: {len(df_s)} units, TF-responsive "
              f"{frac_s:.1f}% (paper {PAPER_LO:.0f}-{PAPER_HI:.0f})")

    # Figures.
    fa = plot_fraction_by_region(region_dfs, FIG_DIR / "glm_khilkevich_fraction_by_region.png")
    fb = plot_exemplars(region_dfs, region_designs, region_fits, cfg,
                        FIG_DIR / "glm_khilkevich_exemplars.png")
    print(f"\nFigures:\n  {fa}\n  {fb}")

    # Validation gate.
    if frac_s is not None and np.isfinite(frac_c) and np.isfinite(frac_s):
        gate = (frac_c > frac_s) and (frac_c > 0.0)
        print(f"\nValidation gate (cortex>striatum>0): "
              f"{'PASSED' if gate else 'FAILED'} "
              f"({a.region}={frac_c:.1f}% vs {a.striatum_region}={frac_s:.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
