"""Presentation figures for the BG TF-encoding GLM result (BG_046 DMS vs BG_039 cortex).

Reads the two per-unit CSVs written by run_tf_glm.py and emits:
  - figures/tf_responsiveness/glm_bg_fraction_by_region.png
      bars of TF-responsive % (C2) for BG_046 DMS vs BG_039 cortex, with the
      Khilkevich positive-control values (VISp 27%, CP 15%) as reference points
      and the paper's 5-45% band.
  - figures/tf_responsiveness/glm_bg_exemplars.png
      2-3 top TF-responsive DMS units: TF FIR kernel (weight vs lag) + dense
      held-out predicted-vs-actual baseline trace. If no DMS units pass, the
      figure is still emitted with a clear note.

Exemplars are RE-FIT on the fly from the BG_046 pkls (the per-unit CSV stores
metrics, not coefficients). Pass --dms-csv / --cortex-csv to point at the run
outputs and --dms-pkl-dir for the exemplar re-fit.

Example
-------
PYTHONPATH=src py scripts/tf_responsiveness/make_tf_glm_figures.py \
    --dms-csv data/cache/tf_glm/bg046_dms.csv \
    --cortex-csv data/cache/tf_glm/bg039_cortex.csv \
    --dms-pkl-dir "E:/.../data/pkls/BG_046"
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from visdetect.core.session import load_session
from visdetect.analysis.tf_glm import (
    TFGLMConfig, assemble_design, count_vector, fit_poisson_cv,
    make_trial_folds, _lag_offsets,
)
from visdetect.analysis.tf_glm_data import session_trial_regressors
from visdetect.viz.plotting import set_style, despine

FIG_DIR = _REPO / "figures" / "tf_responsiveness"
PAPER_LO, PAPER_HI = 5.0, 45.0
VISP_REF, CP_REF = 27.0, 15.0           # Khilkevich positive-control fractions
MIN_SPIKES = 500
DMS_COLOR = "#d8743b"                    # striatum (warm)
CORTEX_COLOR = "#3b7dd8"                 # cortex (cool)


def _frac(df):
    n = len(df)
    return (100.0 * df["is_responsive"].mean() if n else 0.0), n


def plot_fraction(dms_df, cortex_df, out_path):
    """Bars: TF-responsive % for our two regions + Khilkevich reference points."""
    set_style("talk")
    dms_frac, dms_n = _frac(dms_df)
    ctx_frac, ctx_n = _frac(cortex_df)

    labels = ["BG_046\nDMS (striatum)", "BG_039\ncortex / M2"]
    fracs = [dms_frac, ctx_frac]
    ns = [dms_n, ctx_n]
    colors = [DMS_COLOR, CORTEX_COLOR]

    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    ax.axhspan(PAPER_LO, PAPER_HI, color="0.88", zorder=0,
               label=f"Khilkevich-Lohse range ({PAPER_LO:.0f}-{PAPER_HI:.0f}%)")
    bars = ax.bar(labels, fracs, color=colors, width=0.58, zorder=2,
                  edgecolor="0.2", linewidth=0.8)
    for bar, frac, n in zip(bars, fracs, ns):
        ax.annotate(f"{frac:.1f}%\n(n={n})",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=13,
                    xytext=(0, 4), textcoords="offset points")

    # Khilkevich positive-control reference points (VISp = cortex, CP = striatum).
    ax.scatter([1], [VISP_REF], marker="D", s=90, color=CORTEX_COLOR,
               edgecolor="k", zorder=4, label=f"Khilkevich VISp cortex ({VISP_REF:.0f}%)")
    ax.scatter([0], [CP_REF], marker="D", s=90, color=DMS_COLOR,
               edgecolor="k", zorder=4, label=f"Khilkevich CP striatum ({CP_REF:.0f}%)")

    ax.set_ylabel("% of neurons encoding stimulus speed (TF)")
    ax.set_ylim(0, max(50.0, max(fracs + [VISP_REF]) * 1.3))
    ax.set_title("Fraction of neurons encoding stimulus speed (TF) -\n"
                 "our data, lick-controlled GLM", fontsize=15)
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    despine(ax)
    fig.text(0.5, -0.03,
             "C2 criterion (paired full-vs-reduced TF-ablation test, p<0.01). "
             "Diamonds are the authors' own VISp/CP\nfractions on the validated "
             "pipeline. Cortex (BG_039) is the in-house positive control.",
             ha="center", va="top", fontsize=8.5, color="0.3")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path, dms_frac, dms_n, ctx_frac, ctx_n


def _refit_unit(pkl_dir, session_name, unit_id, cfg):
    """Re-fit one DMS unit's full GLM; return (design, y, full_fit) or None."""
    pkl = Path(pkl_dir) / f"{session_name}.pkl"
    if not pkl.exists():
        return None
    session = load_session(str(pkl))
    trials, units = session_trial_regressors(session, cfg)
    if int(unit_id) not in units:
        return None
    design = assemble_design(trials, cfg)
    folds = make_trial_folds(design.trial_index, cfg.n_folds, cfg.seed)
    y = count_vector(trials, units[int(unit_id)], design)
    full = fit_poisson_cv(design.X, y, cfg, folds)
    return design, y, full


def plot_exemplars(dms_df, pkl_dir, cfg, out_path, n_exemplars=3):
    """Top TF-responsive DMS units: TF kernel + held-out predicted-vs-actual."""
    set_style("talk")
    passing = dms_df[dms_df["is_responsive"] == True]  # noqa: E712
    passing = passing.sort_values("c1_r", ascending=False).head(n_exemplars)

    if not len(passing) or pkl_dir is None:
        fig, ax = plt.subplots(figsize=(10, 3))
        msg = ("No DMS units passed the TF-responsive criterion (C2 p<0.01)."
               if not len(passing) else
               "No --dms-pkl-dir supplied for exemplar re-fit.")
        ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=14)
        ax.axis("off")
        fig.suptitle("TF-responsive exemplars - BG_046 DMS", fontsize=14)
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return out_path

    refits = []
    for _, r in passing.iterrows():
        got = _refit_unit(pkl_dir, str(r["session"]), int(r["unit"]), cfg)
        if got is not None:
            refits.append((r, got))

    if not refits:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, "Top DMS units could not be re-fit "
                "(pkl/unit missing).", ha="center", va="center", fontsize=14)
        ax.axis("off")
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return out_path

    nrow = len(refits)
    fig = plt.figure(figsize=(10, 2.5 * nrow + 0.6))
    gs = gridspec.GridSpec(nrow, 2, width_ratios=[1.0, 1.3],
                           hspace=0.6, wspace=0.3)
    tf_lags = _lag_offsets(cfg.kern["tf"], cfg.bin_s) * cfg.bin_s

    for ri, (r, (design, y, full)) in enumerate(refits):
        # LEFT: TF FIR kernel (mean weight per lag across folds).
        ax0 = fig.add_subplot(gs[ri, 0])
        sl = design.col_groups["tf"]
        K = np.vstack([cf[sl] for cf in full.coef_by_fold]).mean(axis=0)
        ax0.plot(tf_lags, K, color=DMS_COLOR, lw=2)
        ax0.axhline(0, color="0.6", lw=0.8)
        ax0.set_xlabel("Lag (s)")
        ax0.set_ylabel("GLM weight")
        ax0.set_title("Firing tracks a speed pulse over time", fontsize=10)
        despine(ax0)

        # RIGHT: dense held-out predicted-vs-actual baseline trace.
        ax1 = fig.add_subplot(gs[ri, 1])
        held = np.isfinite(full.pred) & np.isfinite(y)
        m, s = np.nanmean(y[held]), np.nanstd(y[held])
        yz = (y.astype(float) - m) / (s if s > 1e-9 else 1.0)
        idx = np.where(held)[0]
        win_bins = min(idx.size, int(6.0 / cfg.bin_s))
        seg = idx[:win_bins]
        t_seg = np.arange(seg.size) * cfg.bin_s
        ax1.plot(t_seg, yz[seg], color="0.4", lw=1.0, label="actual (z-scored)")
        ax1.plot(t_seg, full.pred[seg], color=DMS_COLOR, lw=2,
                 label="full-model predicted")
        ax1.set_xlabel("Time within held-out segment (s)")
        ax1.set_ylabel("Firing (z / predicted)")
        ax1.set_title(f"{r['session']} unit {int(r['unit'])} | held-out "
                      f"r={float(r['c1_r']):.2f} (vs no-TF "
                      f"{float(r['r_red_mean']):.2f}), p={float(r['c2_p']):.1e}",
                      fontsize=9)
        ax1.legend(loc="best", frameon=False, fontsize=9)
        despine(ax1)

    fig.suptitle("Most clearly TF-responsive striatal (DMS) neurons - BG_046",
                 fontsize=14, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dms-csv", required=True)
    p.add_argument("--cortex-csv", required=True)
    p.add_argument("--dms-pkl-dir", default=None,
                   help="BG_046 pkl dir for exemplar re-fit")
    a = p.parse_args(argv)

    cfg = TFGLMConfig(include_phase=False, fast_fit=True)
    dms_df = pd.read_csv(a.dms_csv) if Path(a.dms_csv).exists() else pd.DataFrame()
    cortex_df = (pd.read_csv(a.cortex_csv)
                 if Path(a.cortex_csv).exists() else pd.DataFrame())

    fa, dms_frac, dms_n, ctx_frac, ctx_n = plot_fraction(
        dms_df, cortex_df, FIG_DIR / "glm_bg_fraction_by_region.png")
    fb = plot_exemplars(dms_df, a.dms_pkl_dir, cfg,
                        FIG_DIR / "glm_bg_exemplars.png")

    print(f"BG_046 DMS:    {dms_frac:.1f}% TF-responsive (n={dms_n})")
    print(f"BG_039 cortex: {ctx_frac:.1f}% TF-responsive (n={ctx_n})")
    print(f"Figures:\n  {fa}\n  {fb}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
