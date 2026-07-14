"""Row-smoothed companion to heatmap_transient_sustained.py (ADDED, nothing overwritten).

WHY. In the unsmoothed heatmap the eye cannot read the width->duration gradient, because a
SINGLE ROW is a noisy readout: per-cell "late-minus-early" response vs true kernel width is
only rho=+0.37 (p=4e-15) — real, but modest. So individual rows in the transient block can
LOOK prolonged purely from noise, even though no cell is misplaced (verified: the transient
block spans true widths 0.026-0.139 s and the sustained block 0.167-0.691 s — ZERO overlap,
and the within-block sort is strictly monotone).

WHAT THIS DOES. Rows are sorted by kernel width, so smoothing ACROSS NEIGHBOURING ROWS is a
local average over cells of SIMILAR WIDTH. That is exactly the quantity of interest, and it
suppresses the per-cell noise the eye keeps mistaking for structure. It is a continuous
version of the width-binned PSTH families.

HONESTY RULES OBEYED HERE:
  * smoothing is WITHIN each class block only — never across the transient/sustained
    boundary (that would manufacture a smooth ramp across the very division under test);
  * it is a DISPLAY aid, not new data or a new statistic. No number in the project changes.
    Every statistic is still computed on the UNSMOOTHED per-cell traces;
  * the unsmoothed figure remains the primary one — compare the two;
  * NaN-aware (a cell with a missing trace does not leak zeros into its neighbours).

Inherits every correction made to the parent figure: ALL ~41k pulses/session, pulse traces
SIGN-ALIGNED BY THE GLM KERNEL (never by the trace's own post-window — that is circular),
rows sorted by kernel WIDTH (never by each cell's own peak latency — that manufactures a
diagonal out of noise), and NO peak-normalisation.

Usage:  py scripts/tf_responsiveness/state_conditioned/heatmap_transient_sustained_rowsmooth.py
        [--sigma-rows 5]
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import gaussian_filter1d

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from representative_cells import REPO                                   # noqa: E402
from transient_vs_sustained import TCOL, SCOL                           # noqa: E402
from heatmap_transient_sustained import ALIGN, TITLES, XLAB, SHARED_CACHE  # noqa: E402
from heatmap_continuum import _kernel_sign, _join_width                 # noqa: E402

OUT = Path(REPO) / "FIGURES/tf_glm_bg046/heatmap_transient_sustained_rowsmooth"
KEYS = ["pulse", "change", "fa"]


def _rowsmooth(M, blocks, sigma_rows):
    """Gaussian smooth ACROSS rows (cells) WITHIN each block, NaN-aware.

    Rows are width-sorted, so this averages cells of similar width. Never crosses a block
    boundary. The NaN-aware form (smooth values and a validity mask, then divide) stops a
    missing trace from leaking zeros into its neighbours."""
    out = np.full_like(M, np.nan, dtype=float)
    for lo, hi in blocks:
        sub = M[lo:hi].astype(float)
        val = np.nan_to_num(sub, nan=0.0)
        msk = np.isfinite(sub).astype(float)
        num = gaussian_filter1d(val, sigma_rows, axis=0, mode="nearest")
        den = gaussian_filter1d(msk, sigma_rows, axis=0, mode="nearest")
        with np.errstate(invalid="ignore", divide="ignore"):
            out[lo:hi] = np.where(den > 1e-6, num / den, np.nan)
    return out


def main(sigma_rows=5.0):
    D = {k: v for k, v in np.load(SHARED_CACHE, allow_pickle=True).items()}
    cls_all = D["meta_cls"].astype(str)
    keep = np.isin(cls_all, ("transient", "sustained"))
    D = {k: (v[keep] if (k.startswith("mat_") or k.startswith("meta_")) else v)
         for k, v in D.items()}
    cls = D["meta_cls"].astype(str)

    D["mat_pulse"] = D["mat_pulse"] * _kernel_sign(D)[:, None]   # GLM-kernel sign (not circular)
    w = _join_width(D)

    order = []
    for c in ("transient", "sustained"):
        idx = np.where(cls == c)[0]
        idx = idx[np.argsort(np.nan_to_num(w[idx], nan=1e9))]    # sort by TRUE width
        order.append(idx)
    n_tr = len(order[0])
    order = np.concatenate(order)
    blocks = [(0, n_tr), (n_tr, len(order))]                     # never smooth across these

    OUT.mkdir(parents=True, exist_ok=True)
    try:
        from visdetect.viz.plotting import set_style
        set_style("talk")
    except Exception:
        pass
    plt.rcParams.update({"font.size": 11})

    fig = plt.figure(figsize=(17, 10.5))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 3.1], hspace=0.20, wspace=0.22)
    ims = {}
    for j, k in enumerate(KEYS):
        t = D[f"t_{k}"]
        M_raw = D[f"mat_{k}"][order]
        M = _rowsmooth(M_raw, blocks, sigma_rows)

        # ── top: class-mean PSTH (computed on the UNSMOOTHED per-cell traces) ──
        axp = fig.add_subplot(gs[0, j])
        for c, col in (("transient", TCOL), ("sustained", SCOL)):
            sub = M_raw[cls[order] == c]
            if not len(sub):
                continue
            mean = np.nanmean(sub, 0)
            sem = np.nanstd(sub, 0) / np.sqrt(np.sum(np.isfinite(sub), 0).clip(1))
            axp.plot(t, mean, color=col, lw=2.2, label=c)
            axp.fill_between(t, mean - 1.96 * sem, mean + 1.96 * sem, color=col, alpha=0.2)
        axp.axvline(0, color="0.6", lw=0.8); axp.axhline(0, color="0.85", lw=0.8)
        axp.set_xlim(t[0], t[-1])
        axp.set_title(f"{TITLES[k]}  — class mean ±95% CI\n(from the UNSMOOTHED cells)",
                      fontsize=11)
        if j == 0:
            axp.set_ylabel("sign-aligned z (pop mean)", fontsize=11)
            axp.legend(frameon=False, fontsize=10)
        for sp in ("top", "right"):
            axp.spines[sp].set_visible(False)

        # ── heatmap: row-smoothed ──────────────────────────────────────────────
        axh = fig.add_subplot(gs[1, j])
        if k == "pulse":
            pmax = float(np.nanpercentile(np.abs(M), 99)) or 1.0
            imkw = dict(norm=TwoSlopeNorm(vmin=-pmax, vcenter=0.0, vmax=pmax))
        else:
            imkw = dict(norm=TwoSlopeNorm(vmin=-0.6, vcenter=0.0, vmax=1.2))
        ims[k] = axh.imshow(M, aspect="auto", cmap="RdBu_r",
                            extent=[t[0], t[-1], len(M), 0],
                            interpolation="nearest", **imkw)
        axh.axhline(n_tr, color="k", lw=1.6)
        axh.axvline(0, color="k", lw=1.0, ls="--")
        axh.set_xlabel(XLAB[k], fontsize=12)
        if j == 0:
            axh.text(-0.13, n_tr / 2, "transient", rotation=90, va="center", ha="center",
                     transform=axh.get_yaxis_transform(), color=TCOL, fontsize=13,
                     fontweight="bold")
            axh.text(-0.13, n_tr + (len(M) - n_tr) / 2, "sustained", rotation=90, va="center",
                     ha="center", transform=axh.get_yaxis_transform(), color=SCOL,
                     fontsize=13, fontweight="bold")
            axh.set_ylabel("cells, sorted by kernel width (narrow → broad) within each block",
                           fontsize=10)
        else:
            axh.set_yticks([])

    cb1 = fig.colorbar(ims["pulse"], ax=fig.axes[1], fraction=0.05, pad=0.03)
    cb1.set_label("pulse: sign-aligned z (row-smoothed)", fontsize=9)
    cb2 = fig.colorbar(ims["fa"], ax=[fig.axes[3], fig.axes[5]], fraction=0.02, pad=0.015)
    cb2.set_label("change / FA:  z (row-smoothed)", fontsize=10)

    n_su = len(order) - n_tr
    fig.suptitle(
        f"ROW-SMOOTHED view — transient (n={n_tr}) vs sustained (n={n_su})   "
        f"[companion to heatmap_transient_sustained; that one is the primary, unsmoothed figure]\n"
        f"Rows are sorted by kernel width, so smoothing across neighbouring rows (Gaussian σ="
        f"{sigma_rows:g} cells, WITHIN each block only) = a local average over cells of SIMILAR width.\n"
        "It reveals the width→duration gradient that per-cell noise hides (a single row is a poor "
        "duration readout: per-cell late−early vs width ρ=+0.37). DISPLAY AID ONLY — every statistic "
        "is still computed on the unsmoothed cells.",
        fontsize=10.5, y=1.015)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"heatmap_transient_sustained_rowsmooth.{ext}", dpi=170,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/heatmap_transient_sustained_rowsmooth.png (+.pdf)  "
          f"[transient={n_tr}, sustained={n_su}, sigma_rows={sigma_rows:g}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma-rows", type=float, default=5.0)
    main(sigma_rows=ap.parse_args().sigma_rows)
