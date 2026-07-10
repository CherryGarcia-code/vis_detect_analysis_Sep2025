"""Continuum re-render of the FA-lick activity figure (fa_lick_activity.py).

The class figure ordered the FA-lick-aligned population by each cell's pre-lick
ramp and contrasted two transient/sustained CLASS means, showing WHY sustained
cells are 89% lick-responsive (they carry the strong pre-lick motor ramp) while
transient cells' ramp is weaker. The spectrum result (docs 2026-07
transient-sustained-spectrum) showed kernel width is a GRADED axis, so here every
view is a CONTINUOUS function of kernel width (interp_fwhm):

  * Panel A: per-cell pre-lick FA ramp (mean z over the (-0.3, -0.15) s window)
    vs continuous width, as a decile binned-trend + Spearman (replaces the
    transient-vs-sustained ramp comparison).
  * Panel B: the FA-lick heatmap, rows ordered by CONTINUOUS width ASCENDING
    (narrow TOP -> broad BOTTOM), TwoSlopeNorm(-1.5, 0, 3) RdBu_r, with a
    continuous-width viridis colour strip AND a lick-responsive overlay strip on
    the left. Orientation matches heatmap_continuum.py (strip + numeric width
    colorbar both run narrow-top -> broad-bottom, end-labelled).
  * Panel C: the FA PSTH FAMILY over 5 equal-count width bins (width_bin_assign),
    mean z-trace + 95% CI (+/-1.96 SEM) shading along the viridis gradient; each
    legend entry carries the bin width range, n, and % lick-responsive (among
    labeled cells). FA is predominantly excitatory (motor ramp) so NO sign-flip
    is needed (unlike the fast pulse); broad-width bins carry the larger ramp.

Lick labels: the lick_acquisition_cells.csv screen labels ALL 520 responsive
cells (transient/intermediate/sustained) in the current cache, so the lick
overlay + per-bin % use real labels for every cell (0 greyed). The greying path
(GREY for any cell whose lick label is genuinely missing) is retained as a
robustness fallback and reported.

Cache-only: reads peth_traces_all.npz (all 520 responsive cells) +
kernel_width_continuous.csv (via continuum_common) + lick_acquisition_cells.csv.
No session reload.

Usage:  py scripts/tf_responsiveness/state_conditioned/fa_lick_continuum.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import TwoSlopeNorm, Normalize, to_rgb
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch
from scipy.stats import spearmanr

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from continuum_common import (load_width_metrics, binned_trend, width_bin_assign,  # noqa: E402
                              WIDTH, REPO, _cmap)

NPZ = Path(REPO) / "data/cache/tf_glm_bg046/peth_traces_all.npz"
LICK = Path(REPO) / "FIGURES/tf_glm_bg046/lick_acquisition/lick_acquisition_cells.csv"
OUT = Path(REPO) / "FIGURES/tf_glm_bg046/fa_lick_continuum"

PRE = (-0.3, -0.15)      # canonical pre-lick ramp window (== FA-lick responsiveness window)
N_WIDTH_BINS = 5

LICK_TRUE = "#d94801"    # lick-responsive
LICK_FALSE = "#4292c6"   # not lick-responsive
LICK_MISSING = "#bdbdbd"  # no lick label (greyed) — robustness fallback


def _join_width(D):
    """One continuous width (interp_fwhm) per npz cell, keyed (subject, session, unit)."""
    d = load_width_metrics()
    wmap = {(str(r.subject), str(r.session), int(r.unit)): float(getattr(r, WIDTH))
            for r in d.itertuples()}
    return np.array([wmap.get((str(D["meta_subject"][i]), str(D["meta_session"][i]),
                               int(D["meta_unit"][i])), np.nan)
                     for i in range(len(D["meta_unit"]))])


def _join_lick(D):
    """Per-cell lick-responsive label (float: 1.0 True / 0.0 False / NaN missing)."""
    lk = pd.read_csv(LICK)
    lmap = {(str(r.subject), str(r.session), int(r.unit)): bool(r.lick_sig)
            for r in lk.itertuples()}
    out = np.full(len(D["meta_unit"]), np.nan)
    for i in range(len(D["meta_unit"])):
        v = lmap.get((str(D["meta_subject"][i]), str(D["meta_session"][i]),
                      int(D["meta_unit"][i])), None)
        if v is not None:
            out[i] = 1.0 if v else 0.0
    return out


def _mean_ci(rows):
    """Per-timepoint mean and 95% CI half-width (1.96 * SEM) across cells in a bin."""
    mean = np.nanmean(rows, axis=0)
    n = np.sum(np.isfinite(rows), axis=0).clip(1)
    sem = np.nanstd(rows, axis=0, ddof=1) / np.sqrt(n)
    return mean, 1.96 * sem


def _lick_rgb(lick_sorted):
    """RGB column image for the lick overlay strip: True/False colours, GREY for missing."""
    rgb = np.zeros((len(lick_sorted), 1, 3))
    for i, v in enumerate(lick_sorted):
        if not np.isfinite(v):
            rgb[i, 0] = to_rgb(LICK_MISSING)
        elif v >= 0.5:
            rgb[i, 0] = to_rgb(LICK_TRUE)
        else:
            rgb[i, 0] = to_rgb(LICK_FALSE)
    return rgb


def main():
    D = {k: v for k, v in np.load(NPZ, allow_pickle=True).items()}
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        from visdetect.viz.plotting import set_style
        set_style("talk")
    except Exception:
        pass
    plt.rcParams.update({"font.size": 12})

    n_all = len(D["meta_unit"])
    t = D["t_fa"]
    M = D["mat_fa"]
    cls = D["meta_cls"].astype(str)
    w = _join_width(D)
    lick = _join_lick(D)

    n_w = int(np.isfinite(w).sum())
    n_lab = int(np.isfinite(lick).sum())
    n_grey = n_all - n_lab
    if n_w < n_all:
        print(f"WARNING: {n_all - n_w} cells missing interp_fwhm — dropped from ordering")

    # pre-lick FA ramp per cell = mean z over the (-0.3, -0.15) s window
    pre = (t >= PRE[0]) & (t <= PRE[1])
    ramp = np.nanmean(M[:, pre], axis=1)

    # ── global row order = ascending continuous width (narrow -> plotted at TOP) ──
    order = np.argsort(w, kind="stable")
    order = order[np.isfinite(w[order])]
    Msort = M[order]
    w_sorted = w[order]
    lick_sorted = lick[order]
    n = len(order)
    wmin, wmax = float(np.nanmin(w)), float(np.nanmax(w))

    # 5 equal-count width bins (idx on ORIGINAL cell order)
    bin_idx, edges = width_bin_assign(w, n=N_WIDTH_BINS)
    cmap = _cmap()
    bin_colors = cmap(np.linspace(0.08, 0.95, N_WIDTH_BINS))
    bin_counts, bin_pctlick, bin_ramp, bin_prezmean = [], [], [], []
    for b in range(N_WIDTH_BINS):
        sel = bin_idx == b
        bin_counts.append(int(sel.sum()))
        lb = lick[sel]
        lb = lb[np.isfinite(lb)]
        bin_pctlick.append(100.0 * np.mean(lb) if lb.size else np.nan)
        bin_ramp.append(float(np.nanmean(ramp[sel])) if sel.any() else np.nan)

    # global ramp~width Spearman (finite pairs)
    fin = np.isfinite(w) & np.isfinite(ramp)
    rho, p = spearmanr(w[fin], ramp[fin])

    # ── figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15.5, 11.5))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 2.7], width_ratios=[1.0, 1.18],
                           hspace=0.28, wspace=0.20)

    # ── Panel A: pre-lick ramp vs continuous width ──────────────────────────────
    axA = fig.add_subplot(gs[0, 0])
    binned_trend(axA, w, ramp, n_bins=10, color="#238b45",
                 label="decile mean +/- boot-CI")
    axA.axhline(0, color="0.8", lw=0.8, zorder=0)
    axA.axvspan(0, 0, color="none")
    axA.set_xlabel("kernel width  interp_fwhm (s)", fontsize=12)
    axA.set_ylabel(f"pre-lick FA ramp  (mean z, {PRE[0]}..{PRE[1]} s)", fontsize=12)
    axA.set_title("A  broad-width cells carry a larger pre-lick FA ramp",
                  fontsize=12.5, fontweight="bold", loc="left")
    axA.legend(frameon=False, fontsize=9.5, loc="lower right")
    axA.tick_params(labelsize=10)

    # ── Panel C: FA PSTH family by width bin (mean + 95% CI) ─────────────────────
    axC = fig.add_subplot(gs[0, 1])
    axC.axvspan(PRE[0], PRE[1], color="0.88", zorder=0, label="pre-lick window")
    for b in range(N_WIDTH_BINS):
        rows = M[bin_idx == b]
        if rows.size == 0:
            continue
        mean, ci = _mean_ci(rows)
        bin_prezmean.append(float(np.nanmean(mean[pre])))
        axC.fill_between(t, mean - ci, mean + ci, color=bin_colors[b], alpha=0.18, lw=0)
        pct = bin_pctlick[b]
        pct_s = f"{pct:.0f}% lick" if np.isfinite(pct) else "no label"
        lab = f"{edges[b]:.3f}-{edges[b+1]:.3f}s  (n={bin_counts[b]}, {pct_s})"
        axC.plot(t, mean, color=bin_colors[b], lw=2.2, label=lab)
    axC.axvline(0, color="k", lw=1.0, ls="--")
    axC.axhline(0, color="0.85", lw=0.8)
    axC.set_xlim(t[0], t[-1])
    axC.set_xlabel("t from FA lick (s)", fontsize=12)
    axC.set_ylabel("z-score (pop mean)", fontsize=12)
    axC.set_title("C  FA PSTH family by width bin  (+/-95% CI)",
                  fontsize=12.5, fontweight="bold", loc="left")
    axC.legend(frameon=False, fontsize=8.6,
               title="width bin (interp_fwhm, s) — n~104 each",
               title_fontsize=9.0, loc="upper left", handlelength=1.3)
    axC.tick_params(labelsize=10)
    for sp in ("top", "right"):
        axC.spines[sp].set_visible(False)

    # ── Panel B: width-ordered FA heatmap (+ width strip + lick strip) ───────────
    axH = fig.add_subplot(gs[1, :])
    im = axH.imshow(Msort, aspect="auto", cmap="RdBu_r",
                    norm=TwoSlopeNorm(vmin=-1.5, vcenter=0.0, vmax=3.0),
                    extent=[t[0], t[-1], n, 0], interpolation="nearest")
    axH.axvspan(PRE[0], PRE[1], color="k", alpha=0.08, zorder=1)
    axH.axvline(0, color="k", lw=1.0, ls="--")
    axH.set_xlabel("t from FA lick (s)", fontsize=13)
    axH.set_yticks([])
    axH.set_title("B  FA-lick heatmap — rows ordered by continuous width (narrow top -> broad bottom)",
                  fontsize=12.5, fontweight="bold", loc="left")
    axH.tick_params(labelsize=10)

    # continuous-width viridis strip: narrow(top) -> broad(bottom), matching rows
    wstrip = axH.inset_axes([-0.070, 0.0, 0.024, 1.0])
    wstrip.imshow(w_sorted[:, None], aspect="auto", origin="upper", cmap=cmap,
                  vmin=wmin, vmax=wmax, interpolation="nearest")
    wstrip.set_xticks([]); wstrip.set_yticks([])
    wstrip.set_title("narrow", fontsize=8.5, pad=3)
    wstrip.set_xlabel("broad", fontsize=8.5, labelpad=3)
    wstrip.text(-1.9, 0.5, "cells ordered by width", rotation=90, va="center",
                ha="center", transform=wstrip.transAxes, fontsize=10)

    # lick-responsive overlay strip (GREY = no label), same row order
    lstrip = axH.inset_axes([-0.040, 0.0, 0.024, 1.0])
    lstrip.imshow(_lick_rgb(lick_sorted), aspect="auto", origin="upper",
                  interpolation="nearest")
    lstrip.set_xticks([]); lstrip.set_yticks([])
    lstrip.set_title("lick", fontsize=8.5, pad=3)

    # z-score colorbar (right)
    cbz = fig.colorbar(im, ax=axH, fraction=0.018, pad=0.012, ticks=[-1, 0, 1, 2, 3])
    cbz.set_label("FA z-score (per-unit baseline)", fontsize=11)
    cbz.ax.tick_params(labelsize=9)

    # numeric continuous-width colorbar (left) — INVERTED so narrow is at TOP,
    # matching the strip + row order.
    sm = ScalarMappable(norm=Normalize(vmin=wmin, vmax=wmax), cmap=cmap)
    cbw = fig.colorbar(sm, ax=axH, location="left", fraction=0.016, pad=0.11, aspect=40)
    cbw.set_label("kernel width  interp_fwhm (s)  [narrow top]", fontsize=11)
    cbw.ax.invert_yaxis()
    cbw.ax.tick_params(labelsize=9)

    # lick-strip legend
    handles = [Patch(facecolor=LICK_TRUE, label="lick-responsive"),
               Patch(facecolor=LICK_FALSE, label="not lick-responsive")]
    if n_grey > 0:
        handles.append(Patch(facecolor=LICK_MISSING, label="no lick label (grey)"))
    axH.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, -0.16),
               ncol=len(handles), frameon=False, fontsize=9.5,
               title="lick overlay strip", title_fontsize=9.5)

    n_tr = int((cls == "transient").sum())
    n_su = int((cls == "sustained").sum())
    n_in = int((cls == "intermediate").sum())
    fig.suptitle(
        "FA-lick activity of TF-responsive cells ordered by continuous width; "
        "broad-width cells carry the pre-lick ramp\n"
        f"n={n} cells ({n_tr} transient, {n_in} intermediate, {n_su} sustained) — "
        f"ramp~width Spearman rho={rho:+.2f} (p={p:.1e}); "
        f"lick labels cover {n_lab}/{n_all} cells ({n_grey} greyed)",
        fontsize=12.5, y=0.995,
    )
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fa_lick_continuum.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # ── stats ───────────────────────────────────────────────────────────────
    lines = [
        "FA-lick continuum: pre-lick ramp vs continuous width + width-ordered FA",
        "heatmap + width-binned FA PSTH family",
        "(continuum re-render of fa_lick_activity.py; class comparison -> graded width)",
        "",
        f"n cells = {n} (finite interp_fwhm; total npz cells = {n_all})",
        f"  class composition: {n_tr} transient, {n_in} intermediate, {n_su} sustained",
        f"pre-lick ramp window = {PRE} s (mean z of mat_fa over this window, per cell)",
        "",
        f"ramp ~ width Spearman: rho = {rho:+.4f}, p = {p:.3e}, n = {int(fin.sum())}",
        "  (POSITIVE = broad-width cells carry the larger pre-lick FA ramp)",
        "",
        f"row ordering = CONTINUOUS interp_fwhm ASCENDING; NARROW at TOP, BROAD at BOTTOM",
        f"width range = {wmin:.4f} - {wmax:.4f} s   median = {np.nanmedian(w):.4f} s",
        "heatmap: RdBu_r, TwoSlopeNorm(vmin=-1.5, vcenter=0, vmax=3); FA is predominantly",
        "  excitatory (motor ramp) so NO sign-flip is applied (unlike the fast pulse).",
        "left strip + numeric width colorbar both run NARROW(top) -> BROAD(bottom).",
        "",
        f"{N_WIDTH_BINS} EQUAL-COUNT width bins (equal n, unequal width span; right-skewed):",
        "  edges (s) = [" + ", ".join(f"{e:.4f}" for e in edges) + "]",
    ]
    for b in range(N_WIDTH_BINS):
        pct = bin_pctlick[b]
        pct_s = f"{pct:.1f}%" if np.isfinite(pct) else "n/a"
        lines.append(
            f"    bin {b}: [{edges[b]:.4f}, {edges[b+1]:.4f}) s   n = {bin_counts[b]}"
            f"   %lick-resp = {pct_s}   mean pre-lick ramp = {bin_ramp[b]:+.4f} z")
    lines += [
        "",
        "% lick-responsive rises across width bins (labeled cells only): "
        + " -> ".join(f"{v:.0f}%" for v in bin_pctlick if np.isfinite(v)),
        "per-bin mean pre-lick ramp (broad bins larger): "
        + " -> ".join(f"{v:+.3f}" for v in bin_ramp),
        "per-bin PSTH mean over the pre-lick window (all positive = excitatory, no cancel): "
        + " -> ".join(f"{v:+.3f}" for v in bin_prezmean),
        "",
        "LICK LABEL COVERAGE:",
        f"  lick_acquisition_cells.csv labels {n_lab}/{n_all} of the responsive cells "
        f"({n_grey} greyed).",
        "  NOTE: the task brief expected the lick screen to cover the 414 transient/",
        "  sustained cells only (intermediates greyed). In the CURRENT cache the CSV is a",
        "  FULL screen and labels ALL 520 cells incl. the intermediates, so 0 are greyed.",
        "  Per-class %lick-responsive forms a clean continuum: "
        f"transient {100*np.nanmean(lick[cls=='transient']):.0f}% -> "
        f"intermediate {100*np.nanmean(lick[cls=='intermediate']):.0f}% -> "
        f"sustained {100*np.nanmean(lick[cls=='sustained']):.0f}%.",
        "  The GREY-for-missing overlay path is retained as a robustness fallback.",
    ]
    (OUT / "fa_lick_continuum_stats.txt").write_text("\n".join(lines), encoding="utf-8")

    print(f"wrote {OUT}/fa_lick_continuum.png (+.pdf, +_stats.txt)  "
          f"[n={n} cells, ramp~width rho={rho:+.2f} p={p:.1e}, "
          f"lick labels {n_lab}/{n_all} ({n_grey} greyed)]")
    for s in lines:
        print("  " + s.encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
