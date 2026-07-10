"""Continuum re-render of the §3 transient/sustained heatmap figure.

The class figure (`heatmap_transient_sustained.py`) ordered the population heatmap
rows by a transient/sustained CLASS block and summarised each alignment with two
class-mean PSTHs. The spectrum result (docs 2026-07 transient-sustained-spectrum)
showed kernel width is a GRADED axis, not two classes, so here:

  * every row is ordered by the CONTINUOUS kernel width (`interp_fwhm`) ascending —
    one global order shared across all three alignments (narrow at TOP, broad at
    BOTTOM), so one heatmap row is one cell everywhere (no transient/sustained blocks);
  * a continuous-width colour strip (viridis) runs down the left of each heatmap,
    oriented to MATCH the rows (narrow top -> broad bottom) and the numeric width
    colorbar;
  * above each heatmap, a PSTH FAMILY replaces the two class-mean traces: cells are
    split into 5 equal-COUNT width bins (`width_bin_assign`; equal n per bin, so the
    top/broad bin spans a wider width range because the width distribution is
    right-skewed) and the mean z-trace per bin is drawn along the viridis gradient,
    with 95% CI (+/-1.96 SEM) shading.

FAST TF PULSE — SIGN-ALIGNED. ~half of TF-responsive cells are SUPPRESSION-type
(fire less to a fast pulse; the GLM responsiveness is sign-agnostic), so a signed
pop-mean cancels excitation against suppression. For the pulse panel ONLY we flip
each cell to its own post-pulse response sign (suppression cells * -1) so the
population response is coherent; the % flipped per bin is annotated. Change / FA are
predominantly excitatory and are shown raw.

Per-panel colour scaling: change / FA on an asymmetric `TwoSlopeNorm(-1.5, 0, 3)`
baseline-z (preserves magnitude); the sign-aligned pulse on a robust percentile-
clipped baseline-z (cleaner than the old peak-norm, which amplified weak-cell noise).
Cache-only: reads the rebuilt peth_traces_all.npz (all 520 responsive cells, incl.
the ~106 intermediates the class figure dropped) + kernel_width_continuous.csv.

Usage:  py scripts/tf_responsiveness/state_conditioned/heatmap_continuum.py
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
from matplotlib.colors import TwoSlopeNorm, Normalize
from matplotlib.cm import ScalarMappable
from scipy.ndimage import gaussian_filter1d

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from continuum_common import load_width_metrics, width_bin_assign, WIDTH, REPO, _cmap  # noqa: E402

NPZ = Path(REPO) / "data/cache/tf_glm_bg046/peth_traces_all.npz"
OUT = Path(REPO) / "FIGURES/tf_glm_bg046/heatmap_continuum"

KEYS = ["pulse", "change", "fa"]
TITLES = {"pulse": "fast TF pulse (sign-aligned)", "change": "Change_ON (hit trials)",
          "fa": "FA (early lick)"}
XLAB = {"pulse": "t from fast TF pulse (s)", "change": "t from Change_ON (s)",
        "fa": "t from FA lick (s)"}
N_WIDTH_BINS = 5
PULSE_SIGN_WIN = (0.0, 0.4)   # post-pulse window used to define each cell's response sign


def _join_width(D):
    """One continuous width (interp_fwhm) per npz cell, keyed (subject, session, unit)."""
    d = load_width_metrics()
    wmap = {(str(r.subject), str(r.session), int(r.unit)): float(getattr(r, WIDTH))
            for r in d.itertuples()}
    w = np.array([wmap.get((str(D["meta_subject"][i]), str(D["meta_session"][i]),
                            int(D["meta_unit"][i])), np.nan)
                  for i in range(len(D["meta_unit"]))])
    return w


def _pulse_sign(M, t, win=PULSE_SIGN_WIN):
    """Per-cell sign of the post-pulse deflection (+1 excitation / -1 suppression)."""
    post = (t >= win[0]) & (t <= win[1])
    s = np.sign(np.nanmean(M[:, post], axis=1))
    s[~np.isfinite(s) | (s == 0)] = 1.0
    return s


def _mean_ci(rows):
    """Per-timepoint mean and 95% CI half-width (1.96 * SEM) across cells in a bin."""
    mean = np.nanmean(rows, axis=0)
    n = np.sum(np.isfinite(rows), axis=0).clip(1)
    sem = np.nanstd(rows, axis=0, ddof=1) / np.sqrt(n)
    return mean, 1.96 * sem


def main():
    D = {k: v for k, v in np.load(NPZ, allow_pickle=True).items()}
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        from visdetect.viz.plotting import set_style
        set_style("talk")
    except Exception:
        pass
    plt.rcParams.update({"font.size": 11})

    n = len(D["meta_unit"])
    w = _join_width(D)
    n_fin = int(np.isfinite(w).sum())
    if n_fin < n:
        print(f"WARNING: {n - n_fin} cells missing interp_fwhm — dropped from ordering")

    # one global row order = ascending continuous width (narrow first -> plotted at TOP)
    order = np.argsort(w, kind="stable")
    order = order[np.isfinite(w[order])]
    w_sorted = w[order]
    wmin, wmax = float(np.nanmin(w)), float(np.nanmax(w))

    # 5 equal-count width bins (idx on ORIGINAL cell order; equal n, unequal width span)
    bin_idx, edges = width_bin_assign(w, n=N_WIDTH_BINS)
    cmap = _cmap()
    bin_colors = cmap(np.linspace(0.08, 0.95, N_WIDTH_BINS))
    bin_counts = [int(np.sum(bin_idx == b)) for b in range(N_WIDTH_BINS)]

    # sign-align the fast-pulse traces (flip suppression cells); track % flipped / bin
    psign = _pulse_sign(D["mat_pulse"], D["t_pulse"])
    mat_signed = {"pulse": D["mat_pulse"] * psign[:, None],
                  "change": D["mat_change"], "fa": D["mat_fa"]}
    pct_flip_bin = [100.0 * np.mean(psign[bin_idx == b] < 0) if np.any(bin_idx == b) else 0.0
                    for b in range(N_WIDTH_BINS)]
    pct_flip_all = 100.0 * np.mean(psign < 0)
    # robust colour range for the (sign-aligned) pulse heatmap
    pv = np.abs(mat_signed["pulse"])
    pmax = float(np.nanpercentile(pv, 97)) or 1.0

    fig = plt.figure(figsize=(17, 11.5))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 3.0], hspace=0.24, wspace=0.24)

    ims, hax = {}, {}
    for j, k in enumerate(KEYS):
        t = D[f"t_{k}"]
        Mfull = mat_signed[k]
        M = Mfull[order]
        if k == "pulse":
            # per-cell pulse response is near the baseline z-noise floor; light
            # display smoothing reveals the (sign-aligned) response band.
            M = gaussian_filter1d(M, 1.3, axis=1)

        # ── top panel: width-binned PSTH family (mean + 95% CI shading) ─────────
        axp = fig.add_subplot(gs[0, j])
        for b in range(N_WIDTH_BINS):
            rows = Mfull[bin_idx == b]
            if rows.size == 0:
                continue
            mean, ci = _mean_ci(rows)
            axp.fill_between(t, mean - ci, mean + ci, color=bin_colors[b], alpha=0.20, lw=0)
            extra = f", {pct_flip_bin[b]:.0f}% supp" if k == "pulse" else ""
            lab = f"{edges[b]:.3f}–{edges[b+1]:.3f}s (n={bin_counts[b]}{extra})"
            axp.plot(t, mean, color=bin_colors[b], lw=2.0, label=lab)
        axp.axvline(0, color="0.6", lw=0.8)
        axp.axhline(0, color="0.85", lw=0.8)
        axp.set_xlim(t[0], t[-1])
        yl = "sign-aligned z (pop mean)" if k == "pulse" else "z-score (pop mean)"
        axp.set_ylabel(yl, fontsize=12)
        axp.set_title(f"{TITLES[k]}  — width families (±95% CI)", fontsize=12.5, fontweight="bold")
        axp.tick_params(labelsize=10)
        for sp in ("top", "right"):
            axp.spines[sp].set_visible(False)
        # each panel carries its own legend so n / width-ranges are self-explanatory
        ttl = "width bin (interp_fwhm, s) — n=104 each" if k != "pulse" else \
              "width bin — n=104 each (% suppression flipped)"
        axp.legend(frameon=False, fontsize=7.6, title=ttl, title_fontsize=8.2,
                   loc="upper left", handlelength=1.2)

        # ── heatmap: rows ordered by continuous width (narrow top -> broad bottom) ─
        axh = fig.add_subplot(gs[1, j])
        hax[k] = axh
        if k == "pulse":
            imkw = dict(norm=TwoSlopeNorm(vmin=-pmax, vcenter=0.0, vmax=pmax))
        else:
            imkw = dict(norm=TwoSlopeNorm(vmin=-1.5, vcenter=0.0, vmax=3.0))
        ims[k] = axh.imshow(M, aspect="auto", cmap="RdBu_r",
                            extent=[t[0], t[-1], len(M), 0],
                            interpolation="nearest", **imkw)
        axh.axvline(0, color="k", lw=1.0, ls="--")
        axh.set_xlabel(XLAB[k], fontsize=13)
        axh.tick_params(labelsize=10)
        axh.set_yticks([])

        # continuous-width colour strip (viridis), narrow(top) -> broad(bottom),
        # matching the row order; end-labelled to remove any orientation ambiguity.
        strip = axh.inset_axes([-0.060, 0.0, 0.030, 1.0])
        strip.imshow(w_sorted[:, None], aspect="auto", origin="upper",
                     cmap=cmap, vmin=wmin, vmax=wmax, interpolation="nearest")
        strip.set_xticks([]); strip.set_yticks([])
        if j == 0:
            strip.set_title("narrow", fontsize=9, pad=3)
            strip.set_xlabel("broad", fontsize=9, labelpad=3)
            strip.text(-1.7, 0.5, "cells ordered by width", rotation=90, va="center",
                       ha="center", transform=strip.transAxes, fontsize=10)

    # ── colorbars ───────────────────────────────────────────────────────────
    cbz = fig.colorbar(ims["fa"], ax=[hax["change"], hax["fa"]], fraction=0.02, pad=0.015)
    cbz.set_label("change / FA:  z-score (per-unit, baseline)", fontsize=12)
    cbz.ax.tick_params(labelsize=10)
    fig.canvas.draw()
    pp = hax["pulse"].get_position()
    cax = fig.add_axes([pp.x0, pp.y0 - 0.075, pp.width, 0.014])
    cbp = fig.colorbar(ims["pulse"], cax=cax, orientation="horizontal")
    cbp.set_label("pulse: sign-aligned z (suppression flipped +)", fontsize=10)
    cbp.ax.tick_params(labelsize=9)
    # numeric continuous-width colorbar (far left) — INVERTED so narrow is at TOP,
    # matching the left strips and the heatmap row order.
    sm = ScalarMappable(norm=Normalize(vmin=wmin, vmax=wmax), cmap=cmap)
    cbw = fig.colorbar(sm, ax=[hax["pulse"], hax["change"], hax["fa"]],
                       location="left", fraction=0.015, pad=0.07, aspect=40)
    cbw.set_label("kernel width  interp_fwhm (s)  [narrow top]", fontsize=11)
    cbw.ax.invert_yaxis()
    cbw.ax.tick_params(labelsize=9)

    n_interm = int(np.sum(D["meta_cls"].astype(str) == "intermediate"))
    fig.suptitle(
        f"TF-responsive cells ordered by CONTINUOUS kernel width (interp_fwhm, narrow top -> broad bottom) "
        f"— n={len(order)} cells, no transient/sustained blocks\n"
        f"all responsive cells incl. the {n_interm} intermediate-width cells; PSTH families = "
        f"mean +/-95% CI per equal-count width bin; pulse sign-aligned ({pct_flip_all:.0f}% suppression flipped)",
        fontsize=12.5, y=0.998,
    )
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"heatmap_continuum.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # ── stats ───────────────────────────────────────────────────────────────
    lines = [
        "Width-ordered population heatmap + width-binned PSTH families (mean +/-95% CI)",
        "(continuum re-render of the transient/sustained class heatmap)",
        "",
        f"n cells plotted = {len(order)} (finite interp_fwhm; total npz cells = {n})",
        "row ordering = CONTINUOUS kernel width interp_fwhm, ASCENDING; NARROW at TOP, BROAD at BOTTOM",
        "  (one global order shared across pulse / change / FA; NO transient/sustained blocks)",
        f"width range = {wmin:.4f} - {wmax:.4f} s   median = {np.nanmedian(w):.4f} s",
        "",
        f"{N_WIDTH_BINS} EQUAL-COUNT width bins (equal n, UNEQUAL width span — the width",
        "distribution is right-skewed, so the broad bin spans a wider range):",
        "  edges (s) = [" + ", ".join(f"{e:.4f}" for e in edges) + "]",
    ]
    for b in range(N_WIDTH_BINS):
        lines.append(f"    bin {b}: [{edges[b]:.4f}, {edges[b+1]:.4f}) s   n = {bin_counts[b]} cells"
                     f"   suppression-flipped (pulse) = {pct_flip_bin[b]:.0f}%")
    lines += [
        "",
        f"FAST PULSE SIGN-ALIGNMENT: each cell flipped to its post-pulse sign over "
        f"{PULSE_SIGN_WIN} s; {pct_flip_all:.1f}% of cells were suppression-type (flipped).",
        "  (a signed pop-mean cancels exc vs supp -> the raw family looks flat; sign-",
        "   alignment makes the population pulse response coherent.)",
        "",
        "per-panel colour scaling:",
        f"  pulse  : SIGN-ALIGNED baseline-z, RdBu_r, TwoSlopeNorm(+/-{pmax:.2f}) "
        "(robust 97th-pct clip; replaces the noisy peak-norm)",
        "  change : baseline-z, RdBu_r, TwoSlopeNorm(vmin=-1.5, vcenter=0, vmax=3)",
        "  fa     : baseline-z, RdBu_r, TwoSlopeNorm(vmin=-1.5, vcenter=0, vmax=3)",
        "left strip + numeric width colorbar both run NARROW(top) -> BROAD(bottom).",
    ]
    (OUT / "heatmap_continuum_stats.txt").write_text("\n".join(lines), encoding="utf-8")

    print(f"wrote {OUT}/heatmap_continuum.png (+.pdf, +_stats.txt)  [n={len(order)} cells, "
          f"{pct_flip_all:.0f}% pulse-suppression flipped]")
    for s in lines:
        print("  " + s.encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
