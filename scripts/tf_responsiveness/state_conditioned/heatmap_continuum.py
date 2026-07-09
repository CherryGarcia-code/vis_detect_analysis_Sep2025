"""Continuum re-render of the §3 transient/sustained heatmap figure.

The class figure (`heatmap_transient_sustained.py`) ordered the population heatmap
rows by a transient/sustained CLASS block (then within-block by fast-pulse peak
latency) and summarised each alignment with two class-mean PSTHs. The spectrum
result ([[tf_transient_sustained_state_jul2026]] / docs 2026-07
transient-sustained-spectrum) showed kernel width is a GRADED axis, not two
classes, so here:

  * every row is ordered by the CONTINUOUS kernel width (`interp_fwhm`) ascending
    — one global order shared across all three alignments, so one heatmap row is
    one cell everywhere (no transient/sustained blocks);
  * a continuous-width colour strip (viridis) runs down the left of each heatmap
    to make the width gradient explicit;
  * above each heatmap, a PSTH FAMILY replaces the two class-mean traces: cells
    are split into 5 equal-count width bins (`width_bin_assign`) and the mean
    z-trace per bin is drawn along the viridis gradient (the continuum analogue
    of the class-mean PSTHs).

Per-panel colour scaling matches the class heatmap: change / FA on an asymmetric
`TwoSlopeNorm(-1.5, 0, 3)` baseline-z (preserves magnitude), the fast TF pulse
peak-normalised per unit (the ~1 Hz pulse is small vs ongoing-firing SD, so a
baseline-z scale washes out its width/SHAPE). Cache-only: reads the rebuilt
peth_traces_all.npz (all 520 responsive cells, incl. the ~106 intermediates the
class figure dropped) + kernel_width_continuous.csv. No session reloads.

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

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from continuum_common import load_width_metrics, width_bin_assign, WIDTH, REPO, _cmap  # noqa: E402

NPZ = Path(REPO) / "data/cache/tf_glm_bg046/peth_traces_all.npz"
OUT = Path(REPO) / "FIGURES/tf_glm_bg046/heatmap_continuum"

KEYS = ["pulse", "change", "fa"]
TITLES = {"pulse": "fast TF pulse", "change": "Change_ON (hit trials)", "fa": "FA (early lick)"}
XLAB = {"pulse": "t from fast TF pulse (s)", "change": "t from Change_ON (s)",
        "fa": "t from FA lick (s)"}
N_WIDTH_BINS = 5


def _join_width(D):
    """One continuous width (interp_fwhm) per npz cell, keyed (subject, session, unit)."""
    d = load_width_metrics()
    wmap = {(str(r.subject), str(r.session), int(r.unit)): float(getattr(r, WIDTH))
            for r in d.itertuples()}
    w = np.array([wmap.get((str(D["meta_subject"][i]), str(D["meta_session"][i]),
                            int(D["meta_unit"][i])), np.nan)
                  for i in range(len(D["meta_unit"]))])
    return w


def _panel_matrix(M, key):
    """Apply the per-panel transform (pulse -> per-unit peak-norm; else raw z)."""
    M = np.asarray(M, float).copy()
    if key == "pulse":
        pk = np.nanmax(np.abs(M), axis=1, keepdims=True)
        pk[~np.isfinite(pk) | (pk < 1e-9)] = 1.0
        M = M / pk
    return M


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
        # keep only cells with a continuous width (should be all 520)
        print(f"WARNING: {n - n_fin} cells missing interp_fwhm — dropped from ordering")

    # one global row order = ascending continuous width (stable => deterministic ties)
    order = np.argsort(w, kind="stable")
    # push any NaN-width cells (finite last in argsort) — keep only finite for plotting
    order = order[np.isfinite(w[order])]
    w_sorted = w[order]
    wmin, wmax = float(np.nanmin(w)), float(np.nanmax(w))

    # 5 equal-count width bins (assignment is on the ORIGINAL cell order; idx[i] = bin of cell i)
    bin_idx, edges = width_bin_assign(w, n=N_WIDTH_BINS)
    cmap = _cmap()
    bin_colors = cmap(np.linspace(0.08, 0.95, N_WIDTH_BINS))
    bin_counts = [int(np.sum(bin_idx == b)) for b in range(N_WIDTH_BINS)]

    fig = plt.figure(figsize=(17, 11))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 3.0], hspace=0.20, wspace=0.24)

    ims, hax = {}, {}
    for j, k in enumerate(KEYS):
        t = D[f"t_{k}"]
        Mfull = _panel_matrix(D[f"mat_{k}"], k)   # transformed, in ORIGINAL cell order
        M = Mfull[order]                          # width-ordered rows for the heatmap

        # ── top panel: width-binned PSTH family ─────────────────────────────
        axp = fig.add_subplot(gs[0, j])
        for b in range(N_WIDTH_BINS):
            rows = Mfull[bin_idx == b]
            if rows.size == 0:
                continue
            mean = np.nanmean(rows, axis=0)
            lab = f"{edges[b]:.3f}–{edges[b + 1]:.3f}s (n={bin_counts[b]})"
            axp.plot(t, mean, color=bin_colors[b], lw=2.0, label=lab)
        axp.axvline(0, color="0.6", lw=0.8)
        axp.axhline(0, color="0.85", lw=0.8)
        axp.set_xlim(t[0], t[-1])
        yl = "peak-norm (pop mean)" if k == "pulse" else "z-score (pop mean)"
        axp.set_ylabel(yl, fontsize=12)
        axp.set_title(f"{TITLES[k]}  — width families", fontsize=13.5, fontweight="bold")
        axp.tick_params(labelsize=10)
        for sp in ("top", "right"):
            axp.spines[sp].set_visible(False)
        if j == 1:
            # width bins are identical across the three families -> one legend suffices;
            # place it on the change panel (clean upper-left, traces rise from 0 there).
            axp.legend(frameon=False, fontsize=8.5, title="kernel width bin (interp_fwhm)",
                       title_fontsize=9, loc="upper left")

        # ── heatmap: rows ordered by continuous width ───────────────────────
        axh = fig.add_subplot(gs[1, j])
        hax[k] = axh
        if k == "pulse":
            imkw = dict(vmin=-1.0, vmax=1.0)   # peak-norm shape
        else:
            imkw = dict(norm=TwoSlopeNorm(vmin=-1.5, vcenter=0.0, vmax=3.0))  # baseline-z magnitude
        ims[k] = axh.imshow(M, aspect="auto", cmap="RdBu_r",
                            extent=[t[0], t[-1], len(M), 0],
                            interpolation="nearest", **imkw)
        axh.axvline(0, color="k", lw=1.0, ls="--")
        axh.set_xlabel(XLAB[k], fontsize=13)
        axh.tick_params(labelsize=10)
        axh.set_yticks([])

        # continuous-width colour strip down the left of each heatmap (viridis)
        strip = axh.inset_axes([-0.055, 0.0, 0.030, 1.0])
        strip.imshow(w_sorted[:, None], aspect="auto", origin="upper",
                     cmap=cmap, vmin=wmin, vmax=wmax, interpolation="nearest")
        strip.set_xticks([])
        strip.set_yticks([])
        if j == 0:
            strip.set_ylabel("cells: narrow → broad width", fontsize=11)

    # ── colorbars ───────────────────────────────────────────────────────────
    # change / FA share the baseline-z magnitude scale (right)
    cbz = fig.colorbar(ims["fa"], ax=[hax["change"], hax["fa"]], fraction=0.02, pad=0.015)
    cbz.set_label("change / FA:  z-score (per-unit, baseline)", fontsize=12)
    cbz.ax.tick_params(labelsize=10)
    # pulse peak-norm scale (its own bar) — horizontal, in a dedicated axes below the
    # pulse column so it does NOT shrink the heatmap (all three heatmaps keep equal
    # height => one row is one cell at the same scale everywhere).
    fig.canvas.draw()  # realise the gridspec positions before querying them
    pp = hax["pulse"].get_position()
    cax = fig.add_axes([pp.x0, pp.y0 - 0.075, pp.width, 0.014])
    cbp = fig.colorbar(ims["pulse"], cax=cax, orientation="horizontal")
    cbp.set_label("pulse: peak-norm", fontsize=11)
    cbp.ax.tick_params(labelsize=9)
    # single continuous-width colorbar (explains the left strips), far left
    sm = ScalarMappable(norm=Normalize(vmin=wmin, vmax=wmax), cmap=cmap)
    cbw = fig.colorbar(sm, ax=[hax["pulse"], hax["change"], hax["fa"]],
                       location="left", fraction=0.015, pad=0.07, aspect=40)
    cbw.set_label("kernel width  interp_fwhm (s)", fontsize=11)
    cbw.ax.tick_params(labelsize=9)

    n_interm = int(np.sum(D["meta_cls"].astype(str) == "intermediate"))
    fig.suptitle(
        f"TF-responsive cells ordered by CONTINUOUS kernel width (interp_fwhm) "
        f"— n={len(order)} cells, no transient/sustained blocks\n"
        f"all responsive cells incl. the {n_interm} intermediate-width cells; "
        "PSTH families = mean z per width bin (viridis gradient)",
        fontsize=13, y=0.998,
    )
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"heatmap_continuum.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # ── stats ───────────────────────────────────────────────────────────────
    lines = [
        "Width-ordered population heatmap + width-binned PSTH families",
        "(continuum re-render of the transient/sustained class heatmap)",
        "",
        f"n cells plotted = {len(order)} (finite interp_fwhm; total npz cells = {n})",
        "row ordering = CONTINUOUS kernel width interp_fwhm, ASCENDING "
        "(one global order shared across pulse / change / FA; NO transient/sustained blocks)",
        f"width range = {wmin:.4f} – {wmax:.4f} s   median = {np.nanmedian(w):.4f} s",
        "",
        f"{N_WIDTH_BINS} equal-count width bins (width_bin_assign):",
        "  edges = [" + ", ".join(f"{e:.4f}" for e in edges) + "]",
    ]
    for b in range(N_WIDTH_BINS):
        lines.append(f"    bin {b}: [{edges[b]:.4f}, {edges[b+1]:.4f}) s   n = {bin_counts[b]}")
    lines += [
        "",
        "per-panel colour scaling (matches the class heatmap):",
        "  pulse  : per-unit peak-norm, RdBu_r, vmin=-1 vmax=1 "
        "(baseline-z washes out the ~1 Hz pulse SHAPE)",
        "  change : baseline-z, RdBu_r, TwoSlopeNorm(vmin=-1.5, vcenter=0, vmax=3)",
        "  fa     : baseline-z, RdBu_r, TwoSlopeNorm(vmin=-1.5, vcenter=0, vmax=3)",
        "left strip = continuous width (viridis, narrow->broad top->bottom).",
    ]
    (OUT / "heatmap_continuum_stats.txt").write_text("\n".join(lines), encoding="utf-8")

    print(f"wrote {OUT}/heatmap_continuum.png (+.pdf, +_stats.txt)  [n={len(order)} cells]")
    for s in lines:
        print("  " + s.encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
