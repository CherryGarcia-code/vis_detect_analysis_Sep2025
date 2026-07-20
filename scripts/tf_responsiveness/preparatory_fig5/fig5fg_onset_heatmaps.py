"""Fig5 f/g: onset heatmaps of pre-lick preparatory activity.

Within-striatum port of Khilkevich & Lohse (Nature 2024) Fig 5 f/g, replacing the
brain-area grouping with the transient->sustained kernel-width axis.

  * Panel f (TF-responsive): rows = width deciles of interp_fwhm; per-decile mean
    fraction of active units above baseline (|z|>2.576, bootstrap over neurons);
    rows SORTED by population activation onset; onset points overlaid (black);
    left strip = decile median width (viridis).
  * Panel g (non-TF reference): rows = individual TF-non-responsive cells, each a
    binary active(t)=|z|>2.576 raster, ordered by that cell's own pre-lick onset
    (NaN onsets last). Same colour scale as f, NO width gradient — the reference
    showing no width-ordered recruitment wave.

PER-REGION ALWAYS: pooled + DMS + VMS. Cache-only (prep_<lick>.npz). No reload.

Usage:  py fig5fg_onset_heatmaps.py [--lick hit|fa]
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
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import prep_common as C  # noqa: E402
from visdetect.analysis.preparatory import (  # noqa: E402
    active_mask, bootstrap_fraction_ci, population_onset, cell_onset, width_deciles)

FIGROOT = C.REPO / "FIGURES/preparatory_fig5"
REGIONS = [("pooled", None), ("DMS", "DMS"), ("VMS", "VMS")]
N_DEC = 10
N_BOOT = 5000
HEAT_CMAP = "magma"          # sequential fraction/active map (distinct from viridis width strip)


def _panel_f_deciles(A_resp, interp, t, base_mask):
    """Per-width-decile mean fraction (10 x nBins), population onset, median width, n."""
    idx, _edges = width_deciles(interp, n=N_DEC)
    mat = np.full((N_DEC, len(t)), np.nan)
    onset = np.full(N_DEC, np.nan)
    medw = np.full(N_DEC, np.nan)
    ncell = np.zeros(N_DEC, int)
    for d in range(N_DEC):
        sel = idx == d
        ncell[d] = int(sel.sum())
        if ncell[d] == 0:
            continue
        mean, lo, _hi = bootstrap_fraction_ci(A_resp[sel], baseline_bins=base_mask, n=N_BOOT)
        mat[d] = mean
        onset[d] = population_onset(t, mean, lo)
        medw[d] = float(np.median(interp[sel]))
    return mat, onset, medw, ncell


def _sort_by_onset(onset):
    """Row order sorting ascending by onset; NaN onsets pushed last (stable)."""
    key = np.where(np.isnan(onset), np.inf, onset)
    return np.argsort(key, kind="stable")


def main(lick: str = "hit") -> None:
    path = C.REPO / f"data/cache/preparatory_fig5/prep_{lick}.npz"
    if not path.exists():
        raise SystemExit(f"cache missing: {path} — run build_prep_cache.py --lick {lick}")

    D = np.load(path, allow_pickle=True)
    t = np.asarray(D["t"], float)
    z = np.asarray(D["z"], float)
    resp = np.asarray(D["resp"], bool)
    region = D["region"].astype(str)
    interp = np.asarray(D["interp_fwhm"], float)

    A = active_mask(z)
    base_mask = (t >= C.BASE_FRAC_WIN[0]) & (t <= C.BASE_FRAC_WIN[1])
    lick_lbl = lick.upper()

    try:
        from visdetect.viz.plotting import set_style
        set_style("talk")
    except Exception:
        pass

    for rname, rval in REGIONS:
        rmask = np.ones(len(resp), bool) if rval is None else (region == rval)

        # Panel f population: TF-responsive AND finite interp_fwhm
        f_sel = rmask & resp & np.isfinite(interp)
        A_resp = A[f_sel]
        w_resp = interp[f_sel]
        mat, onset, medw, ncell = _panel_f_deciles(A_resp, w_resp, t, base_mask)
        order_f = _sort_by_onset(onset)
        mat_s = mat[order_f]
        onset_s = onset[order_f]
        medw_s = medw[order_f]
        ncell_s = ncell[order_f]

        # Panel g population: TF-non-responsive cells, own pre-lick onset
        g_sel = rmask & (~resp)
        A_non = A[g_sel]
        z_non = z[g_sel]
        onset_cell = np.array([cell_onset(t, z_non[i]) for i in range(A_non.shape[0])])
        order_g = _sort_by_onset(onset_cell)
        G = A_non[order_g].astype(float)
        onset_g_sorted = onset_cell[order_g]

        # shared colour scale (panel g uses same scale as f)
        vmax = float(np.nanmax(mat)) if np.any(np.isfinite(mat)) else 1.0
        vmax = max(vmax, 1e-3)
        norm = Normalize(vmin=0.0, vmax=vmax)

        fig = plt.figure(figsize=(15.0, 7.0))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1.0, 1.0], wspace=0.22)

        # ── Panel f ──────────────────────────────────────────────────────────
        axF = fig.add_subplot(gs[0, 0])
        axF.imshow(mat_s, aspect="auto", cmap=HEAT_CMAP, norm=norm,
                   extent=[float(t[0]), float(t[-1]), N_DEC, 0], interpolation="nearest")
        # onset overlay (black points + connecting line), row centre = i+0.5
        yrows = np.arange(N_DEC) + 0.5
        fin = np.isfinite(onset_s)
        if fin.any():
            axF.plot(onset_s[fin], yrows[fin], "-o", color="k", lw=1.4, ms=5, zorder=5)
        axF.axvline(0, color="w", lw=1.0, ls="--")
        axF.set_xlabel(f"time from {lick_lbl} lick (s)")
        axF.set_ylabel("width decile (sorted by onset)")
        axF.set_yticks([])
        axF.set_title("f  TF-responsive — width-decile onset heatmap",
                      fontsize=13, loc="left")

        # median-width viridis strip (rows match sorted order)
        strip = axF.inset_axes([-0.055, 0.0, 0.022, 1.0])
        strip.imshow(medw_s[:, None], aspect="auto", origin="upper", cmap=C.WIDTH_CMAP,
                     interpolation="nearest")
        strip.set_xticks([]); strip.set_yticks([])
        strip.set_title("width", fontsize=9, pad=3)

        # ── Panel g ──────────────────────────────────────────────────────────
        axG = fig.add_subplot(gs[0, 1])
        ng = G.shape[0]
        axG.imshow(G, aspect="auto", cmap=HEAT_CMAP, norm=norm,
                   extent=[float(t[0]), float(t[-1]), ng, 0], interpolation="nearest")
        axG.axvline(0, color="w", lw=1.0, ls="--")
        axG.set_xlabel(f"time from {lick_lbl} lick (s)")
        axG.set_ylabel(f"non-TF cells (n={ng}, sorted by onset)")
        axG.set_yticks([])
        axG.set_title("g  non-TF reference — cells by own onset (no width gradient)",
                      fontsize=13, loc="left")

        sm = ScalarMappable(norm=norm, cmap=HEAT_CMAP)
        cb = fig.colorbar(sm, ax=[axF, axG], fraction=0.020, pad=0.015)
        cb.set_label("fraction active above baseline")

        fig.suptitle(
            f"Fig5 f/g  preparatory-activity onset — {rname} ({lick_lbl}); "
            f"f: {int(f_sel.sum())} TF-responsive cells / {N_DEC} width deciles; "
            f"g: {ng} non-TF cells",
            fontsize=13, y=1.00)

        outdir = FIGROOT / rname
        outdir.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(outdir / f"fig5fg_{lick}.{ext}", dpi=170, bbox_inches="tight")
        plt.close(fig)

        # ── stats ────────────────────────────────────────────────────────────
        rows = []
        for i in range(N_DEC):
            rows.append({"panel": "f", "region": rname, "row_id": i,
                         "decile_median_width": float(medw_s[i]) if np.isfinite(medw_s[i]) else np.nan,
                         "onset_s": onset_s[i], "n_units": int(ncell_s[i])})
        for i in range(ng):
            rows.append({"panel": "g", "region": rname, "row_id": i,
                         "decile_median_width": np.nan,
                         "onset_s": onset_g_sorted[i], "n_units": 1})
        pd.DataFrame(rows).to_csv(outdir / f"fig5fg_{lick}_stats.csv", index=False)

        n_on_f = int(np.isfinite(onset).sum())
        n_on_g = int(np.isfinite(onset_cell).sum())
        print(f"[{rname}] f: {int(f_sel.sum())} resp cells, {n_on_f}/{N_DEC} deciles with finite onset; "
              f"g: {ng} non-TF cells, {n_on_g} with finite onset", flush=True)

    print(f"wrote {FIGROOT}/{{pooled,DMS,VMS}}/fig5fg_{lick}.{{png,pdf}} (+_stats.csv)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lick", choices=["hit", "fa"], default="hit")
    a = ap.parse_args()
    main(lick=a.lick)
