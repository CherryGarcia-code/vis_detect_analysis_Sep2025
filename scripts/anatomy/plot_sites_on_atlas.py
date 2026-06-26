# scripts/anatomy/plot_sites_on_atlas.py
"""Coronal Allen-atlas schematic with localized recording sites overlaid.

Presentation-ready "where did we record" figure: a coronal slice at the probe's AP
(soft region fills + boundaries), the per-shank tracks (cortex -> striatum), and the
recording sites colored by shank (default) or by an arbitrary per-site value (hook for
future unit-activity overlays). Two panels: whole slice (context) + striatum zoom.

Reusable for any subject with a built channel atlas + track artifact.
"""
from __future__ import annotations

import argparse
import os
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from visdetect.anatomy.atlas import AllenAtlas
from visdetect.anatomy.tracks import load_track_artifact
from visdetect.anatomy.stereotaxic import CoordMap, pia_dv_um

# Soft schematic fills per coarse region (light so overlaid sites/tracks pop).
COARSE_FILL = {
    "CP": "#c6dbef", "CTX": "#d9f0d3", "WM": "#e8e8e8", "VS": "#ffffff",
    "GPe": "#fdd0a2", "other": "#f4f4f4", "out": "#ffffff", "unknown": "#f7f7f7",
}
# Readable names for figure titles, keyed by coarse class.
COARSE_NAME = {
    "CP": "CPu", "CTX": "cortex", "WM": "white matter", "VS": "ventral striatum",
    "GPe": "GPe", "other": "mixed", "out": "—", "unknown": "—",
}
_OUTLINE = (0.55, 0.55, 0.55)


def dominant_region_label(ch) -> str:
    """Readable label for where the bank sits: the dominant coarse region, plus a
    secondary if the probe straddles regions (e.g. 'cortex/WM'). Used in titles so
    a cortical probe isn't mislabelled 'CPu'."""
    if "region_coarse" not in ch.columns or ch.empty:
        return "—"
    frac = ch["region_coarse"].value_counts(normalize=True)
    parts = [COARSE_NAME.get(frac.index[0], frac.index[0])]
    if len(frac) > 1 and frac.iloc[1] >= 0.2:
        parts.append(COARSE_NAME.get(frac.index[1], frac.index[1]))
    return "/".join(parts)


def coronal_coarse_image(atlas: AllenAtlas, ap_um: float):
    """Return (rgb image [DV, ML, 3], extent [ml0,ml1,dv1,dv0]) for a coronal slice,
    coloured by coarse region with thin region boundaries."""
    res = atlas.resolution_um
    ap_idx = int(round(ap_um / res))
    ap_idx = max(0, min(ap_idx, atlas.annotation.shape[0] - 1))
    sl = np.asarray(atlas.annotation[ap_idx])              # (ML, DV)
    img = np.ones(sl.shape + (3,), float)
    for uid in np.unique(sl):
        coarse = atlas.id_to_coarse.get(int(uid), "out" if uid == 0 else "other")
        img[sl == uid] = mcolors.to_rgb(COARSE_FILL.get(coarse, "#ffffff"))
    # region boundaries
    edge = np.zeros(sl.shape, bool)
    edge[1:, :] |= sl[1:, :] != sl[:-1, :]
    edge[:, 1:] |= sl[:, 1:] != sl[:, :-1]
    img[edge] = _OUTLINE
    ml_ext = sl.shape[0] * res
    dv_ext = sl.shape[1] * res
    return img.transpose(1, 0, 2), [0.0, ml_ext, dv_ext, 0.0]


def _draw_slice(ax, atlas, ap_um, cm):
    img, extent = coronal_coarse_image(atlas, ap_um)
    img, extent = cm.image(img, extent)
    ax.imshow(img, extent=extent, origin="upper", interpolation="nearest", aspect="equal")
    ax.set_xlabel(cm.xlabel); ax.set_ylabel(cm.ylabel)
    return extent


def plot_sites_on_atlas(subject: str, atlas_csv, tracks_json, out_png,
                        atlas: Optional[AllenAtlas] = None,
                        color_by: str = "shank", values: Optional[Sequence[float]] = None,
                        cmap: str = "viridis", value_label: str = "value",
                        coords: str = "ccf") -> str:
    """Two-panel coronal schematic (context + striatum zoom) with sites overlaid.

    color_by="shank" (default) colours dots by probe shank; pass `values` (one per
    atlas row) to colour by a continuous quantity (e.g. firing rate) with a colorbar.
    coords="stereotaxic" relabels/flips the axes to Bregma-referenced mm (ML from
    midline with anatomical left on the left; DV as depth below the brain surface).
    """
    atlas = atlas or AllenAtlas()
    ch = pd.read_csv(atlas_csv)
    art = load_track_artifact(tracks_json)
    ap_um = float(ch["ccf_ap"].median())
    cm = CoordMap(coords, pia_dv_um(art))

    shanks = sorted(ch["shank"].unique())
    shank_colors = {s: plt.cm.viridis(t) for s, t in zip(shanks, np.linspace(0.05, 0.9, len(shanks)))}

    fig = plt.figure(figsize=(13, 6.2), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0])
    axA, axB = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    # zoom window around the sites, computed in plot coordinates (handles the ML flip)
    xs = np.sort(cm.x(np.array([ch.ccf_ml.min() - 700, ch.ccf_ml.max() + 700])))
    ys = np.sort(cm.y(np.array([ch.ccf_dv.min() - 1500, ch.ccf_dv.max() + 500])))  # incl. cortex
    xlo, xhi = float(xs[0]), float(xs[1]); ylo, yhi = float(ys[0]), float(ys[1])

    # colour spec for sites
    norm = sm = None
    if values is not None:
        values = np.asarray(values, float)
        norm = mcolors.Normalize(np.nanmin(values), np.nanmax(values))
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    for ax, zoom in ((axA, False), (axB, True)):
        _draw_slice(ax, atlas, ap_um, cm)
        # tracks (cortex -> CP)
        for s in art.shanks:
            poly = np.asarray(s.ccf_polyline, float)
            ax.plot(cm.x(poly[:, 1]), cm.y(poly[:, 2]), "-", lw=1.0, color="0.35", alpha=0.8, zorder=4)
        # sites (larger in the zoom panel)
        sc_kw = dict(s=34 if zoom else 11, edgecolors="white",
                     linewidths=0.4 if zoom else 0.2, zorder=5)
        if values is not None:
            ax.scatter(cm.x(ch.ccf_ml), cm.y(ch.ccf_dv), c=values, cmap=cmap, norm=norm, **sc_kw)
        else:
            ax.scatter(cm.x(ch.ccf_ml), cm.y(ch.ccf_dv),
                       c=[shank_colors[s] for s in ch["shank"]], **sc_kw)
        if zoom:
            ax.set_xlim(xlo, xhi); ax.set_ylim(yhi, ylo)
            ax.set_title("B. Striatum zoom — recording sites", fontweight="bold", fontsize=12)
            # scale bar (500 um -> 0.5 mm in stereotaxic)
            x0 = xlo + 0.08 * (xhi - xlo); y0 = yhi - 0.08 * (yhi - ylo)
            ax.plot([x0, x0 + cm.length(500)], [y0, y0], "k-", lw=2.5)
            ax.text(x0 + cm.length(250), y0 - 0.02 * (yhi - ylo), "500 µm",
                    ha="center", va="bottom", fontsize=8)
        else:
            ax.add_patch(Rectangle((xlo, ylo), xhi - xlo, yhi - ylo, fill=False,
                                   ec="k", lw=1.0, ls="--", zorder=6))
            ax.set_title(f"A. Coronal section ({cm.ap_title(ap_um)})", fontweight="bold", fontsize=12)
            ax.text(0.02, 0.02, f"n = {len(ch)} sites", transform=ax.transAxes,
                    fontsize=8, color="0.3", va="bottom")

    # legend / colorbar
    if values is not None:
        cb = fig.colorbar(sm, ax=axB, fraction=0.046, pad=0.04)
        cb.set_label(value_label, fontsize=9)
    else:
        handles = [Line2D([0], [0], marker="o", ls="", mec="white", mew=0.3,
                          mfc=shank_colors[s], label=f"shank {s}") for s in shanks]
        axB.legend(handles=handles, title="probe shank", loc="upper right",
                   frameon=False, fontsize=8, title_fontsize=8)

    hemi = art.hemisphere
    fig.suptitle(f"{subject} — recording site localization ({hemi} {dominant_region_label(ch)})",
                 fontsize=13, fontweight="bold")
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--anatomy-dir", default=None, help="defaults to data/anatomy/<subject>")
    ap.add_argument("--coords", choices=["ccf", "stereotaxic"], default="ccf",
                    help="ccf (raw Allen microns; default) or stereotaxic "
                         "(Bregma-referenced mm, ML flipped so anatomical left is on the left)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.anatomy_dir = args.anatomy_dir or os.path.join("data", "anatomy", args.subject)
    atlas_csv = os.path.join(args.anatomy_dir, f"{args.subject}_channel_atlas.csv")
    tracks = os.path.join(args.anatomy_dir, f"{args.subject}_shank_tracks.json")
    suffix = "_stereotaxic" if args.coords == "stereotaxic" else ""
    out = args.out or os.path.join("FIGURES", "anatomy", args.subject,
                                   f"{args.subject}_sites_on_atlas{suffix}.png")
    plot_sites_on_atlas(args.subject, atlas_csv, tracks, out, coords=args.coords)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
