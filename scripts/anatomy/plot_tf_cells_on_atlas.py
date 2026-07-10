# scripts/anatomy/plot_tf_cells_on_atlas.py
"""Population map of TF-responsive / kernel-width (transient vs sustained) cells on a
coronal Allen slice.

The probe is fixed across a subject's sessions, so every localized unit shares one
geometry — we pool ALL of them onto a single coronal slice and mark the TF cells.
Two external label sources are joined to data/anatomy/<subj>/unit_anatomy.csv on
(canonical session, cluster_id):
  - TF-responsive flag  `resp_log2`     from data/cache/tf_responsive/<subj>_tf_responsive.csv
  - continuous FWHM     `interp_fwhm`(s) from data/cache/tf_glm_bg046/kernel_width_continuous.csv

Metrics:
  tf_responsive : all units faint grey + TF-responsive units highlighted (location map).
  kernel_width  : TF-responsive cells coloured by continuous FWHM, transient(narrow) <->
                  sustained(wide), diverging at the subject's median.
  width_class   : TF-responsive cells split transient/sustained at the median FWHM.
                  NB the project's own modality analysis says this axis is a floor-
                  dominated SPECTRUM, not two clean classes — the split is a derived cut.

Coverage: BG_046, BG_039, BG_031 (the striatal TF-GLM mice). BG_038 (cortex) has no TF data.

Usage:
  py scripts/anatomy/plot_tf_cells_on_atlas.py --subject BG_031 \
     --metric tf_responsive kernel_width width_class [--coords stereotaxic]
"""
from __future__ import annotations

import argparse
import os
import re
import sys

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
from visdetect.analysis.config import canonical_session_id as _canon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_sites_on_atlas import coronal_coarse_image, dominant_region_label

TF_DIR = os.path.join("data", "cache", "tf_responsive")
KW_CSV = os.path.join("data", "cache", "tf_glm_bg046", "kernel_width_continuous.csv")

TF_COLOR = "#6a51a3"     # TF-responsive highlight (purple; distinct from CP-blue / cortex-green fills)
BG_COLOR = "#d9d9d9"     # non-TF background units
TRANSIENT_COLOR = "#4575b4"   # narrow kernel
SUSTAINED_COLOR = "#d73027"   # wide kernel

METRIC_INFO = {
    "tf_responsive": dict(label="TF-responsive cells", kind="tf"),
    "kernel_width": dict(label="TF-kernel width (s)  transient ←→ sustained", kind="width"),
    "width_class": dict(label="transient vs sustained (median split)", kind="class"),
}


def _sess_key_from_registry(session_date) -> str:
    """Canonical 8-digit key from a TF-registry session_date ('01042025_v2' -> '01042025')."""
    return _canon(re.sub(r"_.*$", "", str(session_date)))


def _sess_key_from_prefixed(session) -> str:
    """Canonical key from a subject-prefixed session ('BG_046_01072025' -> '01072025')."""
    return _canon(re.sub(r"^BG_\d+_", "", str(session)))


def load_tf_labels(subject, anatomy_dir=None, tf_dir=TF_DIR, kw_csv=KW_CSV) -> pd.DataFrame:
    """unit_anatomy for `subject` left-joined with TF-responsive flag + continuous FWHM.

    Returns df with the anatomy columns plus `resp_log2` (bool, NaN if not in registry)
    and `interp_fwhm` (s, NaN if no kernel width). Keyed per-session (session, cluster_id)."""
    anatomy_dir = anatomy_dir or os.path.join("data", "anatomy", subject)
    anat = pd.read_csv(os.path.join(anatomy_dir, "unit_anatomy.csv"))
    anat["skey"] = anat["session_name"].map(_canon)
    anat["unit"] = anat["cluster_id"].astype(int)

    reg_path = os.path.join(tf_dir, f"{subject.lower().replace('_', '')}_tf_responsive.csv")
    if os.path.exists(reg_path):
        reg = pd.read_csv(reg_path, dtype={"session": str, "session_date": str})
        reg["skey"] = reg["session_date"].map(_sess_key_from_registry)
        reg["unit"] = reg["unit"].astype(int)
        reg = reg.drop_duplicates(["skey", "unit"])
        anat = anat.merge(reg[["skey", "unit", "resp_log2"]], on=["skey", "unit"], how="left")
    else:
        anat["resp_log2"] = np.nan

    if os.path.exists(kw_csv):
        kw = pd.read_csv(kw_csv)
        kw = kw[kw["subject"] == subject].copy()
        if len(kw):
            kw["skey"] = [_sess_key_from_prefixed(s) for s in kw["session"]]
            kw["unit"] = kw["unit"].astype(int)
            kw = kw.drop_duplicates(["skey", "unit"])
            anat = anat.merge(kw[["skey", "unit", "interp_fwhm"]], on=["skey", "unit"], how="left")
    if "interp_fwhm" not in anat.columns:
        anat["interp_fwhm"] = np.nan
    return anat


def _jitter(vals, sd, rng):
    return np.asarray(vals, float) + rng.normal(0.0, sd, size=len(vals))


def plot_tf_cells(subject, df, art, metric, out_png, atlas=None, *, coords="ccf",
                  jitter_um=18.0, seed=0) -> str:
    """2-panel (whole section + zoom) population map for one metric. `df` = load_tf_labels output."""
    atlas = atlas or AllenAtlas()
    info = METRIC_INFO[metric]
    cm = CoordMap(coords, pia_dv_um(art))
    rng = np.random.default_rng(seed)

    tf = df[df["resp_log2"] == True].copy()                       # noqa: E712
    if info["kind"] in ("width", "class"):
        tf = tf[np.isfinite(pd.to_numeric(tf["interp_fwhm"], errors="coerce"))].copy()
    focus = tf if len(tf) else df                                 # units that set the zoom window
    if len(focus) == 0 or (info["kind"] in ("width", "class") and len(tf) == 0):
        return None                                               # width/class need TF cells with a kernel
    ap_um = float(focus["ccf_ap"].median())

    xs = np.sort(cm.x(np.array([focus.ccf_ml.min() - 700, focus.ccf_ml.max() + 700])))
    ys = np.sort(cm.y(np.array([focus.ccf_dv.min() - 1500, focus.ccf_dv.max() + 500])))
    xlo, xhi = float(xs[0]), float(xs[1]); ylo, yhi = float(ys[0]), float(ys[1])

    # colour spec
    med = float(pd.to_numeric(tf["interp_fwhm"], errors="coerce").median()) if len(tf) else np.nan
    norm = cmap = None
    if info["kind"] == "width":
        vmax = float(np.nanmax(np.abs(pd.to_numeric(tf["interp_fwhm"]) - med))) or 0.05
        norm = mcolors.TwoSlopeNorm(vcenter=med, vmin=med - vmax, vmax=med + vmax)
        cmap = plt.get_cmap("coolwarm")

    def _draw(ax, zoom):
        img, extent = coronal_coarse_image(atlas, ap_um)
        img, extent = cm.image(img, extent)
        ax.imshow(img, extent=extent, origin="upper", interpolation="nearest", aspect="equal")
        for s in art.shanks:
            poly = np.asarray(s.ccf_polyline, float)
            ax.plot(cm.x(poly[:, 1]), cm.y(poly[:, 2]), "-", lw=0.9, color="0.4", alpha=0.7, zorder=3)
        js = jitter_um if zoom else jitter_um * 0.6
        def _xy(d):
            return cm.x(_jitter(d.ccf_ml, js, rng)), cm.y(_jitter(d.ccf_dv, js, rng))
        s_bg, s_fg = (10, 26) if zoom else (3, 9)
        if info["kind"] == "tf":
            ng = df[df["resp_log2"] != True]                      # noqa: E712
            xg, yg = _xy(ng); ax.scatter(xg, yg, s=s_bg, c=BG_COLOR, alpha=0.35, lw=0, zorder=4)
            xt, yt = _xy(tf); ax.scatter(xt, yt, s=s_fg, c=TF_COLOR, alpha=0.9,
                                         edgecolors="white", linewidths=0.3, zorder=6)
        elif info["kind"] == "width":
            xt, yt = _xy(tf)
            ax.scatter(xt, yt, s=s_fg, c=pd.to_numeric(tf["interp_fwhm"]), cmap=cmap, norm=norm,
                       alpha=0.95, edgecolors="white", linewidths=0.3, zorder=6)
        else:  # class
            for lab, col in (("transient", TRANSIENT_COLOR), ("sustained", SUSTAINED_COLOR)):
                sel = tf[(tf["interp_fwhm"] <= med) == (lab == "transient")]
                xt, yt = _xy(sel); ax.scatter(xt, yt, s=s_fg, c=col, alpha=0.9,
                                              edgecolors="white", linewidths=0.3, zorder=6, label=lab)
        ax.set_xlabel(cm.xlabel); ax.set_ylabel(cm.ylabel)
        if zoom:
            ax.set_xlim(xlo, xhi); ax.set_ylim(yhi, ylo)

    fig = plt.figure(figsize=(13, 6.2), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0])
    axA, axB = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    _draw(axA, False); _draw(axB, True)
    axA.add_patch(Rectangle((xlo, ylo), xhi - xlo, yhi - ylo, fill=False, ec="k", lw=1.0, ls="--", zorder=7))

    n_tf = len(tf)
    reg = dominant_region_label(df[df["resp_log2"] == True]) if n_tf else dominant_region_label(df)  # noqa: E712
    if info["kind"] == "tf":
        n_lab = f"n = {n_tf} TF-responsive / {len(df)} units"
    elif info["kind"] == "width":
        n_lab = f"n = {n_tf} TF cells with kernel width · median FWHM {med:.3f}s"
    else:
        n_lab = (f"n = {int((tf['interp_fwhm'] <= med).sum())} transient / "
                 f"{int((tf['interp_fwhm'] > med).sum())} sustained (split @ {med:.3f}s)")
    axA.set_title(f"A. Coronal section ({cm.ap_title(ap_um)})", fontweight="bold", fontsize=12)
    axA.text(0.02, 0.02, n_lab, transform=axA.transAxes, fontsize=8, color="0.3", va="bottom")
    reg_zoom = reg.split("/")[0]; reg_zoom = reg_zoom[:1].upper() + reg_zoom[1:]
    axB.set_title(f"B. {reg_zoom} zoom", fontweight="bold", fontsize=12)
    x0 = xlo + 0.08 * (xhi - xlo); y0 = yhi - 0.08 * (yhi - ylo)
    axB.plot([x0, x0 + cm.length(500)], [y0, y0], "k-", lw=2.5)
    axB.text(x0 + cm.length(250), y0 - 0.02 * (yhi - ylo), "500 µm", ha="center", va="bottom", fontsize=8)

    if info["kind"] == "tf":
        handles = [Line2D([0], [0], marker="o", ls="", mec="white", mew=0.3, mfc=TF_COLOR, label="TF-responsive"),
                   Line2D([0], [0], marker="o", ls="", mfc=BG_COLOR, mec="none", label="other units")]
        axB.legend(handles=handles, loc="upper right", frameon=False, fontsize=8)
    elif info["kind"] == "width":
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cb = fig.colorbar(sm, ax=axB, fraction=0.046, pad=0.04)
        cb.set_label("FWHM (s)   transient ←→ sustained", fontsize=9)
    else:
        handles = [Line2D([0], [0], marker="o", ls="", mec="white", mew=0.3, mfc=c, label=l)
                   for l, c in (("transient", TRANSIENT_COLOR), ("sustained", SUSTAINED_COLOR))]
        axB.legend(handles=handles, title="kernel width", loc="upper right", frameon=False,
                   fontsize=8, title_fontsize=8)

    fig.suptitle(f"{subject} — {info['label']}  ·  {art.hemisphere} {reg}  (all sessions pooled)",
                 fontsize=13, fontweight="bold")
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--metric", nargs="+", default=["tf_responsive", "kernel_width", "width_class"],
                    choices=list(METRIC_INFO))
    ap.add_argument("--coords", choices=["ccf", "stereotaxic"], default="ccf")
    ap.add_argument("--anatomy-dir", default=None)
    args = ap.parse_args()
    anatomy_dir = args.anatomy_dir or os.path.join("data", "anatomy", args.subject)
    df = load_tf_labels(args.subject, anatomy_dir)
    art = load_track_artifact(os.path.join(anatomy_dir, f"{args.subject}_shank_tracks.json"))
    suffix = "_stereotaxic" if args.coords == "stereotaxic" else ""
    for m in args.metric:
        out = os.path.join("FIGURES", "anatomy", args.subject, f"{args.subject}_{m}{suffix}.png")
        res = plot_tf_cells(args.subject, df, art, m, out, coords=args.coords)
        print(f"wrote {res}" if res else f"{m}: no TF cells to plot for {args.subject}")


if __name__ == "__main__":
    main()
