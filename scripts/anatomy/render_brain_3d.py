# scripts/anatomy/render_brain_3d.py
"""3D whole-brain render (brainrender) of probe tracks + recording sites.

SELF-CONTAINED: reads a track-artifact JSON + a sites CSV (columns ccf_ap/ccf_ml/ccf_dv,
optionally `shank` or a numeric value column). No visdetect import, so run it with the
brainrender conda env, NOT the project .venv:

    /c/Users/Ben/anaconda3/envs/napari-env/python.exe scripts/anatomy/render_brain_3d.py \
        --tracks data/anatomy/BG_046/BG_046_shank_tracks.json \
        --sites  data/anatomy/BG_046/BG_046_channel_atlas.csv \
        --color-col shank --out FIGURES/anatomy/BG_046/BG_046_brain3d_shank.png

Coordinate note: our artifact/CSV store (AP, ML, DV) µm; brainrender/brainglobe meshes
expect (AP, DV, ML). Verified: a point at our CP tip lands inside the left-CP mesh.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from brainrender import Scene, settings
from brainrender.actors import Points

# medial -> lateral, matching the 2D slice figures (viridis 4-step)
VIRIDIS4 = ["#440154", "#31688e", "#35b779", "#fde725"]


def _to_br(ap, ml, dv):
    """(AP, ML, DV) µm  ->  brainrender (AP, DV, ML) µm."""
    return np.column_stack([np.asarray(ap, float), np.asarray(dv, float), np.asarray(ml, float)])


def render(tracks_json, sites_csv, out_png, *, color_col="shank", cmap="viridis",
           region="CP", hemisphere="left", radius=28, title=None,
           azimuth=150.0, elevation=12.0, zoom=2.0):
    settings.OFFSCREEN = True
    settings.SHOW_AXES = False
    settings.SHADER_STYLE = "plastic"   # no cartoon silhouette -> cleaner brain
    try:
        sc = Scene(atlas_name="allen_mouse_25um", title=title)
    except Exception as e:  # noqa
        raise RuntimeError(
            "brainrender/VTK could not create an OpenGL context. 3D rendering needs a "
            "machine with a real GL display (a remote-desktop/SSH/headless session "
            "usually has none). Run this on the LOCAL machine.") from e
    try:
        sc.root.alpha(0.06)             # faint whole-brain so the striatum/shanks stand out
    except Exception:
        pass
    sc.add_brain_region(region, alpha=0.22, color="lightblue", hemisphere=hemisphere)

    if tracks_json and os.path.exists(tracks_json):
        art = json.loads(open(tracks_json, encoding="utf-8").read())
        for sh in art["shanks"]:
            idx = int(sh.get("probe_shank_index", 0))
            poly = np.asarray(sh["ccf_polyline"], float)  # (AP, ML, DV)
            br = _to_br(poly[:, 0], poly[:, 1], poly[:, 2])
            # Render the track as a dense brainrender Points line (NOT a raw vedo.Tube):
            # brainrender applies a coordinate transform to its own actors via add()/
            # _prepare_actor that a raw vedo mesh does not get, which would offset the
            # track from the (brainrender Points) sites. Using Points for both guarantees
            # they share one frame. Per-shank colour so the bank reads as beads on its line.
            sc.add(Points(br[::4], radius=12, colors=VIRIDIS4[idx % len(VIRIDIS4)], alpha=0.6))

    df = pd.read_csv(sites_csv)
    coords = _to_br(df["ccf_ap"], df["ccf_ml"], df["ccf_dv"])

    if color_col == "shank" and "shank" in df.columns:
        shanks = sorted(df["shank"].unique())
        cmap_s = {s: VIRIDIS4[i % len(VIRIDIS4)] for i, s in enumerate(shanks)}
        for s in shanks:
            m = (df["shank"] == s).to_numpy()
            sc.add(Points(coords[m], radius=radius, colors=cmap_s[s], alpha=0.9))
    elif color_col in df.columns:
        v = pd.to_numeric(df[color_col], errors="coerce").to_numpy()
        m = np.isfinite(v)
        coords, v = coords[m], v[m]
        if v.min() < 0 < v.max():
            norm = mcolors.TwoSlopeNorm(0.0, -np.abs(v).max(), np.abs(v).max())
        else:
            norm = mcolors.Normalize(v.min(), v.max())
        mp = cm.get_cmap(cmap)
        cols = [mcolors.to_hex(mp(norm(x))) for x in v]
        sc.add(Points(coords, radius=radius, colors=cols))   # brainrender Points (same frame as tracks)
    else:
        sc.add(Points(coords, radius=radius, colors="red"))

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    # Avoid brainrender's named cameras ("frontal"/"three_quarters" hang offscreen):
    # render with the default camera, then rotate directly. azimuth=90 -> coronal-facing
    # (shows the 4 shanks spread in ML); elevation adds a three-quarter tilt.
    sc.render(interactive=False, zoom=zoom)
    cam = sc.plotter.camera
    cam.Azimuth(azimuth)
    cam.Elevation(elevation)
    sc.plotter.render()
    sc.screenshot(name=out_png, scale=2)
    sc.close()
    return out_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--sites", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--color-col", default="shank")
    ap.add_argument("--cmap", default="viridis")
    ap.add_argument("--region", default="CP")
    ap.add_argument("--hemisphere", default="left")
    ap.add_argument("--radius", type=float, default=32,
                    help="recording-site marker radius (µm); sits on the per-shank track line")
    ap.add_argument("--azimuth", type=float, default=150.0,
                    help="camera azimuth after render; ~150 = 3/4 view showing the 4-shank ML spread")
    ap.add_argument("--elevation", type=float, default=12.0, help="camera tilt for a 3/4 view")
    ap.add_argument("--zoom", type=float, default=2.0)
    a = ap.parse_args()
    out = render(a.tracks, a.sites, a.out, color_col=a.color_col, cmap=a.cmap,
                 region=a.region, hemisphere=a.hemisphere, radius=a.radius,
                 azimuth=a.azimuth, elevation=a.elevation, zoom=a.zoom)
    print("wrote", out)


if __name__ == "__main__":
    main()
