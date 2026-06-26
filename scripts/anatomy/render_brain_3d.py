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


def _build_scene(tracks_json, sites_csv, *, color_col="shank", cmap="viridis",
                 region="CP", hemisphere="left", radius=28, title=None,
                 shader="cartoon", wireframe=False, root_alpha=0.06, coarse_region=None):
    """Build a brainrender Scene with the brain, CP region, per-shank tracks and sites.
    Everything is a brainrender Points actor so tracks and sites share one coord frame."""
    settings.OFFSCREEN = True
    settings.SHOW_AXES = False
    # SHADER_STYLE: 'plastic'/'glossy'/'metallic'/'shiny' = smooth surface; 'cartoon' =
    # silhouette outlines; 'default' = vedo default. Set before Scene creation.
    settings.SHADER_STYLE = shader
    try:
        sc = Scene(atlas_name="allen_mouse_25um", title=title)
    except Exception as e:  # noqa
        raise RuntimeError(
            "brainrender/VTK could not create an OpenGL context. 3D rendering needs a "
            "machine with a real GL display (a remote-desktop/SSH/headless session "
            "usually has none). Run this on the LOCAL machine.") from e
    try:
        sc.root.alpha(root_alpha)       # faint whole-brain so the striatum/shanks stand out
        if wireframe:
            sc.root.mesh.wireframe(True); sc.root.mesh.lw(1); sc.root.alpha(0.25)
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
    if coarse_region and "region_coarse" in df.columns:   # e.g. cortical sites only (CTX)
        df = df[df["region_coarse"] == coarse_region].reset_index(drop=True)
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
    return sc


def render(tracks_json, sites_csv, out_png, *, azimuth=150.0, elevation=12.0, zoom=1.3,
           scale=2, **scene_kw):
    sc = _build_scene(tracks_json, sites_csv, **scene_kw)
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    # Avoid brainrender's named cameras ("frontal"/"three_quarters" hang offscreen):
    # render with default camera, rotate directly, then reset_camera() to FIT the whole
    # brain at that angle (prevents edge clipping); zoom is an optional post-fit factor
    # (>1 closer, <1 more margin).
    sc.render(interactive=False)
    cam = sc.plotter.camera
    cam.Azimuth(azimuth); cam.Elevation(elevation)
    sc.plotter.reset_camera()
    if zoom and zoom != 1.0:
        cam.Zoom(zoom)
    sc.plotter.render()
    sc.screenshot(name=out_png, scale=scale)
    sc.close()
    return out_png


def render_spin(tracks_json, sites_csv, out_gif, *, frames=160, fps=8, width=640,
                start_azimuth=150.0, elevation=12.0, zoom=1.3, **scene_kw):
    """Render a slow 360° azimuth rotation as an animated GIF. Speed = frames/fps seconds."""
    import tempfile
    import imageio.v2 as imageio
    from PIL import Image
    sc = _build_scene(tracks_json, sites_csv, **scene_kw)
    sc.render(interactive=False)
    cam = sc.plotter.camera
    # Fit to the WIDEST projection (sagittal) once, so no frame clips during the spin;
    # then keep the camera distance fixed and only rotate azimuth (consistent size).
    cam.Elevation(elevation); cam.Azimuth(0.0)
    sc.plotter.reset_camera()
    if zoom and zoom != 1.0:
        cam.Zoom(zoom)
    cam.Azimuth(start_azimuth)
    tmp = tempfile.mkdtemp()
    step = 360.0 / frames
    imgs = []
    for i in range(frames):
        sc.plotter.render()
        p = os.path.join(tmp, f"f{i:03d}.png")
        sc.screenshot(name=p, scale=1)
        im = Image.open(p).convert("RGB")
        if width and im.width > width:
            im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
        imgs.append(np.asarray(im))
        os.remove(p)
        cam.Azimuth(step)            # advance the rotation
    sc.close()
    os.rmdir(tmp)
    os.makedirs(os.path.dirname(out_gif) or ".", exist_ok=True)
    imageio.mimsave(out_gif, imgs, fps=fps, loop=0)
    return out_gif


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
    ap.add_argument("--zoom", type=float, default=1.3,
                    help="post-fit zoom factor (1.0 = whole brain fits/no clip; >1 closer)")
    ap.add_argument("--shader", default="cartoon",
                    choices=["plastic", "glossy", "metallic", "shiny", "cartoon", "default"],
                    help="brain-surface style; 'cartoon' (default) = silhouette outlines")
    ap.add_argument("--wireframe", action="store_true", help="render the whole-brain outline as a mesh/wireframe")
    ap.add_argument("--coarse-region", default=None,
                    help="plot only sites in this coarse region (e.g. CTX); pair with --region to "
                         "highlight the matching mesh (e.g. Isocortex)")
    ap.add_argument("--spin", action="store_true",
                    help="render a slow 360° rotation as an animated GIF (use a .gif --out)")
    ap.add_argument("--frames", type=int, default=160, help="number of frames for --spin (more = smoother)")
    ap.add_argument("--fps", type=int, default=8, help="frames/sec for --spin GIF (lower = slower; duration = frames/fps s)")
    a = ap.parse_args()
    scene_kw = dict(color_col=a.color_col, cmap=a.cmap, region=a.region,
                    hemisphere=a.hemisphere, radius=a.radius, shader=a.shader,
                    wireframe=a.wireframe, coarse_region=a.coarse_region)
    if a.spin:
        out = render_spin(a.tracks, a.sites, a.out, frames=a.frames, fps=a.fps,
                          start_azimuth=a.azimuth, elevation=a.elevation, zoom=a.zoom, **scene_kw)
    else:
        out = render(a.tracks, a.sites, a.out, azimuth=a.azimuth, elevation=a.elevation,
                     zoom=a.zoom, **scene_kw)
    print("wrote", out)


if __name__ == "__main__":
    main()
