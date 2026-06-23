# scripts/anatomy/import_brainglobe_tracks.py
"""Adapt brainglobe-segmentation atlas-space track output -> our track artifact.

brainglobe-segmentation writes, per shank, an atlas_space spline ``<name>.npy`` of
shape (N, 3) in **microns**, axis order **(AP, DV, ML)** (allen_mouse 'asr' frame),
plus a ``<name>.csv`` listing the Allen region at each spline point. This adapter
converts those splines into a validated `TrackArtifact` consumed by
build_channel_atlas.py.

Coordinate transform (verified against brainglobe's own region labels at 100%):
    npy (AP, DV, ML) um  ->  our (AP, ML, DV) um   [permutation (0, 2, 1), scale 1]
Polylines are reordered **deepest-first** (index 0 = largest DV = closest to the tip).

probe_shank_index is assigned by the explicit ``shank_order`` list (index 0 = first
entry), NOT by the orientation/barcode convention — the caller supplies the
medial->lateral (or whatever) order that matches the probe's electrode geometry
(channel shank 0 = smallest x). validate_shank_order still enforces monotonic tip-ML
and ~250 um spacing.

Usage:
    py scripts/anatomy/import_brainglobe_tracks.py --subject BG_046 \
        --tracks-dir data/anatomy/BG_046/segmentation/atlas_space/tracks \
        --hemisphere left \
        --shank-order shank1_med shank2_fit shank3 shank4
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional

import numpy as np

from visdetect.anatomy.tracks import (
    ShankTrack, TrackArtifact, save_track_artifact, validate_track_artifact,
)
from visdetect.anatomy.orientation import validate_shank_order


def brainglobe_npy_to_polyline(npy_path) -> np.ndarray:
    """Load a brainglobe atlas_space spline .npy and return our (AP, ML, DV) um
    polyline, ordered deepest-first (row 0 = largest DV)."""
    arr = np.asarray(np.load(npy_path), float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{npy_path}: expected (N,3) array, got {arr.shape}")
    poly = np.stack([arr[:, 0], arr[:, 2], arr[:, 1]], axis=1)  # (AP,DV,ML)->(AP,ML,DV)
    if poly[0, 2] < poly[-1, 2]:           # ensure deepest (largest DV) first
        poly = poly[::-1].copy()
    return poly


def _track_length_um(poly: np.ndarray) -> float:
    """Arc length of a polyline (tip -> surface) in um."""
    return float(np.linalg.norm(np.diff(np.asarray(poly, float), axis=0), axis=1).sum())


def import_brainglobe_tracks(tracks_dir, subject: str, hemisphere: str,
                             shank_order: List[str], out_json,
                             *, tip_y_um: float = 0.0,
                             insertion_depth_um: Optional[float] = None,
                             sigma_um: float = 50.0,
                             sigma_growth_k: float = 0.0, method: str = "brainreg_traced",
                             atlas: str = "allen_mouse_25um",
                             barcode_orientation: str = "forward",
                             source_tool: str = "brainglobe-segmentation",
                             created: str = "2026-06-23") -> TrackArtifact:
    """Build a TrackArtifact from brainglobe-segmentation .npy tracks.

    Depth calibration (the channel y_um that maps to the deepest traced point):
    - ``insertion_depth_um`` set → per-shank ``tip_y_um = insertion_depth_um -
      track_length`` (anchors the traced cortical surface to channel y =
      insertion_depth_um). This is the recommended knob — recalibrate later by just
      re-running with a different depth.
    - else → the scalar ``tip_y_um`` for every shank (default 0 = deepest dye == tip).
    """
    tracks_dir = Path(tracks_dir)
    shanks = []
    for idx, stem in enumerate(shank_order):
        poly = brainglobe_npy_to_polyline(tracks_dir / f"{stem}.npy")
        this_tip_y = (insertion_depth_um - _track_length_um(poly)
                      if insertion_depth_um is not None else tip_y_um)
        shanks.append(ShankTrack(
            probe_shank_index=idx, ccf_polyline=poly, tip_y_um=float(this_tip_y),
            method=method, sigma_along_um=float(sigma_um), sigma_across_um=float(sigma_um),
            sigma_growth_k=float(sigma_growth_k), planned_entry=None, planned_vector=None,
        ))
    art = TrackArtifact(
        subject=subject, atlas=atlas, hemisphere=hemisphere,
        barcode_orientation=barcode_orientation, source_tool=source_tool,
        created=created, shanks=shanks,
    )
    validate_track_artifact(art)
    validate_shank_order(art)
    save_track_artifact(art, out_json)
    return art


def _auto_shank_order(tracks_dir, lateral_first: bool = False) -> List[str]:
    """Discover *.npy tracks and order them medial->lateral (default) by tip ML.

    Note: medial/lateral here is by absolute ML; the caller must still confirm that
    the medial track maps to probe electrode-shank 0. Pass an explicit --shank-order
    when in doubt.
    """
    stems = [os.path.splitext(f)[0] for f in sorted(os.listdir(tracks_dir))
             if f.endswith(".npy")]
    tip_ml = {}
    for s in stems:
        poly = brainglobe_npy_to_polyline(os.path.join(tracks_dir, f"{s}.npy"))
        tip_ml[s] = float(poly[0, 1])
    ordered = sorted(stems, key=lambda s: tip_ml[s], reverse=lateral_first)
    return ordered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--tracks-dir", required=True)
    ap.add_argument("--hemisphere", required=True, choices=["left", "right"])
    ap.add_argument("--shank-order", nargs="+", default=None,
                    help="npy stems for probe_shank_index 0..N-1 (electrode shank 0 first). "
                         "If omitted, auto-order medial->lateral by tip ML.")
    ap.add_argument("--lateral-first", action="store_true",
                    help="with auto-order, put the lateral track at index 0")
    ap.add_argument("--tip-y-um", type=float, default=0.0,
                    help="channel y at the deepest traced point (default 0 = deepest dye == tip). "
                         "Ignored if --insertion-depth-um is given.")
    ap.add_argument("--insertion-depth-um", type=float, default=None,
                    help="tip depth below the cortical surface (um). Sets per-shank tip_y_um = "
                         "depth - track_length, anchoring the traced surface to channel y=depth. "
                         "The recommended depth knob; re-run with a new value to recalibrate.")
    ap.add_argument("--sigma-um", type=float, default=50.0)
    ap.add_argument("--out", default=None,
                    help="defaults to data/anatomy/<subject>_shank_tracks.json")
    args = ap.parse_args()

    order = args.shank_order or _auto_shank_order(args.tracks_dir, args.lateral_first)
    out = args.out or os.path.join("data", "anatomy", f"{args.subject}_shank_tracks.json")
    art = import_brainglobe_tracks(
        args.tracks_dir, args.subject, args.hemisphere, order, out,
        tip_y_um=args.tip_y_um, insertion_depth_um=args.insertion_depth_um, sigma_um=args.sigma_um,
    )
    print(f"{args.subject}: wrote {out}")
    for s in art.shanks:
        print(f"  shank {s.probe_shank_index}: tip ML={s.ccf_polyline[0,1]:.0f}um, "
              f"DV {s.ccf_polyline[0,2]:.0f}->{s.ccf_polyline[-1,2]:.0f}um, tip_y_um={s.tip_y_um:.0f}")


if __name__ == "__main__":
    main()
