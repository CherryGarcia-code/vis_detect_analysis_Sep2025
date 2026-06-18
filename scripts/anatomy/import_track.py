# scripts/anatomy/import_track.py
"""Adapt a tracing export (brainglobe-segmentation / Pinpoint, re-exported to our
CSV+JSON contract) into a validated track artifact. See docs/anatomy/registration_recipe.md."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from visdetect.anatomy.tracks import (
    ShankTrack, TrackArtifact, save_track_artifact, validate_track_artifact,
)
from visdetect.anatomy.orientation import assign_probe_shank_indices, validate_shank_order


def _vec(x):
    return None if x is None else np.asarray(x, float)


def import_track(points_csv, meta_json, out_json) -> TrackArtifact:
    pts = pd.read_csv(points_csv)
    meta = json.loads(Path(meta_json).read_text(encoding="utf-8"))
    have_idx = pts["probe_shank_index"].notna().all() if "probe_shank_index" in pts else False

    # group points into shank polylines (by index if present, else by an order key)
    group_col = "probe_shank_index" if have_idx else "shank_group"
    if group_col not in pts:
        raise ValueError("points need either probe_shank_index or shank_group")

    shanks = []
    for g, d in pts.sort_values([group_col, "point_order"]).groupby(group_col):
        poly = d[["ap_um", "ml_um", "dv_um"]].to_numpy(float)
        m = meta["shanks"][str(int(g))]
        shanks.append(ShankTrack(
            probe_shank_index=int(g) if have_idx else -1,
            ccf_polyline=poly, tip_y_um=float(m["tip_y_um"]), method=m["method"],
            sigma_along_um=float(m["sigma_along_um"]),
            sigma_across_um=float(m["sigma_across_um"]),
            sigma_growth_k=float(m["sigma_growth_k"]),
            planned_entry=_vec(m.get("planned_entry")),
            planned_vector=_vec(m.get("planned_vector")),
        ))

    if not have_idx:
        shanks = assign_probe_shank_indices(
            shanks, meta["barcode_orientation"], meta["hemisphere"])

    art = TrackArtifact(
        subject=meta["subject"], atlas=meta["atlas"], hemisphere=meta["hemisphere"],
        barcode_orientation=meta["barcode_orientation"], source_tool=meta["source_tool"],
        created=meta["created"], shanks=sorted(shanks, key=lambda s: s.probe_shank_index),
    )
    validate_track_artifact(art)
    validate_shank_order(art)
    save_track_artifact(art, out_json)
    return art


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import_track(a.points, a.meta, a.out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
