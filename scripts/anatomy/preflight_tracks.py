# scripts/anatomy/preflight_tracks.py
"""Pure-anatomy pre-flight for a subject's traced probe tracks (no sort needed).

Reads the brainglobe-segmentation atlas_space tracks (track_*.npy + .csv) and
reports, per shank, everything you can know from the histology alone — BEFORE
any Kilosort/UnitMatch output exists:

  * npy->(AP,ML,DV) transform check vs the brainglobe .csv region labels
  * hemisphere (Allen 'asr' midline ML=5700: >5700 = LEFT)
  * tip AP/ML/DV, track arc-length, and tip-ML ordering (medial<->lateral)
  * coarse regions traversed along each track + the tip region
  * (with --insertion-depth-um) the implied per-shank tip_y_um and the CCF-DV
    window a recording bank would occupy for a given probe-y span — so you can
    sanity-check "the bank was in cortex" without the chanmap.

This is the human-tracing -> localization gate; it does NOT need channel_positions
or pkls. Run with the project venv:

    py scripts/anatomy/preflight_tracks.py --subject BG_012 --insertion-depth-um 4590
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visdetect.anatomy.atlas import AllenAtlas
from import_brainglobe_tracks import brainglobe_npy_to_polyline, _track_length_um

MIDLINE_ML_UM = 5700.0   # Allen asr: ML<5700 = RIGHT, >5700 = LEFT


def preflight(subject, tracks_dir=None, insertion_depth_um=None, bank_span_um=720.0,
              atlas: AllenAtlas | None = None) -> pd.DataFrame:
    tracks_dir = tracks_dir or os.path.join("data", "anatomy", subject,
                                            "segmentation", "atlas_space", "tracks")
    atlas = atlas or AllenAtlas()
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(tracks_dir) if f.endswith(".npy"))
    rows = []
    for stem in stems:
        poly = brainglobe_npy_to_polyline(os.path.join(tracks_dir, f"{stem}.npy"))  # (AP,ML,DV) deepest-first
        csv_path = os.path.join(tracks_dir, f"{stem}.csv")
        match = ""
        if os.path.exists(csv_path):
            csv_acrs = set(pd.read_csv(csv_path)["Region acronym"].astype(str))
            idxs = np.linspace(0, len(poly) - 1, min(25, len(poly))).astype(int)
            hits = sum(atlas.region_at(poly[i])["acronym"] in csv_acrs for i in idxs)
            match = f"{hits}/{len(idxs)}"
        idxs = np.linspace(0, len(poly) - 1, min(25, len(poly))).astype(int)
        coarse = pd.Series([atlas.region_at(p)["coarse"] for p in poly[idxs]]
                           ).value_counts(normalize=True).round(2).to_dict()
        tip, surf = poly[0], poly[-1]
        length = _track_length_um(poly)
        row = dict(stem=stem, n=len(poly), tip_ap=round(tip[0]), tip_ml=round(tip[1]),
                   tip_dv=round(tip[2]), surf_dv=round(surf[2]), length=round(length),
                   hemi="LEFT" if tip[1] > MIDLINE_ML_UM else "RIGHT", match=match,
                   tip_region=atlas.region_at(tip)["acronym"],
                   tip_coarse=atlas.region_at(tip)["coarse"], along=coarse)
        if insertion_depth_um is not None:
            # tip_y_um maps the deepest electrode (probe y=0) onto the track; a bank
            # spanning probe-y [a,b] sits at track depth (along-arc) below the deepest
            # traced point by (a-tip_y)..(b-tip_y). We report the CCF-DV window for a
            # bank at the TOP of the probe (cortical) vs near the tip, as a guide.
            row["tip_y_um"] = round(insertion_depth_um - length)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--tracks-dir", default=None)
    ap.add_argument("--insertion-depth-um", type=float, default=None)
    a = ap.parse_args()
    df = preflight(a.subject, a.tracks_dir, a.insertion_depth_um)
    pd.set_option("display.width", 240, "display.max_columns", 40)
    cols = [c for c in ["stem", "n", "tip_ap", "tip_ml", "tip_dv", "surf_dv", "length",
                        "tip_y_um", "hemi", "match", "tip_region", "tip_coarse"] if c in df.columns]
    print(df[cols].to_string(index=False))
    print("\ncoarse regions along each track (sampled tip->surface):")
    for _, r in df.iterrows():
        print(f"  {r['stem']}: {r['along']}")
    mls = df["tip_ml"].tolist()
    steps = np.diff(mls)
    print(f"\ntip-ML order {mls}  steps {[int(s) for s in steps]}  "
          + ("monotonic" if (np.all(steps > 0) or np.all(steps < 0)) else "NON-monotonic"))
    print("hemisphere(s):", sorted(df["hemi"].unique()))


if __name__ == "__main__":
    main()
