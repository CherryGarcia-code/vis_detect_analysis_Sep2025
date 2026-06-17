# src/visdetect/anatomy/localize.py
"""Place channels on shank polylines -> CCF + region + confidence; build atlas."""
from __future__ import annotations

from math import erf, sqrt
from typing import Tuple

import numpy as np
import pandas as pd

from visdetect.anatomy.atlas import AllenAtlas
from visdetect.anatomy.channel_geometry import assign_shanks
from visdetect.anatomy.tracks import ShankTrack, TrackArtifact

ATLAS_COLUMNS = [
    "subject", "chanmap_signature", "channel", "shank", "x_um", "y_um",
    "ccf_ap", "ccf_ml", "ccf_dv", "sigma_um",
    "region_acronym", "region_name", "region_coarse", "region_confidence", "loc_method",
]


def place_channel_on_track(track: ShankTrack, y_um: float) -> Tuple[np.ndarray, float]:
    poly = np.asarray(track.ccf_polyline, float)
    seg = np.diff(poly, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])  # arc length from polyline[0]
    s = float(y_um - track.tip_y_um)
    total = cum[-1]
    if s <= 0:
        return poly[0].copy(), track.sigma_along_um
    if s <= total:
        j = int(np.searchsorted(cum, s) - 1)
        j = max(0, min(j, len(seg) - 1))
        frac = (s - cum[j]) / seg_len[j] if seg_len[j] > 0 else 0.0
        xyz = poly[j] + frac * seg[j]
        return xyz, track.sigma_along_um
    # extrapolate above the top of the traced polyline
    overshoot = s - total
    if track.planned_vector is not None:
        direction = np.asarray(track.planned_vector, float)
    else:
        direction = seg[-1]
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    xyz = poly[-1] + overshoot * direction
    sigma = track.sigma_along_um + track.sigma_growth_k * overshoot
    return xyz, sigma


def region_confidence(sigma_um: float, border_distance_um: float) -> float:
    z = border_distance_um / max(sigma_um, 1e-3)
    cdf = 0.5 * (1.0 + erf(z / sqrt(2.0)))   # P(jittered location stays on this side)
    return float(min(1.0, max(0.0, cdf)))


def build_channel_atlas(subject: str, art: TrackArtifact, channel_positions: np.ndarray,
                        signature: str, atlas: AllenAtlas) -> pd.DataFrame:
    pos = np.asarray(channel_positions, float)
    shank_of = assign_shanks(pos)
    track_by_idx = {s.probe_shank_index: s for s in art.shanks}
    rows = []
    for ch in range(len(pos)):
        x_um, y_um = float(pos[ch, 0]), float(pos[ch, 1])
        sh = int(shank_of[ch])
        track = track_by_idx[sh]
        xyz, sigma = place_channel_on_track(track, y_um)
        reg = atlas.region_at(xyz)
        bd = atlas.border_distance_um(xyz)
        rows.append({
            "subject": subject, "chanmap_signature": signature, "channel": ch,
            "shank": sh, "x_um": x_um, "y_um": y_um,
            "ccf_ap": float(xyz[0]), "ccf_ml": float(xyz[1]), "ccf_dv": float(xyz[2]),
            "sigma_um": float(sigma),
            "region_acronym": reg["acronym"], "region_name": reg["name"],
            "region_coarse": reg["coarse"],
            "region_confidence": region_confidence(sigma, bd),
            "loc_method": track.method,
        })
    return pd.DataFrame(rows, columns=ATLAS_COLUMNS)
