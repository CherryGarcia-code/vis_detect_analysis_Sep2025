# src/visdetect/anatomy/orientation.py
"""Probe barcode-orientation handling: associate traced shanks with probe shank
indices, and guard the medial/lateral ordering. See spec §7."""
from __future__ import annotations

from typing import List

import numpy as np

from visdetect.anatomy.tracks import ShankTrack, TrackArtifact, TrackArtifactError


def _index_increasing_with_ml(barcode_orientation: str, hemisphere: str) -> bool:
    """Does probe_shank_index increase with CCF ML (True) or decrease (False)?

    Convention (recorded here, guarded by validate_shank_order): forward+right
    -> index increases with ML (index 0 = most medial / smallest ML). Each of
    {backward, left} flips it.
    """
    increasing = True
    if barcode_orientation == "backward":
        increasing = not increasing
    if hemisphere == "left":
        increasing = not increasing
    return increasing


def assign_probe_shank_indices(shanks: List[ShankTrack], barcode_orientation: str,
                               hemisphere: str, n_shanks: int = 4) -> List[ShankTrack]:
    if len(shanks) != n_shanks:
        raise ValueError(f"expected {n_shanks} shanks, got {len(shanks)}")
    ml = np.array([s.ccf_polyline[0, 1] for s in shanks])  # tip ML
    order = np.argsort(ml)  # medial(small ML) -> lateral(large ML)
    if not _index_increasing_with_ml(barcode_orientation, hemisphere):
        order = order[::-1]
    out = []
    for new_idx, src in enumerate(order):
        s = shanks[src]
        s.probe_shank_index = int(new_idx)
        out.append(s)
    return sorted(out, key=lambda s: s.probe_shank_index)


def validate_shank_order(art: TrackArtifact, shank_pitch_um: float = 250.0,
                         tol_um: float = 120.0) -> None:
    shanks = sorted(art.shanks, key=lambda s: s.probe_shank_index)
    ml = np.array([s.ccf_polyline[0, 1] for s in shanks])
    diffs = np.diff(ml)
    if not (np.all(diffs > 0) or np.all(diffs < 0)):
        raise TrackArtifactError(f"tip ML not monotonic in shank index: {ml.tolist()}")
    if np.any(np.abs(np.abs(diffs) - shank_pitch_um) > tol_um):
        raise TrackArtifactError(
            f"shank ML spacing {np.abs(diffs).tolist()} deviates from pitch "
            f"{shank_pitch_um}±{tol_um} um"
        )
