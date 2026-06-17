"""Tool-agnostic probe-track artifact: the contract between histology tracing
(brainreg / brainglobe-segmentation / Pinpoint) and the in-repo localizer.

See docs/superpowers/specs/2026-06-17-channel-anatomical-localization-design.md (§5).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

VALID_METHODS = {"brainreg_traced", "extended_from_tip", "pinpoint_planned"}
VALID_ORIENTATIONS = {"forward", "backward"}
VALID_HEMISPHERES = {"left", "right"}


class TrackArtifactError(ValueError):
    """Raised when a track artifact violates its schema."""


@dataclass
class ShankTrack:
    probe_shank_index: int
    ccf_polyline: np.ndarray            # (N, 3) float um, (AP, ML, DV), deepest-first
    tip_y_um: float                     # channel y_um at polyline[0]
    method: str                         # one of VALID_METHODS
    sigma_along_um: float
    sigma_across_um: float
    sigma_growth_k: float               # extra sigma per um of upward extension
    planned_entry: Optional[np.ndarray] = None   # (3,) or None
    planned_vector: Optional[np.ndarray] = None  # (3,) unit-ish, points tip->entry


@dataclass
class TrackArtifact:
    subject: str
    atlas: str
    hemisphere: str
    barcode_orientation: str
    source_tool: str
    created: str
    shanks: List[ShankTrack] = field(default_factory=list)


def validate_track_artifact(art: TrackArtifact) -> None:
    if art.barcode_orientation not in VALID_ORIENTATIONS:
        raise TrackArtifactError(
            f"barcode_orientation {art.barcode_orientation!r} not in {sorted(VALID_ORIENTATIONS)}"
        )
    if art.hemisphere not in VALID_HEMISPHERES:
        raise TrackArtifactError(
            f"hemisphere {art.hemisphere!r} not in {sorted(VALID_HEMISPHERES)}"
        )
    seen = set()
    for sh in art.shanks:
        if sh.method not in VALID_METHODS:
            raise TrackArtifactError(
                f"shank {sh.probe_shank_index}: method {sh.method!r} not in {sorted(VALID_METHODS)}"
            )
        poly = np.asarray(sh.ccf_polyline)
        if poly.ndim != 2 or poly.shape[1] != 3 or poly.shape[0] < 2:
            raise TrackArtifactError(
                f"shank {sh.probe_shank_index}: ccf_polyline must be (N>=2, 3), got {poly.shape}"
            )
        if sh.probe_shank_index in seen:
            raise TrackArtifactError(f"duplicate probe_shank_index {sh.probe_shank_index}")
        seen.add(sh.probe_shank_index)
        if sh.method != "brainreg_traced" and (sh.planned_vector is None):
            raise TrackArtifactError(
                f"shank {sh.probe_shank_index}: method {sh.method!r} requires planned_vector"
            )


def _shank_to_dict(sh: ShankTrack) -> dict:
    return {
        "probe_shank_index": int(sh.probe_shank_index),
        "ccf_polyline": np.asarray(sh.ccf_polyline, float).tolist(),
        "tip_y_um": float(sh.tip_y_um),
        "method": sh.method,
        "sigma_along_um": float(sh.sigma_along_um),
        "sigma_across_um": float(sh.sigma_across_um),
        "sigma_growth_k": float(sh.sigma_growth_k),
        "planned_entry": None if sh.planned_entry is None else np.asarray(sh.planned_entry, float).tolist(),
        "planned_vector": None if sh.planned_vector is None else np.asarray(sh.planned_vector, float).tolist(),
    }


def _shank_from_dict(d: dict) -> ShankTrack:
    def _arr(x):
        return None if x is None else np.asarray(x, float)
    return ShankTrack(
        probe_shank_index=int(d["probe_shank_index"]),
        ccf_polyline=np.asarray(d["ccf_polyline"], float),
        tip_y_um=float(d["tip_y_um"]),
        method=d["method"],
        sigma_along_um=float(d["sigma_along_um"]),
        sigma_across_um=float(d["sigma_across_um"]),
        sigma_growth_k=float(d["sigma_growth_k"]),
        planned_entry=_arr(d.get("planned_entry")),
        planned_vector=_arr(d.get("planned_vector")),
    )


def save_track_artifact(art: TrackArtifact, path) -> None:
    payload = {
        "subject": art.subject, "atlas": art.atlas, "hemisphere": art.hemisphere,
        "barcode_orientation": art.barcode_orientation, "source_tool": art.source_tool,
        "created": art.created, "shanks": [_shank_to_dict(s) for s in art.shanks],
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_track_artifact(path) -> TrackArtifact:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    art = TrackArtifact(
        subject=d["subject"], atlas=d["atlas"], hemisphere=d["hemisphere"],
        barcode_orientation=d["barcode_orientation"], source_tool=d["source_tool"],
        created=d["created"], shanks=[_shank_from_dict(s) for s in d["shanks"]],
    )
    validate_track_artifact(art)
    return art
