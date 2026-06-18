"""Allen Mouse CCF region lookup over an annotation volume.

Annotation/resolution are injectable for testing; the default loads the real
atlas via brainglobe-atlasapi (cached download). Coordinates are microns
(AP, ML, DV) in atlas space.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

# Coarse classes used by analyses. Keys are Allen acronyms or acronym prefixes;
# resolution is by exact acronym first, then by the prefixes in _COARSE_PREFIXES.
COARSE_MAP: Dict[str, str] = {
    "CP": "CP",            # caudoputamen (dorsal striatum, the target)
    "ACB": "VS",           # nucleus accumbens (ventral striatum)
    "GPe": "GPe", "GPi": "GPe",
    "VL": "VS", "V3": "VS", "VS": "VS",   # ventricles + ventral striatum -> VS (non-target, pooled)
    "root": "out", "": "out",
}
# prefix fallbacks (longest match wins)
_COARSE_PREFIXES = [
    ("VIS", "CTX"), ("SS", "CTX"), ("MO", "CTX"), ("RSP", "CTX"),
    ("PTLp", "CTX"), ("ACA", "CTX"), ("AI", "CTX"),
    ("cc", "WM"), ("ec", "WM"), ("int", "WM"), ("fi", "WM"), ("or", "WM"),
    ("ccg", "WM"), ("ccb", "WM"),
]


def coarse_region(acronym: str) -> str:
    """Acronym-based coarse class (fallback used when no ontology map is available,
    e.g. injected test atlases). The real atlas uses _ontology_coarse_map instead."""
    if acronym in COARSE_MAP:
        return COARSE_MAP[acronym]
    best = ("", "other")
    for pre, cls in _COARSE_PREFIXES:
        if acronym.startswith(pre) and len(pre) > len(best[0]):
            best = (pre, cls)
    return best[1]


# Allen ontology anchors: a region's coarse class is the first anchor whose
# structure id appears in the region's ancestry (structure_id_path). These
# subtrees are disjoint, so priority only affects unlisted catch-alls. This is
# the robust replacement for acronym-prefix matching — it captures every white-
# matter tract (cing/scwm/sm/...), every isocortical area, etc. by lineage.
_ONTOLOGY_ANCHORS = [
    ("fiber tracts", "WM"),
    ("Isocortex", "CTX"),
    ("STRd", "CP"),      # dorsal striatum (caudoputamen) = the target
    ("STRv", "VS"),      # ventral striatum (ACB/OT/...) pooled as VS (non-target)
    ("PAL", "GPe"),      # pallidum (GPe/GPi/...) pooled as GPe
    ("VS", "VS"),        # ventricular systems
]


def _ontology_coarse_map(bg) -> Dict[int, str]:
    """Map every Allen structure id to a coarse class by ontology ancestry.

    Requires a brainglobe atlas object exposing ``lookup_df`` (columns
    acronym/id) and ``structures[id]["structure_id_path"]`` (root-first ancestor
    ids). Not used by the injectable test path, which falls back to coarse_region.
    """
    acr_to_id = dict(zip(bg.lookup_df["acronym"], bg.lookup_df["id"].astype(int)))
    anchors = [(acr_to_id[a], c) for a, c in _ONTOLOGY_ANCHORS if a in acr_to_id]
    out: Dict[int, str] = {}
    for sid in bg.lookup_df["id"].astype(int):
        sid = int(sid)
        try:
            path = {int(p) for p in bg.structures[sid]["structure_id_path"]}
        except Exception:
            continue
        cls = "other"
        for aid, c in anchors:
            if aid in path:
                cls = c
                break
        out[sid] = cls
    root_id = acr_to_id.get("root")
    if root_id is not None:
        out[root_id] = "out"   # bare root voxels = outside any defined region
    out[0] = "out"             # background id
    return out


class AllenAtlas:
    def __init__(self, annotation: Optional[np.ndarray] = None, resolution_um: float = 25.0,
                 id_to_acronym: Optional[dict] = None, id_to_name: Optional[dict] = None,
                 id_to_coarse: Optional[dict] = None,
                 atlas_name: str = "allen_mouse_25um"):
        if annotation is None:
            from brainglobe_atlasapi import BrainGlobeAtlas
            bg = BrainGlobeAtlas(atlas_name)
            # BrainGlobe annotation axis order is (AP, DV, ML) ("asr"); standardize to
            # our (AP, ML, DV) convention so region_at indexing matches track coords.
            # Verified against allen_mouse_25um: (528,320,456)->(528,456,320), and
            # lookup_df columns are acronym/id/name.
            annotation = np.transpose(np.asarray(bg.annotation), (0, 2, 1))
            resolution_um = float(bg.resolution[0])
            lut = bg.lookup_df  # DataFrame: columns id, acronym, name
            id_to_acronym = dict(zip(lut["id"].astype(int), lut["acronym"]))
            id_to_name = dict(zip(lut["id"].astype(int), lut["name"]))
            id_to_coarse = _ontology_coarse_map(bg)  # robust coarse via ancestry
        self.annotation = np.asarray(annotation)
        self.resolution_um = float(resolution_um)
        self.id_to_acronym = id_to_acronym or {}
        self.id_to_name = id_to_name or {}
        self.id_to_coarse = id_to_coarse or {}

    def _coarse_of(self, rid: int) -> str:
        """Coarse class for a region id: ontology map (real atlas) if present,
        else the acronym-based fallback (injected test atlases)."""
        if rid in self.id_to_coarse:
            return self.id_to_coarse[rid]
        return coarse_region(self.id_to_acronym.get(rid, ""))

    def _voxel(self, ccf_xyz):
        return tuple(int(np.floor(c / self.resolution_um)) for c in ccf_xyz)

    def _in_bounds(self, vox) -> bool:
        return all(0 <= v < n for v, n in zip(vox, self.annotation.shape))

    def region_at(self, ccf_xyz) -> dict:
        vox = self._voxel(ccf_xyz)
        if not self._in_bounds(vox):
            return {"id": 0, "acronym": "", "name": "out of atlas", "coarse": "out"}
        rid = int(self.annotation[vox])
        acr = self.id_to_acronym.get(rid, "")
        name = self.id_to_name.get(rid, "")
        return {"id": rid, "acronym": acr, "name": name, "coarse": self._coarse_of(rid)}

    def border_distance_um(self, ccf_xyz, max_search_um: float = 300.0) -> float:
        """Approx distance to the nearest voxel of a different region id, by
        expanding-radius search along +/- each axis. Returns max_search_um if none."""
        vox = self._voxel(ccf_xyz)
        if not self._in_bounds(vox):
            return 0.0
        rid = int(self.annotation[vox])
        if self._coarse_of(rid) == "out":
            return 0.0
        r_vox = int(np.ceil(max_search_um / self.resolution_um))
        for r in range(1, r_vox + 1):
            for ax in range(3):
                for sgn in (-1, 1):
                    nb = list(vox); nb[ax] += sgn * r
                    if self._in_bounds(tuple(nb)) and int(self.annotation[tuple(nb)]) != rid:
                        return r * self.resolution_um
        return max_search_um
