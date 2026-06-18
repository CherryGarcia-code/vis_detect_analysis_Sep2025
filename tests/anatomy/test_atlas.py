import numpy as np
from visdetect.anatomy.atlas import AllenAtlas, COARSE_MAP

def _toy_atlas():
    # 10x10x10 voxels @ 25um: left half region id 1 (CP), right half id 2 (GPe)
    ann = np.zeros((10, 10, 10), dtype=int)
    ann[:, :, :5] = 1
    ann[:, :, 5:] = 2
    id_to_acr = {0: "root", 1: "CP", 2: "GPe"}
    id_to_name = {0: "root", 1: "Caudoputamen", 2: "Globus pallidus external"}
    return AllenAtlas(annotation=ann, resolution_um=25.0,
                      id_to_acronym=id_to_acr, id_to_name=id_to_name)

def test_region_at_returns_acronym():
    a = _toy_atlas()
    r = a.region_at((50., 50., 25.))   # dv index 1 -> id 1 -> CP
    assert r["acronym"] == "CP"
    assert r["coarse"] == "CP"

def test_region_at_other_half():
    a = _toy_atlas()
    r = a.region_at((50., 50., 200.))  # dv index 8 -> id 2 -> GPe
    assert r["acronym"] == "GPe"
    assert r["coarse"] == "GPe"

def test_out_of_volume_is_out():
    a = _toy_atlas()
    r = a.region_at((-100., 50., 25.))
    assert r["coarse"] == "out"

def test_border_distance_small_near_boundary():
    a = _toy_atlas()
    near = a.border_distance_um((50., 50., 112.))   # dv ~ index 4.5, near 4/5 border
    far = a.border_distance_um((50., 50., 12.))      # dv index 0, deep in region 1
    assert near < far
    assert near == 25.0

def test_coarse_map_has_core_classes():
    for acr in ("CP", "GPe"):
        assert acr in COARSE_MAP

def test_coarse_region_prefix_and_default():
    from visdetect.anatomy.atlas import coarse_region
    assert coarse_region("VISp") == "CTX"
    assert coarse_region("cc") == "WM"
    assert coarse_region("ACB") == "VS"
    assert coarse_region("ZZZ") == "other"

def test_border_distance_zero_for_out_voxel():
    import numpy as np
    from visdetect.anatomy.atlas import AllenAtlas
    ann = np.zeros((10, 10, 10), dtype=int)  # all id 0 -> root -> "out"
    a = AllenAtlas(annotation=ann, resolution_um=25.0,
                   id_to_acronym={0: "root"}, id_to_name={0: "root"})
    assert a.border_distance_um((50., 50., 50.)) == 0.0


# ── Robust coarse mapping via Allen ontology ancestry ────────────────────────

class _FakeBG:
    """Minimal stand-in for a BrainGlobeAtlas: lookup_df + structures with
    structure_id_path ancestry (root-first), matching the real API shape."""
    def __init__(self):
        import pandas as pd
        rows = [
            # acronym,        id,        ancestry (structure_id_path, root-first)
            ("root",          997,       [997]),
            ("grey",            8,       [997, 8]),
            ("fiber tracts", 1009,       [997, 1009]),
            ("cing",          940,       [997, 1009, 940]),         # WM tract
            ("scwm",     484682512,      [997, 1009, 484682512]),   # WM tract
            ("STR",           477,       [997, 8, 477]),
            ("STRd",          485,       [997, 8, 477, 485]),
            ("CP",            672,       [997, 8, 477, 485, 672]),   # target
            ("STRv",          493,       [997, 8, 477, 493]),
            ("ACB",           56,        [997, 8, 477, 493, 56]),    # ventral striatum
            ("PAL",           803,       [997, 8, 803]),
            ("GPe",         1022,        [997, 8, 803, 1022]),
            ("Isocortex",     315,       [997, 8, 315]),
            ("MOp5",          648,       [997, 8, 315, 648]),        # cortex
            ("VS",             73,       [997, 73]),
            ("VL",             81,       [997, 73, 81]),             # ventricle
            ("TH",            549,       [997, 8, 549]),             # grey, non-target
        ]
        self.lookup_df = pd.DataFrame(
            {"acronym": [r[0] for r in rows], "id": [r[1] for r in rows],
             "name": [r[0] for r in rows]})
        self.structures = {r[1]: {"id": r[1], "acronym": r[0],
                                  "structure_id_path": r[2]} for r in rows}


def test_ontology_coarse_map_uses_ancestry():
    from visdetect.anatomy.atlas import _ontology_coarse_map
    m = _ontology_coarse_map(_FakeBG())
    assert m[940] == "WM"          # cing -> under fiber tracts
    assert m[484682512] == "WM"    # scwm -> under fiber tracts
    assert m[672] == "CP"          # CP -> under STRd
    assert m[56] == "VS"           # ACB -> under STRv (ventral striatum)
    assert m[1022] == "GPe"        # GPe -> under PAL
    assert m[648] == "CTX"         # MOp5 -> under Isocortex
    assert m[81] == "VS"           # VL -> ventricular system
    assert m[549] == "other"       # thalamus: grey but not a target subtree
    assert m[997] == "out"         # root


def test_region_at_prefers_id_to_coarse_override():
    # 'cing' has no acronym rule (-> would be 'other'); ontology id_to_coarse fixes it.
    ann = np.ones((4, 4, 4), dtype=int)  # all region id 1
    a = AllenAtlas(annotation=ann, resolution_um=25.0,
                   id_to_acronym={1: "cing"}, id_to_coarse={1: "WM"})
    r = a.region_at((10., 10., 10.))
    assert r["acronym"] == "cing"
    assert r["coarse"] == "WM"


def test_region_at_acronym_fallback_without_ontology():
    # Without an id_to_coarse map, the acronym fallback can't classify 'cing'.
    ann = np.ones((4, 4, 4), dtype=int)
    a = AllenAtlas(annotation=ann, resolution_um=25.0, id_to_acronym={1: "cing"})
    assert a.region_at((10., 10., 10.))["coarse"] == "other"
