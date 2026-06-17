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

def test_out_of_volume_is_out():
    a = _toy_atlas()
    r = a.region_at((-100., 50., 25.))
    assert r["coarse"] == "out"

def test_border_distance_small_near_boundary():
    a = _toy_atlas()
    near = a.border_distance_um((50., 50., 112.))   # dv ~ index 4.5, near 4/5 border
    far = a.border_distance_um((50., 50., 12.))      # dv index 0, deep in region 1
    assert near < far

def test_coarse_map_has_core_classes():
    for acr in ("CP", "GPe"):
        assert acr in COARSE_MAP
