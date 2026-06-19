import os, pytest
from visdetect.analysis.tf_glm_data import load_khilkevich_session

BASE = r"X:/public/projects/MoHa_20260212_dmdmTemporalExpectation/data/npx_converted"


@pytest.mark.skipif(not os.path.isdir(BASE), reason="ceph not mounted")
def test_load_one_khilkevich_session():
    animal = sorted(os.listdir(BASE))[0]
    sess = sorted(os.listdir(os.path.join(BASE, animal)))[0]
    ks = load_khilkevich_session(os.path.join(BASE, animal, sess))
    assert len(ks.units) > 0
    assert ks.trials.shape[0] > 0
    assert ks.change_on.ndim == 1
    # at least one region label present
    assert len(set(ks.regions.values())) >= 1
