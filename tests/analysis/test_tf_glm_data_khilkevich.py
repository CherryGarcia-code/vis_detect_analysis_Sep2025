import os, pytest
import numpy as np
from visdetect.analysis.tf_glm import TFGLMConfig, trial_bin_edges
from visdetect.analysis.tf_glm_data import (
    load_khilkevich_session, khilkevich_trial_regressors,
)

BASE = r"X:/public/projects/MoHa_20260212_dmdmTemporalExpectation/data/npx_converted"


def _first_session():
    animal = sorted(os.listdir(BASE))[0]
    sess = sorted(os.listdir(os.path.join(BASE, animal)))[0]
    return os.path.join(BASE, animal, sess)


@pytest.mark.skipif(not os.path.isdir(BASE), reason="ceph not mounted")
def test_load_one_khilkevich_session():
    ks = load_khilkevich_session(_first_session())
    assert len(ks.units) > 0
    assert ks.trials.shape[0] > 0
    assert ks.change_on.ndim == 1
    # at least one region label present
    assert len(set(ks.regions.values())) >= 1


@pytest.mark.skipif(not os.path.isdir(BASE), reason="ceph not mounted")
def test_khilkevich_movement_regressors_populated():
    """include_movement=True populates per-trial motion_bins/pupil_bins of the
    right per-trial length and non-constant; airpuff_time is finite on some
    trials. Reduced (default) build leaves them unset."""
    sd = _first_session()
    ks = load_khilkevich_session(sd)

    # Reduced (default): movement fields stay None/NaN.
    cfg_red = TFGLMConfig(include_movement=False)
    tr_red, _ = khilkevich_trial_regressors(ks, cfg_red, region=None)
    assert all(t.motion_bins is None and t.pupil_bins is None for t in tr_red)

    # Full model: motion/pupil populated, per-trial length, non-constant.
    cfg = TFGLMConfig(include_movement=True)
    trials, _ = khilkevich_trial_regressors(ks, cfg, region=None)
    checked = 0
    for t in trials:
        if t.motion_bins is None:
            continue
        e = trial_bin_edges(t.t_start, t.t_end, cfg.bin_s)
        assert t.motion_bins.size == e.size, "motion_bins not per-trial length"
        assert t.pupil_bins.size == e.size, "pupil_bins not per-trial length"
        checked += 1
        if checked >= 20:
            break
    assert checked > 0, "no trials carried movement bins"
    # at least one trial's motion-energy is non-constant
    assert any(t.motion_bins is not None and np.nanstd(t.motion_bins) > 0
               for t in trials), "motion_energy is constant everywhere"
    assert any(t.pupil_bins is not None and np.nanstd(t.pupil_bins) > 0
               for t in trials), "pupil is constant everywhere"
    # some trial carries a finite air-puff time
    assert any(np.isfinite(t.airpuff_time) for t in trials), "no airpuff times"
