# tests/anatomy/test_plot_units_on_atlas.py
import matplotlib
matplotlib.use("Agg")
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "anatomy"))
from plot_units_on_atlas import plot_units_on_atlas, _state_contrast_cmap, METRIC_INFO
from visdetect.anatomy.atlas import AllenAtlas
from visdetect.anatomy.tracks import TrackArtifact, ShankTrack


def _atlas():
    ann = np.ones((200, 160, 160), dtype=int)  # all CP
    return AllenAtlas(annotation=ann, resolution_um=25.0, id_to_acronym={1: "CP"},
                      id_to_name={1: "Caudoputamen"}, id_to_coarse={1: "CP"})


def _art():
    mls = [1600.0, 1850.0, 2100.0, 2350.0]
    sh = [ShankTrack(i, np.array([[2500., mls[i], 2900.], [2500., mls[i], 2000.]]),
                     0.0, "brainreg_traced", 30., 30., 0.0) for i in range(4)]
    return TrackArtifact("BG_046", "allen_mouse_25um", "left", "forward", "t", "2026-06-24", sh)


def _df(values):
    k = len(values)
    return pd.DataFrame({"cluster_id": range(k), "ccf_ap": [2500.0] * k,
                         "ccf_ml": np.linspace(1600, 2350, k),
                         "ccf_dv": np.linspace(2600, 3200, k), "value": values})


def test_state_contrast_cmap_endpoints():
    cm = _state_contrast_cmap()
    # low end = Impulsive-ish (reddish: R>B), high end = StimSens-ish (bluish: B>R)
    lo, hi = cm(0.0), cm(1.0)
    assert lo[0] > lo[2] and hi[2] > hi[0]


def test_units_continuous_diverging_png(tmp_path):
    out = tmp_path / "sc.png"
    assert os.path.exists(plot_units_on_atlas(
        "BG_046", "17092025", "state_contrast", _df([-2., -1., 0., 1., 2., 3.]),
        _art(), out, atlas=_atlas()))


def test_units_all_positive_diverging_png(tmp_path):
    # change_response all-positive must still build (symmetric TwoSlopeNorm)
    out = tmp_path / "cr.png"
    assert os.path.exists(plot_units_on_atlas(
        "BG_046", "17092025", "change_response", _df([0., 1., 2., 3., 4., 5.]),
        _art(), out, atlas=_atlas()))


def test_units_sequential_png(tmp_path):
    out = tmp_path / "fr.png"
    assert os.path.exists(plot_units_on_atlas(
        "BG_046", "17092025", "fr", _df([1., 5., 10., 20., 30., 45.]),
        _art(), out, atlas=_atlas()))


def test_units_categorical_png(tmp_path):
    out = tmp_path / "ps.png"
    df = _df(["StimSens", "Impulsive", "StimSens", "Impulsive", "StimSens", "StimSens"])
    assert os.path.exists(plot_units_on_atlas(
        "BG_046", "17092025", "preferred_state", df, _art(), out, atlas=_atlas()))


def test_units_stereotaxic_png(tmp_path):
    out = tmp_path / "sc_stereo.png"
    assert os.path.exists(plot_units_on_atlas(
        "BG_046", "17092025", "fr", _df([1., 5., 10., 20., 30., 45.]),
        _art(), out, atlas=_atlas(), coords="stereotaxic"))


def test_lick_metrics_registered():
    for m in ("lick_hit", "lick_fa", "lick_contrast"):
        assert METRIC_INFO[m]["diverging"] is True


def test_fdr_significant_masks_and_handles_nan():
    from plot_units_on_atlas import fdr_significant
    # clear signal + clear nulls + a NaN (too-few-trials) -> NaN never significant
    pvals = [1e-6, 1e-5, 0.9, 0.8, float("nan")]
    sig = fdr_significant(pvals, alpha=0.05)
    assert sig[0] and sig[1] and not sig[2] and not sig[3] and not sig[4]
    # all-NaN -> nothing significant, no crash
    assert not fdr_significant([float("nan"), float("nan")]).any()
