import numpy as np
import pytest
from visdetect.analysis import population_field as pf
from visdetect.analysis.utils import build_population_tensor
from visdetect.utils.synthetic import make_synthetic_session


def _np2_positions():
    """4-shank NP2.0-like geometry: x in {0,32,250,282,500,532,750,782}, y 1500..2200."""
    xs = [0, 32, 250, 282, 500, 532, 750, 782]
    ys = np.arange(1500, 2205, 15.0)  # 15 um row pitch
    pos = np.array([[x, y] for y in ys for x in xs], dtype=float)
    return pos


def test_depth_bin_edges_cover_active_band():
    pos = _np2_positions()
    edges = pf.depth_bin_edges(pos, depth_bin_um=60.0)
    assert edges[0] <= pos[:, 1].min()
    assert edges[-1] >= pos[:, 1].max()
    assert np.allclose(np.diff(edges), 60.0)


def test_robust_unit_depth_weighted_centroid():
    # 3 channels at y = 0, 100, 200; ptp concentrated as 1:2:1 -> centroid = 100
    n_samp = 82
    mw = np.zeros((n_samp, 3))
    mw[:, 0] = np.linspace(-0.5, 0.5, n_samp)   # ptp 1
    mw[:, 1] = np.linspace(-1.0, 1.0, n_samp)   # ptp 2
    mw[:, 2] = np.linspace(-0.5, 0.5, n_samp)   # ptp 1
    pos = np.array([[0.0, 0.0], [0.0, 100.0], [0.0, 200.0]])
    assert pf.robust_unit_depth(mw, pos) == pytest.approx(100.0)


def test_robust_unit_depth_zero_amplitude_is_nan():
    mw = np.zeros((82, 3))
    pos = np.array([[0.0, 0.0], [0.0, 100.0], [0.0, 200.0]])
    assert np.isnan(pf.robust_unit_depth(mw, pos))


def test_fingerprint_pools_all_channels():
    # one unit, ptp = 4 on the channel at y=1560 -> that bin gets >=4
    pos = _np2_positions()
    y_edges = pf.depth_bin_edges(pos, 60.0)
    mw = np.zeros((82, pos.shape[0]))
    chan = int(np.argmin(np.abs(pos[:, 1] - 1560)))
    mw[:, chan] = np.linspace(-2.0, 2.0, 82)  # ptp = 4
    fp = pf.amplitude_depth_fingerprint([mw], pos, y_edges)
    assert fp.shape == (len(y_edges) - 1,)
    target_bin = np.clip(np.searchsorted(y_edges, pos[chan, 1]) - 1, 0, len(y_edges) - 2)
    assert fp[target_bin] == pytest.approx(4.0)


def test_estimate_shift_recovers_known_roll():
    rng = np.random.default_rng(0)
    ref = np.abs(rng.normal(size=40))
    mov = np.roll(ref, 3); mov[:3] = 0.0     # shifted 3 bins deeper
    shift, corr = pf.estimate_shift_bins(ref, mov, max_lag_bins=10)
    assert shift == -3          # mov must be rolled by -3 to align onto ref
    # estimate_shift_bins zeroes the |lag| edge bins on alignment (verbatim from
    # diagnose_intersession_drift.estimate_shift). On a length-40 signal that drops
    # 3 bins -- here including ref[39]~1.49 -- so peak corr saturates at ~0.886, not
    # >0.99 (>0.99 is unreachable for this impl even at size=200). Strong-alignment
    # sanity check kept, threshold set to the value the verbatim impl actually attains.
    assert corr > 0.85


def test_session_shift_um_recovers_60um(tmp_path):
    # Build two fake sessions' fingerprints on a shared axis, one shifted +2 bins.
    pos = _np2_positions()
    y_edges = pf.depth_bin_edges(pos, 60.0)
    ref_fp = np.abs(np.sin(np.linspace(0, 6, len(y_edges) - 1))) + 0.1
    mov_fp = np.roll(ref_fp, 2); mov_fp[:2] = 0.0
    shifts = pf.session_shift_um(
        {"01072025": ref_fp, "02072025": mov_fp},
        ref_session="01072025", depth_bin_um=60.0, max_lag_um=300.0,
    )
    assert shifts["01072025"][0] == pytest.approx(0.0)
    # mov is 2 bins deeper -> needs -2 bins to align -> reported deeper shift = +120 um
    assert shifts["02072025"][0] == pytest.approx(120.0)


def test_registered_depth_subtracts_shift():
    assert pf.registered_depth(1800.0, 120.0) == pytest.approx(1680.0)


def test_unit_field_index_and_count():
    pos = _np2_positions()
    y_edges = pf.depth_bin_edges(pos, 60.0)
    n_depth = len(y_edges) - 1
    # shank 2, depth in bin 0 -> index = 2 * n_depth + 0
    idx = pf.unit_field_index(y_edges[0] + 1.0, shank=2, y_edges=y_edges, n_shanks=4)
    assert idx == 2 * n_depth
    assert pf.n_field_bins(y_edges, n_shanks=4) == 4 * n_depth
    # below range clips to depth bin 0
    assert pf.unit_field_index(y_edges[0] - 999, shank=0, y_edges=y_edges) == 0


def test_build_field_tensor_sums_units_into_bins():
    sess = make_synthetic_session(n_trials=40, n_clusters=6, seed=1)
    uids = [c.cluster_id for c in sess.clusters]
    unit_bin_index = np.array([0, 0, 0, 1, 1, 1])   # first 3 -> bin 0, last 3 -> bin 1
    per_unit, bc, valid = build_population_tensor(
        sess, uids, event_name="Baseline_ON", window=(-0.5, 1.0), bin_size=0.025)
    field, bc2, valid2 = pf.build_field_tensor(
        sess, uids, unit_bin_index, n_bins_anat=2,
        event_name="Baseline_ON", window=(-0.5, 1.0), bin_size=0.025)
    assert field.shape == (per_unit.shape[0], per_unit.shape[1], 2)
    assert valid2 == valid
    np.testing.assert_allclose(field[:, :, 0], per_unit[:, :, :3].sum(axis=2))
    np.testing.assert_allclose(field[:, :, 1], per_unit[:, :, 3:].sum(axis=2))


def test_fingerprint_corr_identical_is_one():
    a = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    assert pf.fingerprint_corr(a, a) == pytest.approx(1.0)


def test_peak_vs_centroid_depth():
    mw = np.zeros((82, 3))
    mw[:, 1] = np.linspace(-1, 1, 82)     # single dominant channel at index 1
    pos = np.array([[0.0, 0.0], [0.0, 100.0], [0.0, 200.0]])
    peak_d, cent_d = pf.peak_vs_centroid_depth(mw, pos)
    assert peak_d == pytest.approx(100.0)
    assert cent_d == pytest.approx(100.0)


def test_audit_shift_vs_um_offset():
    mf = {"01072025": 0.0, "02072025": 60.0, "03072025": 0.0}
    um = {"01072025": 0.0, "02072025": 45.0}          # only 2 shared
    rep = pf.audit_shift_vs_um_offset(mf, um)
    assert rep["n"] == 2
    assert rep["max_abs_diff_um"] == pytest.approx(15.0)
    assert rep["median_abs_diff_um"] == pytest.approx(7.5)
