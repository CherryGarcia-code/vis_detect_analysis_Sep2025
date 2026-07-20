import sys
import numpy as np
sys.path.insert(0, "scripts/tf_responsiveness/preparatory_fig5")
import build_prep_cache as B  # noqa: E402
from visdetect.analysis import preparatory as P  # noqa: E402


def _poisson_spikes(rate_fn, t0, t1, seed):
    rng = np.random.default_rng(seed)
    rmax = 120.0
    n = rng.poisson(rmax * (t1 - t0))
    cand = np.sort(rng.uniform(t0, t1, n))
    keep = rng.random(cand.size) < (rate_fn(cand) / rmax)
    return cand[keep]


def test_unit_lick_ztrace_detects_prelick_ramp():
    lick_times = np.arange(10) * 20.0 + 100.0
    change_times = np.arange(10) * 20.0 + 108.0  # change 8 s past the lick (far)

    def rate(ts):
        r = np.full(ts.shape, 5.0)
        for L in lick_times:
            d = ts - L
            r += 60.0 * np.exp(-((d + 0.2) ** 2) / (2 * 0.15 ** 2))  # bump ~0.2 s pre-lick
        return r
    spk = _poisson_spikes(rate, 0, 400, seed=7)
    z, t, n = B.unit_lick_ztrace(spk, list(lick_times), list(change_times),
                                 lick_win=(-2.0, 1.5), base_win=(-2.0, 0.0),
                                 bin_s=0.025, sigma_bins=1.0)
    assert n == 10
    onset = P.cell_onset(t, z)
    assert -0.7 < onset < 0.1           # ramp onset just before the lick
    assert np.nanmax(z) > 2.576         # clearly active


def test_unit_lick_ztrace_flat_unit_no_onset():
    lick_times = np.arange(10) * 20.0 + 100.0
    change_times = np.arange(10) * 20.0 + 108.0
    spk = _poisson_spikes(lambda ts: np.full(ts.shape, 8.0), 0, 400, seed=3)
    z, t, n = B.unit_lick_ztrace(spk, list(lick_times), list(change_times),
                                 lick_win=(-2.0, 1.5), base_win=(-2.0, 0.0),
                                 bin_s=0.025, sigma_bins=1.0)
    assert np.isnan(P.cell_onset(t, z))  # no sustained supra-threshold activity


def test_label_shuffle_flattens_onset_gradient():
    """The null machinery must FLATTEN a genuine onset-decreases-with-width gradient:
    a real negative onset~width correlation must exceed the 95th percentile of the
    width-shuffled null (otherwise the null is not actually breaking the link)."""
    import nulls_and_hardening as H  # noqa: E402
    rng = np.random.default_rng(0)
    # synthetic: onset genuinely decreases with width (wider -> earlier / more negative)
    width = np.linspace(0.03, 0.5, 200)
    onset = -0.2 * (width - 0.03) / 0.47 + rng.normal(0, 0.02, 200)
    r_obs = H.width_onset_corr(width, onset)
    assert r_obs < 0                                    # real effect is negative
    null = H.width_shuffle_corr_null(onset, width, n=300, seed=1)
    assert np.nanmedian(np.abs(null)) < abs(r_obs)      # null flattens toward 0
    assert abs(r_obs) > np.percentile(np.abs(null), 95)  # observed beats shuffled labels
