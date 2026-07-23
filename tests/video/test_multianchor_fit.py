import numpy as np
from visdetect.core import video_sync as vs


def _anchors(nidaq, video, event="baseline_on"):
    out = []
    for i, (n, v) in enumerate(zip(nidaq, video)):
        out.append({"trial_index": i, "event_type": event,
                    "nidaq_event_s": float(n), "video_time_s": float(v)})
    return out


def test_loo_cv_small_n_returns_finite():
    x = np.array([0., 1., 2., 3., 4.])
    y = 1.0 * x + 0.5
    assert vs._loo_cv(x, y) < 1.0  # near-perfect line -> tiny cv (ms)


def test_fit_multianchor_recovers_slope_offset():
    nidaq = np.linspace(10, 600, 8)
    slope_true, off_true = 1.00002, -3.2
    video = slope_true * nidaq + off_true          # cam(video) = fn(nidaq)... invert below
    # anchors store video_time_s and nidaq_event_s; fit models nidaq = slope*cam + off
    anchors = _anchors(nidaq, video)
    res = vs.fit_multianchor_clock(anchors, n_baseline_on=8)
    # cam_s = video, nidaq = nidaq -> slope ~ 1/slope_true, offset ~ -off/slope_true
    pred = res.slope * np.asarray(video) + res.offset
    assert np.allclose(pred, nidaq, atol=1e-3)
    assert res.detection_method == "manual_multianchor"
    assert res.cv_rmse_ms < 5.0
    assert res.quality == "good"


def test_fit_multianchor_mad_rejects_outlier():
    nidaq = np.linspace(10, 600, 8)
    video = 1.0 * nidaq + 0.0
    video[3] += 0.5  # 500 ms bad anchor
    anchors = _anchors(nidaq, video)
    res = vs.fit_multianchor_clock(anchors, n_baseline_on=8)
    assert res.n_anchors == 7  # one rejected


def test_quality_manual_multianchor_review_and_failed():
    good = vs.SyncResult(slope=1.0, offset=0.0, n_anchors=6, n_baseline_on=6,
                         rmse_ms=10, max_residual_ms=15, cv_rmse_ms=15,
                         slope_ppm=5, durbin_watson=2.0,
                         detection_method="manual_multianchor")
    review = vs.SyncResult(slope=1.0, offset=0.0, n_anchors=6, n_baseline_on=6,
                           rmse_ms=30, max_residual_ms=35, cv_rmse_ms=30,
                           slope_ppm=5, durbin_watson=2.0,
                           detection_method="manual_multianchor")
    failed = vs.SyncResult(slope=1.0, offset=0.0, n_anchors=6, n_baseline_on=6,
                           rmse_ms=80, max_residual_ms=90, cv_rmse_ms=80,
                           slope_ppm=5, durbin_watson=2.0,
                           detection_method="manual_multianchor")
    assert good.quality == "good"
    assert review.quality == "review"
    assert failed.quality == "failed"


def test_quality_manual_multianchor_strict_boundaries():
    # Thresholds are strict `<`: cv==GOOD_CV (20) is NOT good but IS review;
    # cv==REVIEW_CV (40) is NOT review -> failed.
    at_good = vs.SyncResult(slope=1.0, offset=0.0, n_anchors=6, n_baseline_on=6,
                            rmse_ms=10, max_residual_ms=15, cv_rmse_ms=20.0,
                            slope_ppm=5, durbin_watson=2.0,
                            detection_method="manual_multianchor")
    at_review = vs.SyncResult(slope=1.0, offset=0.0, n_anchors=6, n_baseline_on=6,
                              rmse_ms=10, max_residual_ms=15, cv_rmse_ms=40.0,
                              slope_ppm=5, durbin_watson=2.0,
                              detection_method="manual_multianchor")
    assert at_good.quality == "review"
    assert at_review.quality == "failed"
