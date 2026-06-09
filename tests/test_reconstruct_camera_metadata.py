"""Tests for camera-metadata reconstruction (header-only/empty timestamp logs).

Some sessions (e.g. BG_046 12082025, 260725) saved a header-only camera
metadata CSV, so the per-frame timestamps the video-sync pipeline needs were
never written. These helpers reconstruct a steady-fps timestamp log from the
video's own frame count and container fps. Justified empirically: the eye/front
cameras run metronomic ~50 fps with no frame drops (see reference session
140825), so a linear reconstruction is accurate to ~1 ms and fit_sync's slope
absorbs any constant fps error.
"""
from __future__ import annotations

import numpy as np
import pytest

from visdetect.core import video_sync as vs


# ---------------------------------------------------------------------------
# build_reconstructed_timestamps
# ---------------------------------------------------------------------------


def test_build_reconstructed_timestamps_length_matches_frame_count():
    ts = vs.build_reconstructed_timestamps(frame_count=100, fps=50.0)
    assert len(ts) == 100


def test_build_reconstructed_timestamps_starts_at_zero_and_steps_by_dt():
    ts = vs.build_reconstructed_timestamps(frame_count=5, fps=50.0)
    assert ts[0] == 0.0
    # dt = 1000/50 = 20 ms
    np.testing.assert_allclose(np.diff(ts), 20.0)


def test_build_reconstructed_timestamps_strictly_increasing():
    ts = vs.build_reconstructed_timestamps(frame_count=1000, fps=50.0355)
    assert np.all(np.diff(ts) > 0)


def test_build_reconstructed_timestamps_rejects_nonpositive_fps():
    with pytest.raises(ValueError):
        vs.build_reconstructed_timestamps(frame_count=10, fps=0.0)


def test_build_reconstructed_timestamps_rejects_nonpositive_frame_count():
    with pytest.raises(ValueError):
        vs.build_reconstructed_timestamps(frame_count=0, fps=50.0)


# ---------------------------------------------------------------------------
# metadata_is_header_only
# ---------------------------------------------------------------------------


def _write_header_only(path):
    path.write_text("Timestamp (ms), Acquired frames, Saved frames\n")


def _write_real_metadata(path, n=10, fps=50.0):
    """Write a plausible real metadata file: n data rows + terminal zero-row."""
    dt = 1000.0 / fps
    lines = ["Timestamp (ms),Acquired frames,Saved frames"]
    for i in range(n):
        lines.append(f"{i * dt},{i + 1},{i + 1}")
    lines.append("0,0,0")  # terminal zero-row convention
    path.write_text("\n".join(lines) + "\n")


def test_metadata_is_header_only_true_for_empty_file(tmp_path):
    p = tmp_path / "Eye_cam_metadata.csv"
    _write_header_only(p)
    assert vs.metadata_is_header_only(str(p)) is True


def test_metadata_is_header_only_false_for_real_file(tmp_path):
    p = tmp_path / "Eye_cam_metadata.csv"
    _write_real_metadata(p, n=10)
    assert vs.metadata_is_header_only(str(p)) is False


# ---------------------------------------------------------------------------
# backup_header_only_metadata
# ---------------------------------------------------------------------------


def test_backup_renames_original_to_bak(tmp_path):
    p = tmp_path / "BG_046_120825_Eye_cam_metadata.csv"
    _write_header_only(p)

    bak = vs.backup_header_only_metadata(str(p))

    expected = tmp_path / "BG_046_120825_Eye_cam_metadata.header_only.bak"
    assert bak == str(expected)
    assert expected.exists()
    assert not p.exists()  # original moved, not copied


def test_backup_preserves_existing_bak(tmp_path):
    """Second call must NOT overwrite the first (true-original) backup."""
    p = tmp_path / "BG_046_120825_Eye_cam_metadata.csv"
    bak_path = tmp_path / "BG_046_120825_Eye_cam_metadata.header_only.bak"
    bak_path.write_text("ORIGINAL_HEADER_ONLY\n")
    # p now holds a (reconstructed) file standing in for a re-run
    p.write_text("reconstructed,data,here\n")

    returned = vs.backup_header_only_metadata(str(p))

    assert returned == str(bak_path)
    # The pre-existing backup is untouched.
    assert bak_path.read_text() == "ORIGINAL_HEADER_ONLY\n"


# ---------------------------------------------------------------------------
# write_reconstructed_metadata  (round-trips through load_camera_metadata)
# ---------------------------------------------------------------------------


def test_written_metadata_round_trips_through_loader(tmp_path):
    p = tmp_path / "Eye_cam_metadata.csv"
    vs.write_reconstructed_metadata(str(p), frame_count=100, fps=50.0)

    ts_ms, acq, saved = vs.load_camera_metadata(str(p))

    assert len(ts_ms) == 100  # frame_count preserved (no spurious terminal drop)
    assert ts_ms[0] == 0.0
    np.testing.assert_allclose(np.diff(ts_ms), 20.0)
    assert np.all(np.diff(ts_ms) > 0)
