import os
import pytest
from visdetect.analysis import config


@pytest.mark.parametrize("raw,expected", [
    ("05032025", "05032025"),   # already 8-digit DDMMYYYY
    ("050325", "05032025"),     # 6-digit DDMMYY (BG_031/039 early sessions)
    (5032025, "05032025"),      # int, leading-zero day dropped -> 7 digits
    ("BG_031_050325", "05032025"),   # subject-prefixed
    ("BG_039_01042025_b", "01042025"),  # subject-prefixed + re-record suffix
])
def test_canonical_camera_session(raw, expected):
    assert config.canonical_camera_session(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("05032025", "050325"),
    ("050325", "050325"),
    ("BG_039_01042025_b", "010425"),
])
def test_camera_dir_token(raw, expected):
    assert config.camera_dir_token(raw) == expected


def test_subject_dirs_are_namespaced(monkeypatch):
    d046 = config.subject_video_sync_dir("BG_046")
    d031 = config.subject_video_sync_dir("BG_031")
    assert d046.endswith(os.path.join("video_sync", "BG_046"))
    assert d031.endswith(os.path.join("video_sync", "BG_031"))
    assert d046 != d031


from visdetect.core import video_sync


def _make_cam_dir(tmp_path, subject, token, cams=("Eye_cam", "Front_cam")):
    d = tmp_path / f"{subject}_{token}"
    d.mkdir()
    for c in cams:
        (d / f"{subject}_{token}_{c}.mp4").write_bytes(b"x")
        (d / f"{subject}_{token}_{c}_metadata.csv").write_text("Timestamp (ms)\n")
    return d


def test_find_camera_files_6digit_subject(tmp_path):
    _make_cam_dir(tmp_path, "BG_031", "050325")
    files = video_sync.find_camera_files(
        "050325", camera_root=str(tmp_path), subject="BG_031")
    assert "eye_cam" in files and "front_cam" in files
    assert files["eye_cam"]["video"].endswith("Eye_cam.mp4")


def test_camera_dir_to_session_tolerates_suffix():
    assert video_sync.camera_dir_to_session("BG_039_010425_b") == "01042025"
    assert video_sync.camera_dir_to_session("BG_046_010725") == "01072025"
