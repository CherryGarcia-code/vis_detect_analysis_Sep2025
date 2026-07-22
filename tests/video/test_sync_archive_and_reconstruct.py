import json, os

import numpy as np

from visdetect.core import video_sync as vs


def test_archive_moves_existing_sync_and_anchor(tmp_path):
    d = tmp_path
    (d / "01072025_video_sync.json").write_text(json.dumps({"session_name": "01072025"}))
    (d / "01072025_anchor.json").write_text(json.dumps({"anchors": []}))
    arch = vs.archive_sync_artifacts("01072025", sync_dir=str(d), when="2026-07-21")
    assert arch is not None
    assert not (d / "01072025_video_sync.json").exists()
    assert os.path.exists(os.path.join(arch, "01072025_video_sync.json"))
    assert os.path.exists(os.path.join(arch, "01072025_anchor.json"))


def test_archive_noop_when_nothing_exists(tmp_path):
    assert vs.archive_sync_artifacts("01072025", sync_dir=str(tmp_path)) is None


def test_reconstruction_writes_local_not_camera_root(tmp_path, monkeypatch):
    # Camera root is treated as READ-ONLY: a reconstructed CSV must NOT appear there.
    cam_root = tmp_path / "X"
    cam_dir = cam_root / "BG_046_010725"
    cam_dir.mkdir(parents=True)
    (cam_dir / "BG_046_010725_Eye_cam.mp4").write_bytes(b"x")
    (cam_dir / "BG_046_010725_Eye_cam_metadata.csv").write_text("Timestamp (ms)\n")  # header-only
    local = tmp_path / "sync"
    monkeypatch.setattr(vs, "CAMERA_ROOT", str(cam_root))
    monkeypatch.setattr(
        "visdetect.analysis.config.VIDEO_SYNC_DIR", str(local), raising=False)

    out = vs.write_local_reconstructed_metadata(
        "01072025", "eye_cam", frame_count=100, fps=50.0, subject="BG_046")
    assert str(local) in out and out.endswith("reconstructed.csv")
    # camera dir untouched apart from the original header-only file
    names = sorted(p.name for p in cam_dir.iterdir())
    assert names == ["BG_046_010725_Eye_cam.mp4", "BG_046_010725_Eye_cam_metadata.csv"]
    ts, _, _ = vs.load_camera_metadata(out)
    assert len(ts) == 100 and abs(ts[1] - 20.0) < 1e-6
