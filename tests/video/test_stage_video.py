import os
from visdetect.core import video_sync as vs


def test_stage_copies_and_leaves_source_intact(tmp_path, monkeypatch):
    cam_root = tmp_path / "X"
    cam_dir = cam_root / "BG_046_010725"
    cam_dir.mkdir(parents=True)
    (cam_dir / "BG_046_010725_Eye_cam.mp4").write_bytes(b"video")
    (cam_dir / "BG_046_010725_Eye_cam_metadata.csv").write_text("Timestamp (ms)\n0\n")
    staging = tmp_path / "stage"

    out = vs.stage_session_video(
        "01072025", subject="BG_046", cams=("eye_cam",),
        camera_root=str(cam_root), staging_dir=str(staging))
    assert os.path.exists(out["eye_cam"]["video"])
    assert str(staging) in out["eye_cam"]["video"]
    # source untouched
    assert (cam_dir / "BG_046_010725_Eye_cam.mp4").read_bytes() == b"video"
    # idempotent (force=False -> no error, returns same paths)
    out2 = vs.stage_session_video(
        "01072025", subject="BG_046", cams=("eye_cam",),
        camera_root=str(cam_root), staging_dir=str(staging))
    assert out2["eye_cam"]["video"] == out["eye_cam"]["video"]

    vs.unstage_session_video("01072025", subject="BG_046", staging_dir=str(staging))
    assert not os.path.exists(out["eye_cam"]["video"])
