import os
from visdetect.core import video_sync as vs


def test_stage_copies_and_leaves_source_intact(tmp_path):
    cam_root = tmp_path / "X"
    cam_dir = cam_root / "BG_046_010725"
    cam_dir.mkdir(parents=True)
    src_video = cam_dir / "BG_046_010725_Eye_cam.mp4"
    src_meta = cam_dir / "BG_046_010725_Eye_cam_metadata.csv"
    src_video.write_bytes(b"video")
    src_meta.write_text("Timestamp (ms)\n0\n")
    staging = tmp_path / "stage"

    out = vs.stage_session_video(
        "01072025", subject="BG_046", cams=("eye_cam",),
        camera_root=str(cam_root), staging_dir=str(staging))
    staged_video = out["eye_cam"]["video"]
    assert os.path.exists(staged_video)
    assert str(staging) in staged_video
    # source untouched
    assert src_video.read_bytes() == b"video"

    # Idempotency: mark the staged copy, then re-stage force=False. If no
    # re-copy happens the sentinel SURVIVES (a re-copy would clobber it).
    with open(staged_video, "wb") as f:
        f.write(b"SENTINEL")
    out2 = vs.stage_session_video(
        "01072025", subject="BG_046", cams=("eye_cam",),
        camera_root=str(cam_root), staging_dir=str(staging))
    assert out2["eye_cam"]["video"] == staged_video
    with open(staged_video, "rb") as f:
        assert f.read() == b"SENTINEL"  # proves no re-copy on force=False

    # force=True re-copies from source, restoring the original bytes.
    vs.stage_session_video(
        "01072025", subject="BG_046", cams=("eye_cam",),
        camera_root=str(cam_root), staging_dir=str(staging), force=True)
    with open(staged_video, "rb") as f:
        assert f.read() == b"video"

    # metadata source file bytes are unchanged after staging (X: read-only).
    assert src_meta.read_text() == "Timestamp (ms)\n0\n"

    vs.unstage_session_video("01072025", subject="BG_046", staging_dir=str(staging))
    assert not os.path.exists(staged_video)
