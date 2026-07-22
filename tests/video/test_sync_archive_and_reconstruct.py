import json, os
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
