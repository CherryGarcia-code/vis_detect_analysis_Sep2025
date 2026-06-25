import pandas as pd

import curate_dant


def _pair(cmd, flag):
    """Return the single value following `flag` in an argv list."""
    return cmd[cmd.index(flag) + 1]


def test_write_curation_registry_keeps_positive_uids(tmp_path):
    src = tmp_path / "dant_registry.csv"
    pd.DataFrame({
        "session": ["01072025", "01072025", "02072025", "02072025"],
        "ks_unit_id": [3, 4, 5, 6],
        "dant_uid": [-1, 37, 0, 37],     # -1 untracked, 0 untracked, 37 tracked x2
    }).to_csv(src, index=False)
    out = tmp_path / "sub" / "dant_registry_curation.csv"

    n_rows, n_uids = curate_dant.write_curation_registry(src, out)

    assert out.exists()                       # parent dir created
    got = pd.read_csv(out, dtype={"session": str})
    assert list(got.columns) == ["session", "ks_unit_id", "dant_uid"]
    assert n_rows == 2 and n_uids == 1        # two rows of uid 37
    assert set(got["dant_uid"]) == {37}       # -1 and 0 dropped


def test_write_curation_registry_preserves_session_leading_zero(tmp_path):
    src = tmp_path / "dant_registry.csv"
    pd.DataFrame({
        "session": ["01072025"], "ks_unit_id": [3], "dant_uid": [37],
    }).to_csv(src, index=False)
    out = tmp_path / "dant_registry_curation.csv"

    curate_dant.write_curation_registry(src, out)

    got = pd.read_csv(out, dtype={"session": str})
    assert got["session"].iloc[0] == "01072025"   # not 1072025


def _paths(tmp_path):
    return curate_dant.DantCurationPaths.default(
        worktree_root=tmp_path / "wt", primary_root=tmp_path / "primary")


def test_default_paths_target_dant_dir_not_um(tmp_path):
    p = _paths(tmp_path)
    # out-dir under tracking_dant (NOT tracking_qc), so UM curation is untouched
    assert "tracking_dant" in str(p.out_dir).replace("\\", "/")
    assert "tracking_qc" not in str(p.out_dir).replace("\\", "/")
    assert p.out_dir.name == "curation"
    assert p.sheets_dir == p.out_dir / "sheets"
    # raw waveforms + pkls live under PRIMARY
    assert str(p.raw_wf_root).replace("\\", "/").endswith(
        "primary/data/unit_match/input/BG_046")
    assert str(p.pkl_dir).replace("\\", "/").endswith("primary/data/pkls/BG_046")
    # the existing CLIs we drive
    assert p.curate_script.name == "curate_tracks.py"
    assert p.render_script.name == "render_curation_sheets.py"


def test_build_curate_cmd_has_critical_flags(tmp_path):
    p = _paths(tmp_path)
    cmd = curate_dant.build_curate_cmd("py.exe", p, rebuild_cache=True)
    assert cmd[:2] == ["py.exe", str(p.curate_script)]
    # flag/value pairs must be present and correct
    assert _pair(cmd, "--liberal-col") == "dant_uid"
    assert _pair(cmd, "--drift-source") == "none"
    assert _pair(cmd, "--min-span") == "2"
    assert _pair(cmd, "--registry") == str(p.registry_curation)
    assert _pair(cmd, "--states-dir") == str(p.states_empty)
    assert _pair(cmd, "--out-dir") == str(p.out_dir)
    assert _pair(cmd, "--cache-path") == str(p.cache_path)
    assert _pair(cmd, "--raw-wf-root") == str(p.raw_wf_root)
    assert _pair(cmd, "--pkl-dir") == str(p.pkl_dir)
    assert "--rebuild-cache" in cmd


def test_build_curate_cmd_omits_rebuild_when_false(tmp_path):
    cmd = curate_dant.build_curate_cmd("py.exe", _paths(tmp_path), rebuild_cache=False)
    assert "--rebuild-cache" not in cmd


def test_build_render_cmd_has_critical_flags(tmp_path):
    p = _paths(tmp_path)
    cmd = curate_dant.build_render_cmd("py.exe", p, tier="trusted", max_uids=25)
    assert _pair(cmd, "--liberal-col") == "dant_uid"
    assert _pair(cmd, "--tier") == "trusted"
    assert _pair(cmd, "--registry") == str(p.registry_curation)
    assert _pair(cmd, "--tracks") == str(p.out_dir / "curated_tracks.csv")
    assert _pair(cmd, "--out-dir") == str(p.sheets_dir)
    assert _pair(cmd, "--max-uids") == "25"
    assert "--no-pair-scores" in cmd


def test_build_render_cmd_uids_and_no_max(tmp_path):
    cmd = curate_dant.build_render_cmd(
        "py.exe", _paths(tmp_path), tier="review", uids=[1, 2, 3])
    assert "--max-uids" not in cmd
    i = cmd.index("--uids")
    assert cmd[i + 1:i + 4] == ["1", "2", "3"]


import json


def test_write_validation_json_writes_to_given_dir(tmp_path):
    result = {"trusted": {"auc": 0.9, "n_matched": 5, "n_nonmatched": 7}}
    out_dir = tmp_path / "FIGURES" / "tracking_dant" / "BG_046" / "curation"

    p = curate_dant.write_validation_json(result, out_dir)

    assert p == out_dir / "curation_validation.json"
    assert p.exists()                                  # parent dirs created
    assert json.loads(p.read_text())["trusted"]["auc"] == 0.9
    # clobber-safety: nothing written outside the given dir
    assert "tracking_qc" not in str(p).replace("\\", "/")


def test_build_summary_table_rows_and_yardstick():
    tier_counts = {"trusted": 40, "review": 300, "suspect": 80}
    auc_by_tier = {
        "trusted": {"auc": 0.81, "n_matched": 120, "n_nonmatched": 200},
        "review": {"auc": 0.70, "n_matched": 90, "n_nonmatched": 150},
    }
    df = curate_dant.build_summary_table(tier_counts, auc_by_tier)

    assert list(df["tier"]) == ["trusted", "review", "suspect"]
    trusted = df[df.tier == "trusted"].iloc[0]
    assert trusted["dant_n_tracks"] == 40
    assert trusted["dant_auc"] == 0.81
    assert trusted["um_n_tracks"] == 22           # yardstick wired in
    assert trusted["um_auc"] == 0.96
    # a tier with no AUC entry still produces a row (suspect)
    suspect = df[df.tier == "suspect"].iloc[0]
    assert suspect["dant_n_tracks"] == 80
    assert suspect["dant_n_matched"] == 0


import pytest


def test_parse_steps_default_order():
    assert curate_dant.parse_steps("registry,curate,validate,render,summary") == [
        "registry", "curate", "validate", "render", "summary"]


def test_parse_steps_subset_canonical_order():
    # given out of order, returns canonical order; whitespace tolerated
    assert curate_dant.parse_steps("summary, registry") == ["registry", "summary"]


def test_parse_steps_rejects_unknown():
    with pytest.raises(ValueError):
        curate_dant.parse_steps("registry,bogus")
