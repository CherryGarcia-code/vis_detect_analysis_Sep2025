import pandas as pd

import curate_dant


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
