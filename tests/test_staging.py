"""
Smoke test for stage_sessions — verifies the staging manifest can be generated
and that the output CSV has the expected structure.

Requires real session pickle files to run (skips if data not present).
"""
import sys
import pytest
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]


@pytest.fixture
def staging_output(tmp_path):
    """Run stage_sessions and return the output CSV path."""
    subject_dir = repo_root / "data" / "pkls" / "BG_046"
    if not subject_dir.exists() or not list(subject_dir.glob("*.pkl")):
        pytest.skip("BG_046 session pickles not available")

    # Import and run
    from scripts.analysis.stage_sessions import stage_sessions
    output_csv = str(tmp_path / "staging_test.csv")
    stage_sessions(str(subject_dir), "BG_046", output_csv)
    return Path(output_csv)


def test_staging_manifest_columns(staging_output):
    """Verify the staging manifest has all expected columns."""
    import pandas as pd
    df = pd.read_csv(staging_output)

    required_cols = [
        "session_name", "date", "path", "hits", "misses", "fas", "crs",
        "n_go", "n_catch", "hit_rate", "fa_rate", "d_prime",
        "qc_fail", "early_licks", "aborts", "stage",
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"


def test_staging_valid_stages(staging_output):
    """All stages should be from the expected set."""
    import pandas as pd
    df = pd.read_csv(staging_output)
    valid_stages = {"Naive", "Learning", "Expert", "Disengaged", "Excluded"}
    actual_stages = set(df["stage"].unique())
    assert actual_stages.issubset(valid_stages), f"Unexpected stages: {actual_stages - valid_stages}"


def test_staging_excluded_have_nan_dprime(staging_output):
    """Excluded sessions should have NaN d'."""
    import pandas as pd
    df = pd.read_csv(staging_output)
    excluded = df[df["stage"] == "Excluded"]
    if len(excluded) > 0:
        assert excluded["d_prime"].isna().all(), "Excluded sessions should have NaN d'"


def test_staging_monotonic_transitions(staging_output):
    """Stage transitions should be one-way: Naive → Learning → Expert."""
    import pandas as pd
    df = pd.read_csv(staging_output)

    # Filter to non-excluded, non-disengaged sessions
    valid = df[~df["stage"].isin(["Excluded", "Disengaged"])].reset_index(drop=True)
    if len(valid) == 0:
        pytest.skip("No valid sessions")

    stage_order = {"Naive": 0, "Learning": 1, "Expert": 2}
    max_seen = -1
    for _, row in valid.iterrows():
        level = stage_order.get(row["stage"], -1)
        if level >= 0:
            assert level >= max_seen, (
                f"Stage regression: {row['session_name']} is {row['stage']} "
                f"after seeing level {max_seen}"
            )
            max_seen = max(max_seen, level)
