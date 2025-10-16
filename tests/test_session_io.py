from pathlib import Path
from src.session_io import load_session, session_summary


def test_load_example_session():
    repo_root = Path(__file__).resolve().parents[1]
    pkl = repo_root / "data" / "BG_031_260325.pkl"
    assert pkl.exists(), "Example pkl missing"
    session = load_session(str(pkl))
    summary = session_summary(session)
    # Basic assertions based on the example file we inspected earlier
    assert summary["n_trials"] > 0
    assert summary["n_clusters"] > 0
