from pathlib import Path
from src.session_io import load_session
from src.align import compute_peth_for_session


def test_compute_peth_basic():
    repo_root = Path(__file__).resolve().parents[1]
    pkl = repo_root / "data" / "BG_031_260325.pkl"
    session = load_session(str(pkl))
    peths = compute_peth_for_session(
        session, "Change_ON", window=(-0.5, 1.0), bin_size=0.05
    )
    # Basic checks
    assert isinstance(peths, dict)
    # If any peth exists, check shape
    if peths:
        cid, info = next(iter(peths.items()))
        assert "peth" in info and "n_trials" in info and "bin_centers" in info
        assert len(info["peth"]) == len(info["bin_centers"])
