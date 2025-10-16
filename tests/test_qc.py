from pathlib import Path
from src.session_io import load_session
from src.qc import run_qc


def test_run_qc_example(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    pkl = repo_root / 'data' / 'BG_031_260325.pkl'
    session = load_session(str(pkl))
    outdir = tmp_path / 'qc_out'
    res = run_qc(session, str(outdir))
    # Check files exist
    for k in ['summary_path', 'clusters_qc_path', 'trials_qc_path']:
        p = Path(res[k])
        assert p.exists()
    # Expect at least one cluster and one trial
    assert session.clusters
    assert session.trials
