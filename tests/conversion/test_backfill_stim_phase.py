import json, numpy as np
from pathlib import Path
from visdetect.core.session import Session, Trial, save_session, load_session
from scripts.conversion.backfill_stim_phase import backfill_session

def _make_raw(dirpath, n=3, nframes=5):
    sess = dirpath / "Session"; sess.mkdir(parents=True)
    trials = [{"trialoutcome": "Hit", "Stim2TF": 1.5,
               "vbl": list(np.arange(nframes) * 0.0166 + 100.0 + i),
               "TF": [0.0] * nframes,
               "phase": [[k, 0] for k in range(nframes)]} for i in range(n)]
    (sess / "run1__trials.json").write_text(json.dumps(trials))

def test_backfill_attaches_phase(tmp_path):
    raw = tmp_path / "BG_999_01012025"; _make_raw(raw, n=3, nframes=5)
    s = Session(trials=[Trial(trialoutcome="Hit") for _ in range(3)],
                session_name="BG_999_01012025")
    pkl = tmp_path / "in.pkl"; save_session(s, str(pkl))
    out = tmp_path / "out.pkl"
    info = backfill_session(str(pkl), str(raw), str(out))
    assert info["n_trials"] == 3 and info["n_with_phase"] == 3 and info["matched"]
    s2 = load_session(str(out))
    assert s2.trials[0].stim_phase.shape == (5, 2)
    assert s2.trials[1].stim_vbl.shape == (5,)

def test_backfill_count_mismatch_flags_unmatched(tmp_path):
    raw = tmp_path / "BG_999_01012025"; _make_raw(raw, n=2)   # 2 raw trials
    s = Session(trials=[Trial() for _ in range(3)], session_name="BG_999_01012025")  # 3 pkl trials
    pkl = tmp_path / "in.pkl"; save_session(s, str(pkl))
    info = backfill_session(str(pkl), str(raw), str(tmp_path / "out.pkl"))
    assert info["matched"] is False
