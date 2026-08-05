import os
import pickle
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "QC_technical"))
from repair_trial_event_alignment import backup_pkl, repair_session  # noqa: E402

from visdetect.core.session import Session, Trial


def _make_pkl(tmp_path, n_pad=25, n=60):
    trials, bon, con = [], [], []
    rng = np.random.default_rng(0)
    for _ in range(n_pad):
        t0 = len(bon) * 10.0
        bon.append(t0); con.append(t0 + 5.0)
    for i in range(n):
        t0 = (n_pad + i) * 10.0
        ct = round(float(rng.uniform(6.0, 11.0)), 3)
        if i % 2 == 0:
            con.append(t0 + ct); outcome = "Hit"
        else:
            con.append(np.nan); outcome = "FA"
        bon.append(t0)
        trials.append(Trial(trialoutcome=outcome, change_time=ct))
    s = Session(trials=trials, subject="BG_TEST", session_name="01012025")
    s.ni_events = {
        "Baseline_ON": np.array(bon, float),
        "Change_ON": np.array(con, float),
        "Valve_L": np.zeros(len(bon), float),
    }
    p = tmp_path / "BG_TEST_01012025.pkl"
    with open(p, "wb") as f:
        pickle.dump(s, f)
    return str(p)


def test_backup_is_written_before_mutation(tmp_path):
    p = _make_pkl(tmp_path)
    b = backup_pkl(p)
    assert os.path.exists(b)
    assert "qc1_backup" in b


def test_repair_writes_index_map_and_preserves_behaviour(tmp_path):
    p = _make_pkl(tmp_path, n_pad=25, n=60)
    with open(p, "rb") as f:
        before = pickle.load(f)
    outcomes_before = [t.trialoutcome for t in before.trials]

    row = repair_session(p)

    assert row["solved"] is True
    assert row["event_offset"] == 25
    assert row["agreement"] == pytest.approx(1.0)

    with open(p, "rb") as f:
        after = pickle.load(f)
    # behaviour untouched
    assert [t.trialoutcome for t in after.trials] == outcomes_before
    assert len(after.trials) == len(before.trials)
    # map correct
    assert np.array_equal(after.trial_event_index, np.arange(25, 85))


def test_dry_run_does_not_mutate(tmp_path):
    p = _make_pkl(tmp_path)
    row = repair_session(p, dry_run=True)
    assert row["solved"] is True
    with open(p, "rb") as f:
        after = pickle.load(f)
    assert getattr(after, "trial_event_index", None) is None


def test_gate_refuses_when_a_reference_session_is_missing(tmp_path, monkeypatch):
    """The destructive script must refuse to run if it cannot verify the solver."""
    import repair_trial_event_alignment as R
    monkeypatch.setattr(R, "_ROOT", str(tmp_path))   # no data/pkls/ under tmp_path
    with pytest.raises(SystemExit) as exc:
        R.verify_realdata_gate()
    assert "REFUSING TO RUN" in str(exc.value)


def test_repair_session_does_not_invoke_the_gate(tmp_path):
    """repair_session() is called on synthetic pkls by tests; it must not gate-check."""
    p = _make_pkl(tmp_path)
    row = repair_session(p, dry_run=True)      # would SystemExit if the gate ran
    assert row["solved"] is True


def test_unsolvable_session_gets_all_minus_one(tmp_path):
    p = _make_pkl(tmp_path)
    with open(p, "rb") as f:
        s = pickle.load(f)
    s.ni_events["Change_ON"] = np.full(len(s.ni_events["Change_ON"]), np.nan)
    with open(p, "wb") as f:
        pickle.dump(s, f)

    row = repair_session(p)
    assert row["solved"] is False
    with open(p, "rb") as f:
        after = pickle.load(f)
    assert (after.trial_event_index == -1).all()


def test_post_write_check_raises_if_map_did_not_round_trip(tmp_path, monkeypatch):
    """The post-write integrity gate must RAISE (not a strippable assert) when
    the written map does not survive the reload. We force the verification
    reload to return a session whose trial_event_index differs from what was
    written, simulating a write that silently failed to persist the map.
    """
    import repair_trial_event_alignment as R
    p = _make_pkl(tmp_path)

    real_load = R.load_session
    calls = {"n": 0}

    def fake_load(path):
        s = real_load(path)
        calls["n"] += 1
        if calls["n"] >= 2:  # the verification reload inside repair_session
            s.trial_event_index = np.full(len(s.trials or []), -999, dtype=int)
        return s

    monkeypatch.setattr(R, "load_session", fake_load)
    with pytest.raises(RuntimeError, match="did not round-trip through the write"):
        repair_session(p)


def test_post_write_check_raises_if_behaviour_changed(tmp_path, monkeypatch):
    """If the reloaded behaviour differs from before the write, the gate must
    RAISE RuntimeError (not a strippable assert) and point to the backup.
    """
    import repair_trial_event_alignment as R
    p = _make_pkl(tmp_path)

    real_load = R.load_session
    calls = {"n": 0}

    def fake_load(path):
        s = real_load(path)
        calls["n"] += 1
        if calls["n"] >= 2 and s.trials:  # the verification reload
            s.trials[0].trialoutcome = "CORRUPTED"
        return s

    monkeypatch.setattr(R, "load_session", fake_load)
    with pytest.raises(RuntimeError, match="REPAIR CORRUPTED BEHAVIOUR"):
        repair_session(p)
