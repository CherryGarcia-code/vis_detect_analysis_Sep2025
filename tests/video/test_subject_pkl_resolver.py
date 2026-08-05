"""Subject-aware pkl resolver in visdetect.suite.loader (A1-pilot fix).

fit_sync/tag_session accept --subject but must NOT resolve the behavioural pkl
through the import-frozen config.SUBJECT (VISDETECT_SUBJECT, default BG_046).
resolve_subject_pkl / load_session_for_subject honour an explicit subject and
match tokens by canonical camera-session id (6-digit DDMMYY vs 8-digit DDMMYYYY).

All synthetic: pkl "files" are empty stubs; the resolver only lists directory
names and never unpickles, and the loader is monkeypatched, so no real 200 MB
pkl is touched.
"""
import os

import pytest

from visdetect.suite import loader


def _make_pkl(root, subject, token):
    """Create an empty stub data/pkls/<subject>/<subject>_<token>.pkl; return path."""
    d = os.path.join(root, "data", "pkls", subject)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{subject}_{token}.pkl")
    with open(path, "wb") as f:
        f.write(b"")  # content irrelevant: the resolver never opens it
    return path


# ── resolve_subject_pkl ───────────────────────────────────────────────

def test_resolves_subject_pkl_ignoring_frozen_default(tmp_path, monkeypatch):
    # Frozen default is BG_046 but we ask for BG_031: must find BG_031's pkl.
    monkeypatch.setattr(loader, "ROOT", str(tmp_path))
    monkeypatch.setattr(loader, "SUBJECT", "BG_046")
    _make_pkl(str(tmp_path), "BG_046", "09042025")   # decoy same-date BG_046 pkl
    expected = _make_pkl(str(tmp_path), "BG_031", "09042025")
    got = loader.resolve_subject_pkl("09042025", subject="BG_031")
    assert got == expected


def test_matches_6digit_token_to_8digit_request(tmp_path, monkeypatch):
    # BG_031/039 can carry 6-digit DDMMYY tokens; an 8-digit DDMMYYYY request
    # must still match via canonical_camera_session.
    monkeypatch.setattr(loader, "ROOT", str(tmp_path))
    expected = _make_pkl(str(tmp_path), "BG_031", "090425")     # DDMMYY
    got = loader.resolve_subject_pkl("09042025", subject="BG_031")  # DDMMYYYY
    assert got == expected


def test_matches_leading_zero_day(tmp_path, monkeypatch):
    # Day 1-9 leading-zero footgun: request as int-form string must still resolve.
    monkeypatch.setattr(loader, "ROOT", str(tmp_path))
    expected = _make_pkl(str(tmp_path), "BG_031", "01072025")
    assert loader.resolve_subject_pkl("01072025", subject="BG_031") == expected


def test_returns_none_when_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(loader, "ROOT", str(tmp_path))
    _make_pkl(str(tmp_path), "BG_031", "09042025")
    # Different date -> None; different (empty) subject dir -> None.
    assert loader.resolve_subject_pkl("01012020", subject="BG_031") is None
    assert loader.resolve_subject_pkl("09042025", subject="BG_999") is None


def test_defaults_to_frozen_subject_when_none(tmp_path, monkeypatch):
    monkeypatch.setattr(loader, "ROOT", str(tmp_path))
    monkeypatch.setattr(loader, "SUBJECT", "BG_046")
    expected = _make_pkl(str(tmp_path), "BG_046", "01072025")
    assert loader.resolve_subject_pkl("01072025") == expected  # subject=None -> SUBJECT


# ── load_session_for_subject ──────────────────────────────────────────

def test_load_session_for_subject_raises_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(loader, "ROOT", str(tmp_path))
    with pytest.raises(FileNotFoundError) as exc:
        loader.load_session_for_subject("09042025", subject="BG_031")
    msg = str(exc.value)
    # Clear, not a bare traceback: names the subject and the session.
    assert "BG_031" in msg
    assert "09042025" in msg


def test_load_session_for_subject_loads_by_path(tmp_path, monkeypatch):
    monkeypatch.setattr(loader, "ROOT", str(tmp_path))
    path = _make_pkl(str(tmp_path), "BG_031", "09042025")
    captured = {}

    def _fake_raw(p):
        captured["path"] = p
        return "SENTINEL_SESSION"

    # Patch the PATH-based core loader so no real pkl is unpickled.
    monkeypatch.setattr(loader, "_load_session_raw", _fake_raw)
    out = loader.load_session_for_subject("09042025", subject="BG_031")
    assert out == "SENTINEL_SESSION"
    assert captured["path"] == path
