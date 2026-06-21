"""Tests for the Engine-A generative module (`decision_latents_generative`).

Task 0.9: the expert-anchor contingency GATE — `select_expert_anchors` returns
one of three modes (expert / pooled / fallback) given the Task-0.8 inventory.
"""
import numpy as np
import pandas as pd
import pytest

from visdetect.analysis import decision_latents_generative as dlg


def _inv(rows):
    """Build an inventory DataFrame from (session, dprime, n_impu, n_stim) tuples."""
    return pd.DataFrame(rows, columns=["session", "dprime", "n_impu", "n_stim"])


# ── Mode: expert ────────────────────────────────────────────────────────────
def test_expert_mode_three_qualifying():
    """>=3 qualifying sessions -> mode 'expert', anchors = the qualifying ids."""
    df = _inv([
        ("01072025", 1.5, 100, 100),   # qualifies
        ("02072025", 1.2, 50, 40),     # qualifies
        ("03072025", 0.9, 30, 25),     # qualifies
        ("04072025", 0.2, 5, 5),       # fails dprime + n
    ])
    out = dlg.select_expert_anchors(df, min_d=0.7, min_mood_n=20, min_anchors=3)
    assert out["mode"] == "expert"
    assert out["anchors"] == ["01072025", "02072025", "03072025"]


def test_expert_mode_preserves_inventory_order():
    """When the inventory is ordered (chronological), anchors keep that order."""
    df = _inv([
        ("05072025", 1.0, 50, 50),
        ("01072025", 1.0, 50, 50),
        ("03072025", 1.0, 50, 50),
    ])
    out = dlg.select_expert_anchors(df)
    assert out["mode"] == "expert"
    assert out["anchors"] == ["05072025", "01072025", "03072025"]  # inventory order kept


def test_qualification_requires_both_moods_above_min():
    """A high-d' session with one mood below min_mood_n does NOT qualify."""
    df = _inv([
        ("01072025", 1.5, 100, 100),
        ("02072025", 1.5, 100, 100),
        ("03072025", 1.5, 100, 5),     # n_stim too low -> NOT qualifying
        ("04072025", 1.5, 5, 100),     # n_impu too low -> NOT qualifying
    ])
    out = dlg.select_expert_anchors(df, min_d=0.7, min_mood_n=20, min_anchors=3)
    # only 2 qualify -> cannot be expert
    assert out["mode"] != "expert"


# ── Mode: pooled ────────────────────────────────────────────────────────────
def test_pooled_mode_tops_up_to_min_anchors():
    """1 qualifying + extra sessions -> 'pooled' with exactly min_anchors anchors,
    topped up by the best remaining sessions (dprime desc, then recency)."""
    df = _inv([
        ("01072025", 1.5, 100, 100),   # qualifies
        ("02072025", 0.9, 100, 5),     # fails (n_stim) but high d'
        ("03072025", 0.8, 5, 100),     # fails (n_impu)
        ("04072025", 0.3, 5, 5),       # weak
    ])
    out = dlg.select_expert_anchors(df, min_d=0.7, min_mood_n=20, min_anchors=3)
    assert out["mode"] == "pooled"
    assert len(out["anchors"]) == 3
    # the sole qualifier must be in the pool
    assert "01072025" in out["anchors"]
    # top-ups are the two best remaining by dprime-desc -> 02072025 (0.9), 03072025 (0.8)
    assert "02072025" in out["anchors"]
    assert "03072025" in out["anchors"]
    assert "04072025" not in out["anchors"]


def test_pooled_mode_zero_qualifying_but_enough_sessions():
    """No qualifier but >= min_anchors sessions -> pooled, length min_anchors."""
    df = _inv([
        ("01072025", 0.6, 100, 5),
        ("02072025", 0.5, 5, 100),
        ("03072025", 0.4, 10, 10),
        ("04072025", 0.1, 1, 1),
    ])
    out = dlg.select_expert_anchors(df, min_d=0.7, min_mood_n=20, min_anchors=3)
    assert out["mode"] == "pooled"
    assert len(out["anchors"]) == 3
    # best three by dprime-desc
    assert out["anchors"] == ["01072025", "02072025", "03072025"]


# ── Mode: fallback ──────────────────────────────────────────────────────────
def test_fallback_mode_too_few_sessions():
    """Fewer than min_anchors sessions total and none qualify -> fallback."""
    df = _inv([
        ("01072025", 0.5, 10, 5),
        ("02072025", 0.3, 5, 5),
    ])
    out = dlg.select_expert_anchors(df, min_d=0.7, min_mood_n=20, min_anchors=3)
    assert out["mode"] == "fallback"
    assert len(out["anchors"]) < 3  # whatever qualifies (here, none)


def test_fallback_mode_empty_inventory():
    """Empty inventory -> fallback, empty anchors."""
    df = _inv([])
    out = dlg.select_expert_anchors(df)
    assert out["mode"] == "fallback"
    assert out["anchors"] == []


def test_fallback_returns_qualifying_when_present_but_below_min():
    """2 qualify, no other sessions to top up, min_anchors=3 -> fallback with the
    2 qualifiers (downstream ships proxies for the rest)."""
    df = _inv([
        ("01072025", 1.5, 100, 100),
        ("02072025", 1.2, 50, 50),
    ])
    out = dlg.select_expert_anchors(df, min_d=0.7, min_mood_n=20, min_anchors=3)
    assert out["mode"] == "fallback"
    assert out["anchors"] == ["01072025", "02072025"]


# ── Column-name robustness (real inventory uses `session_name`) ─────────────
def test_accepts_session_name_column():
    """Real Task-0.8 inventory uses `session_name`; the gate must accept it."""
    df = pd.DataFrame({
        "session_name": ["01072025", "02072025", "03072025"],
        "dprime": [1.5, 1.2, 0.9],
        "n_impu": [100, 50, 30],
        "n_stim": [100, 40, 25],
    })
    out = dlg.select_expert_anchors(df)
    assert out["mode"] == "expert"
    assert out["anchors"] == ["01072025", "02072025", "03072025"]


def test_int_session_ids_are_zfilled_to_8_digits():
    """CSV-read session ids come back as int64; the leading zero must be
    restored (`1072025` -> `01072025`) so downstream loaders find the session."""
    df = pd.DataFrame({
        "session_name": [1072025, 2072025, 3072025],   # int64, leading zero lost
        "dprime": [1.5, 1.2, 0.9],
        "n_impu": [100, 50, 30],
        "n_stim": [100, 40, 25],
    })
    out = dlg.select_expert_anchors(df)
    assert out["mode"] == "expert"
    assert out["anchors"] == ["01072025", "02072025", "03072025"]


# ── Real inventory smoke test ───────────────────────────────────────────────
def test_real_inventory_is_expert_with_30(real_inventory_csv):
    """On the real Task-0.8 inventory, the gate yields expert mode with 30 anchors."""
    df = pd.read_csv(real_inventory_csv)
    out = dlg.select_expert_anchors(df, min_d=0.7, min_mood_n=20, min_anchors=3)
    assert out["mode"] == "expert"
    assert len(out["anchors"]) == 30


@pytest.fixture
def real_inventory_csv():
    import os
    p = os.path.join("data", "cache", "decision_latents",
                     "b8p2_expert_anchor_inventory.csv")
    if not os.path.exists(p):
        pytest.skip(f"real inventory CSV not found at {p}")
    return p
