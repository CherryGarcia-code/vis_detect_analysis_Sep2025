"""fa_lick is a canonical PSTH condition, centralized in PSTH_CONDITIONS so scripts
do not re-derive its event / outcome / window (change #1)."""
from visdetect.analysis.tracking_qc import PSTH_CONDITIONS, extract_unit_psths
from visdetect.analysis.constants import (
    EVENT_VALID_OUTCOMES, EVENT_RESPONSIVENESS_WINDOWS)
from visdetect.utils.synthetic import make_synthetic_session


def test_fa_lick_condition_registered():
    assert "fa_lick" in PSTH_CONDITIONS
    cfg = PSTH_CONDITIONS["fa_lick"]
    assert cfg["event"] == "FA"
    assert cfg["outcomes"] == EVENT_VALID_OUTCOMES["FA"]      # {"fa"}
    assert cfg["sizes"] is None


def test_fa_lick_window_spans_canonical_baseline():
    """The extraction window must contain the canonical FA baseline window so a
    baseline-subtracted FA PSTH can be taken from the condition alone."""
    (lo, hi), _ = EVENT_RESPONSIVENESS_WINDOWS["FA"]          # baseline (-1.75, -1.25)
    w0, w1 = PSTH_CONDITIONS["fa_lick"]["window"]
    assert w0 <= lo and hi <= w1


def test_extract_unit_psths_includes_fa_lick():
    """extract_unit_psths now emits an fa_lick entry (a (psth, centers, n) tuple)."""
    sess = make_synthetic_session(n_trials=60, n_clusters=3, seed=2)
    out = extract_unit_psths(sess, ks_unit_id=0)
    assert "fa_lick" in out
    assert len(out["fa_lick"]) == 3
