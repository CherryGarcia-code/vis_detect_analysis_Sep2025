"""Focused test for the B8 Phase 2 label-reliability confidence gate (Task 0.7).

Covers the pure seam `session_reliability_row`: the >=80%-of-trials-with-
confidence>0.7 rule and the mood-proportion accounting. Imports the script
module by path (it lives under scripts/, not the package).
"""
import importlib.util
from pathlib import Path

import pandas as pd

_SCRIPT = (Path(__file__).resolve().parents[2]
           / "scripts" / "analysis" / "decision_latents" / "_label_reliability.py")


def _load_script_module():
    spec = importlib.util.spec_from_file_location("_label_reliability", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _labels(confidences, moods):
    """Build a labels DataFrame like dl.load_state_labels returns."""
    return pd.DataFrame(
        {"state_label": moods, "state_confidence": confidences},
        index=pd.Index(range(len(moods)), name="trial_idx"),
    )


def test_reliable_when_at_least_80pct_above_threshold():
    mod = _load_script_module()
    # 8/10 trials with confidence > 0.7 -> exactly 80% -> reliable (>=).
    conf = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.5, 0.5]
    moods = ["Impulsive"] * 5 + ["StimSens"] * 3 + ["Disengaged", "Abort"]
    row = mod.session_reliability_row("01072025", _labels(conf, moods))
    assert row["naive_reliable"] is True
    assert abs(row["frac_conf_gt_0.7"] - 0.80) < 1e-9
    # Mood proportions sum to 1 and match the counts.
    assert abs(row["Impulsive"] - 0.5) < 1e-9
    assert abs(row["StimSens"] - 0.3) < 1e-9
    assert abs(row["Disengaged"] - 0.1) < 1e-9
    assert abs(row["Abort"] - 0.1) < 1e-9


def test_unreliable_when_below_80pct():
    mod = _load_script_module()
    # 7/10 trials above 0.7 -> 70% -> UNRELIABLE.
    conf = [0.9] * 7 + [0.5, 0.5, 0.5]
    moods = ["Impulsive"] * 10
    row = mod.session_reliability_row("02072025", _labels(conf, moods))
    assert row["naive_reliable"] is False
    assert abs(row["frac_conf_gt_0.7"] - 0.70) < 1e-9


def test_threshold_is_strict_greater_than():
    mod = _load_script_module()
    # Confidence exactly at 0.7 does NOT count (strict >), so all-0.7 -> 0% high.
    conf = [0.7] * 10
    moods = ["StimSens"] * 10
    row = mod.session_reliability_row("03072025", _labels(conf, moods))
    assert row["frac_conf_gt_0.7"] == 0.0
    assert row["naive_reliable"] is False
