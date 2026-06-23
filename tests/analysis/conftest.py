import os, sys
import pytest, pandas as pd
from visdetect.utils.synthetic import make_synthetic_session

# ── Make the decision-latents script helpers importable by their bare module
# name (e.g. ``from _recovery_fixtures import make_recovery_design``). The Task
# 3.0 recovery fixtures live there (reusable builder), not under ``src`` —
# this path-insert lets both these conftest fixtures and the test module import
# them without a package install. ──
_DL_SCRIPTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "scripts", "analysis", "decision_latents",
)
if _DL_SCRIPTS not in sys.path:
    sys.path.insert(0, _DL_SCRIPTS)


@pytest.fixture
def synth_session():
    return make_synthetic_session(n_trials=40, n_clusters=2, seed=0)


@pytest.fixture
def synth_state_labels():
    # alternate Impulsive/StimSens, with a couple Disengaged/Abort to exercise filtering
    labels = []
    for i in range(40):
        m = ["Impulsive", "StimSens"][i % 2]
        if i in (5, 6): m = "Disengaged"
        if i in (7,):   m = "Abort"
        labels.append({"trial_idx": i, "state_label": m, "state_confidence": 0.9})
    return pd.DataFrame(labels).set_index("trial_idx")[["state_label", "state_confidence"]]


# ── Task 3.0: shared recovery fixtures (expert-like & naive-like ground truth) ──
# These feed every recovery test (3.2-3.5). Each is the (Design, true_theta,
# ParamSpec) triple from the reusable builder in
# scripts/analysis/decision_latents/_recovery_fixtures.py.
@pytest.fixture(scope="session")
def recovery_design_expert():
    """Expert-like recovery ground truth: change-driven licks (high v, low z)."""
    from _recovery_fixtures import make_recovery_design
    return make_recovery_design("expert", n_trials=2000, seed=0)


@pytest.fixture(scope="session")
def recovery_design_naive():
    """Naive-like recovery ground truth: flat-evidence hair-trigger licks (low v, high z)."""
    from _recovery_fixtures import make_recovery_design
    return make_recovery_design("naive", n_trials=2000, seed=0)
