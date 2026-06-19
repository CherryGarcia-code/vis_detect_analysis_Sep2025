import pytest, pandas as pd
from visdetect.utils.synthetic import make_synthetic_session


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
