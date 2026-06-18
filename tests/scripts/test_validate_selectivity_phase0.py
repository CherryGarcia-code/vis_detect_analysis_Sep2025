"""Smoke test for the selectivity early-validation gate's pure seam."""
import importlib.util
from pathlib import Path

import numpy as np

from visdetect.core.session import Session, Trial, Cluster
from visdetect.analysis.tf_selectivity import TFSelectivityConfig

_SCRIPT = (Path(__file__).resolve().parents[2]
           / "scripts" / "tf_responsiveness" / "validate_selectivity_phase0.py")


def _load_script_module():
    spec = importlib.util.spec_from_file_location("validate_selectivity_phase0", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tiny_both_pulse_session():
    base_on = (np.arange(20) * 300.0).astype(float)
    change_on = base_on + 250.0
    trials, fast_t = [], []
    for k in range(20):
        bv = np.ones(3 * 200)
        for j, idx in enumerate(range(40, 160, 10)):
            val = 2.0 if (j % 2 == 0) else 0.5
            bv[3 * idx] = val
            if val == 2.0:
                fast_t.append(base_on[k] + idx * 0.05)
        trials.append(Trial(trialoutcome="Hit", reactiontimes={"RT": 0.3},
                            change_size=2.0, change_time=250.0,
                            baseline_values=bv, n_seen=None))
    spikes = [np.arange(0.0, float(change_on[-1] + 10), 0.05)]
    for tp in fast_t:
        spikes.append(np.arange(tp + 0.005, tp + 0.155, 1.0 / 140.0))
    spikes = np.sort(np.concatenate(spikes))
    ni = {"Baseline_ON": base_on, "Change_ON": change_on}
    return Session(trials=trials, clusters=[Cluster(cluster_id=0, spike_times=spikes,
                   quality="good")], subject="SYN", session_name="SEL",
                   good_cluster_ids=[0], ni_events=ni)


def test_build_feature_table_runs():
    mod = _load_script_module()
    sess = _tiny_both_pulse_session()
    cfg = TFSelectivityConfig(n_shuffles=20)
    df = mod.build_feature_table(sess, [0], cfg)
    assert len(df) == 1
    assert {"cluster_id", "sel_peak", "shuffle_p"}.issubset(df.columns)
