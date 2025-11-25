from pathlib import Path
import numpy as np
from visdetect.core.legacy_io import Session, Trial, Cluster, save_session

# Determine repo root from this file's location
REPO = Path(__file__).resolve().parents[1]

# Synthesize a simple session
n_trials = 60
# Even spacing of baseline onsets
baseline_times = np.arange(n_trials, dtype=float) * 2.0
# Change_ON occurs 0.5s after baseline
change_times = baseline_times + 0.5

# Cycle outcomes for variety
outcomes_cycle = ["Hit", "Miss", "FA", "Abort"]
trials = []
for i in range(n_trials):
    o = outcomes_cycle[i % len(outcomes_cycle)]
    # Simple reaction time dicts
    if o == "Hit":
        rt = {"RT": 0.3}
    elif o == "Miss":
        rt = {"Miss": 0.6}
    elif o == "FA":
        rt = {"FA": 0.25}
    else:
        rt = {}
    # store change_time relative to baseline for robustness
    trials.append(Trial(trialoutcome=o, reactiontimes=rt, change_time=0.5))

# Build a spike train with a modest bump after Baseline_ON
rng = np.random.default_rng(42)
window = (-0.5, 1.0)
rate_baseline = 7.0  # Hz
rate_bump = 18.0     # Hz during [0, 0.2] after event
spike_times = []
for et in baseline_times:
    # baseline Poisson spikes in window
    dur = window[1] - window[0]
    n_base = rng.poisson(rate_baseline * dur)
    if n_base > 0:
        spike_times.extend(et + rng.uniform(window[0], window[1], size=n_base))
    # short bump post event
    n_bump = rng.poisson(rate_bump * 0.2)
    if n_bump > 0:
        spike_times.extend(et + rng.uniform(0.0, 0.2, size=n_bump))

spike_times = np.sort(np.asarray(spike_times, dtype=float))
cluster = Cluster(cluster_id=1, spike_times=spike_times)

ni_events = {
    "Baseline_ON": baseline_times.copy(),
    "Change_ON": change_times.copy(),
}

sess = Session(
    trials=trials,
    clusters=[cluster],
    subject="SYN",
    session_name="demo",
    good_cluster_ids=[1],
    ni_events=ni_events,
)

# Save to tmp folder inside repo
out_dir = REPO / "tmp_demo_qc"
out_dir.mkdir(parents=True, exist_ok=True)
pkl_path = out_dir / "synthetic_session.pkl"
save_session(sess, str(pkl_path))
print(f"WROTE SYNTHETIC SESSION: {pkl_path}")
