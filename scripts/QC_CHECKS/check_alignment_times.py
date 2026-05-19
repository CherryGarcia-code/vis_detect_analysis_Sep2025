"""Check alignment times for a session (prints NI events and per-trial computed times).

Run with the repo conda environment (we expect to run via the conda run wrapper).
"""
import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parents[2]

from visdetect.core.session import load_session
from visdetect.analysis import align as align_mod
import numpy as np

p = repo_root / 'data' / 'BG_046_15082025.pkl'
print('Loading:', p)
s = load_session(str(p))
print('\nSession summary:')
print(' subject:', s.subject)
print(' session_name:', s.session_name)
print(' n_trials:', len(s.trials))
print(' n_clusters:', len(s.clusters))
print('\nNI event keys:', list(s.ni_events.keys()))

for k in ['Baseline_ON', 'Change_ON']:
    arr = s.ni_events.get(k, None)
    if arr is None:
        print(k, '-> None')
    else:
        arr = np.asarray(arr).flatten()
        print(f"{k}: count={arr.size}, min={arr.min() if arr.size else None}, max={arr.max() if arr.size else None}")
        print(' first10:', arr[:10])

print('\nFirst 12 trials:')
for i,t in enumerate(s.trials[:12]):
    print(f' trial {i}: outcome={t.trialoutcome} change_time_field={t.change_time} reactiontimes={t.reactiontimes}')
    crt = align_mod.compute_true_reaction_time(t, s.ni_events, i)
    print('   compute_true_reaction_time ->', crt)

# show get_event_times variations
for ev in ['Change_ON','Baseline_ON','Hit','Miss','FA']:
    et = align_mod.get_event_times(s, ev)
    print(f"get_event_times('{ev}') -> n={len(et)} sample(10):", et[:10])

# show example spike times around first Change_ON for cluster 0 if present
if len(s.clusters)>0:
    st = np.asarray(s.clusters[0].spike_times).flatten()
    print('\ncluster 0 spikes sample (first 30):', st[:30])
    # if there is a first change time, show aligned spikes relative to it
    if 'Change_ON' in s.ni_events and len(s.ni_events['Change_ON'])>0:
        t0 = float(np.asarray(s.ni_events['Change_ON']).flatten()[0])
        rel = st - t0
        print('cluster 0 spikes relative to first Change_ON (first 30):', rel[:30])

print('\nDone.')
