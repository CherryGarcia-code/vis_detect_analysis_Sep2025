from pathlib import Path
import numpy as np
import sys
repo_root = Path(__file__).resolve().parents[2]
from visdetect.core.session import load_session
p = repo_root / 'data' / 'BG_046_15082025.pkl'
s = load_session(str(p))
ch = np.asarray(s.ni_events.get('Change_ON', [])).flatten()
print('Change_ON length:', ch.size)
mask_nan = np.array([np.isnan(x) for x in ch])
print(' n_nan in Change_ON array:', int(mask_nan.sum()))
print(' n_trials in session:', len(s.trials))
from collections import Counter
nan_indices = np.where(mask_nan)[0]
outcomes = []
for idx in nan_indices:
    if idx < len(s.trials):
        outcomes.append(s.trials[idx].trialoutcome)
    else:
        outcomes.append('INDEX_OUT_OF_RANGE')
ctr = Counter(outcomes)
print('Outcomes for trials with NaN Change_ON (counts):')
for k,v in ctr.items():
    print(' ', k, v)
# Also counts for non-NaN
notnan_indices = np.where(~mask_nan)[0]
outcomes2 = []
for idx in notnan_indices:
    if idx < len(s.trials):
        outcomes2.append(s.trials[idx].trialoutcome)
    else:
        outcomes2.append('INDEX_OUT_OF_RANGE')
ctr2 = Counter(outcomes2)
print('\nOutcomes for trials with NON-NaN Change_ON (counts):')
for k,v in ctr2.items():
    print(' ', k, v)
