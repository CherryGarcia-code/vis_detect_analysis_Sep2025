import sys
from pathlib import Path

from visdetect.core.session import load_session, Session
from visdetect.analysis import align
import numpy as np
import matplotlib.pyplot as plt

pkl_path = "data/pkls/BG_046/BG_046_01072025.pkl"
session = load_session(pkl_path)

# Get absolute times
# Change ON (Scheduled)
change_times = align.get_event_times_by_trial(session, 'Change_ON')
# FA Lick Times
fa_times = align.get_event_times_by_trial(session, 'FA') # This returns NaNs for non-FAs

deltas = []
for i in range(len(session.trials)):
    if i >= len(change_times) or i >= len(fa_times): break
    
    t_ch = change_times[i]
    t_fa = fa_times[i]
    
    # We only care if it's an FA trial
    if not np.isnan(t_fa):
        # We need the scheduled change time for this Specific FA trial
        # But wait, does 'Change_ON' exist for FA trials in the NI data?
        # Usually checking NI 'Change_ON' events might strictly be Actual Changes.
        # But our `get_event_times_by_trial` function tries to fill from `trial.change_time`.
        # Let's see if we have valid change times for FAs.
        
        if not np.isnan(t_ch):
            deltas.append(t_fa - t_ch)

print(f"Found {len(deltas)} FA trials with both Lick Time and Scheduled Change Time.")
if deltas:
    deltas = np.array(deltas)
    print(f"Mean (Lick - Change): {np.mean(deltas):.4f} s")
    print(f"Median: {np.median(deltas):.4f} s")
    print(f"Min: {np.min(deltas):.4f} s")
    print(f"Max: {np.max(deltas):.4f} s")
    print(f"Std: {np.std(deltas):.4f} s")
    
    # Histogram
    # plt.hist(deltas, bins=20)
    # plt.savefig('fa_timing_check.png')
