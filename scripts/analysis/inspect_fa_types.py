import sys
from pathlib import Path

from visdetect.core.session import load_session
import numpy as np
import pandas as pd

pkl_path = "data/pkls/BG_046/BG_046_01072025.pkl"
session = load_session(pkl_path)

print(f"Total Trials: {len(session.trials)}")

fas = []
for i, t in enumerate(session.trials):
    if t.trialoutcome == 'FA':
        fas.append({
            'idx': i,
            'change_size': t.change_size,
            'change_time': t.change_time,
            'rt': t.reactiontimes.get('FA') if t.reactiontimes else None
        })

df = pd.DataFrame(fas)
if not df.empty:
    print(f"Total FAs: {len(df)}")
    print("Change Size counts for FAs:")
    print(df['change_size'].value_counts(dropna=False))
    
    # Check if RT is before Change Time
    # Assuming RT is from Baseline, and Change Time is from Baseline
    # Or RT is reaction time ? usually absolute?
    # inspection said: RT: {'FA': 9.044}, Change Time: 11.495.
    # If timings are from session start (absolute), then 9.044 < 11.495.
    # Let's check relation
    
    # We don't have absolute times here easily without NI events, but let's assume numeric comparison works
    # if both are floats.
    
    # Actually, let's just check if change_size is None or 0 for majority.
else:
    print("No FAs found.")
