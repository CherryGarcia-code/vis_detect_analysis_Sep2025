import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from visdetect.core.session import load_session
import numpy as np

pkl_path = "data/pkls/BG_046/BG_046_01072025.pkl"

try:
    s = load_session(pkl_path)
    print(f"Total Trials: {len(s.trials)}")
    
    hits = 0
    misses = 0
    fas = 0
    refs = 0
    aborts = 0
    
    for t in s.trials:
        if t.trialoutcome == 'Hit': hits += 1
        elif t.trialoutcome == 'Miss': misses += 1
        elif t.trialoutcome == 'FA': fas += 1
        elif t.trialoutcome == 'Ref': refs += 1
        elif t.trialoutcome == 'abort': aborts += 1
        
    print(f"Hits: {hits}")
    print(f"Misses: {misses}")
    print(f"FAs: {fas}")
    print(f"Refs: {refs}")
    print(f"Aborts: {aborts}")
    print(f"Sum (Hit+Miss): {hits+misses}")
    
    if s.ni_events:
        print("NI Events Counts:")
        for k, v in s.ni_events.items():
            if isinstance(v, (list, np.ndarray)):
                print(f"  {k}: {len(v)}")
            else:
                print(f"  {k}: {v} (scalar/other)")

except Exception as e:
    import traceback
    traceback.print_exc()
