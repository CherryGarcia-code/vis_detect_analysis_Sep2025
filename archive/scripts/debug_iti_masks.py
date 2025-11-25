import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from session_io import load_session
from unit_tracking import extract_iti_spikes

session = load_session(Path(__file__).parent.parent / "data" / "BG_046_02072025.pkl")
print(f'Session has {len(session.clusters)} clusters')
print(f'Good clusters: {len(session.good_cluster_ids)}')

# Extract ITI masks
iti_masks = extract_iti_spikes(session, method='trial_field', min_iti_duration=0.5)
print(f'\nITI masks created: {len(iti_masks)}')
print(f'Good clusters: {len(session.good_cluster_ids)}')
print(f'Mismatch: {len(session.good_cluster_ids) - len(iti_masks)} clusters missing ITI masks')

# Check which good clusters don't have ITI masks
missing = [cid for cid in session.good_cluster_ids if cid not in iti_masks]
print(f'\nGood clusters without ITI masks: {len(missing)}')
if len(missing) > 0:
    print(f'First 10 missing: {missing[:10]}')
    
# Check if any clusters have empty masks
empty_masks = [cid for cid, mask in iti_masks.items() if mask.sum() == 0]
print(f'\nClusters with empty ITI masks (no ITI spikes): {len(empty_masks)}')
if len(empty_masks) > 0:
    print(f'First 10 empty: {empty_masks[:10]}')
