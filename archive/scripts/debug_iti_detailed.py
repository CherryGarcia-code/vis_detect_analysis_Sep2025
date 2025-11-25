import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from session_io import load_session
from unit_tracking import extract_iti_spikes
import numpy as np

session = load_session(Path(__file__).parent.parent / "data" / "BG_046_02072025.pkl")

print(f'Session has {len(session.trials)} trials')
print(f'Session has {len(session.clusters)} clusters')

# Extract ITI masks
iti_masks = extract_iti_spikes(session, method='trial_field', min_iti_duration=0.5)

print(f'\nITI masks created: {len(iti_masks)}')

# Check coverage
for cluster_id, mask in list(iti_masks.items())[:5]:
    n_spikes = len(mask)
    n_iti = mask.sum()
    pct = 100 * n_iti / n_spikes if n_spikes > 0 else 0
    print(f'Cluster {cluster_id}: {n_iti}/{n_spikes} spikes in ITI ({pct:.1f}%)')

# Check first cluster in detail
cluster_0 = session.clusters[0]
mask_0 = iti_masks[cluster_0.cluster_id]
iti_spike_times = cluster_0.spike_times[mask_0]

print(f'\nCluster {cluster_0.cluster_id} detail:')
print(f'  Total spikes: {len(cluster_0.spike_times)}')
print(f'  ITI spikes: {len(iti_spike_times)}')
if len(iti_spike_times) > 0:
    print(f'  ITI spike times (first 10): {iti_spike_times[:10]}')
    print(f'  ITI spike times range: [{iti_spike_times.min():.2f}s, {iti_spike_times.max():.2f}s]')

# Check trial ITI values
print(f'\n Trial ITI values (first 10):')
for i, trial in enumerate(session.trials[:10]):
    iti_val = getattr(trial, 'ITI', None)
    print(f'  Trial {i}: ITI = {iti_val}')

# Check ni_events
print(f'\nni_events keys: {list(session.ni_events.keys())}')
if 'Baseline_ON' in session.ni_events:
    baseline_on = session.ni_events['Baseline_ON']
    if isinstance(baseline_on, dict) and 'rise_t' in baseline_on:
        times = np.array(baseline_on['rise_t']).flatten()
    else:
        times = np.array(baseline_on).flatten()
    print(f'Baseline_ON times (first 10): {times[:10]}')
