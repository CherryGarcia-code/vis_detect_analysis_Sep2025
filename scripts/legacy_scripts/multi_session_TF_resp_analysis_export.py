# Multi-Session Pulse Response Analysis
# This script is exported from the Jupyter notebook for sharing and review.

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 100

# Utility function to load analysis results
def load_analysis_results(session_id, results_dir='pkls'):
    pkl_path = os.path.join(results_dir, f'fast_slow_results_{session_id}.pkl')
    with open(pkl_path, 'rb') as f:
        results = pickle.load(f)
    print(f"Loaded results from {pkl_path}")
    return results

# Specify sessions and load data
session_ids = [
    'BG_031_26032025', 'BG_031_27022025'  # Add more session IDs as needed
]
os.chdir('E:\python_analysis\git_repos\vis_detect_analysis_Apr2023')
session_data = {}
for session_id in session_ids:
    results = load_analysis_results(session_id)
    session_data[session_id] = results

# Filtering and cluster selection functions
def select_clusters(z_fast, t_vec, threshold=4.0, window=[0, 0.5]):
    window_mask = (t_vec >= window[0]) & (t_vec < window[1])
    return [clu for clu, z in z_fast.items() if np.max(z[window_mask]) > threshold]

def filter_pre_pulse(z_fast, t_vec, clusters, pre_window=[-0.05, 0], pre_activity_threshold=2.0):
    pre_mask = (t_vec >= pre_window[0]) & (t_vec < pre_window[1])
    filtered = []
    for clu in clusters:
        pre_max = np.max(np.abs(z_fast[clu][pre_mask]))
        if pre_max < pre_activity_threshold:
            filtered.append(clu)
    return filtered

def calculate_z_difference(z_fast, z_slow):
    return {clu: np.abs(z_fast[clu] - z_slow[clu]) for clu in z_fast.keys()}

def filter_post_pulse(z_diff, t_vec, clusters, post_window=[0, 0.5], responsivity_threshold=5.0):
    post_mask = (t_vec >= post_window[0]) & (t_vec < post_window[1])
    final = []
    for clu in clusters:
        max_resp = np.max(z_diff[clu][post_mask])
        if max_resp > responsivity_threshold:
            final.append(clu)
    return final

# Process and filter clusters for each session
filtered_clusters_per_session = {}
z_diff_per_session = {}
for session_id, results in session_data.items():
    z_fast = results['z_fast']
    z_slow = results['z_slow']
    t_vec = results['t_vec']
    selected = select_clusters(z_fast, t_vec, threshold=3.0)
    z_diff = calculate_z_difference(z_fast, z_slow)
    selected = filter_post_pulse(z_diff, t_vec, selected)
    if 295 in selected:
        selected.remove(295)
    filtered_clusters_per_session[session_id] = selected
    z_diff_per_session[session_id] = z_diff
    print(f"Session {session_id}: {len(selected)} clusters selected.")

# Plotting functions and further analysis can be added here as needed.
# For full functionality, see the original notebook.
