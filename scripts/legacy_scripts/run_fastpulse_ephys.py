import sys
import os
import numpy as np
from joblib import Parallel, delayed
from ephys_npxls_analysis_June02_2025 import load_mat_file, mat_struct_to_dict, get_session_dict, get_drilled_down, plot_pfpsth_raster_grid_sorted

# # Set up paths and parameters
# mat_path = 'BG_031_010525.mat'  # Update as needed
# data = load_mat_file(mat_path)
# data_dict1 = mat_struct_to_dict(data)
# data_dict = get_session_dict(data_dict1)
# session = get_drilled_down(data_dict, 'BG_031', 'BG_031_01052025')

# --- Parallelized version of the cluster computation ---
def compute_cluster_fastpulse(clu, spike_times, cluster_ids, all_fast_pulse_times, window, min_spikes_per_window, min_fast_pulses, bins, window_length):
    spike_times_clu = spike_times[cluster_ids == clu]
    peri_spike_times = [spike_times_clu - t_fp for t_fp in all_fast_pulse_times]
    peri_spike_times = [aligned[(aligned >= window[0]) & (aligned <= window[1])] for aligned in peri_spike_times]
    n_fast_pulses_with_spikes = sum(len(s) >= min_spikes_per_window for s in peri_spike_times)
    if n_fast_pulses_with_spikes > min_fast_pulses:
        all_aligned = np.concatenate(peri_spike_times) if len(peri_spike_times) > 0 else np.array([])
        counts, _ = np.histogram(all_aligned, bins=bins)
        mean_rate = np.sum(counts) / (len(all_fast_pulse_times) * window_length)
        return (clu, peri_spike_times, counts, mean_rate)
    else:
        return None

# Parameters (should match your notebook)
fast_pulse_thresh = 0.25
window = [-0.5, 0.5]
bin_size = 0.025
clusters_per_row = 10
min_fast_pulses = 20
min_spikes_per_window = 1
save_figure = True

npx_probes = session['NPX_probes']
trials, outcomes = session['behav_data']['trials_data_exp'], None
spike_times = npx_probes['st']
cluster_ids = npx_probes['clu']
good_clusters = npx_probes.get('cluster_id_KS_good', np.unique(cluster_ids))
ni_events = session['NI_events']

# --- Collect all valid fast pulse times across all trials (same as notebook logic) ---
all_fast_pulse_times = []
for trial_idx, trial in enumerate(trials):
    TF_vec_full = np.array(trial['St1TrialVector'])
    TF_vec = TF_vec_full[::3]
    if 'Baseline_ON' in ni_events:
        baseline_on = ni_events['Baseline_ON']
        if isinstance(baseline_on, dict) and 'rise_t' in baseline_on:
            baseline_on_times = np.array(baseline_on['rise_t']).flatten()
        else:
            baseline_on_times = np.array(baseline_on).flatten()
        t0 = baseline_on_times[trial_idx]
    else:
        continue
    if 'Change_ON' in ni_events:
        change_on = ni_events['Change_ON']
        if isinstance(change_on, dict) and 'rise_t' in change_on:
            change_on_times = np.array(change_on['rise_t']).flatten()
        else:
            change_on_times = np.array(change_on).flatten()
        t_change = change_on_times[trial_idx] if trial_idx < len(change_on_times) else None
    else:
        t_change = None
    outcome = trial['trialoutcome']
    reactiontimes = trial.get('reactiontimes', {})
    if outcome in ['FA', 'abort']:
        t_outcome = reactiontimes.get(outcome, np.nan)
        if not np.isnan(t_outcome):
            t_outcome = t0 + t_outcome
        else:
            t_outcome = None
    else:
        t_outcome = None
    log2_TF = np.log2(TF_vec + 1e-8)
    fast_pulse_bins = np.where(log2_TF > fast_pulse_thresh)[0]
    fast_pulse_times = fast_pulse_bins * 0.05 + t0
    valid_fast_pulse_times = []
    for fp_time in fast_pulse_times:
        if fp_time < t0 + 1.0:
            continue
        if t_change is not None:
            if fp_time > t_change - 1.0:
                continue
        elif outcome in ['FA', 'abort'] and t_outcome is not None:
            if fp_time > t_outcome - 2.0:
                continue
        valid_fast_pulse_times.append(fp_time)
    all_fast_pulse_times.extend(valid_fast_pulse_times)
all_fast_pulse_times = np.array(all_fast_pulse_times)
print(f"Total valid fast pulses in session: {len(all_fast_pulse_times)}")
bins = np.arange(window[0], window[1] + bin_size, bin_size)
window_length = window[1] - window[0]

# --- Parallel computation over clusters ---
results = Parallel(n_jobs=8, verbose=5)(
    delayed(compute_cluster_fastpulse)(
        clu, spike_times, cluster_ids, all_fast_pulse_times, window, min_spikes_per_window, min_fast_pulses, bins, window_length
    ) for clu in good_clusters
)
# Filter out None results
results = [r for r in results if r is not None]
if not results:
    print("No clusters passed the fast-pulse filter. Adjust your parameters.")
    sys.exit(0)
# Unpack results for plotting
clusters_to_plot, rasters_for_plot, counts_for_plot, mean_rates = zip(*results)
# Sort by mean_rates descending
sort_idx = np.argsort(mean_rates)[::-1]
clusters_to_plot = [clusters_to_plot[i] for i in sort_idx]
rasters_for_plot = [rasters_for_plot[i] for i in sort_idx]
counts_for_plot = [counts_for_plot[i] for i in sort_idx]
mean_rates = [mean_rates[i] for i in sort_idx]
# Plot using the original plotting code (reuse your notebook's plotting logic, or call the function with precomputed data if you refactor)
# For now, just print summary:
print(f"Finished parallel computation for {len(clusters_to_plot)} clusters.")
# Optionally, you can call your plotting function here if you refactor it to accept precomputed data.
# plot_pfpsth_raster_grid_sorted(session, ...)  # If you want to plot as before
