import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

def collect_valid_pulses(session, fast_pulse_thresh=0.25, slow_pulse_thresh=0.0, pre_window=[-0.4, 0], post_window=[0, 0.5]):
    """
    Collect valid fast and slow pulses that the subject actually saw.
    Returns: fast_pulse_times, slow_pulse_times
    """
    trials = session['behav_data']['trials_data_exp']
    ni_events = session['NI_events']
    all_fast_pulse_times = []
    all_slow_pulse_times = []
    for trial_idx, trial in enumerate(trials):
        TF_vec_full = np.array(trial['St1TrialVector'])
        TF_vec = TF_vec_full[::3]
        # Only consider pulses up to the last seen value
        n_seen = trial.get('n_seen', len(TF_vec))  # fallback if not present
        TF_vec = TF_vec[:n_seen]
        if 'Baseline_ON' in ni_events:
            baseline_on = ni_events['Baseline_ON']
            if isinstance(baseline_on, dict) and 'rise_t' in baseline_on:
                baseline_on_times = np.array(baseline_on['rise_t']).flatten()
            else:
                baseline_on_times = np.array(baseline_on).flatten()
            t0 = baseline_on_times[trial_idx]
        else:
            continue
        log2_TF = np.log2(TF_vec)
        fast_pulse_bins = np.where(log2_TF >= fast_pulse_thresh)[0]
        slow_pulse_bins = np.where(log2_TF <= slow_pulse_thresh)[0]
        fast_pulse_times = (fast_pulse_bins * 0.05) + t0
        slow_pulse_times = (slow_pulse_bins * 0.05) + t0
        all_fast_pulse_times.extend(fast_pulse_times)
        all_slow_pulse_times.extend(slow_pulse_times)
    return np.array(all_fast_pulse_times), np.array(all_slow_pulse_times)

def mean_activity_per_neuron(session, pulse_times, pre_window, post_window, dt, sigma_ms):
    npx_probes = session['NPX_probes']
    spike_times = npx_probes['st']
    cluster_ids = npx_probes['clu']
    good_clusters = npx_probes.get('cluster_id_KS_good', np.unique(cluster_ids))
    full_window = [pre_window[0], post_window[1]]
    t_vec = np.arange(full_window[0], full_window[1], dt)
    sigma = sigma_ms / 1000 / dt
    mean_activities = {}
    for clu in good_clusters:
        spike_times_clu = spike_times[cluster_ids == clu]
        all_smooth = []
        for t_pulse in pulse_times:
            aligned_spikes = spike_times_clu - t_pulse
            mask = (aligned_spikes >= full_window[0]) & (aligned_spikes < full_window[1])
            spikes_in_window = aligned_spikes[mask]
            spike_train = np.zeros_like(t_vec)
            spike_indices = np.searchsorted(t_vec, spikes_in_window)
            spike_indices = spike_indices[(spike_indices >= 0) & (spike_indices < len(spike_train))]
            spike_train[spike_indices] = 1
            smooth_activity = gaussian_filter1d(spike_train, sigma=sigma)
            all_smooth.append(smooth_activity)
        if all_smooth:
            mean_activities[clu] = np.mean(all_smooth, axis=0)
        else:
            mean_activities[clu] = np.zeros_like(t_vec)
    return mean_activities, t_vec

def pre_pulse_stats(mean_activities, t_vec, pre_window):
    pre_mask = (t_vec >= pre_window[0]) & (t_vec < pre_window[1])
    stats = {}
    for clu, activity in mean_activities.items():
        pre_vals = activity[pre_mask]
        stats[clu] = {'mean': np.mean(pre_vals), 'std': np.std(pre_vals)}
    return stats

def zscore_activities(mean_activities, stats):
    z_activities = {}
    for clu, activity in mean_activities.items():
        mean = stats[clu]['mean']
        std = stats[clu]['std']
        if std > 0:
            z_activities[clu] = (activity - mean) / std
        else:
            z_activities[clu] = activity * 0
    return z_activities

def plot_fast_slow_psth(z_fast, z_slow, t_vec, clusters=None, n_cols=5):
    """
    Plot mean PSTH for each cluster, with fast and slow pulse responses.
    """
    if clusters is None:
        clusters = sorted(set(z_fast.keys()) & set(z_slow.keys()))
    n_plot = len(clusters)
    n_rows = int(np.ceil(n_plot / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 1.8 * n_rows), sharex=True, sharey=True)
    axes = axes.flatten()
    for idx, clu in enumerate(clusters):
        ax = axes[idx]
        ax.plot(t_vec, z_fast[clu], color='r', label='Fast')
        ax.plot(t_vec, z_slow[clu], color='b', label='Slow')
        ax.axvline(0, color='k', linestyle='--', lw=0.8)
        ax.set_ylim(bottom=0)
        ax.set_title(f'Clu {clu}', fontsize=9)
        if idx % n_cols == 0:
            ax.set_ylabel('Z-scored activity')
        if idx // n_cols == n_rows - 1:
            ax.set_xlabel('Time from pulse (s)')
        if idx == 0:
            ax.legend(fontsize=8)
    for ax in axes[n_plot:]:
        ax.axis('off')
    plt.tight_layout(h_pad=0.2, w_pad=0.05)
    plt.show()
