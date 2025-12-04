import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import time

def find_tf_responsive_units_activity(
    session,
    fast_pulse_thresh=0.25,
    pre_window=[-0.4, 0],
    post_window=[0, 0.5],
    dt=0.001,  # 1 ms bins
    # min_fast_pulses=20,
    # min_spikes_per_window=1,
    z_thresh=4.0,
    sigma_ms=13.3,
    max_fast_pulses=None
):
    print("[INFO] Starting find_tf_responsive_units_activity()")
    start_time = time.time()
    npx_probes = session['NPX_probes']
    trials = session['behav_data']['trials_data_exp']
    print(f"[INFO] Number of trials: {len(trials)}")
    spike_times = npx_probes['st']
    cluster_ids = npx_probes['clu']
    good_clusters = npx_probes.get('cluster_id_KS_good', np.unique(cluster_ids))
    print(f"[INFO] Number of clusters: {len(np.unique(cluster_ids))}, Good clusters: {len(good_clusters)}")
    ni_events = session['NI_events']

    # --- Collect all valid fast pulse times across all trials ---
    all_fast_pulse_times = []
    for trial_idx, trial in enumerate(trials):
        TF_vec_full = np.array(trial['St1TrialVector'])
        TF_vec = TF_vec_full[::3]
        # Get Baseline_ON for this trial
        if 'Baseline_ON' in ni_events:
            baseline_on = ni_events['Baseline_ON']
            if isinstance(baseline_on, dict) and 'rise_t' in baseline_on:
                baseline_on_times = np.array(baseline_on['rise_t']).flatten()
            else:
                baseline_on_times = np.array(baseline_on).flatten()
            t0 = baseline_on_times[trial_idx]
        else:
            print(f"[WARN] Skipping trial {trial_idx}: Baseline_ON not found.")
            continue
        # Get Change_ON for this trial (if available)
        if 'Change_ON' in ni_events:
            change_on = ni_events['Change_ON']
            if isinstance(change_on, dict) and 'rise_t' in change_on:
                change_on_times = np.array(change_on['rise_t']).flatten()
            else:
                change_on_times = np.array(change_on).flatten()
            t_change = change_on_times[trial_idx] if trial_idx < len(change_on_times) else None
        else:
            t_change = None
        # Get outcome and outcome time for this trial
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

        log2_TF = np.log2(TF_vec)
        fast_pulse_bins = np.where(log2_TF >= fast_pulse_thresh)[0]
        fast_pulse_times = (fast_pulse_bins * 0.05) + t0
        # Apply filtering conditions
        valid_fast_pulse_times = []
        for fp_time in fast_pulse_times:
            if fp_time < t0 + 1.0:
                print(f"[DEBUG] Skipping fast pulse at {fp_time:.3f}s (before t0+1s)")
                continue
            if t_change is not None:
                if fp_time > t_change - 1.0:
                    print(f"[DEBUG] Skipping fast pulse at {fp_time:.3f}s (after t_change-1s)")
                    continue
            if outcome in ['FA', 'abort'] and t_outcome is not None:
                if fp_time > t_outcome - 2.0:
                    print(f"[DEBUG] Skipping fast pulse at {fp_time:.3f}s (after t_outcome-2s)")
                    continue
            valid_fast_pulse_times.append(fp_time)
        print(f"[INFO] Trial {trial_idx}: {len(valid_fast_pulse_times)} valid fast pulses")
        all_fast_pulse_times.extend(valid_fast_pulse_times)
    all_fast_pulse_times = np.array(all_fast_pulse_times)
    print(f"[INFO] Total valid fast pulses collected: {len(all_fast_pulse_times)}")
    if max_fast_pulses is not None and len(all_fast_pulse_times) > max_fast_pulses:
        rng = np.random.default_rng(seed=42)  # for reproducibility
        selected_idx = rng.choice(len(all_fast_pulse_times), size=max_fast_pulses, replace=False)
        all_fast_pulse_times = all_fast_pulse_times[selected_idx]
        all_fast_pulse_times = np.sort(all_fast_pulse_times)
        print(f"[INFO] Random subsampling: using {max_fast_pulses} randomly selected fast pulses for screening.")
    print(f"[INFO] Total valid fast pulses in session: {len(all_fast_pulse_times)}")
    print(f"[INFO] Total clusters to check: {len(good_clusters)}")

    tf_responsive_clusters = set()
    z_scores = []

    # Window setup
    full_window = [pre_window[0], post_window[1]]  # e.g. -0.4 to 0.5
    t_vec = np.arange(full_window[0], full_window[1], dt)
    sigma = sigma_ms / 1000 / dt  # convert ms to bins

    for i, clu in enumerate(good_clusters):
        if i % 10 == 0:
            elapsed = time.time() - start_time
            print(f"[INFO] Processing cluster {i+1}/{len(good_clusters)} (clu={clu})... Elapsed: {elapsed:.1f}s")
        spike_times_clu = spike_times[cluster_ids == clu]
        responsive = False
        for t_fp in all_fast_pulse_times:
            aligned_spikes = spike_times_clu - t_fp
            mask = (aligned_spikes >= full_window[0]) & (aligned_spikes < full_window[1])
            spikes_in_window = aligned_spikes[mask]
            # Kernel density estimate (KDE) instead of binarization
            smooth_activity = spike_density_estimate(spikes_in_window, t_vec, sigma)
            # Pre-pulse indices
            pre_mask = (t_vec >= pre_window[0]) & (t_vec < pre_window[1])
            # Mean and std in pre-pulse
            mean_pre = np.mean(smooth_activity[pre_mask])
            std_pre = np.std(smooth_activity[pre_mask])
            # Z-score
            if std_pre > 0:
                z = (smooth_activity - mean_pre) / std_pre
                if (z[400:] >= z_thresh).any():  # Check if any z-score after pre-window exceeds threshold
                    print(f"[INFO] Cluster {clu} is TF-responsive (z >= {z_thresh}) at pulse {t_fp:.3f}s")
                    tf_responsive_clusters.add(clu)
                    z_scores.append(z)
                    responsive = True
                    break  # Only need to be responsive for one pulse
        if not responsive:
            print(f"[DEBUG] Cluster {clu} is NOT TF-responsive.")
        # Optionally, collect z-scores for all pulses/clusters if needed

    tf_responsive_clusters = list(tf_responsive_clusters)
    print(f"[RESULT] Done. Total TF-responsive clusters: {len(tf_responsive_clusters)}")
    print(f"[RESULT] Total elapsed time: {time.time() - start_time:.1f}s")
    return tf_responsive_clusters, z_scores, all_fast_pulse_times

def spike_density_estimate(spike_times, t_vec, sigma):
    """
    Kernel density estimate of firing rate from spike times.
    Each spike is convolved with a Gaussian kernel (width=sigma, in bins).
    Returns firing rate in Hz (spikes/sec).
    """
    rate = np.zeros_like(t_vec)
    for t in spike_times:
        rate += np.exp(-0.5 * ((t_vec - t) / sigma) ** 2)
    # Normalize: area under each Gaussian = 1, so scale to Hz
    norm = 1.0 / (sigma * np.sqrt(2 * np.pi))
    rate *= norm
    return rate

def plot_tf_responsive_psths_activity(
    session,
    tf_responsive_clusters,
    all_fast_pulse_times,
    pre_window=[-0.4, 0],
    post_window=[0, 0.5],
    dt=0.001,
    sigma_ms=40,
    min_mean_firing_rate=0.01
):
    print("[INFO] Starting plot_tf_responsive_psths_activity()")
    npx_probes = session['NPX_probes']
    spike_times = npx_probes['st']
    cluster_ids = npx_probes['clu']
    full_window = [pre_window[0], post_window[1]]
    t_vec = np.arange(full_window[0], full_window[1], dt)
    sigma = sigma_ms / 1000 / dt

    filtered_clusters = []
    filtered_psths = []
    for clu in tf_responsive_clusters:
        spike_times_clu = spike_times[cluster_ids == clu]
        all_smooth = []
        for t_fp in all_fast_pulse_times:
            aligned_spikes = spike_times_clu - t_fp
            mask = (aligned_spikes >= full_window[0]) & (aligned_spikes < full_window[1])
            spikes_in_window = aligned_spikes[mask]
            spike_train = np.zeros_like(t_vec)
            spike_indices = np.searchsorted(t_vec, spikes_in_window)
            spike_indices = spike_indices[(spike_indices >= 0) & (spike_indices < len(spike_train))]
            spike_train[spike_indices] = 1
            smooth_activity = gaussian_filter1d(spike_train, sigma=sigma)
            all_smooth.append(smooth_activity)
        if len(all_smooth) > 0:
            mean_smooth = np.mean(all_smooth, axis=0)
            mean_rate = np.mean(mean_smooth)
            print(f"[INFO] Cluster {clu}: mean firing rate {mean_rate:.4f}")
            if mean_rate > min_mean_firing_rate:
                filtered_clusters.append(clu)
                filtered_psths.append(mean_smooth)
            else:
                print(f"[DEBUG] Cluster {clu} excluded (mean firing rate below threshold)")
    n_plot = len(filtered_clusters)
    print(f"[INFO] Plotting {n_plot} clusters (min_mean_firing_rate={min_mean_firing_rate})")
    n_cols = 5
    n_rows = int(np.ceil(n_plot / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 1.8 * n_rows), sharex=True, sharey=True)
    axes = axes.flatten()
    for idx, (clu, psth) in enumerate(zip(filtered_clusters, filtered_psths)):
        ax = axes[idx]
        ax.plot(t_vec, psth, color='k')
        ax.axvline(0, color='r', linestyle='--', lw=0.8)
        ax.set_ylim(bottom=0)
        ax.set_title(f'Clu {clu}', fontsize=9)
        if idx % n_cols == 0:
            ax.set_ylabel('Smoothed activity')
        if idx // n_cols == n_rows - 1:
            ax.set_xlabel('Time from fast pulse (s)')
    for ax in axes[n_plot:]:
        ax.axis('off')
    plt.tight_layout(h_pad=0.2, w_pad=0.05)
    plt.show()
    print("[INFO] Finished plotting PSTHs.")