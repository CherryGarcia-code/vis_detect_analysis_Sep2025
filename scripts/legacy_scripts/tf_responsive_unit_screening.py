import numpy as np
import matplotlib.pyplot as plt
import time

def find_tf_responsive_units(session, fast_pulse_thresh=0.25, window=[-0.5, 0.75], bin_size=0.025, min_fast_pulses=20, min_spikes_per_window=1, ref_window=[-0.4, 0], resp_window=[0, 0.5], z_thresh=0.5, max_fast_pulses=None):
    """
    Identify TF-responsive units by z-scoring activity in 500ms post fast pulse to 400ms pre fast pulse.
    Returns (responsive cluster IDs, their z-scores, and the fast pulse times used).
    If max_fast_pulses is set, only the first N fast pulses are used (for fast testing/debugging).
    """
    start_time = time.time()
    npx_probes = session['NPX_probes']
    trials = session['behav_data']['trials_data_exp']
    spike_times = npx_probes['st']
    cluster_ids = npx_probes['clu']
    good_clusters = npx_probes.get('cluster_id_KS_good', np.unique(cluster_ids))
    ni_events = session['NI_events']
    # --- Collect all valid fast pulse times across all trials ---
    all_fast_pulse_times = []
    for trial_idx, trial in enumerate(trials):
        TF_vec_full = np.array(trial['St1TrialVector'])
        TF_vec = TF_vec_full[::3]
        # print('vector:',(TF_vec))
        # print('length of vector:',len(TF_vec))
        # print('TF vector is:',TF_vec)
        # Get Baseline_ON for this trial
        if 'Baseline_ON' in ni_events:
            baseline_on = ni_events['Baseline_ON']
            if isinstance(baseline_on, dict) and 'rise_t' in baseline_on:
                baseline_on_times = np.array(baseline_on['rise_t']).flatten()
            else:
                baseline_on_times = np.array(baseline_on).flatten()
            t0 = baseline_on_times[trial_idx]
        else:
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
        print('outcome:', outcome)
        reactiontimes = trial.get('reactiontimes', {})
        if outcome in ['FA', 'abort']:
            t_outcome = reactiontimes.get(outcome, np.nan)
            if not np.isnan(t_outcome):
                t_outcome = t0 + t_outcome
            else:
                t_outcome = None
        else:
            t_outcome = None
        print('outcome time:',t_outcome)
        print('t0:',t0)


        log2_TF = np.log2(TF_vec)
        # print('log2_TF:', log2_TF)
        fast_pulse_bins = np.where(log2_TF > fast_pulse_thresh)[0]
        # print the actual values of the fast pulses that passed the threshold:
        # print(log2_TF[fast_pulse_bins])
        # print('fast pulses found:', fast_pulse_bins)
        # print('fast pulse bins', fast_pulse_bins)
        fast_pulse_times = (fast_pulse_bins * 0.05) + t0
        # print('fast pulse times', fast_pulse_times)
        # Apply filtering conditions
        valid_fast_pulse_times = []
        for fp_time in fast_pulse_times:
            # Condition c: at least 1s after Baseline_ON
            if fp_time < t0 + 1.0:
                continue
            # Condition a: up to 1s before Change_ON (if Change_ON exists)
            if t_change is not None:
                if fp_time > t_change - 1.0:
                    print('skip change')
                    continue
            # Condition b: for FA/abort, up to 2s before outcome time (if no Change_ON)
            if outcome in ['FA', 'abort'] and t_outcome is not None:
                print('ch: ', t_change)

                if fp_time > t_outcome - 2.0:
                    print('skip EL')
                    continue
            valid_fast_pulse_times.append(fp_time)
        
        print('fast pulse times for trial' , trial_idx, valid_fast_pulse_times)
        # print('fast pulse times:' ,len(fast_pulse_times))
        all_fast_pulse_times.extend(valid_fast_pulse_times)
    all_fast_pulse_times = np.array(all_fast_pulse_times)
    if max_fast_pulses is not None and len(all_fast_pulse_times) > max_fast_pulses:
        rng = np.random.default_rng(seed=42)  # for reproducibility
        selected_idx = rng.choice(len(all_fast_pulse_times), size=max_fast_pulses, replace=False)
        all_fast_pulse_times = all_fast_pulse_times[selected_idx]
        all_fast_pulse_times = np.sort(all_fast_pulse_times)
        print(f"Random subsampling: using {max_fast_pulses} randomly selected fast pulses for screening.")
    print(f"Total valid fast pulses in session: {len(all_fast_pulse_times)}")
    print(f"Total clusters to check: {len(good_clusters)}")
    tf_responsive_clusters = []

    z_scores = []
    for i, clu in enumerate(good_clusters):
        if i % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Processing cluster {i+1}/{len(good_clusters)} (clu={clu})... Elapsed: {elapsed:.1f}s")
        spike_times_clu = spike_times[cluster_ids == clu]
        peri_spike_times = []
        for t_fp in all_fast_pulse_times:
            aligned = spike_times_clu - t_fp
            peri_spike_times.append(aligned[(aligned >= window[0]) & (aligned <= window[1])])
        n_fast_pulses_with_spikes = sum(len(s) >= min_spikes_per_window for s in peri_spike_times)
        if n_fast_pulses_with_spikes < min_fast_pulses:
            continue
        # For each fast pulse, count spikes in ref and resp windows
        ref_counts = []
        resp_counts = []
        for aligned in peri_spike_times:
            ref_count = np.sum((aligned >= ref_window[0]) & (aligned < ref_window[1]))
            resp_count = np.sum((aligned >= resp_window[0]) & (aligned < resp_window[1]))
            ref_counts.append(ref_count)
            resp_counts.append(resp_count)
        ref_counts = np.array(ref_counts)
        resp_counts = np.array(resp_counts)
        ref_mean = np.mean(ref_counts)
        ref_std = np.std(ref_counts) 
        resp_mean = np.mean(resp_counts)

        # print(resp_count, resp_mean)
        z = (resp_mean - ref_mean) / ref_std
        if np.abs(z) >= z_thresh:
            tf_responsive_clusters.append(clu)
            z_scores.append(z)
    print(f"Done. Total TF-responsive clusters: {len(tf_responsive_clusters)}")
    print(f"Total elapsed time: {time.time() - start_time:.1f}s")
    return tf_responsive_clusters, z_scores, all_fast_pulse_times

def plot_tf_responsive_psths(session, tf_responsive_clusters, all_fast_pulse_times, window=[-0.5, 0.5], bin_size=0.025, min_mean_firing_rate=0.01):
    npx_probes = session['NPX_probes']
    spike_times = npx_probes['st']
    cluster_ids = npx_probes['clu']
    bins = np.arange(window[0], window[1] + bin_size, bin_size)
    filtered_clusters = []
    filtered_psths = []
    for clu in tf_responsive_clusters:
        spike_times_clu = spike_times[cluster_ids == clu]
        peri_spike_times = []
        for t_fp in all_fast_pulse_times:
            aligned = spike_times_clu - t_fp
            peri_spike_times.append(aligned[(aligned >= window[0]) & (aligned <= window[1])])
        all_aligned = np.concatenate(peri_spike_times) if len(peri_spike_times) > 0 else np.array([])
        counts, _ = np.histogram(all_aligned, bins=bins)
        psth = counts / (len(all_fast_pulse_times) * bin_size) if len(all_fast_pulse_times) > 0 else np.zeros(len(bins)-1)
        mean_rate = np.mean(psth)
        if mean_rate > min_mean_firing_rate:
            filtered_clusters.append(clu)
            filtered_psths.append(psth)
    n_plot = len(filtered_clusters)
    n_cols = 5
    n_rows = int(np.ceil(n_plot / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 1.8 * n_rows), sharex=True, sharey=True)
    axes = axes.flatten()
    for idx, (clu, psth) in enumerate(zip(filtered_clusters, filtered_psths)):
        ax = axes[idx]
        ax.step(bins[:-1], psth, where='post', color='k')
        ax.axvline(0, color='r', linestyle='--', lw=0.8)
        ax.set_ylim(bottom=0)
        ax.set_title(f'Clu {clu}', fontsize=9)
        if idx % n_cols == 0:
            ax.set_ylabel('Firing rate (Hz)')
        if idx // n_cols == n_rows - 1:
            ax.set_xlabel('Time from fast pulse (s)')
    for ax in axes[n_plot:]:
        ax.axis('off')
    plt.tight_layout(h_pad=0.2, w_pad=0.05)
    plt.show()

# Example usage (to be run in a notebook cell):
# from tf_responsive_unit_screening import find_tf_responsive_units, plot_tf_responsive_psths
# tf_units, z_scores, fast_pulse_times = find_tf_responsive_units(session)
# if len(tf_units) > 0:
#     plot_tf_responsive_psths(session, tf_units, fast_pulse_times)
