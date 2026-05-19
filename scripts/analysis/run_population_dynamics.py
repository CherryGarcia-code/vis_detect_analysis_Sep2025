
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


from visdetect.core.session import load_session, Session
from visdetect.analysis import align
from visdetect.analysis import behavior

def run_population_analysis(session_path: str, output_dir: str):
    """
    Perform population coding direction analysis (Hit vs Miss).
    Uses visdetect.analysis.align for alignment.
    Separates traces by behavioral state (Balanced, Impulsive, Disengaged).
    """
    print(f"Loading session: {session_path}")
    session = load_session(session_path)
    
    # 1. Compute Behavioral States
    print("Computing behavioral states...")
    df_beh = behavior.compute_rolling_performance(session)
    if df_beh.empty:
        print("Behavior dataframe empty.")
        return

    # 2. Get Event Times (Change_ON) — only hit/miss trials are valid
    #    (FA/abort trials never saw the change stimulus)
    change_times_all = align.get_event_times_by_trial(session, 'Change_ON')

    # Define Groups based on State + Outcome
    # Only Hit and Miss trials have valid Change_ON times
    groups = {
        'Balanced_Hit': [],
        'Balanced_Miss': [],
        'Impulsive_Hit': [],
        'Impulsive_Miss': [],
        'Disengaged_Miss': [],
        'Disengaged_Hit': []
    }
    
    for idx, row in df_beh.iterrows():
        if idx >= len(change_times_all): break
        
        ct = change_times_all[idx]
        if np.isnan(ct): continue
        
        state = row['state'] # impulsive, disengaged, balanced
        outcome = row['outcome'] # hit, miss, fa, etc.
        
        key = f"{state.capitalize()}_{outcome.capitalize()}"
        if key in groups:
            groups[key].append(ct)
            
    # Print counts
    for k, v in groups.items():
        print(f"  {k}: {len(v)} trials")

    # Check Requirements for CD (Balanced Hit & Miss)
    if len(groups['Balanced_Hit']) < 5 or len(groups['Balanced_Miss']) < 5:
        print("Not enough 'Balanced' trials to define Coding Direction.")
        # Fallback? Maybe use all Hits/Misses if balanced count is low?
        # For now, simplistic fallback to all if specialized fails, or just return.
        if len(groups['Balanced_Hit']) < 5 and len(groups['Impulsive_Hit']) > 5:
             print("WARNING: Using Impulsive Hits for CD definition due to lack of Balanced Hits.")
             groups['Balanced_Hit'] = groups['Impulsive_Hit'] # Fallback hack for robust plotting
        else:
             print("Cannot define CD. Aborting.")
             return

    # 3. Select Units (Good Clusters) — prefer stable > good > all, with FR filter
    if getattr(session, "good_and_stable_ids", None):
        candidates = set(int(x) for x in session.good_and_stable_ids)
    elif session.good_cluster_ids:
        candidates = set(int(x) for x in session.good_cluster_ids)
    else:
        candidates = {int(c.cluster_id) for c in session.clusters}
    units = [c for c in session.clusters if int(c.cluster_id) in candidates]
    
    print(f"Analyzing {len(units)} units.")
    if len(units) == 0: return

    # 4. Compute PSTHs per group
    window = [-2.5, 1.5]
    bin_size = 0.02
    
    # Dictionary to store mean PSTHs per group: groups_psth[group_name] = (n_bins, n_units) -> Wait, (n_units, n_bins)
    groups_data = {k: [] for k in groups.keys()} 
    
    bin_centers = None
    
    valid_units_idx = []

    for i, unit in enumerate(units):
        unit_valid = True
        
        # We need data for the defining groups (Balanced Hit/Miss) to compute CD
        # For projection groups, we can tolerate missing data (as long as we don't use them for CD)
        
        # Temp storage for this unit
        unit_psths = {} 
        
        for grp_name, times in groups.items():
            if len(times) == 0:
                unit_psths[grp_name] = None
                continue
                
            mat, bins = align.align_spikes_to_events(unit.spike_times, times, window, bin_size)
            if bin_centers is None: bin_centers = bins
            
            if mat.shape[0] < 1: # Strict check?
                unit_psths[grp_name] = None
            else:
                # Smooth and mean
                psth = np.mean(mat, axis=0) # (n_bins,)
                psth_hz = psth / bin_size
                psth_smooth = gaussian_filter1d(psth_hz, sigma=2)
                unit_psths[grp_name] = psth_smooth

        # Check if we have defining data
        if unit_psths['Balanced_Hit'] is None or unit_psths['Balanced_Miss'] is None:
            continue
            
        # Store data
        for k in groups_data:
            val = unit_psths[k]
            if val is None:
                # If a unit is missing data for a passive group (e.g. Disengaged Miss), fill with NaNs or Zeros?
                # Better to fill with zeros so matrix operation works, keeping in mind it might skew mean if we did population mean.
                # But here we do projection: dot(CD, R). R is (n_units, n_time).
                # If a unit has no spikes for Disengaged Mod, its rate is 0? Or unknown?
                # If no trials, we can't estimate rate. 
                # We should exclude this unit from the projection completely? 
                # Or just exclude it from *that trace*?
                # We need the SAME units for CD definition and Projection.
                # If we drop unit X because it has no Disengaged Misses, we lose it for Balanced Hit.
                # Ideally, we only drop if defining groups are missing.
                # For Missing non-defining groups, we just can't plot that trace properly if MANY units are missing.
                # BUT, population projection is a sum over units. missing entries -> 0 contribution.
                # Let's assume 0 rate if no trials (it is weird but mechanically works) or better yet, zeros.
                groups_data[k].append(np.zeros(len(bin_centers)))
            else:
                groups_data[k].append(val)
                
        valid_units_idx.append(i)

    # Check we have units
    if len(valid_units_idx) == 0:
        print("No units had enough data.")
        return

    # Convert to arrays: (n_units, n_times)
    R_arrays = {}
    for k, v in groups_data.items():
        R_arrays[k] = np.array(v)

    # 5. Define Coding Direction (CD)
    # CD defined as vector difference of means in interval [-2.0, 0]s using Balanced trials
    # This captures the "Pre-Stimulus State" difference
    t_mask = (bin_centers >= -2.0) & (bin_centers <= 0.0)
    
    vec_hit = np.mean(R_arrays['Balanced_Hit'][:, t_mask], axis=1)
    vec_miss = np.mean(R_arrays['Balanced_Miss'][:, t_mask], axis=1)
    
    cd_vector = vec_hit - vec_miss
    norm_cd = np.linalg.norm(cd_vector)
    if norm_cd == 0:
        print("CD vector magnitude is zero.")
        return
    cd_vector = cd_vector / norm_cd
    
    # 6. Project population activity onto CD (Stimulus Aligned)
    projections_stim = {}
    for k, R in R_arrays.items():
        projections_stim[k] = np.dot(cd_vector, R)
    
    # 7. Plot Stimulus Aligned
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Define styles
    styles = {
        'Balanced_Hit': {'color': 'green', 'label': 'Balanced Hit', 'ls': '-'},
        'Balanced_Miss': {'color': 'purple', 'label': 'Balanced Miss', 'ls': '-'},
        'Balanced_Fa': {'color': 'orange', 'label': 'Balanced FA', 'ls': '-'},
        'Impulsive_Hit': {'color': 'lime', 'label': 'Impulsive Hit', 'ls': '--'},
        'Impulsive_Miss': {'color': 'magenta', 'label': 'Impulsive Miss', 'ls': '--'},
        'Impulsive_Fa': {'color': 'gold', 'label': 'Impulsive FA', 'ls': '--'},
        'Disengaged_Miss': {'color': 'gray', 'label': 'Disengaged Miss', 'ls': ':'},
        'Disengaged_Hit': {'color': 'darkgreen', 'label': 'Disengaged Hit', 'ls': ':'},
        'Disengaged_Fa': {'color': 'brown', 'label': 'Disengaged FA', 'ls': ':'}
    }
    
    for k, proj in projections_stim.items():
        if len(groups[k]) > 0:
             # Skip FA traces in Stimulus Aligned plot
             if 'Fa' in k or 'FA' in k: continue
             ax1.plot(bin_centers, proj, **styles.get(k, {'label': k}))
    
    ax1.axvline(0, color='k', linestyle='--', label='Stimulus Onset')
    ax1.set_title(f"Stimulus Aligned Projection\n(CD = Pre-Stim State [-2,0]s)")
    ax1.set_xlabel("Time from Change (s)")
    ax1.set_ylabel("Proj. Activity (a.u.)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- LICK ALIGNED ANALYSIS ---
    print("Computing Lick-Aligned Projections...")
    
    # 1. Get Lick Times for Valid Trials (Hits and FAs only)
    # We need to grab them by index to match behavioral states
    # get_event_times_by_trial returns array of size n_trials with NaNs where event absent
    # or explicit times.
    # For Hit/FA, we need the response time.
    # 'Hit' returns absolute lick time for Hit trials. 'FA' returns abs lick time for FA trials.
    
    times_hit = align.get_event_times_by_trial(session, 'Hit')
    times_fa = align.get_event_times_by_trial(session, 'FA')
    
    groups_lick = {
        'Balanced_Hit': [],
        'Balanced_Fa': [],
        'Impulsive_Hit': [],
        'Impulsive_Fa': [],
        'Disengaged_Hit': [],
        'Disengaged_Fa': []
    }
    
    for idx, row in df_beh.iterrows():
        if idx >= len(times_hit) or idx >= len(times_fa): break
        
        state = row['state']
        outcome = row['outcome']
        
        t_lick = np.nan
        if outcome == 'hit':
            t_lick = times_hit[idx]
        elif outcome == 'fa':
            t_lick = times_fa[idx]
            
        if np.isnan(t_lick): continue
        
        key = f"{state.capitalize()}_{outcome.capitalize()}"
        if key in groups_lick:
            groups_lick[key].append(t_lick)

    # 2. Compute PSTHs (Lick Aligned)
    # Using slightly different window for motor: -1.0 to 0.5
    window_lick = [-1.0, 0.5]
    
    lick_psths = {k: [] for k in groups_lick.keys()}
    bin_centers_lick = None
    
    for i, unit in enumerate(units):
        # Must match unit index from stim analysis for dot product
        # BUT we previously filtered using valid_units_idx!
        # If we re-iterate 'units', we match. But if we skipped units inside the loop...
        # Wait, inside stim loop we did: valid_units_idx.append(i). 
        # And R_arrays was built by appending.
        # We need to strictly use only the units that went into 'cd_vector'.
        # 'units' list is the same. We need to check if we skip same way.
        
        # Check if this unit was used in Stim Analysis
        # We stored valid_units_idx.
        if i not in valid_units_idx:
            continue
            
        # Compute Lick Aligned
        for grp_name, times in groups_lick.items():
            if len(times) == 0:
                lick_psths[grp_name].append(np.zeros_like(bin_centers_lick) if bin_centers_lick is not None else [])
                continue
                
            mat, bins = align.align_spikes_to_events(unit.spike_times, times, window_lick, bin_size)
            if bin_centers_lick is None: bin_centers_lick = bins
            
            if mat.shape[0] < 1:
                psth_smooth = np.zeros(len(bins))
            else:
                psth = np.mean(mat, axis=0) / bin_size
                psth_smooth = gaussian_filter1d(psth, sigma=2)
            
            lick_psths[grp_name].append(psth_smooth)

    # 3. Project
    if bin_centers_lick is not None:
        for k, psths in lick_psths.items():
            # Check for empty lists if first unit failed
            if len(psths) == 0: continue
            
            R_lick = np.array(psths) # (n_units, n_times_lick)
            
            # Check shape
            if R_lick.shape[0] != cd_vector.shape[0]:
                 # Should not happen if filtered correctly
                 print(f"Shape mismatch for {k}: R={R_lick.shape}, CD={cd_vector.shape}")
                 continue
                 
            proj_lick = np.dot(cd_vector, R_lick)
            
            # Plot
            ax2.plot(bin_centers_lick, proj_lick, **styles.get(k, {'label': k}))

    ax2.axvline(0, color='k', linestyle='--', label='Lick Onset')
    ax2.set_title("Lick Aligned Projection\n(Projected on Stimulus CD)")
    ax2.set_xlabel("Time from Lick (s)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    out_file = os.path.join(output_dir, f"CD_projection_states_DUAL_{session.session_name}.png")
    plt.tight_layout()
    plt.savefig(out_file)
    print(f"Saved dual plot to {out_file}")
    plt.close()


if __name__ == "__main__":
    import glob
    # Running on one example expert session for now as a test
    pkls = glob.glob("data/pkls/BG_046/*Expert*.pkl") # Need to check if filenames have stage or use manifest
    
    # Fallback to specific file if glob fails or just pick one
    manifest = "data/BG_046_staging_manifest.csv"
    if os.path.exists(manifest):
        import pandas as pd
        df = pd.read_csv(manifest)
        expert_sess = df[df['stage'] == 'Expert'].iloc[0]
        path = expert_sess['path']
        # Fix path windows/linux slashes if needed
        path = path.replace("\\", "/")
        run_population_analysis(path, "FIGURES/tf") # Saving to existing folder for now
    else:
        print("Manifest not found, run stage_sessions.py first")

