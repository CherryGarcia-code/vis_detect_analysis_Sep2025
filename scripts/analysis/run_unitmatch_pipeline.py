
import sys
import os
import argparse
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

# UnitMatch imports
try:
    import UnitMatchPy.utils as util
    import UnitMatchPy.overlord as ov
    import UnitMatchPy.bayes_functions as bf
    import UnitMatchPy.assign_unique_id as aid
    import UnitMatchPy.save_utils as su
    import UnitMatchPy.default_params as default_params
except ImportError:
    print("Error: UnitMatchPy not found. Make sure you are in the correct environment.")
    sys.exit(1)

# Setup paths
repo_root = Path(".").resolve()
data_root = repo_root / "data" / "unit_match" / "input" / "BG_046"
output_dir = repo_root / "data" / "unit_match" / "output" / "BG_046"
output_dir.mkdir(parents=True, exist_ok=True)

def parse_date(name):
    """
    Parse date from session name string.
    Expects DDMMYYYY or BG_046_DDMMYYYY.
    """
    # Try raw name first (e.g. "01072025")
    if len(name) == 8 and name.isdigit():
        try:
            return datetime.strptime(name, "%d%m%Y")
        except ValueError:
            pass

    # Try split by underscore (e.g. "BG_046_01072025")
    parts = name.split('_')
    if len(parts) >= 2:
        d_str = parts[-1] 
        if len(d_str) == 8 and d_str.isdigit():
            try:
                return datetime.strptime(d_str, "%d%m%Y")
            except ValueError:
                pass
    
    # Debug print for failed parses if needed, but for now return min
    # print(f"Warning: Could not parse date from {name}")
    return datetime.min

def sync_tsv_with_waveforms(ks_dir):
    """
    Ensure cluster_group.tsv only contains units present in RawWaveforms
    to prevent UnitMatch from trying to load missing files.
    """
    ks_path = Path(ks_dir)
    wav_dir = ks_path / "RawWaveforms"
    tsv_path = ks_path / "cluster_group.tsv"
    if not tsv_path.exists():
        tsv_path = ks_path / "cluster_KSLabel.tsv"
    
    if not tsv_path.exists():
        print(f"Warning: No cluster group TSV found in {ks_dir}")
        # Create one?
        return

    if not wav_dir.exists():
        return

    # Get available unit IDs from .npy files
    available_ids = set()
    for f in wav_dir.glob("*.npy"):
        try:
             # Try plain int (old style)
            available_ids.add(int(f.stem))
        except ValueError:
            # Try UnitX_RawSpikes (new style)
            if f.stem.startswith("Unit") and "RawSpikes" in f.stem:
                try:
                    # Parse "Unit123_RawSpikes" -> 123
                    # Split by _ and remove Unit
                    uid_str = f.stem.split('_')[0].replace('Unit', '')
                    available_ids.add(int(uid_str))
                except ValueError:
                    pass
            
    if not available_ids:
        print(f"Warning: No waveforms found in {wav_dir}")
        return

    # Read TSV
    try:
        df = pd.read_csv(tsv_path, sep='\t')
        # Filter: keep only available units
        original_count = len(df)
        df_filtered = df[df['cluster_id'].isin(available_ids)].copy()
        
        # Ensure they are marked 'good' so UnitMatch picks them up
        if 'group' in df_filtered.columns:
            df_filtered['group'] = 'good'
        elif 'KSLabel' in df_filtered.columns:
            df_filtered['KSLabel'] = 'good'
            
        # Save back
        df_filtered.to_csv(tsv_path, sep='\t', index=False)
        print(f"Synced {tsv_path.name} in {ks_path.name}: {original_count} -> {len(df_filtered)} units.")
    except Exception as e:
        print(f"Error syncing TSV in {ks_dir}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run UnitMatch pipeline on prepared data.")
    parser.add_argument('--sessions', nargs='+', help='List of session dates (DDMMYYYY) or names to process explicitly')
    args = parser.parse_args()

    # Find session directories
    session_dirs = [d for d in data_root.iterdir() if d.is_dir() and (d / "RawWaveforms").exists()]
    
    # Filter if sessions specified
    if args.sessions:
        targets = args.sessions
        # Check if target is in directory name
        session_dirs = [d for d in session_dirs if any(t in d.name for t in targets)]
        print(f"Filtered to {len(session_dirs)} sessions based on input args: {[d.name for d in session_dirs]}")

    # Sort DESCENDING (Latest first) as requested
    session_dirs.sort(key=lambda x: parse_date(x.name), reverse=True)

    print(f"Found {len(session_dirs)} sessions prepared.")
    if not session_dirs:
        print("No prepared sessions found. Run prep_unitmatch_full_trial_waveforms.py first.")
        sys.exit(1)

def run_pipeline_batch(session_dirs, output_dir, label_suffix=""):
    print(f"\n=======================================================")
    print(f"Running batch {label_suffix}: {[d.name for d in session_dirs]}")
    print(f"=======================================================\n")
    
    # Check output dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sync TSVs
    print("Syncing cluster info files...")
    for d in session_dirs:
        sync_tsv_with_waveforms(d)

    # Convert to string paths
    KS_dirs = [str(d) for d in session_dirs]
    print("\nProcessing Sequence (Latest -> Earliest):")
    for k in KS_dirs:
        print(f" - {Path(k).name}")

    # Initialize Parameters
    param = default_params.get_default_param()
    param['KS_dirs'] = KS_dirs
    
    # --- MEMORY OPTIMIZATION ---
    # With 10 sessions (~2000 units), the default 100 bins creates a 19GB array in apply_naive_bayes.
    # RAM usage is approx N^2 * bins * 8 bytes.
    # For Batch=12 (N~2400), 100 bins uses ~27GB, which failed allocation on 64GB RAM.
    # We reduce bin resolution to 50 bins (step 0.02) to reduce memory usage by 50% (to ~13.5GB).
    if len(KS_dirs) >= 8:
        print(f"Optimization: Large batch ({len(KS_dirs)} sessions) detected.")
        print("Reducing histogram bin resolution (100 -> 50 bins) to fit in RAM while maintaining good quality.")
        param['bins'] = np.linspace(0, 1, 51) # 50 intervals
        # Fix: score_vector must match the bins (midpoints)
        param['score_vector'] = (param['bins'][:-1] + param['bins'][1:]) / 2
        
        # AGGRESSIVE OPTIMIZATION:
        # The crash occurs in get_recentered_euclidean_dist computing (3, N, 23, 2, N) float64
        # For N=3891, this is 3*3891*23*2*3891 * 8 bytes ~ 16.7 GB
        # We must Monkey-Patch numpy to default to float32 for metric_functions if possible,
        # OR we reduce the batch size further.
        # But wait, extracting metrics calls `mf.get_recentered_euclidean_dist` which uses broadcasting.
        # We can try to cast input waveforms to float32? They likely already are.
        # The issue is the temporary broadcasting `x1 - x2`.
        # Code: tmp_euclid = np.linalg.norm(x1-x2, axis=0)
        # x1, x2 are (dims, N, ...)
        
    # ---------------------------

    # Load Data
    print("\nLoading UnitMatch data...")
    try:
        # We need to suppress print output from this if it's too verbose?
        # UnitMatch prints a LOT.
        wave_paths, unit_label_paths, channel_pos = util.paths_from_KS(KS_dirs)
        
        waveform, session_id, session_switch, within_session, good_units, param = util.load_good_waveforms(
            wave_paths, 
            unit_label_paths, 
            param, 
            good_units_only=True
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return
        
    print(f"Loaded {len(np.concatenate(good_units))} total units across sessions.")

    # Create clus_info
    clus_info = {
        'good_units': good_units, 
        'session_switch': session_switch, 
        'session_id': session_id, 
        'original_ids': np.concatenate(good_units) 
    }
    
    # Correct probe geometry
    param = util.get_probe_geometry(channel_pos[0], param)

    # STEP 1: Extract parameters
    print("Extracting waveform parameters...")
    extracted_wave_properties = ov.extract_parameters(waveform, channel_pos, clus_info, param)

    # STEP 2-4: Extract metric scores
    print("Extracting metric scores...")
    total_score, candidate_pairs, scores_to_include, predictors = ov.extract_metric_scores(
        extracted_wave_properties, 
        session_switch, 
        within_session, 
        param, 
        niter=2
    )

    # STEP 5: Probability Analysis
    print("Running probability analysis...")
    n_units = param['n_units']
    prior_match = 1 - (param['n_expected_matches'] / n_units**2)
    priors = np.array((prior_match, 1 - prior_match))
    
    labels = candidate_pairs.astype(int)
    cond = np.unique(labels)
    
    parameter_kernels = bf.get_parameter_kernels(scores_to_include, labels, cond, param, add_one=1)
    
    # MEMORY WARNING: apply_naive_bayes can use O(N^2) memory.
    try:
        probability = bf.apply_naive_bayes(parameter_kernels, priors, predictors, param, cond)
    except MemoryError as e:
        print(f"MEMORY ERROR: {e}. Try reducing batch size.")
        return
    except Exception as e:
        # Catch numpy.core._exceptions.MemoryError like error
        if "Unable to allocate" in str(e):
            print(f"MEMORY ERROR (Allocation): {e}. Try reducing batch size.")
            return
        raise e

    output_prob_matrix = probability[:, 1].reshape(n_units, n_units)
    
    # Thresholding
    match_threshold = 0.5 # You can tune this
    output_threshold = np.zeros_like(output_prob_matrix)
    output_threshold[output_prob_matrix > match_threshold] = 1

    # Assign Unique IDs
    print("Assigning Unique IDs...")
    matches = np.argwhere(output_threshold == 1)
    
    # We might need to curate matches? Skipping GUI, using raw threshold for now.
    # assign_unique_id returns a tuple of 4 arrays: (liberal, intermediate, conservative, original)
    UID_collection = aid.assign_unique_id(output_prob_matrix, param, clus_info)
    
    # We need the full collection for save_to_output
    # But we need a specific single array for our Cell Registry
    # Index 0 = Liberal? Index 1 = Intermediate? 
    # Based on save_utils source: UIDs[0]->Liberal, UIDs[1]->Intermediate, UIDs[2]->Conservative.
    # We will use Index 1 (Intermediate) as the default "UID" for the registry, 
    # as it balances False Pos/Neg.
    if isinstance(UID_collection, (list, tuple)) and len(UID_collection) >= 2:
        UIDs_for_registry = UID_collection[1] 
        print("Using Intermediate UIDs (Index 1) for Registry.")
    elif hasattr(UID_collection, '__len__') and len(UID_collection) == len(clus_info['original_ids']):
        # It returned just one array?
        UIDs_for_registry = UID_collection
        UID_collection = [UID_collection, UID_collection, UID_collection, UID_collection] # Hack to satisfy save_utils
    else:
        # Fallback to index 0
        UIDs_for_registry = UID_collection[0]

    # Save Output
    save_path = str(output_dir)
    print(f"Saving results to {save_path}...")
    
    # Prepare data for saving
    # su.save_to_output expects unpacked components
    amplitude = extracted_wave_properties['amplitude']
    spatial_decay = extracted_wave_properties['spatial_decay']
    avg_centroid = extracted_wave_properties['avg_centroid']
    avg_waveform = extracted_wave_properties['avg_waveform']
    avg_waveform_per_tp = extracted_wave_properties['avg_waveform_per_tp']
    wave_idx = extracted_wave_properties['good_wave_idxs'] 
    
    max_site = extracted_wave_properties['max_site']
    
    su.save_to_output(
        save_path, 
        scores_to_include, 
        matches, 
        output_prob_matrix, 
        avg_centroid, 
        avg_waveform, 
        avg_waveform_per_tp, 
        max_site,
        total_score, 
        output_threshold, 
        clus_info, 
        param, 
        UIDs=UID_collection, 
        matches_curated=None, 
        save_match_table=True
    )

    # --- CUSTOM: Generate Cell Registry and Save Session Metadata ---
    print("Generating Cell Registry (UID -> Session mapping)...")
    
    # 1. Save Session List to preserve order
    session_names = [Path(k).name for k in KS_dirs]
    with open(output_dir / "SessionList.txt", "w") as f:
        for s in session_names:
            f.write(s + "\n")
            
    # 2. Build Registry DataFrame
    sess_ids = clus_info['session_id']
    orig_ids = clus_info['original_ids']

    # Define variants to save
    registry_variants = {}
    if isinstance(UID_collection, (list, tuple)) and len(UID_collection) >= 3:
        registry_variants['Liberal'] = UID_collection[0]
        registry_variants['Intermediate'] = UID_collection[1]
        registry_variants['Conservative'] = UID_collection[2]
        # Index 3 is typically 'Original' (1..N) if present
    else:
        registry_variants['Default'] = UIDs_for_registry

    for var_name, var_uids in registry_variants.items():
        if len(var_uids) != len(sess_ids):
            print(f"Warning: {var_name} UID length mismatch. Skipping.")
            continue
            
        # Create a Long-form table first
        data_list = []
        for i, uid in enumerate(var_uids):
            s_idx = sess_ids[i]
            c_val = orig_ids[i]
            # Convert numpy types to native Python types to avoid "unhashable type" in pivot
            if hasattr(c_val, 'item'):
                c_val = c_val.item()

            if s_idx < len(session_names):
                s_name = session_names[s_idx]
                data_list.append({
                    'UID': uid,
                    'Session': s_name,
                    'ClusterID': c_val,
                    'GlobalIndex': i
                })
        
        df_long = pd.DataFrame(data_list)
        # Only save long table for Intermediate/Default to save space/clutter
        if var_name in ['Intermediate', 'Default']:
             df_long.to_csv(output_dir / "Unit_Long_Table.csv", index=False)
        
        # Pivot to Wide Format (Cell Registry)
        # Handle cases where multiple clusters in one session map to the same UID (merge/split issue)
        if df_long.duplicated(subset=['UID', 'Session']).any():
            print(f"Warning: {var_name} contains multiple clusters for the same UID in a single session. Aggregating with ';'.")
            
            # Helper to join unique cluster IDs
            def agg_clusters(x):
                 # Use set instead of pd.unique to reduce dependency on numpy types
                 return ";".join(str(v) for v in sorted(set(x)))

            df_registry = df_long.pivot_table(
                index='UID', 
                columns='Session', 
                values='ClusterID', 
                aggfunc=agg_clusters
            )
        else:
            df_registry = df_long.pivot(index='UID', columns='Session', values='ClusterID')
        
        # Reorder columns
        existing_cols = [s for s in session_names if s in df_registry.columns]
        df_registry = df_registry[existing_cols]
        
        # Save specific variant
        df_registry.to_csv(output_dir / f"CellRegistry_{var_name}.csv")
        
        # If this is the 'Intermediate' or 'Default' one, also save as main "CellRegistry.csv"
        if var_name in ['Intermediate', 'Default']:
             df_registry.to_csv(output_dir / "CellRegistry.csv")
             print(f"Saved CellRegistry.csv (from {var_name}) with {len(df_registry)} unique units.")
        else:
             print(f"Saved CellRegistry_{var_name}.csv with {len(df_registry)} units.")

    print(f"Batch {label_suffix} completed successfully.")

def main():
    parser = argparse.ArgumentParser(description="Run UnitMatch pipeline on prepared data.")
    parser.add_argument('--sessions', nargs='+', help='List of session dates (DDMMYYYY) or names to process explicitly')
    parser.add_argument('--batch_size', type=int, default=12, help='Number of sessions to process in a sliding window batch. Defaults to 12 (Safe for 64GB RAM with High Quality).')
    args = parser.parse_args()

    # Find session directories
    session_dirs = [d for d in data_root.iterdir() if d.is_dir() and (d / "RawWaveforms").exists()]
    
    # Filter if sessions specified
    if args.sessions:
        targets = args.sessions
        # Check if target is in directory name
        session_dirs = [d for d in session_dirs if any(t in d.name for t in targets)]
        print(f"Filtered to {len(session_dirs)} sessions based on input args: {[d.name for d in session_dirs]}")

    # Sort DESCENDING (Latest first) as requested
    session_dirs.sort(key=lambda x: parse_date(x.name), reverse=True)

    print(f"Found {len(session_dirs)} sessions prepared.")
    if not session_dirs:
        print("No prepared sessions found. Run prep_unitmatch_full_trial_waveforms.py first.")
        sys.exit(1)
        
    # Check if we need batching
    bs = args.batch_size
    n_sessions = len(session_dirs)
    
    if bs >= n_sessions:
        # Process all at once
        run_pipeline_batch(session_dirs, output_dir, label_suffix="ALL")
    else:
        print(f"Running in sliding window batches of size {bs}...")
        # Sliding window, step 1? Or non-overlapping? 
        # For tracking continuity, usually overlapping is good, but outputs will be separate.
        # Let's do overlapping step 1.
        
        # NOTE: Output dirs will be output_dir / "Batch_StartDate_EndDate"
        step = 1
        for i in range(0, n_sessions - bs + 1, step):
            batch_dirs = session_dirs[i : i + bs]
            start_name = batch_dirs[0].name
            end_name = batch_dirs[-1].name
            label = f"{start_name}_to_{end_name}"
            # Keep folder name simple
            sub_out = output_dir / label
            
            run_pipeline_batch(batch_dirs, sub_out, label_suffix=label)
            
            # Optional: just do one batch? No, do all.

if __name__ == "__main__":
    main()

