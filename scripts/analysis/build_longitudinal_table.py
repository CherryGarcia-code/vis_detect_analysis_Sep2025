"""
Build Grand Longitudinal Table
------------------------------
Integrates UnitMatch tracking (ID) with Physiology (TF/Time course), Behavior, and QC metrics.
Produces a long-form DataFrame where each row is a (Global_Unit, Session) tuple.

Metrics Strategy:
- Behavior: Calculated fresh from session PKLs (dPrime, hit rates).
- Physiology (TF): Calculated fresh using standard pipeline logic, but verified against existing CSVs.

Usage:
    conda activate unitmatch_env && python scripts/analysis/build_longitudinal_table.py --n_workers 6
"""

from pathlib import Path
import pandas as pd
import numpy as np
import sys
import argparse
import pickle
from datetime import datetime
from tqdm import tqdm
import math
import concurrent.futures
import gc

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(repo_root / "src"))

from visdetect.core.session import load_session
from visdetect.analysis.behavior import calculate_dprime, compute_session_performance
from visdetect.analysis.tf_pulse import collect_tf_pulse_traces, TFRespPulseConfig
from scipy.signal import correlate

# =============================================================================
# QC METRICS HELPERS
# =============================================================================

def compute_isi_violation_rate(spike_times, ref_period=0.0015, min_spikes=100):
    """
    Computes fraction of ISIs violating the refractory period (< 1.5ms).
    Proxy for contamination.
    """
    if len(spike_times) < min_spikes:
        return np.nan
        
    spike_times = np.sort(spike_times)
    isis = np.diff(spike_times)
    
    n_viol = np.sum(isis < ref_period)
    n_total = len(isis)
    
    return n_viol / n_total

def compute_acg(spike_times, bin_size=0.0005, window=0.050):
    """
    Computes Auto-Correlogram (one-sided) for a spike train.
    Returns: centers (s), rate (Hz)
    Adapted from Single_Unit_Deep_Dive.ipynb
    """
    spike_times = np.sort(spike_times)
    n_spikes = len(spike_times)
    if n_spikes < 10: 
        return None, None
        
    differences = []
    # Check neighbors (vectorized-ish). 
    # For a typical unit firing < 50Hz, checking 100 neighbors covers >2s.
    max_neighbors = 100 
    
    for lag in range(1, max_neighbors):
        # Calculate diffs for this lag
        d = spike_times[lag:] - spike_times[:-lag]
        
        # Keep only those within window
        valid = d <= window
        if not np.any(valid):
            if np.min(d) > window:
                break
        
        differences.append(d[valid])
        
    if not differences:
        return None, None
        
    diffs = np.concatenate(differences)
    
    # Histogram
    bins = np.arange(0, window + bin_size, bin_size)
    counts, _ = np.histogram(diffs, bins=bins)
    
    centers = bins[:-1] + bin_size/2
    
    # Normalize to Rate
    rate = counts / (n_spikes * bin_size)
    
    return centers, rate

def load_mean_waveform(date_str, cluster_id, input_root):
    """
    Loads raw spikes from UnitMatch input folder and computes mean peak waveform.
    Returns: 1D numpy array (samples,) or None
    """
    # Path format: data/unit_match/input/BG_046/{DDMMYYYY}/RawWaveforms/Unit{ID}_RawSpikes.npy
    # Note: date_str in session_cols comes as '01072025' or '20250701'.
    # We need to match the folder name in 'input/BG_046'.
    # We saw folders like '01072025' (DDMMYYYY).
    
    try:
        # Construct path
        wf_dir = input_root / date_str / "RawWaveforms"
        wf_file = wf_dir / f"Unit{cluster_id}_RawSpikes.npy"
        
        if not wf_file.exists():
            # Try alternate date formats if needed?
            return None
            
        # Load (n_spikes, n_samples, n_chans)
        raw = np.load(wf_file)
        
        if raw.size == 0:
            return None
            
        # Mean across spikes -> (n_samples, n_chans)
        mean_wf = np.mean(raw, axis=0)
        
        # Take peak channel (max amplitude range)
        # simplistic: just take 0-th channel if it's "best", or find max ptp
        ptps = np.ptp(mean_wf, axis=0)
        best_chan = np.argmax(ptps)
        
        return mean_wf[:, best_chan]
        
    except Exception as e:
        # print(f"Error loading waveform for {date_str} Unit {cluster_id}: {e}")
        return None

def process_single_unit_qc(args):
    """
    Worker function to compute QC for one unit.
    args: (cluster_id, spike_times, date_part, input_root)
    Returns: (cluster_id, qc_metrics_dict, waveform_or_None, acg_rate_or_None)
    """
    cid, st, date_str, root = args
    
    # Init storage
    metrics = {}
    wf = None
    acg_rate = None
    
    # 1. ISI
    metrics['qc_isi_viol'] = compute_isi_violation_rate(st)
    
    # 2. Firing Rate
    if len(st) > 0:
        t_max = np.max(st) if len(st) > 0 else 1.0
        metrics['qc_firing_rate'] = len(st) / t_max if t_max > 0 else 0.0
    else:
        metrics['qc_firing_rate'] = 0.0
        
    # 3. ACG
    if len(st) > 100:
        centers, rate = compute_acg(st)
        if rate is not None:
             acg_rate = rate
             
    # 4. Waveform
    wf = load_mean_waveform(date_str, cid, root)
    
    return cid, metrics, wf, acg_rate

def compute_waveform_correlations(df_grand, waveforms_dict):
    """
    Computes correlation of each session's waveform to the Global Template (Median).
    """
    corrs = []
    
    # Iterate by Global UID
    for global_uid, group in df_grand.groupby('Global_UID'):
        # Collect all waveforms for this unit
        unit_wfs = []
        indices = []
        
        for idx, row in group.iterrows():
            key = (row['Session_Date'], row['Cluster_ID'])
            if key in waveforms_dict:
                unit_wfs.append(waveforms_dict[key])
                indices.append(idx)
        
        if len(unit_wfs) < 2:
            # If only 1 session, correlation is 1.0 (to itself) or NaN (no template)
            # Let's say 1.0 for self-consistency
            for idx in indices:
                corrs.append({'idx': idx, 'corr': 1.0})
            continue
            
        # Compute Template: Median of waveforms
        # Stack: (n_sessions, n_samples)
        # Pad if lengths differ? Assuming consistent sampling for now.
        try:
            stack = np.stack(unit_wfs, axis=0)
            template = np.median(stack, axis=0)
            
            # Compute Correlation for each
            for i, idx in enumerate(indices):
                wf = unit_wfs[i]
                # Pearson Correlation
                if np.std(wf) == 0 or np.std(template) == 0:
                    r = 0
                else:
                    r = np.corrcoef(wf, template)[0, 1]
                corrs.append({'idx': idx, 'corr': r})
                
        except Exception as e:
            # Dimensions might mismatch?
            # print(f"Waveform stacking error UID {global_uid}: {e}")
            for idx in indices:
                corrs.append({'idx': idx, 'corr': np.nan})
                
    return pd.DataFrame(corrs).set_index('idx')

def compute_acg_correlations(df_grand, acg_dict):
    """
    Computes correlation of each session's ACG to the Global Template (Median).
    dict key: (Session_Date, Cluster_ID) -> acg_rate_vector
    """
    corrs = []
    
    for global_uid, group in df_grand.groupby('Global_UID'):
        unit_acgs = []
        indices = []
        
        # 1. Collect
        for idx, row in group.iterrows():
            key = (row['Session_Date'], row['Cluster_ID'])
            if key in acg_dict and acg_dict[key] is not None:
                unit_acgs.append(acg_dict[key])
                indices.append(idx)
        
        if len(unit_acgs) < 2:
            for idx in indices:
                corrs.append({'idx': idx, 'corr': 1.0})
            continue
            
        try:
            # 2. Template (Median)
            # Ensure same shape
            # ACGs should be fixed bins from compute_acg
            stack = np.stack(unit_acgs, axis=0)
            template = np.median(stack, axis=0)
            
            # 3. Correlation
            for i, idx in enumerate(indices):
                acg = unit_acgs[i]
                if np.std(acg) == 0 or np.std(template) == 0:
                    r = 0
                else:
                    r = np.corrcoef(acg, template)[0, 1]
                corrs.append({'idx': idx, 'corr': r})
                
        except Exception as e:
            # print(f"ACG stacking error UID {global_uid}: {e}")
            for idx in indices:
                corrs.append({'idx': idx, 'corr': np.nan})
                
    return pd.DataFrame(corrs).set_index('idx')

def get_behavior_metrics(session):
    """
    Compute behavioral metrics for a session using the canonical SDT implementation.

    Thin wrapper around ``visdetect.analysis.behavior.compute_session_performance``
    that returns the subset of keys expected by the longitudinal table builder.
    """
    perf = compute_session_performance(session)
    if not perf:
        return {'beh_d_prime': np.nan, 'beh_hit_rate': np.nan, 'beh_fa_rate': np.nan,
                'beh_n_trials': 0, 'beh_n_go': 0, 'beh_n_catch': 0}

    return {
        'beh_d_prime': perf['d_prime'],
        'beh_hit_rate': perf['hit_rate'],
        'beh_fa_rate': perf['fa_rate_total'],
        'beh_n_trials': perf['n_trials'],
        'beh_n_go': perf['n_go'],
        'beh_n_catch': perf['n_catch'],
    }

def get_tf_metrics_batch(session, session_name, existing_metrics_path=None, n_workers=None):
    """
    Calculates TF metrics for ALL units in the session.
    Checks against existing CSV if provided.
    Returns: Dict {cluster_id: {metrics...}}
    """
    print(f"  Computing TF Metrics Fresh for {session_name}...")
    
    # 1. Compute Fresh
    try:
        # Use default config (matches pipeline)
        cfg = TFRespPulseConfig(
            kept_only=False, # We want measures for ALL units in registry, even if 'noise'
            use_constraints=True
        )
        
        # Determine Cache Path dynamically from session_name (e.g. BG_046_01072025)
        parts = session_name.split('_')
        if len(parts) >= 2:
            subject = f"{parts[0]}_{parts[1]}"
        else:
            subject = "unknown"
            
        cache_dir = repo_root / "data" / "cache" / "tf_traces" / subject
        cache_path = cache_dir / f"{session_name}_traces.npz"
        
        if cache_path.exists():
            print(f"    Using cached TF traces: {cache_path.name}")
        
        # collect_tf_pulse_traces returns: times, List[TFUnitTrace]
        # We need to compute traces. This might be slow.
        # But allow parallel=False to avoid spawming overly complex sub-processes here if unstable;
        # parallel=True is faster though.
        use_parallel = True
        if n_workers is not None and n_workers <= 1:
            use_parallel = False
            
        _, unit_traces = collect_tf_pulse_traces(
            session, cfg=cfg, parallel=use_parallel, show_progress=True, n_workers=n_workers,
            cache_path=str(cache_path)
        )
        
        fresh_data = {}
        for ut in unit_traces:
            fresh_data[int(ut.cluster_id)] = {
                'tf_z_max_fast': ut.z_max_fast,
                'tf_z_min_fast': ut.z_min_fast,
                'tf_z_max_slow': ut.z_max_slow,
                'tf_z_min_slow': ut.z_min_slow
            }
            
    except Exception as e:
        print(f"    ERROR computing fresh TF metrics: {e}")
        fresh_data = {}

    # 2. Compare with Existing (Validation)
    if existing_metrics_path and existing_metrics_path.exists():
        try:
            df_old = pd.read_csv(existing_metrics_path)
            # Check overlap
            print(f"  Validating against {existing_metrics_path.name}...")
            
            diffs = []
            for _, row in df_old.iterrows():
                cid = int(row['cluster_id'])
                if cid in fresh_data:
                    old_z = row.get('z_max_fast', np.nan)
                    new_z = fresh_data[cid]['tf_z_max_fast']
                    
                    if not pd.isna(old_z) and not pd.isna(new_z):
                        diff = abs(old_z - new_z)
                        diffs.append(diff)
                        if diff > 0.5: # Tolerance
                            # print(f"    Mismatch CID {cid}: Old={old_z:.2f}, New={new_z:.2f}")
                            pass
            
            if diffs:
                mean_diff = np.mean(diffs)
                print(f"    Mean Abs Diff (Fast Z): {mean_diff:.4f} (N={len(diffs)})")
                if mean_diff > 0.1:
                    print("    WARNING: Significant deviation from cached values.")
            else:
                print("    No overlapping units found for validation.")

        except Exception as e:
            print(f"    Could not validate against file: {e}")

    return fresh_data

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def build_grand_table(registry_path, output_path, n_workers=None):
    print(f"Loading Registry: {registry_path}")
    df_registry = pd.read_csv(registry_path, index_col=0)
    
    # Identify session columns (assuming date format like '30062025' or 'BG_046_...' or 'YYYY-MM-DD')
    session_cols = []
    for c in df_registry.columns:
        if 'BG_' in c:
            session_cols.append(c)
        elif c.replace('-', '').isdigit() and len(c) == 10 and c[4] == '-': # YYYY-MM-DD
            session_cols.append(c)
        elif c.isdigit() and len(c) == 8: # DDMMYYYY
            session_cols.append(c)
            
    print(f"Found {len(session_cols)} sessions in registry.")

    # We will iterate SESSIONS first to efficiently process data, 
    # then map back to the registry.
    
    # Cache for session metrics: session_date -> {cid: {metrics}}
    session_unit_metrics = {} 
    
    # Cache for global session metrics: session_date -> {metrics}
    session_global_metrics = {}
    
    # Dictionary to store QC data for correlation
    # Key: (Session_Date, Cluster_ID) -> Value: data
    global_waveforms = {}
    global_acgs = {}

    data_root = repo_root / "data"
    unit_match_root = repo_root / "data" / "unit_match" / "input" / "BG_046"
    
    # 1. Pre-fetch Data Session-by-Session
    for col in tqdm(session_cols, desc="Processing Sessions"):
        # Parse date from column name
        if 'BG_046' in col:
            date_part = col.split('_')[-1] # BG_046_01072025 -> 01072025
        elif '-' in col and len(col) == 10: # YYYY-MM-DD e.g. 2025-06-23
            try:
                dt = datetime.strptime(col, "%Y-%m-%d")
                date_part = dt.strftime("%d%m%Y") # -> 23062025
            except:
                date_part = col.replace('-', '')
        else:
            date_part = col
            
        session_name = f"BG_046_{date_part}"
        pkl_path = data_root / "pkls" / "BG_046" / f"{session_name}.pkl"
        
        if not pkl_path.exists():
            # Try alternate naming or check previous runs
             pkl_path = data_root / "pkls" / "previous_runs" / "BG_046" / f"{session_name}.pkl"
        
        if not pkl_path.exists():
            # print(f"  Skipping {col} (No PKL found)")
            continue
            
        try:
            sess = load_session(pkl_path)
            
            # A. Behavior
            beh_metrics = get_behavior_metrics(sess)
            session_global_metrics[col] = beh_metrics
            
            # B. TF Metrics
            # Look for existing file
            tf_csv_path = repo_root / "FIGURES" / "tf" / session_name / "tf_pulse_grid_both.csv"
            tf_unit_data = get_tf_metrics_batch(
                sess, session_name, existing_metrics_path=tf_csv_path, n_workers=n_workers
            )
            
            # C. EXTRA QC METRICS (Parallelized)
            qc_tasks = []
            
            # Gather inputs for all clusters found in TF step
            for cid_int in tf_unit_data.keys():
                clust = next((c for c in sess.clusters if int(c.cluster_id) == cid_int), None)
                if clust:
                    st = getattr(clust, 'spike_times', [])
                    qc_tasks.append((cid_int, st, date_part, unit_match_root))
                else:
                    # Append task with empty spikes if cluster not found (unlikely)
                    qc_tasks.append((cid_int, [], date_part, unit_match_root))

            # Run Parallel
            # Determine workers specifically for this QC step
            workers_qc = n_workers if (n_workers is not None and n_workers > 1) else 1
            
            print(f"  Computing QC Metrics for {len(qc_tasks)} units (Workers={workers_qc})...")
            
            qc_results = []
            if workers_qc > 1 and len(qc_tasks) > 0:
                 with concurrent.futures.ProcessPoolExecutor(max_workers=workers_qc) as executor:
                    # Submit tasks
                    future_to_cid = {executor.submit(process_single_unit_qc, task): task[0] for task in qc_tasks}
                    
                    for future in tqdm(concurrent.futures.as_completed(future_to_cid), total=len(qc_tasks), desc="  QC Progress", leave=False):
                         try:
                             qc_results.append(future.result())
                         except Exception as exc:
                             print(f"    QC Worker generated exception: {exc}")
            else:
                # Serial fallback
                for task in tqdm(qc_tasks, desc="  QC Progress (Serial)", leave=False):
                    qc_results.append(process_single_unit_qc(task))

            # Merge Back Results
            for cid, qc_met, wf, acg_rate in qc_results:
                # 1. Update Metrics Dict
                if cid in tf_unit_data:
                    tf_unit_data[cid].update(qc_met)
                
                # 2. Update Global QC Dicts
                if wf is not None:
                    global_waveforms[(col, cid)] = wf
                if acg_rate is not None:
                     global_acgs[(col, cid)] = acg_rate

            session_unit_metrics[col] = tf_unit_data
            
            # Cleanup memory for this session
            del sess
            gc.collect()
            
        except Exception as e:
            print(f"  Error processing session {session_name}: {e}")
            continue

    # 2. Build the Long Table
    long_records = []
    
    print("Building Long Table rows...")
    for global_uid, row in tqdm(df_registry.iterrows(), total=len(df_registry), desc="Mapping Units"):
        
        for col in session_cols:
            cluster_val = row[col]
            if pd.isna(cluster_val):
                continue
                
            # Handle Merged IDs "123;456"
            # Strategy: Split and create separate entries? Or aggregate?
            # Standard: Separate entries, but linking same Global ID.
            cluster_ids = str(cluster_val).split(';')
            
            for cid_str in cluster_ids:
                try:
                    cid = int(float(cid_str))
                except:
                    continue
                
                # Base Record
                record = {
                    'Global_UID': global_uid,
                    'Session_Date': col,
                    'Cluster_ID': cid
                }
                
                # Attach Behavior
                if col in session_global_metrics:
                    record.update(session_global_metrics[col])
                
                # Attach TF
                if col in session_unit_metrics and cid in session_unit_metrics[col]:
                    record.update(session_unit_metrics[col][cid])
                else:
                    # Fill NaNs for missing physiology (maybe unit didn't exist in that sess?)
                    record['tf_z_max_fast'] = np.nan
                
                long_records.append(record)
    
    # Create DataFrame
    print(f"Building DataFrame from {len(long_records)} records...")
    df_grand = pd.DataFrame(long_records)
    
    # QC Correlations (Waveform & ACG)
    print("Computing QC Correlations (Waveform & ACG)...")
    
    if not df_grand.empty:
        # A. Waveform
        if global_waveforms:
            df_wf_corr = compute_waveform_correlations(df_grand, global_waveforms)
            df_grand['qc_waveform_corr'] = df_wf_corr['corr']
        else:
            df_grand['qc_waveform_corr'] = np.nan
            
        # B. ACG
        if global_acgs:
            df_acg_corr = compute_acg_correlations(df_grand, global_acgs)
            df_grand['qc_acg_corr'] = df_acg_corr['corr']
        else:
            df_grand['qc_acg_corr'] = np.nan
    else:
        df_grand['qc_waveform_corr'] = np.nan
        df_grand['qc_acg_corr'] = np.nan
    
    # Post-clean: Sort
    if not df_grand.empty:
        df_grand = df_grand.sort_values(['Global_UID', 'Session_Date'])

    print(f"Built Grand Table with {len(df_grand)} rows.")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_grand.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    
    # 4. Save Waveforms for Reference
    wf_cache_path = output_path.parent / "Grand_Waveforms.pkl"
    with open(wf_cache_path, 'wb') as f:
        pickle.dump(global_waveforms, f)
    print(f"Saved Waveform Cache to {wf_cache_path}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Grand Longitudinal Table")
    parser.add_argument("--registry", 
                      default=str(repo_root / "data" / "deep_unit_match" / "output" / "BG_046" / "DeepUM_CellRegistry.csv"),
                      help="Path to CellRegistry.csv")
    parser.add_argument("--output", 
                      default=str(repo_root / "table_output" / "Grand_Longitudinal_Table.csv"),
                      help="Output CSV path")
    parser.add_argument("--workers", type=int, default=None, 
                      help="Number of workers for parallel TF extraction")
    
    args = parser.parse_args()
    
    reg_file = Path(args.registry)
    out_file = Path(args.output)
    
    if not reg_file.exists():
        print(f"Registry not found at {reg_file}")
    else:
        build_grand_table(reg_file, out_file, n_workers=args.workers)
