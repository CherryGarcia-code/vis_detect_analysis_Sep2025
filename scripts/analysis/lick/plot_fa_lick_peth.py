"""
Analyze lick rate dynamics during False Alarm (FA) trials across sessions.

This script:
1. Identifies FA trials in each session.
2. Aligns lick times to the *first* FA lick (t=0).
3. Computes the Peri-Event Time Histogram (PETH) of lick rate.
4. Generates:
    - A heatmap showing the evolution of FA lick rate profile across sessions.
    - Raster plots of example FA trials from Early, Middle, and Late sessions.

Usage:
    python scripts/analysis/lick/plot_fa_lick_peth.py --manifest data/pkls/BG_046/BG_046_sessions_manifest.csv --out FIGURES/lick/fa_peth_BG_046
"""
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import sys

# Ensure repo root/src is in path
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root / 'src'))

from visdetect.core.session import load_session
from visdetect.analysis.behavior import get_trial_dataframe
from visdetect.analysis.config import load_staging_manifest

def parse_date(session_name):
    try:
        date_str = session_name.split('_')[-1]
        return datetime.strptime(date_str, "%d%m%Y")
    except:
        return datetime.min

def get_event_times(ni_events, key_candidates):
    """Robustly retrieve event times from ni_events dict."""
    if not ni_events:
        return np.array([])
    
    found_times = []
    for key in key_candidates:
        if key in ni_events:
            val = ni_events[key]
            # Handle dict with 'rise_t' or direct array
            if isinstance(val, dict) and 'rise_t' in val:
                found_times.append(np.array(val['rise_t']).flatten())
            else:
                found_times.append(np.array(val).flatten())
            
    if found_times:
        return np.concatenate(found_times)
    return np.array([])

def compute_peth(ref_times, event_times, window, bin_size):
    """
    Compute Peri-Event Time Histogram.
    """
    t_min, t_max = window
    bins = np.arange(t_min, t_max + bin_size, bin_size)
    centers = bins[:-1] + bin_size / 2
    
    all_rel_times = []
    
    # Optimize finding relevant events (vectorized search sorted)
    # Assumes event_times is sorted
    event_times = np.sort(event_times)
    
    for t_ref in ref_times:
        if np.isnan(t_ref):
            continue
            
        # Find events in window [t_ref + t_min, t_ref + t_max]
        start_idx = np.searchsorted(event_times, t_ref + t_min)
        end_idx = np.searchsorted(event_times, t_ref + t_max)
        
        relevant = event_times[start_idx:end_idx]
        rel_times = relevant - t_ref
        all_rel_times.extend(rel_times)
        
    counts, _ = np.histogram(all_rel_times, bins=bins)
    
    # Convert to Rate (Hz)
    # Rate = Count / (N_trials * Bin_Size)
    n_trials = len(ref_times)
    if n_trials == 0:
        return centers, np.zeros_like(centers), []
        
    rate = counts / (n_trials * bin_size)
    
    return centers, rate, all_rel_times

def process_session(pkl_path):
    try:
        session = load_session(pkl_path)
    except Exception as e:
        print(f"Error loading {pkl_path}: {e}")
        return None

    # Get behavior df
    df = get_trial_dataframe(session)
    if df.empty:
        return None
        
    # Get Global Lick Times
    # Case sensitive keys from inspection: 'Lick_L' only as requested
    lick_keys = ['Lick_L', 'lick_L'] 
    lick_times = get_event_times(session.ni_events, lick_keys)
    # Sort in case we combined multiple channels
    if len(lick_times) > 0:
        lick_times = np.sort(np.unique(lick_times))
    
    if len(lick_times) == 0:
        # print(f"No lick times found in {pkl_path.name}") # verbose
        return None
        
    # Get Trial Start Times (Baseline_ON is the reference for RT)
    start_keys = ['Baseline_ON', 'trial_start_times', 'trial_start']
    trial_starts = get_event_times(session.ni_events, start_keys)
    
    if len(trial_starts) == 0:
        # print(f"No trial start times found in {pkl_path.name}")
        return None

    # Identify FA Trials
    fa_trials = df[df['is_fa']]
    
    if fa_trials.empty:
        return None

    alignment_times = []
    valid_fa_indices = []

    for _, row in fa_trials.iterrows():
        idx = int(row['trial_idx'])
        if idx >= len(trial_starts):
            continue
            
        t_start = trial_starts[idx]
        rt = row['rt']
        
        if np.isnan(rt):
            continue
            
        # First FA lick time
        t_align = t_start + rt
        alignment_times.append(t_align)
        valid_fa_indices.append(idx)

    if not alignment_times:
        return None

    return {
        'session_name': session.session_name if session.session_name else pkl_path.stem,
        'alignment_times': np.array(alignment_times),
        'lick_times': lick_times,
        'n_fa': len(alignment_times)
    }

def plot_rasters(example_data, out_dir):
    """Plot example rasters for Early, Middle, Late sessions."""
    fig, axes = plt.subplots(3, 1, figsize=(6, 12), sharex=True, sharey=True)
    window = [-1.0, 2.0]
    
    titles = ['Early Session', 'Middle Session', 'Late Session']
    
    for ax, data, title in zip(axes, example_data, titles):
        if not data:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center')
            continue
            
        aligns = data['alignment_times']
        licks = data['lick_times']
        
        # Select chronologically distributed subset of trials
        n_show = min(100, len(aligns))
        if len(aligns) > n_show:
            # Pick trials evenly spread across the session
            indices = np.linspace(0, len(aligns) - 1, n_show, dtype=int)
        else:
            indices = np.arange(len(aligns))
            
        subset_aligns = aligns[indices]
        
        # Plot from bottom (early in session) to top (late in session)
        for i, t_ref in enumerate(subset_aligns):
            # Find licks in window
            t_min = t_ref + window[0]
            t_max = t_ref + window[1]
            
            trial_licks = licks[(licks >= t_min) & (licks <= t_max)]
            rel_licks = trial_licks - t_ref
            
            ax.vlines(rel_licks, i, i+0.8, color='black', linewidth=1.5)
            
        ax.set_title(f"{title}\n({data['session_name']})")
        ax.set_ylabel('FA Trial #')
        ax.axvline(0, color='crimson', linestyle='--', label='First FA Lick')
    
    axes[-1].set_xlabel('Time from First FA Lick (s)')
    axes[0].set_xlim(window)
        
    plt.tight_layout()
    plt.savefig(out_dir / "fa_lick_rasters.png", dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default=None,
                        help='Path to manifest CSV (default: canonical)')
    parser.add_argument('--out', required=True)
    parser.add_argument('--pkl-dir', help='Optional override for pkl directory')
    parser.add_argument('--no-filter', action='store_true', help='Bypass SESSION_FILTER')
    args = parser.parse_args()
    
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    manifest = load_staging_manifest(manifest_path=args.manifest,
                                     apply_filter=not args.no_filter)
    # Restore leading zeros
    manifest['session_name'] = manifest['session_name'].apply(lambda x: x.zfill(8) if x.isdigit() and len(x) == 7 else x)
    
    subject = manifest.iloc[0]['subject']
    manifest['parsed_date'] = manifest['session_name'].apply(lambda x: parse_date(f"{subject}_{x}"))
    manifest = manifest.sort_values('parsed_date').reset_index(drop=True)
    
    print(f"Found {len(manifest)} sessions for subject {subject}")
    
    # Process Sessions
    peth_window = [-1.0, 3.0]
    bin_size = 0.05
    time_bins = None
    
    peth_matrix = []
    session_labels = []
    
    # Collect all valid sessions for potential raster plotting
    valid_sessions_data = []
    
    for idx, row in manifest.iterrows():
        session_name = row['session_name']
        
        # Locate pickle
        if 'pkl_path' in row and pd.notna(row['pkl_path']):
            pkl_path = Path(row['pkl_path'])
        elif args.pkl_dir:
            candidates = list(Path(args.pkl_dir).glob(f"*{session_name}*.pkl"))
            pkl_path = candidates[0] if candidates else None
        else:
            # Try to construct default path
            repo_pkls = repo_root / "pkls" / subject
            candidates = list(repo_pkls.glob(f"*{session_name}*.pkl"))
            pkl_path = candidates[0] if candidates else None
            
        if not pkl_path or not pkl_path.exists():
            print(f"Pickle not found for {session_name}: {pkl_path}")
            continue

        print(f"Processing {session_name}...")
            
        data = process_session(pkl_path)
        
        if data:
            # Compute PETH
            t_bins, rate, _ = compute_peth(data['alignment_times'], data['lick_times'], peth_window, bin_size)
            
            if time_bins is None:
                time_bins = t_bins
                
            peth_matrix.append(rate)
            session_labels.append(session_name)
            
            # Check if this qualifies as a valid example (>= 5 FAs)
            if data['n_fa'] >= 5:
                # Store full data - it's lightweight enough (just arrays of times)
                valid_sessions_data.append(data)
            
    # --- Select Examples from actually loaded data ---
    examples = [None, None, None]
    n_valid = len(valid_sessions_data)
    if n_valid >= 1:
        examples[0] = valid_sessions_data[0]
        if n_valid >= 2:
            examples[2] = valid_sessions_data[-1]
        if n_valid >= 3:
            examples[1] = valid_sessions_data[n_valid // 2]
            
    # --- Plotting ---
    peth_matrix = np.array(peth_matrix)
    
    # Heatmap
    plt.figure(figsize=(10, 8))
    # Normalize per session? Maybe not, visualize absolute rate change
    sns.heatmap(peth_matrix, cmap='viridis', robust=True)
    
    # Fix ticks
    # X-axis: Time
    n_bins = len(time_bins)
    x_ticks = np.linspace(0, n_bins-1, 5)
    x_labels = np.linspace(peth_window[0], peth_window[1], 5)
    plt.xticks(x_ticks, [f"{x:.1f}" for x in x_labels], rotation=0)
    
    # Y-axis: Session
    y_ticks = np.arange(0.5, len(session_labels), max(1, len(session_labels)//10))
    y_labels = [session_labels[int(y)] for y in y_ticks]
    plt.yticks(y_ticks, y_labels, rotation=0, fontsize=8)
    
    plt.title(f'FA Lick Rate Across Sessions ({subject})\nAligned to First FA Lick')
    plt.xlabel('Time (s) from First FA Lick')
    plt.ylabel('Session')
    plt.axvline(x=len(time_bins) * abs(peth_window[0]) / (peth_window[1] - peth_window[0]), color='white', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(out_dir / "fa_lick_rate_heatmap.png", dpi=300)
    plt.close()
    
    # Rasters
    plot_rasters(examples, out_dir)
    print(f"Results saved to {out_dir}")

if __name__ == "__main__":
    main()
