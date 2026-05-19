#!/usr/bin/env python3
"""Prepare waveforms for UnitMatch using ALL spikes (full trial, not ITI-only).

This script extracts mean waveforms from raw AP binary files for use with UnitMatch.
Unlike the ITI-only version, this uses ALL spikes during the recording, providing
more data for waveform estimation at the cost of potentially including stimulus-
evoked activity.

The script:
1. Loads session metadata from the staging manifest
2. Finds raw Kilosort/SpikeGLX data on X: drive
3. Loads all spike times (not filtered by ITI)
4. Computes mean waveforms from first/second half of spikes (CV splits)
5. Saves in UnitMatch format: Unit{ID}_RawSpikes.npy with shape (time, channels, 2)

Usage:
    python scripts/analysis/prep_unitmatch_full_trial_waveforms.py
    python scripts/analysis/prep_unitmatch_full_trial_waveforms.py --sessions 01072025 02072025
    python scripts/analysis/prep_unitmatch_full_trial_waveforms.py --n_workers 8

Based on: archive/preparation_scripts/prep_unitmatch_iti_waveforms.py
Modified to use full trial data instead of ITI-only.
"""

import sys
import os
import glob
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
import shutil

# Ensure src is in path
repo_root = Path(__file__).resolve().parents[2]
src_dir = repo_root / 'src'

from visdetect.core.session import load_session


def get_session_paths(manifest_path, subject="BG_046"):
    """
    Read the staging manifest to retrieve session info.
    We need:
    1. Session Name
    2. Path to pickle
    3. Path to Kilosort Raw Data (We need to map this)
    """
    df = pd.read_csv(manifest_path, dtype={'session_name': str})
    
    # We need to find the RAW Kilosort locations. 
    # Since they are on X:, we will search for them.
    base_raw_dir = Path(r"X:\public\projects\BeJG_20230130_VisDetect\wEPhys") / subject / "Processed data"
    
    sessions = []
    for _, row in df.iterrows():
        sess_name = row['session_name']
        # Construct expected raw path
        # Search for Kilosort folder
        search_pattern = base_raw_dir / f"*{sess_name}*" / "Kilosort&Phy" / "*imec0"
        found = list(glob.glob(str(search_pattern)))
        
        # Fallback: less specific
        if not found:
            search_pattern = base_raw_dir / f"*{sess_name}*" / "Kilosort&Phy"
            found_base = list(glob.glob(str(search_pattern)))
            if found_base:
                found = list(Path(found_base[0]).glob("*imec0"))

        raw_ks_path = None
        bin_path = None
        
        if found:
            raw_ks_path = Path(found[0])
            # Look for .bin or .cbin file
            bins = list(raw_ks_path.glob("*.ap.cbin")) + list(raw_ks_path.glob("*.ap.bin"))
            if bins:
                bin_path = bins[0]
        
        sessions.append({
            'name': sess_name,
            'pkl_path': repo_root / row['path'],
            'ks_path': raw_ks_path,
            'bin_path': bin_path
        })
        
    return sessions


def load_spikes(ks_path, good_ids=None):
    """
    Load spike times and clusters from Kilosort folder.
    """
    try:
        st = np.load(ks_path / 'spike_times.npy').flatten()
        sc = np.load(ks_path / 'spike_clusters.npy').flatten()
        return st, sc
    except FileNotFoundError:
        return None, None


def process_session(sess_info, output_root, n_workers=1):
    """
    Process a single session:
    1. Load ALL spike times (no ITI filtering).
    2. Compute mean waveforms for first/second half of spikes per cluster.
    3. Save to UnitMatch structure.
    """
    sess_name = str(sess_info['name'])
    print(f"\nProcessing {sess_name}...")
    
    if not sess_info['ks_path'] or not sess_info['bin_path']:
        print(f"Skipping {sess_name}: No binary data found.")
        return

    # Create output dir
    sess_out_dir = output_root / sess_name
    wav_out_dir = sess_out_dir / "RawWaveforms"
    wav_out_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy essential metadata files for UnitMatch (channel info)
    for f_name in ['channel_positions.npy', 'channel_map.npy', 'params.py', 'cluster_group.tsv', 'cluster_KSLabel.tsv']:
        src = sess_info['ks_path'] / f_name
        dst = sess_out_dir / f_name
        if src.exists():
            try:
                shutil.copy2(src, dst)
            except Exception as e:
                print(f"Warning: Failed to copy {f_name}: {e}")
    
    # Load Session (for good_and_stable_ids filtering)
    try:
        session = load_session(sess_info['pkl_path'])
    except Exception as e:
        print(f"Failed to load pickle for {sess_name}: {e}")
        return
        
    # Load Spikes (Samples)
    spike_times, spike_clusters = load_spikes(sess_info['ks_path'])
    if spike_times is None:
        print(f"Failed to load spikes for {sess_name}")
        return

    # Load Channel Map
    meta_files = list(sess_info['ks_path'].glob("*.meta"))
    n_chan = 385  # Default fallback
    fs = 30000.0
    
    if meta_files:
        with open(meta_files[0], 'r') as f:
            for line in f:
                if 'nSavedChans' in line:
                    n_chan = int(line.split('=')[-1])
                if 'imSampRate' in line:
                    fs = float(line.split('=')[-1])

    # Get unique clusters
    unique_clusters = np.unique(spike_clusters).astype(int)
    
    clusters_to_process = unique_clusters
    
    # FILTER: Use only good_and_stable units if available
    good_and_stable = getattr(session, 'good_and_stable_ids', None)
    
    if good_and_stable is not None:
        print(f"  Session has {len(good_and_stable)} good_and_stable_ids.")
        good_set = set(map(int, good_and_stable))
        clusters_to_process = [c for c in unique_clusters if c in good_set]
        print(f"  Filtering: {len(clusters_to_process)} good_and_stable units selected (from {len(unique_clusters)} total).")
    else:
        # Fallback to good_cluster_ids if exists
        good_ids = getattr(session, 'good_cluster_ids', None)
        if good_ids is not None:
            good_set = set(map(int, good_ids))
            clusters_to_process = [c for c in unique_clusters if c in good_set]
            print(f"  Filtering: {len(clusters_to_process)} good_cluster_ids units selected (fallback).")
        else:
            print(f"  No QC filter available - using all {len(unique_clusters)} clusters.")

    clusters_to_process = sorted(list(clusters_to_process))
    
    # Load Channel Map for Slicing
    chan_map_file = sess_out_dir / 'channel_map.npy'
    chan_map_indices = None
    if chan_map_file.exists():
        chan_map_indices = np.load(chan_map_file).flatten()
        print(f"  Loaded channel map: {len(chan_map_indices)} channels.")
        
    # Initialize Memmap for reading waveforms
    bin_path = sess_info['bin_path']
    try:
        file_size = bin_path.stat().st_size
        n_samples_total = file_size // (n_chan * 2)  # int16
        data_map = np.memmap(bin_path, dtype='int16', mode='r', order='C', shape=(n_samples_total, n_chan))
    except Exception as e:
        print(f"Error mapping binary file {bin_path}: {e}")
        return

    # Waveform params
    n_wf_samples = 82 
    pre_samples = 30
    
    def extract_mean_wf(indices):
        """Extract mean waveform from given spike indices."""
        if len(indices) == 0:
            return np.zeros((n_wf_samples, len(chan_map_indices) if chan_map_indices is not None else n_chan))
        
        # Limit max spikes to average (for speed/memory)
        if len(indices) > 500:
            try:
                indices = np.random.choice(indices, 500, replace=False)
            except ValueError:
                pass
        
        # Sort indices for sequential disk reads (important for network drives)
        indices = np.sort(indices)
        
        times = spike_times[indices]
        
        # Check bounds
        valid_t = (times >= pre_samples) & (times < (n_samples_total - (n_wf_samples - pre_samples)))
        times = times[valid_t]
        
        if len(times) == 0:
            return np.zeros((n_wf_samples, len(chan_map_indices) if chan_map_indices is not None else n_chan))
            
        # Accumulate waveforms
        waveforms = np.zeros((len(times), n_wf_samples, n_chan), dtype='float32')
        
        for i, t in enumerate(times):
            start = int(t - pre_samples)
            end = int(start + n_wf_samples)
            waveforms[i] = data_map[start:end, :]
            
        mean_wf = np.mean(waveforms, axis=0)
        
        # Slice channels if map exists
        if chan_map_indices is not None:
            mean_wf = mean_wf[:, chan_map_indices]
             
        return mean_wf

    def process_one_cluster(cid):
        """Process a single cluster - compute CV-split waveforms."""
        try:
            # Get ALL spikes for this cluster (no ITI filtering!)
            all_idx = np.where(spike_clusters == cid)[0]
            
            if len(all_idx) < 10:
                # Skip units with too few spikes
                return
            
            # Chronological split into two halves (for cross-validation)
            mid = len(all_idx) // 2
            idx_set1 = all_idx[:mid]
            idx_set2 = all_idx[mid:]
            
            if len(idx_set1) < 5 or len(idx_set2) < 5:
                return
                
            wf1 = extract_mean_wf(idx_set1)
            wf2 = extract_mean_wf(idx_set2)
            
            # Stack: (Time, Channels, 2)
            final_wf = np.stack([wf1, wf2], axis=2)
            
            # Save using UnitMatch naming convention
            np.save(wav_out_dir / f"Unit{cid}_RawSpikes.npy", final_wf)
        except Exception as e:
            print(f"Error processing cluster {cid}: {e}")

    # Process clusters
    print(f"  Extracting waveforms for {len(clusters_to_process)} units using {n_workers} workers...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        list(tqdm(executor.map(process_one_cluster, clusters_to_process), 
                  total=len(clusters_to_process), 
                  desc=f"  Units {sess_name}"))

    print(f"  Finished {sess_name}: {len(list(wav_out_dir.glob('*.npy')))} waveforms saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare full-trial waveforms for UnitMatch")
    parser.add_argument('--manifest', type=str, default='data/BG_046_staging_manifest.csv',
                        help='Path to staging manifest CSV')
    parser.add_argument('--output', type=str, default='data/unit_match/input/BG_046',
                        help='Output directory for prepared waveforms')
    parser.add_argument('--n_workers', type=int, default=4,
                        help='Number of parallel workers for waveform extraction')
    parser.add_argument('--sessions', nargs='+', 
                        help='List of session dates (DDMMYYYY) or names to process explicitly')
    args = parser.parse_args()
    
    manifest_path = repo_root / args.manifest
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        sys.exit(1)
    
    sessions = get_session_paths(manifest_path)
    print(f"Found {len(sessions)} sessions in manifest.")
    
    # Filter if sessions specified
    if args.sessions:
        targets = args.sessions
        sessions = [s for s in sessions if any(t in s['name'] for t in targets)]
        print(f"Filtered to {len(sessions)} sessions based on input args: {[s['name'] for s in sessions]}")
    else:
        # Sort by date (most recent first)
        from datetime import datetime
        def parse_date(name):
            parts = name.split('_')
            if len(parts) >= 3:
                d_str = parts[-1] 
                if len(d_str) == 8:
                    try:
                        return datetime.strptime(d_str, "%d%m%Y")
                    except ValueError:
                        return datetime.min
            return datetime.min

        sessions.sort(key=lambda x: parse_date(x['name']), reverse=True)
        print(f"Processing all {len(sessions)} sessions (Latest first): {[s['name'] for s in sessions[:5]]}...")
    
    # Ensure output exists
    output_path = repo_root / args.output
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"FULL-TRIAL WAVEFORM EXTRACTION (NOT ITI-ONLY)")
    print(f"Output: {output_path}")
    print(f"{'='*60}\n")
    
    # Process sessions sequentially (to avoid IO thrashing on X: drive)
    for sess in sessions:
        process_session(sess, output_path, args.n_workers)
    
    print(f"\n{'='*60}")
    print("All sessions processed!")
    print(f"Waveforms saved to: {output_path}")
    print(f"{'='*60}")
