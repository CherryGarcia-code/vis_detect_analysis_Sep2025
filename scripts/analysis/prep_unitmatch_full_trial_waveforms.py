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


def _resolve_ks_dirs(session_dir):
    """Resolve the probe folder and Kilosort-output folder for one session.

    Mirrors ``visdetect.core.ingest._find_probe_folder`` + the kilosort4/
    fallback: KS4 sometimes writes its .npy outputs into a ``kilosort4/``
    subfolder of the probe folder (BG_038/BG_039 + most BG_031 sessions),
    while older runs (BG_046, some BG_031) put them directly in the probe
    folder. The raw ``.ap.bin``/``.ap.meta`` always live in the probe folder.

    Returns ``(probe_dir, ks_out_dir, bin_path)`` (any may be None).
    """
    ks_phy = session_dir / "Kilosort&Phy"
    if not ks_phy.exists():
        return None, None, None
    candidates = [d for d in ks_phy.iterdir()
                  if d.is_dir() and not d.name.startswith('.') and 'Sorted' not in d.name]
    if not candidates:
        return None, None, None
    # Prefer an *imec0* probe folder; else the first directory (one probe here).
    probe_dir = next((d for d in candidates if 'imec0' in d.name.lower()),
                     sorted(candidates)[0])

    # Use kilosort4/ only if it actually carries spike data; else the probe dir.
    ks4 = probe_dir / "kilosort4"
    ks_out_dir = ks4 if (ks4 / "spike_times.npy").exists() else probe_dir

    bins = list(probe_dir.glob("*.ap.bin")) + list(probe_dir.glob("*.ap.cbin"))
    bin_path = bins[0] if bins else None
    return probe_dir, ks_out_dir, bin_path


def get_session_paths(manifest_path, subject="BG_046", processed_root=None):
    """
    Read the manifest to retrieve session info.
    We need:
    1. Session Name
    2. Path to pickle
    3. Path to Kilosort Raw Data (probe folder + KS-output folder)

    ``processed_root`` overrides where the per-subject Processed-data folders
    (each holding ``Kilosort&Phy/<probe>/*.ap.bin``) live. Pass the native ceph
    path when running on the cluster (e.g.
    ``/ceph/.../wEPhys/<subject>/Processed data``) -- there is NO ``X:`` drive on
    Linux, so the default below resolves to a nonexistent path and every session
    is skipped "No binary data found". If omitted it falls back to the Windows
    ``X:`` mount for local BG_046 runs.
    """
    df = pd.read_csv(manifest_path, dtype={'session_name': str})

    # Per-subject Processed-data root holding Kilosort&Phy/<probe>/*.ap.bin.
    # On the cluster this MUST be the native ceph path (passed via
    # --processed-root); locally it defaults to the X: mount.
    if processed_root:
        base_raw_dir = Path(processed_root)
    else:
        base_raw_dir = Path(r"X:\public\projects\BeJG_20230130_VisDetect\wEPhys") / subject / "Processed data"

    sessions = []
    for _, row in df.iterrows():
        sess_name = row['session_name']
        # Exact folder match first (session_name == X: Processed-data folder
        # name for the per-subject manifests); glob fallback keeps BG_046's
        # bare-date manifest (session_name '23062025' -> 'BG_046_23062025/')
        # working. Exact-first also avoids the '..._v2' substring ambiguity.
        session_dir = base_raw_dir / str(sess_name)
        if not session_dir.exists():
            cand = sorted(glob.glob(str(base_raw_dir / f"*{sess_name}*")))
            session_dir = Path(cand[0]) if cand else session_dir

        probe_dir, ks_out_dir, bin_path = _resolve_ks_dirs(session_dir)

        sessions.append({
            'name': sess_name,
            'pkl_path': repo_root / row['path'],
            'ks_path': ks_out_dir,    # KS .npy outputs (kilosort4/ or probe dir)
            'probe_dir': probe_dir,   # holds the raw .ap.bin / .ap.meta
            'bin_path': bin_path,
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


def write_curated_cluster_group(sess_out_dir):
    """Rewrite cluster_group.tsv so 'good' == exactly the units we extracted.

    UnitMatch's ``load_good_waveforms`` selects every row whose label column is
    ``'good'`` and then loads that unit's ``Unit{ID}_RawSpikes.npy``. The
    cluster_group.tsv copied straight from Kilosort labels the FULL KS 'good'
    set (a superset of the good_and_stable units we extract, and of units that
    had >=10 spikes), so UnitMatch would try to load files that don't exist and
    crash (UnboundLocalError on the first missing unit). We therefore label as
    'good' exactly the units that have a RawWaveforms file -- mirroring BG_046's
    curated tsv (good-count == waveform-count). Returns the number of good rows.
    """
    from pathlib import Path as _P
    wav_dir = _P(sess_out_dir) / "RawWaveforms"
    ids = sorted(int(p.name[4:].split('_')[0])
                 for p in wav_dir.glob("Unit*_RawSpikes.npy"))
    lines = ["cluster_id\tKSLabel"] + [f"{i}\tgood" for i in ids]
    # write_bytes => LF endings (file is consumed by UnitMatch on Linux/ceph)
    (_P(sess_out_dir) / "cluster_group.tsv").write_bytes(("\n".join(lines) + "\n").encode())
    return len(ids)


def process_session(sess_info, output_root, n_workers=1, skip_existing=True):
    """
    Process a single session:
    1. Load ALL spike times (no ITI filtering).
    2. Compute mean waveforms for first/second half of spikes per cluster.
    3. Save to UnitMatch structure.

    A ``_extraction_complete.txt`` marker is written on success; when
    ``skip_existing`` is True (default) a session with that marker is skipped,
    making the (multi-hour) full run safely resumable after an interrupt.
    """
    sess_name = str(sess_info['name'])
    print(f"\nProcessing {sess_name}...")

    # Resumability: skip sessions already finished in a previous run.
    sess_out_dir = output_root / sess_name
    if skip_existing and (sess_out_dir / "_extraction_complete.txt").exists():
        print(f"Skipping {sess_name}: already complete (marker present).")
        return

    if not sess_info['ks_path'] or not sess_info['bin_path']:
        print(f"Skipping {sess_name}: No binary data found.")
        return

    # Create output dir
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

    # Load .ap.meta from the PROBE folder (the kilosort4/ subfolder has no
    # .meta; reading nSavedChans wrong would corrupt the memmap reshape).
    probe_dir = sess_info.get('probe_dir') or sess_info['ks_path']
    meta_files = list(probe_dir.glob("*.meta"))
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

    n_saved = len(list(wav_out_dir.glob('*.npy')))
    # Curate cluster_group.tsv so UnitMatch's 'good' set == the units we saved.
    n_good = write_curated_cluster_group(sess_out_dir)
    print(f"  Finished {sess_name}: {n_saved} waveforms saved "
          f"(cluster_group.tsv: {n_good} good).")
    # Completion marker for resumable re-runs (see skip_existing).
    (sess_out_dir / "_extraction_complete.txt").write_text(
        f"{n_saved} waveforms\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare full-trial waveforms for UnitMatch")
    parser.add_argument('--subject', type=str, default='BG_046',
                        help='Subject ID; sets the X: Processed-data root')
    parser.add_argument('--processed-root', type=str, default=None,
                        help='Per-subject Processed-data root holding '
                             'Kilosort&Phy/<probe>/*.ap.bin. Pass the native ceph path '
                             'on the cluster (no X: drive on Linux); defaults to the X: '
                             'mount for local BG_046 runs.')
    parser.add_argument('--manifest', type=str, default='data/BG_046_staging_manifest.csv',
                        help='Path to manifest CSV (needs session_name + path columns)')
    parser.add_argument('--output', type=str, default='data/unit_match/input/BG_046',
                        help='Output directory for prepared waveforms')
    parser.add_argument('--n_workers', type=int, default=4,
                        help='Number of parallel workers for waveform extraction')
    parser.add_argument('--sessions', nargs='+',
                        help='List of session dates (DDMMYYYY) or names to process explicitly')
    parser.add_argument('--no-skip-existing', action='store_true',
                        help='Re-extract sessions even if a completion marker exists')
    args = parser.parse_args()

    manifest_path = repo_root / args.manifest
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        sys.exit(1)

    sessions = get_session_paths(manifest_path, subject=args.subject,
                                 processed_root=args.processed_root)
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
        process_session(sess, output_path, args.n_workers,
                        skip_existing=not args.no_skip_existing)
    
    # Fail LOUD if nothing was produced. The extractor prints "Skipping ...:
    # No binary data found." per session and would otherwise exit 0 even when
    # EVERY session was skipped (e.g. wrong --processed-root => no .ap.bin found),
    # making a total no-op look like success ("PREP DONE"). UnitMatch needs >=1
    # session with waveforms, so a zero-yield run must be a hard error.
    done = [str(s['name']) for s in sessions
            if (output_path / str(s['name']) / "_extraction_complete.txt").exists()]
    print(f"\n{'='*60}")
    print(f"{len(done)}/{len(sessions)} sessions have extracted waveforms.")
    if not done:
        print("ERROR: NO sessions produced waveforms. Check --processed-root points at "
              "the dir holding Kilosort&Phy/<probe>/*.ap.bin (ceph path on the cluster).")
        sys.exit(2)
    print(f"Waveforms saved to: {output_path}")
    print(f"{'='*60}")
