#!/usr/bin/env python3
"""Extract mean waveforms from concat-sort output for UnitMatch.

Reads the per-session, per-shank KS4 data produced by the stitching pipeline
and extracts CV-split mean waveforms in the format required by UnitMatchPy.

Data flow:
  final_output/{session}/shank_{N}/  →  spike_times + spike_clusters (global IDs)
  shank_split/{session}/...shankN.ap.bin  →  raw voltage data (96-ch, int16, 30 kHz)
  → output: data/unit_match_concat_sort/input/BG_046/shank_{N}/{session}/RawWaveforms/

Each waveform file:  Unit{global_uid}_RawSpikes.npy  shape (82, n_channels, 2)
  82 time samples (30 pre-peak + 52 post-peak), n_channels, 2 chronological CV halves.

Usage:
    python scripts/pipelines/concat_sort/prep_concat_waveforms.py --shank 0
    python scripts/pipelines/concat_sort/prep_concat_waveforms.py --shank all --n_workers 4
    python scripts/pipelines/concat_sort/prep_concat_waveforms.py --shank 0 --sessions BG_046_01072025
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
import shutil

# Paths
REPO_ROOT = Path(__file__).resolve().parents[3]
FINAL_OUTPUT = Path(r"X:/public/projects/BeJG_20230130_VisDetect/wEPhys/BG_046/concat_sort/final_output")
OUTPUT_ROOT = REPO_ROOT / "data" / "unit_match_concat_sort" / "input" / "BG_046"

# Waveform extraction parameters
N_WF_SAMPLES = 82
PRE_SAMPLES = 30
MAX_SPIKES_PER_HALF = 500
MIN_SPIKES_TOTAL = 10
MIN_SPIKES_PER_HALF = 5


def discover_sessions(shank_id):
    """Find all sessions that have data for the given shank."""
    sessions = []
    for d in sorted(FINAL_OUTPUT.iterdir()):
        if not d.is_dir() or d.name.startswith('_') or d.name.startswith('global'):
            continue
        shank_dir = d / f"shank_{shank_id}"
        if shank_dir.is_dir() and (shank_dir / "spike_clusters.npy").exists():
            sessions.append(d.name)
    return sessions


def parse_params_py(params_path):
    """Extract dat_path, n_channels_dat, sample_rate from params.py."""
    info = {}
    with open(params_path) as f:
        for line in f:
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                key, val = line.split('=', 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'").lstrip('r"').lstrip("r'")
                # Handle r-string prefix properly
                if val.startswith('"') or val.startswith("'"):
                    val = val[1:]
                if val.endswith('"') or val.endswith("'"):
                    val = val[:-1]
                info[key] = val
    return info


def process_session_shank(session_name, shank_id, output_root, n_workers=1):
    """Extract waveforms for one session + shank combination."""
    ks_dir = FINAL_OUTPUT / session_name / f"shank_{shank_id}"
    shank_out = output_root / f"shank_{shank_id}" / session_name
    wav_out = shank_out / "RawWaveforms"

    # Resume: skip if already done
    if wav_out.exists() and len(list(wav_out.glob("Unit*_RawSpikes.npy"))) > 0:
        print(f"  [{session_name}/shank_{shank_id}] Already extracted — skipping.")
        return

    # --- Load spike data ---
    spike_times = np.load(ks_dir / "spike_times.npy", mmap_mode='r').flatten()
    spike_clusters = np.load(ks_dir / "spike_clusters.npy", mmap_mode='r').flatten()

    # --- Read binary path from params.py ---
    params_path = ks_dir / "params.py"
    if not params_path.exists():
        print(f"  [{session_name}/shank_{shank_id}] No params.py — skipping.")
        return

    params_info = parse_params_py(params_path)
    bin_path_str = params_info.get("dat_path", "")
    # Fix r-string artifacts
    bin_path_str = bin_path_str.replace("\\", "/")
    bin_path = Path(bin_path_str)

    n_chan = int(params_info.get("n_channels_dat", 96))
    sample_rate = float(params_info.get("sample_rate", 30000))

    if not bin_path.exists():
        print(f"  [{session_name}/shank_{shank_id}] Binary not found: {bin_path} — skipping.")
        return

    # --- Load channel map ---
    chan_map_path = ks_dir / "channel_map.npy"
    chan_pos_path = ks_dir / "channel_positions.npy"
    if not chan_map_path.exists() or not chan_pos_path.exists():
        print(f"  [{session_name}/shank_{shank_id}] Missing channel map/positions — skipping.")
        return

    channel_map = np.load(chan_map_path).flatten()
    channel_positions = np.load(chan_pos_path)

    # --- Identify clusters to extract ---
    unique_clusters = np.unique(spike_clusters).astype(int)
    print(f"  [{session_name}/shank_{shank_id}] {len(unique_clusters)} clusters, "
          f"{len(spike_times):,} spikes, {n_chan} channels")

    # --- Setup output directory ---
    wav_out.mkdir(parents=True, exist_ok=True)

    # Copy metadata files for UnitMatch
    for fname in ['channel_positions.npy', 'channel_map.npy', 'cluster_KSLabel.tsv']:
        src = ks_dir / fname
        if src.exists():
            shutil.copy2(src, shank_out / fname)

    # Create cluster_group.tsv in the format UnitMatchPy expects.
    # Mark all extracted units as 'good' so UnitMatch processes them.
    # (The KS4 quality filtering can be applied post-hoc on the registry.)
    cg_rows = []
    for cid in unique_clusters:
        cg_rows.append({"cluster_id": int(cid), "KSLabel": "good"})
    pd.DataFrame(cg_rows).to_csv(shank_out / "cluster_group.tsv", sep='\t', index=False)

    # Write a params.py pointing to the shank binary (UnitMatchPy doesn't use it,
    # but keeps provenance)
    with open(shank_out / "params.py", "w") as f:
        f.write(f'dat_path = r"{bin_path}"\n')
        f.write(f"n_channels_dat = {n_chan}\n")
        f.write(f"dtype = 'int16'\n")
        f.write(f"offset = 0\n")
        f.write(f"sample_rate = {int(sample_rate)}\n")
        f.write(f"hp_filtered = True\n")

    # --- Open binary as memmap ---
    file_size = bin_path.stat().st_size
    n_samples_total = file_size // (n_chan * 2)  # int16
    data_map = np.memmap(bin_path, dtype='int16', mode='r', shape=(n_samples_total, n_chan))

    # Pre-compute per-cluster spike indices for fast lookup
    sort_idx = np.argsort(spike_clusters)
    sorted_clusters = spike_clusters[sort_idx]
    boundaries = np.searchsorted(sorted_clusters, unique_clusters, side='left')
    boundaries_right = np.searchsorted(sorted_clusters, unique_clusters, side='right')

    def extract_cv_waveform(cluster_idx):
        """Extract CV-split mean waveform for a single cluster."""
        cid = unique_clusters[cluster_idx]
        left, right = boundaries[cluster_idx], boundaries_right[cluster_idx]
        all_idx = sort_idx[left:right]

        if len(all_idx) < MIN_SPIKES_TOTAL:
            return

        # Chronological split
        mid = len(all_idx) // 2
        halves = [all_idx[:mid], all_idx[mid:]]

        mean_wfs = []
        for half_idx in halves:
            if len(half_idx) < MIN_SPIKES_PER_HALF:
                return

            # Sub-sample if needed
            if len(half_idx) > MAX_SPIKES_PER_HALF:
                rng = np.random.default_rng(seed=cid)
                half_idx = rng.choice(half_idx, MAX_SPIKES_PER_HALF, replace=False)

            # Sort for sequential disk reads
            half_idx = np.sort(half_idx)
            times = spike_times[half_idx]

            # Bounds check
            valid = (times >= PRE_SAMPLES) & (times < (n_samples_total - (N_WF_SAMPLES - PRE_SAMPLES)))
            times = times[valid]
            if len(times) < MIN_SPIKES_PER_HALF:
                return

            # Extract and average
            waveforms = np.zeros((len(times), N_WF_SAMPLES, n_chan), dtype='float32')
            for i, t in enumerate(times):
                start = int(t - PRE_SAMPLES)
                waveforms[i] = data_map[start:start + N_WF_SAMPLES, :]

            mean_wfs.append(np.mean(waveforms, axis=0))

        if len(mean_wfs) != 2:
            return

        # Stack: (time, channels, 2)
        final_wf = np.stack(mean_wfs, axis=2)
        np.save(wav_out / f"Unit{cid}_RawSpikes.npy", final_wf)

    # Run extraction
    if n_workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
            list(tqdm(pool.map(extract_cv_waveform, range(len(unique_clusters))),
                      total=len(unique_clusters),
                      desc=f"  {session_name}/shank_{shank_id}"))
    else:
        for ci in tqdm(range(len(unique_clusters)), desc=f"  {session_name}/shank_{shank_id}"):
            extract_cv_waveform(ci)

    n_saved = len(list(wav_out.glob("Unit*_RawSpikes.npy")))

    # Update cluster_group.tsv to only list clusters with saved waveforms
    saved_ids = set()
    for f in wav_out.glob("Unit*_RawSpikes.npy"):
        uid_str = f.stem.split('_')[0].replace('Unit', '')
        try:
            saved_ids.add(int(uid_str))
        except ValueError:
            pass

    cg_final = [{"cluster_id": cid, "KSLabel": "good"} for cid in sorted(saved_ids)]
    pd.DataFrame(cg_final).to_csv(shank_out / "cluster_group.tsv", sep='\t', index=False)

    print(f"  [{session_name}/shank_{shank_id}] Done: {n_saved}/{len(unique_clusters)} waveforms saved.")


def main():
    parser = argparse.ArgumentParser(
        description="Extract concat-sort waveforms for UnitMatch")
    parser.add_argument('--shank', type=str, required=True,
                        help='Shank ID (0-3) or "all"')
    parser.add_argument('--n_workers', type=int, default=4,
                        help='Threads per session for waveform extraction (default: 4)')
    parser.add_argument('--sessions', nargs='+',
                        help='Specific session names to process (e.g. BG_046_01072025)')
    parser.add_argument('--output', type=str, default=None,
                        help='Override output root (default: data/unit_match_concat_sort/input/BG_046)')
    args = parser.parse_args()

    output_root = Path(args.output) if args.output else OUTPUT_ROOT
    output_root.mkdir(parents=True, exist_ok=True)

    shanks = list(range(4)) if args.shank == 'all' else [int(args.shank)]

    for shank_id in shanks:
        sessions = discover_sessions(shank_id)

        if args.sessions:
            sessions = [s for s in sessions if any(t in s for t in args.sessions)]

        print(f"\n{'='*60}")
        print(f"SHANK {shank_id}: {len(sessions)} sessions")
        print(f"Output: {output_root / f'shank_{shank_id}'}")
        print(f"{'='*60}")

        for sess in sessions:
            process_session_shank(sess, shank_id, output_root, n_workers=args.n_workers)

    print(f"\nAll done. Waveforms saved to {output_root}")


if __name__ == "__main__":
    main()
