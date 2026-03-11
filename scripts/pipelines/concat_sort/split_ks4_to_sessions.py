"""Re-split KS4 multi-file output into per-session KS-format folders.

After KS4 sorts a concatenated window (5 sessions), this script partitions
spikes back into individual sessions, subtracting the cumulative sample
offset so spike times are in the original per-session timebase.

Since KS4 4.1.1 does NOT produce spike_datasets.npy, session assignment is
inferred from spike_times and per-file sample counts (via binary file sizes).

Usage:
    python scripts/pipelines/concat_sort/split_ks4_to_sessions.py \
        --ks4-run-manifest <path_to_ks4_run_manifest.json>

    By default, output goes alongside the KS4 runs:
      .../concat_sort/split_output/<session>/<shank_X>/<window_YYY>/
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Thread-safe print
_print_lock = threading.Lock()


def tprint(*args, **kwargs):
    """Thread-safe print."""
    with _print_lock:
        print(*args, **kwargs)

REPO_ROOT = Path(__file__).resolve().parents[3]

# ── Path translation ─────────────────────────────────────────────────
LINUX_PREFIX = "/ceph/mrsic_flogel/public/"
WIN_PREFIX = "X:/public/"


def to_local_path(p: str) -> str:
    """Translate Linux cluster path to Windows network path if needed."""
    if p.startswith(LINUX_PREFIX):
        return WIN_PREFIX + p[len(LINUX_PREFIX):]
    return p


# ── File lists ────────────────────────────────────────────────────────

# Files that describe the model (shared across sessions from the same sort)
SHARED_FILES = [
    "templates.npy",
    "whitening_mat.npy",
    "whitening_mat_inv.npy",
    "whitening_mat_dat.npy",
    "channel_map.npy",
    "channel_positions.npy",
    "channel_shanks.npy",
    "chanMap.mat",
    "similar_templates.npy",
    "templates_ind.npy",
    "pc_feature_ind.npy",
    "ops.npy",
]

# Per-spike files that need to be split by session
SPIKE_FILES = [
    "spike_times.npy",
    "spike_clusters.npy",
    "spike_templates.npy",
    "spike_detection_templates.npy",
    "amplitudes.npy",
    "spike_positions.npy",
    "pc_features.npy",
    "kept_spikes.npy",
]

# Cluster-level files (shared, just copied)
CLUSTER_FILES = [
    "cluster_group.tsv",
    "cluster_KSLabel.tsv",
    "cluster_Amplitude.tsv",
    "cluster_ContamPct.tsv",
    "cluster_info.tsv",
]


def compute_sample_boundaries(
    bin_paths: List[str], n_channels: int
) -> tuple:
    """Compute cumulative sample offsets and total samples per file.

    Returns:
        offsets: list of length N where offset[i] is the starting sample
                 index for file i in the concatenated timebase.
        n_samples_per_file: list of sample counts per file.
    """
    offsets = []
    n_samples_per_file = []
    cumulative = 0
    for bp in bin_paths:
        local_bp = to_local_path(bp)
        offsets.append(cumulative)
        file_size = os.path.getsize(local_bp)
        n_samples = file_size // (n_channels * 2)  # int16
        n_samples_per_file.append(n_samples)
        cumulative += n_samples
    return offsets, n_samples_per_file


def assign_spikes_to_sessions(
    spike_times: np.ndarray,
    offsets: List[int],
) -> np.ndarray:
    """Assign each spike to a session using sample offsets.

    Uses np.searchsorted: for each spike time, find the rightmost offset
    that does not exceed it.  This is the session that spike belongs to.

    Returns an int array of session indices (0-based).
    """
    offsets_arr = np.array(offsets, dtype=np.int64)
    # searchsorted('right') gives the index of the first offset > spike_time
    # subtract 1 to get the session whose offset is <= spike_time
    session_idx = np.searchsorted(offsets_arr, spike_times, side="right") - 1
    return session_idx.astype(np.int32)


def split_one_window(
    window_info: dict,
    output_dir: Path,
) -> dict:
    """Split one KS4 run (window x shank) into per-session folders.

    Returns a dict of {session_name: {path, n_spikes, ...}}.
    """
    shank_id = window_info["shank_id"]
    sessions = window_info["sessions"]
    bin_paths = window_info["bin_paths"]
    n_channels = window_info["n_channels"]
    window_idx = window_info["window_idx"]

    run_path = Path(to_local_path(window_info["run_dir"]))
    label = f"W{window_idx:02d}/shk{shank_id}"

    # Check that KS4 output exists
    ks_dir = run_path
    for candidate in [run_path / "kilosort4", run_path / "sorter_output", run_path]:
        if (candidate / "spike_times.npy").exists():
            ks_dir = candidate
            break

    if not (ks_dir / "spike_times.npy").exists():
        tprint(f"  SKIP {label}: no spike_times.npy in {ks_dir}")
        return {}

    # ── Load spike arrays (mmap to avoid 2GB CIFS read limit) ──────
    spike_times = np.array(
        np.load(str(ks_dir / "spike_times.npy"), mmap_mode="r").flatten(),
        dtype=np.int64,
    )
    spike_clusters = np.array(
        np.load(str(ks_dir / "spike_clusters.npy"), mmap_mode="r").flatten()
    )

    # Optional per-spike arrays (must have same length as spike_times)
    optional_spike_arrays = {}
    for fname in ["spike_templates.npy", "spike_detection_templates.npy",
                   "amplitudes.npy", "spike_positions.npy"]:
        fpath = ks_dir / fname
        if fpath.exists():
            arr = np.load(str(fpath), mmap_mode="r")
            # Verify first dimension matches spike count
            if arr.shape[0] == len(spike_times):
                optional_spike_arrays[fname] = arr
            else:
                tprint(f"  SKIP {label} {fname}: shape {arr.shape} != {len(spike_times)} spikes")

    # pc_features is (N, n_features, n_channels) — large, keep as mmap
    pc_features = None
    if (ks_dir / "pc_features.npy").exists():
        pc_feat = np.load(str(ks_dir / "pc_features.npy"), mmap_mode="r")
        if pc_feat.shape[0] == len(spike_times):
            pc_features = pc_feat
        else:
            tprint(f"  SKIP {label} pc_features.npy: shape {pc_feat.shape} != {len(spike_times)} spikes")

    # ── Compute session assignments ───────────────────────────────────
    offsets, n_samples_per_file = compute_sample_boundaries(bin_paths, n_channels)

    # Use spike_datasets.npy if KS4 produced it, otherwise infer
    if (ks_dir / "spike_datasets.npy").exists():
        spike_datasets = np.array(
            np.load(str(ks_dir / "spike_datasets.npy"), mmap_mode="r").flatten(),
            dtype=np.int32,
        )
        tprint(f"  {label}: using spike_datasets.npy for session assignment")
    else:
        spike_datasets = assign_spikes_to_sessions(spike_times, offsets)
        tprint(f"  {label}: inferred session assignment from spike_times + offsets")

    tprint(f"  {label}: {len(spike_times):,} total spikes, offsets={offsets}")

    # Sanity check
    assert len(spike_datasets) == len(spike_times), \
        f"spike_datasets length {len(spike_datasets)} != spike_times {len(spike_times)}"
    # Clamp out-of-range indices (a few spikes may land before offset[0] due
    # to KS4 edge effects — assign them to the first/last session)
    n_neg = int((spike_datasets < 0).sum())
    n_over = int((spike_datasets >= len(sessions)).sum())
    if n_neg > 0:
        tprint(f"    WARNING: {label}: {n_neg} spikes with negative session idx, clamping to 0")
    if n_over > 0:
        tprint(f"    WARNING: {label}: {n_over} spikes with session idx >= {len(sessions)}, clamping")
    spike_datasets = np.clip(spike_datasets, 0, len(sessions) - 1)

    # ── Split per session ─────────────────────────────────────────────
    results = {}
    for file_idx, session_name in enumerate(sessions):
        mask = spike_datasets == file_idx
        n_spikes = int(mask.sum())

        sess_out = output_dir / session_name / f"shank_{shank_id}" / f"window_{window_idx:03d}"
        sess_out.mkdir(parents=True, exist_ok=True)

        # Spike times: subtract offset to get per-session sample indices
        sess_spike_times = spike_times[mask] - offsets[file_idx]

        # Verify no negative spike times
        if sess_spike_times.size > 0 and sess_spike_times.min() < 0:
            n_neg = int((sess_spike_times < 0).sum())
            tprint(f"    WARNING: {label}/{session_name} has {n_neg} negative spike times "
                   f"(min={sess_spike_times.min()}). Clamping to 0.")
            sess_spike_times = np.clip(sess_spike_times, 0, None)

        # Verify no spike times exceed session length
        if sess_spike_times.size > 0 and sess_spike_times.max() >= n_samples_per_file[file_idx]:
            n_over = int((sess_spike_times >= n_samples_per_file[file_idx]).sum())
            tprint(f"    WARNING: {label}/{session_name} has {n_over} spike times >= session length. "
                   f"Max={sess_spike_times.max()}, session_samples={n_samples_per_file[file_idx]}")

        np.save(str(sess_out / "spike_times.npy"), sess_spike_times.astype(np.int64))
        np.save(str(sess_out / "spike_clusters.npy"), spike_clusters[mask])

        # Optional per-spike arrays (mmap'd — indexing creates a copy)
        for fname, arr in optional_spike_arrays.items():
            np.save(str(sess_out / fname), np.array(arr[mask]))

        if pc_features is not None:
            np.save(str(sess_out / "pc_features.npy"), pc_features[mask])

        # Copy shared (model) files
        for fname in SHARED_FILES + CLUSTER_FILES:
            src = ks_dir / fname
            if src.exists():
                shutil.copy2(str(src), str(sess_out / fname))

        # Write per-session params.py pointing to the per-shank binary
        local_bin = to_local_path(bin_paths[file_idx])
        params_content = (
            f'dat_path = r"{local_bin}"\n'
            f"n_channels_dat = {n_channels}\n"
            f"dtype = 'int16'\n"
            f"offset = 0\n"
            f"sample_rate = 30000\n"
            f"hp_filtered = True\n"
        )
        (sess_out / "params.py").write_text(params_content)

        n_clusters = int(len(np.unique(spike_clusters[mask]))) if n_spikes > 0 else 0
        results[session_name] = {
            "path": str(sess_out),
            "window_idx": window_idx,
            "shank_id": shank_id,
            "n_spikes": n_spikes,
            "n_clusters": n_clusters,
        }

    # Print per-session summary as a single block (less interleaving)
    lines = [f"  {label}: split complete →"]
    for session_name, info in results.items():
        lines.append(f"    {session_name}: {info['n_spikes']:,} spikes, {info['n_clusters']} clusters")
    tprint("\n".join(lines))

    return results


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Split KS4 multi-file output into per-session KS folders"
    )
    p.add_argument("--ks4-run-manifest", type=Path, required=True,
                   help="Path to ks4_run_manifest.json")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Output directory (default: sibling of ks4_runs)")
    p.add_argument("--n-workers", type=int, default=1,
                   help="Number of parallel workers (default: 1 = sequential). "
                        "Recommended: 4-8 for network I/O. Memory: ~1-2 GB per "
                        "worker without pc_features, ~8 GB with.")
    args = p.parse_args(argv)

    with open(args.ks4_run_manifest) as f:
        run_manifest = json.load(f)

    # Default output: alongside the ks4_runs directory
    if args.output_dir is None:
        manifest_parent = args.ks4_run_manifest.resolve().parent.parent  # .../concat_sort
        args.output_dir = manifest_parent / "split_output"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    n_windows = len(run_manifest["windows"])
    n_workers = min(args.n_workers, n_windows)

    # ── Resume support: load existing manifest and skip completed windows ──
    manifest_path = args.output_dir / "split_manifest.json"
    all_results = {}
    already_done = set()  # (window_idx, shank_id) tuples
    if manifest_path.exists():
        with open(manifest_path) as f:
            all_results = json.load(f)
        for sess, entries in all_results.items():
            for e in entries:
                already_done.add((e["window_idx"], str(e["shank_id"])))
        print(f"Resuming: {len(already_done)} window-shank pairs already done")

    # Filter to windows that still need processing
    windows_to_run = [
        w for w in run_manifest["windows"]
        if (w["window_idx"], str(w["shank_id"])) not in already_done
    ]
    n_todo = len(windows_to_run)
    n_workers = min(n_workers, max(n_todo, 1))

    print(f"KS4 manifest: {args.ks4_run_manifest}")
    print(f"Output dir:   {args.output_dir}")
    print(f"N windows:    {n_windows} total, {n_todo} remaining")
    print(f"Workers:      {n_workers}")
    print()

    if n_todo == 0:
        print("Nothing to do — all windows already split.")
        return 0

    t0 = time.time()

    if n_workers <= 1:
        # Sequential (original behavior)
        for i, win_info in enumerate(windows_to_run):
            w = win_info["window_idx"]
            s = win_info["shank_id"]
            print(f"── [{i+1}/{n_todo}] Window {w} / Shank {s} ──")
            results = split_one_window(win_info, args.output_dir)
            for session_name, info in results.items():
                all_results.setdefault(session_name, []).append(info)
    else:
        # Parallel with ThreadPoolExecutor (I/O bound — numpy releases GIL)
        completed = 0
        futures = {}
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            for win_info in windows_to_run:
                fut = executor.submit(split_one_window, win_info, args.output_dir)
                futures[fut] = win_info

            for fut in as_completed(futures):
                win_info = futures[fut]
                completed += 1
                try:
                    results = fut.result()
                except Exception as exc:
                    w = win_info["window_idx"]
                    s = win_info["shank_id"]
                    tprint(f"  ERROR W{w:02d}/shk{s}: {exc}")
                    continue
                for session_name, info in results.items():
                    all_results.setdefault(session_name, []).append(info)
                if completed % 20 == 0 or completed == n_todo:
                    tprint(f"  Progress: {completed}/{n_todo} windows")

    elapsed = time.time() - t0

    # Save split manifest (overwrites, but all_results includes prior entries)
    with open(manifest_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSplit manifest → {manifest_path}")

    # Summary
    total_spikes = sum(e["n_spikes"] for entries in all_results.values() for e in entries)
    print(f"\n── Summary ──")
    print(f"  Elapsed: {elapsed/60:.1f} min")
    print(f"  Total spike entries: {total_spikes:,}")
    print(f"  Sessions: {len(all_results)}")
    for session_name, entries in sorted(all_results.items()):
        windows = sorted(set(e["window_idx"] for e in entries))
        shanks = sorted(set(str(e["shank_id"]) for e in entries))
        n_sp = sum(e["n_spikes"] for e in entries)
        print(f"  {session_name}: windows {windows}, shanks {shanks}, {n_sp:,} total spikes")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
