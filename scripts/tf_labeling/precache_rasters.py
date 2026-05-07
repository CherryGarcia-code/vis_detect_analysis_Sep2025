#!/usr/bin/env python
"""Pre-cache TF pulse raster data for the labeling GUI.

For each unit in the classification CSV, computes spike times aligned to
every fast and slow TF pulse and saves them as a lightweight NPZ file.
This makes the labeling GUI snappy (no session loading at review time).

Usage
-----
    py scripts/tf_labeling/precache_rasters.py
    py scripts/tf_labeling/precache_rasters.py --max-sessions 5   # dev/test
    py scripts/tf_labeling/precache_rasters.py --force             # recompute all

Output
------
    data/cache/tf_raster_cache/{session}_{cluster}_raster.npz
    Each NPZ contains:
      fast_raster:    object array of float arrays (spike times rel to each pulse)
      slow_raster:    same for slow pulses
      n_fast_pulses:  int
      n_slow_pulses:  int
      t_range:        (pre, post) window in seconds
"""
import argparse
import gc
import os
import sys
import time

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

# Ensure project is importable
_script_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(os.path.dirname(_script_dir))
_src = os.path.join(_root, "src")
if _src not in sys.path:
    sys.path.insert(0, _src)
# Also add analysis_suite for loader/config
_suite = os.path.join(_root, "analysis_suite")
if _suite not in sys.path:
    sys.path.insert(0, _suite)

from visdetect.analysis.tf_labeling import (
    CLASSIFICATION_CSV, RASTER_CACHE_DIR,
)
from visdetect.analysis.tf_pulse import (
    TFRespPulseConfig, _collect_pulses,
)
from visdetect.analysis.constants import TF_PULSE_PRE_WINDOW, TF_PULSE_POST_WINDOW, TF_PULSE_TRACE_PRE
from loader import load_session

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# ── Config ─────────────────────────────────────────────────────────────
PRE_WIN = TF_PULSE_PRE_WINDOW   # (-0.4, 0.0)
POST_WIN = TF_PULSE_POST_WINDOW  # (0.0, 0.5)
T_RANGE = (TF_PULSE_TRACE_PRE, POST_WIN[1])  # (-1.0, 0.5) — wider for detrending
MAX_PULSES_PER_RASTER = 2000  # cap for memory; subsample if more


def _raster_path(session_name, cluster_id):
    return os.path.join(RASTER_CACHE_DIR,
                        f"{str(int(session_name)).zfill(8)}_{int(cluster_id)}_raster.npz")


# ── Worker function (module-level for ProcessPoolExecutor on Windows) ──

def _compute_single_raster(args):
    """Compute raster for one unit. Top-level function for pickling.

    args: (cid, spike_times, fast_pulses, slow_pulses, t_range, out_path)
    Returns: (cid, True) on success, (cid, False) on failure.
    """
    cid, spike_times, fast_pulses, slow_pulses, t_range, out_path = args

    spikes = np.sort(spike_times)

    # Build raster: for each pulse, extract spike times in window
    fast_raster = []
    for tp in fast_pulses:
        lo, hi = tp + t_range[0], tp + t_range[1]
        idx_lo = np.searchsorted(spikes, lo)
        idx_hi = np.searchsorted(spikes, hi)
        rel = spikes[idx_lo:idx_hi] - tp
        fast_raster.append(rel)

    slow_raster = []
    for tp in slow_pulses:
        lo, hi = tp + t_range[0], tp + t_range[1]
        idx_lo = np.searchsorted(spikes, lo)
        idx_hi = np.searchsorted(spikes, hi)
        rel = spikes[idx_lo:idx_hi] - tp
        slow_raster.append(rel)

    # Save as NPZ (object arrays for variable-length trials)
    np.savez_compressed(
        out_path,
        fast_raster=np.array(fast_raster, dtype=object),
        slow_raster=np.array(slow_raster, dtype=object),
        n_fast_pulses=len(fast_pulses),
        n_slow_pulses=len(slow_pulses),
        t_range=np.array(t_range),
    )
    return cid, True


def _compute_rasters_for_session(session_name, cluster_ids, force=False,
                                 n_workers=1):
    """Compute pulse-aligned rasters for all units in one session.

    Uses ProcessPoolExecutor for parallel per-unit raster computation.
    Returns the number of units cached.
    """
    # Check which units actually need caching
    if not force:
        needed = [cid for cid in cluster_ids
                  if not os.path.exists(_raster_path(session_name, cid))]
        if not needed:
            return 0
    else:
        needed = list(cluster_ids)

    if not needed:
        return 0

    # Load session (serial — I/O bound)
    sess = load_session(str(int(session_name)))
    if sess is None:
        print(f"  Warning: could not load session {session_name}")
        return 0

    # Collect pulse times (serial — shared data, computed once per session)
    cfg = TFRespPulseConfig()
    fast_times, slow_times = _collect_pulses(sess, cfg)

    if len(fast_times) == 0 and len(slow_times) == 0:
        del sess
        gc.collect()
        return 0

    # Subsample pulses if too many (for memory)
    rng = np.random.default_rng(42)
    if len(fast_times) > MAX_PULSES_PER_RASTER:
        fast_times_r = np.sort(
            rng.choice(fast_times, MAX_PULSES_PER_RASTER, replace=False))
    else:
        fast_times_r = fast_times

    if len(slow_times) > MAX_PULSES_PER_RASTER:
        slow_times_r = np.sort(
            rng.choice(slow_times, MAX_PULSES_PER_RASTER, replace=False))
    else:
        slow_times_r = slow_times

    # Build per-unit worker args (extract spike times while session is loaded)
    worker_args = []
    for cid in needed:
        cluster = next((c for c in sess.clusters
                        if int(c.cluster_id) == int(cid)), None)
        if cluster is None:
            continue
        spike_times = np.asarray(cluster.spike_times, dtype=float).flatten()
        out_path = _raster_path(session_name, cid)
        worker_args.append((
            int(cid), spike_times, fast_times_r, slow_times_r,
            T_RANGE, out_path,
        ))

    # Free session memory before spawning workers
    del sess
    gc.collect()

    if not worker_args:
        return 0

    # Parallel per-unit raster computation
    n_cached = 0
    actual_workers = min(n_workers, len(worker_args))

    if actual_workers > 1:
        with ProcessPoolExecutor(max_workers=actual_workers) as executor:
            futures = {
                executor.submit(_compute_single_raster, a): a[0]
                for a in worker_args
            }
            for future in as_completed(futures):
                cid = futures[future]
                try:
                    _, ok = future.result()
                    if ok:
                        n_cached += 1
                except Exception as e:
                    print(f"    Warning: cluster {cid} failed: {e}")
    else:
        # Single-worker fallback (avoids multiprocessing overhead for small batches)
        for a in worker_args:
            try:
                _, ok = _compute_single_raster(a)
                if ok:
                    n_cached += 1
            except Exception as e:
                print(f"    Warning: cluster {a[0]} failed: {e}")

    return n_cached


def main():
    parser = argparse.ArgumentParser(
        description="Pre-cache rasters for TF labeling GUI")
    parser.add_argument("--force", action="store_true",
                        help="Recompute even if cache exists")
    parser.add_argument("--max-sessions", type=int, default=None,
                        help="Limit to N sessions (for testing)")
    parser.add_argument("--n-workers", type=int, default=None,
                        help="Parallel workers (default: min(cpu_count, 8))")
    args = parser.parse_args()

    n_workers = args.n_workers or min(os.cpu_count() or 1, 8)

    # Load classification CSV to know which units to cache
    if not os.path.exists(CLASSIFICATION_CSV):
        print(f"ERROR: Classification CSV not found: {CLASSIFICATION_CSV}")
        print("Run g_tf_cell_classifier.py first.")
        sys.exit(1)

    df = pd.read_csv(CLASSIFICATION_CSV)
    sessions = df.groupby("session_name")["cluster_id"].apply(list).to_dict()
    print(f"Found {len(df)} units across {len(sessions)} sessions")
    print(f"Workers: {n_workers}")

    # Create cache dir
    os.makedirs(RASTER_CACHE_DIR, exist_ok=True)

    # Process each session
    session_list = sorted(sessions.keys())
    if args.max_sessions:
        session_list = session_list[:args.max_sessions]

    total_cached = 0
    t0 = time.time()

    for i, sname in enumerate(tqdm(session_list, desc="Sessions")):
        cids = sessions[sname]
        print(f"[{i+1}/{len(session_list)}] Session {sname}: "
              f"{len(cids)} units...", end=" ", flush=True)

        n = _compute_rasters_for_session(sname, cids, force=args.force,
                                         n_workers=n_workers)
        total_cached += n

        if n > 0:
            print(f"cached {n}")
        else:
            print("all cached")

    elapsed = time.time() - t0
    print(f"\nDone: {total_cached} rasters cached in {elapsed:.0f}s")
    print(f"Cache dir: {RASTER_CACHE_DIR}")


if __name__ == "__main__":
    main()
