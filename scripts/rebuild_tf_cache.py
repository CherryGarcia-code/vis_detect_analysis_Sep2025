"""Rebuild all TF pulse NPZ trace caches from scratch.

Deletes existing NPZ files, then recomputes TF pulse traces for every
session in the staging manifest using the updated PKL files.

Usage:
  .venv\\Scripts\\python.exe scripts/rebuild_tf_cache.py
  .venv\\Scripts\\python.exe scripts/rebuild_tf_cache.py --workers 6
  .venv\\Scripts\\python.exe scripts/rebuild_tf_cache.py --qc-only   # non-excluded only
"""

import argparse
import gc
import os
import sys
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

import numpy as np
import pandas as pd

from visdetect.analysis.tf_pulse import collect_tf_pulse_traces, TFRespPulseConfig
from visdetect.analysis.config import STAGING_MANIFEST_PATH, TF_TRACES_DIR, SUBJECT
from visdetect.analysis.constants import TF_PULSE_TRACE_PRE
from visdetect.core.session import load_session as _load_session

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def load_session(session_name):
    """Load a session pkl by name (tries standard paths)."""
    sname = str(session_name)
    pkl_dir = repo_root / "data" / "pkls" / SUBJECT
    pkl_path = pkl_dir / f"{SUBJECT}_{sname}.pkl"
    if not pkl_path.exists():
        # Try zero-padded
        sname_padded = sname.zfill(8)
        pkl_path = pkl_dir / f"{SUBJECT}_{sname_padded}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"PKL not found: {pkl_path}")
    return _load_session(str(pkl_path))


def main():
    parser = argparse.ArgumentParser(description="Rebuild TF pulse NPZ caches from session PKLs")
    parser.add_argument("--workers", type=int, default=6,
                        help="Workers for parallel TF trace computation per session (default: 6)")
    parser.add_argument("--qc-only", action="store_true",
                        help="Only rebuild non-excluded sessions (default: all sessions)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without executing")
    args = parser.parse_args()

    # Load manifest
    manifest = pd.read_csv(STAGING_MANIFEST_PATH, dtype={"session_name": str})
    if args.qc_only:
        manifest = manifest[manifest["stage"] != "Excluded"]
    sessions = manifest[["session_name", "stage"]].values.tolist()

    cache_dir = Path(TF_TRACES_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Count existing NPZ files
    existing_npz = list(cache_dir.glob("*.npz"))
    print("=" * 70)
    print("  TF Pulse NPZ Cache Rebuild")
    print("=" * 70)
    print(f"  Sessions to process: {len(sessions)} ({'QC-only' if args.qc_only else 'all'})")
    print(f"  Existing NPZ files:  {len(existing_npz)}")
    print(f"  Cache dir:           {cache_dir}")
    print(f"  Workers per session: {args.workers}")
    print()

    if args.dry_run:
        print("  [DRY RUN] Would delete existing NPZ files and recompute.")
        for sname, stage in sessions:
            sname_padded = str(sname).zfill(8)
            npz_path = cache_dir / f"{SUBJECT}_{sname_padded}_traces.npz"
            exists = npz_path.exists()
            print(f"    {SUBJECT}_{sname_padded}  [{stage:>10s}]  {'exists' if exists else 'MISSING'}")
        return

    # Phase 1: Delete existing NPZ files
    print("-- Phase 1: Deleting old NPZ caches --")
    deleted = 0
    for npz_file in existing_npz:
        npz_file.unlink()
        deleted += 1
    print(f"  Deleted {deleted} NPZ files.\n")

    # Phase 2: Recompute from PKLs
    print("-- Phase 2: Recomputing TF traces from PKLs --")
    cfg = TFRespPulseConfig(kept_only=False, use_constraints=True,
                            trace_pre=TF_PULSE_TRACE_PRE)
    n_workers = args.workers

    results = []
    t_start = time.time()

    for sname, stage in tqdm(sessions, desc="Sessions"):
        sname_padded = str(sname).zfill(8)
        npz_path = cache_dir / f"{SUBJECT}_{sname_padded}_traces.npz"

        try:
            sess = load_session(sname)
        except FileNotFoundError as e:
            print(f"\n  SKIP {sname}: {e}")
            results.append((sname, stage, 0, "pkl_missing"))
            continue

        n_clusters = len(sess.clusters) if hasattr(sess, "clusters") else 0

        try:
            t_vec, entries = collect_tf_pulse_traces(
                sess,
                cfg=cfg,
                parallel=(n_workers > 1),
                show_progress=False,
                n_workers=n_workers,
                cache_path=str(npz_path),
            )
            n_units = len(entries)
            results.append((sname, stage, n_units, "ok"))
        except Exception as e:
            print(f"\n  ERROR {sname}: {e}")
            results.append((sname, stage, 0, f"error: {e}"))

        del sess
        gc.collect()

    elapsed = time.time() - t_start

    # Summary
    print("\n" + "=" * 70)
    print("  Rebuild Summary")
    print("=" * 70)
    ok_count = sum(1 for r in results if r[3] == "ok")
    total_units = sum(r[2] for r in results)
    missing = sum(1 for r in results if r[3] == "pkl_missing")
    errors = sum(1 for r in results if r[3].startswith("error"))
    print(f"  Processed:      {ok_count}/{len(sessions)} sessions")
    print(f"  Total units:    {total_units}")
    print(f"  PKLs missing:   {missing}")
    print(f"  Errors:         {errors}")
    print(f"  Elapsed:        {elapsed:.1f} s ({elapsed/60:.1f} min)")
    print()

    # Verify new NPZ files
    new_npz = list(cache_dir.glob("*.npz"))
    print(f"  New NPZ files:  {len(new_npz)}")

    # Show per-session detail for failures
    failures = [r for r in results if r[3] != "ok"]
    if failures:
        print("\n  Failures:")
        for sname, stage, _, reason in failures:
            print(f"    {sname} [{stage}]: {reason}")

    print("\nDone.")


if __name__ == "__main__":
    main()
