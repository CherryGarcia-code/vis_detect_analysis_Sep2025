"""Extract raw TF pulse times from session PKLs and save to cache.

Re-runs _collect_pulses() for every non-excluded session and saves the
resulting fast_times / slow_times arrays as per-session CSVs.  This is a
lightweight alternative to rebuild_tf_cache.py — it only extracts pulse
onset times and does NOT recompute per-unit z-scored traces.

Sessions are processed in parallel with ProcessPoolExecutor.

Output:
  data/cache/tf_pulse_times/BG_046/<session_name>_tf_pulses.csv
  Each CSV has two columns (fast_times, slow_times), NaN-padded to equal
  length, with absolute times in seconds.

Usage:
  .venv\\Scripts\\python.exe scripts/extract_tf_pulse_times.py
  .venv\\Scripts\\python.exe scripts/extract_tf_pulse_times.py --workers 8
  .venv\\Scripts\\python.exe scripts/extract_tf_pulse_times.py --qc-only
  .venv\\Scripts\\python.exe scripts/extract_tf_pulse_times.py --no-constraints
"""

import argparse
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


import numpy as np
import pandas as pd

from visdetect.analysis.tf_pulse import _collect_pulses, TFRespPulseConfig
from visdetect.analysis.config import (
    STAGING_MANIFEST_PATH,
    SUBJECT,
    ROOT,
    load_staging_manifest,
)
from visdetect.core.session import load_session as _load_session

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


# ── Cache location ────────────────────────────────────────────────────
PULSE_CACHE_DIR = os.path.join(ROOT, "data", "cache", "tf_pulse_times", SUBJECT)


def _resolve_pkl_path(session_name: str) -> Path:
    """Resolve session pkl path, trying both raw and zero-padded names."""
    pkl_dir = Path(ROOT) / "data" / "pkls" / SUBJECT
    sname = str(session_name).zfill(8)
    pkl_path = pkl_dir / f"{SUBJECT}_{sname}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"PKL not found: {pkl_path}")
    return pkl_path


def _process_session(session_name: str, stage: str, cfg: TFRespPulseConfig) -> dict:
    """Worker: load one session, extract pulse times, return result dict."""
    sname = str(session_name).zfill(8)
    try:
        pkl_path = _resolve_pkl_path(session_name)
        sess = _load_session(str(pkl_path))
        fast_times, slow_times = _collect_pulses(sess, cfg)
        del sess
        return {
            "session_name": sname,
            "stage": stage,
            "n_fast": len(fast_times),
            "n_slow": len(slow_times),
            "fast_times": fast_times,
            "slow_times": slow_times,
            "status": "ok",
        }
    except Exception as e:
        return {
            "session_name": sname,
            "stage": stage,
            "n_fast": 0,
            "n_slow": 0,
            "fast_times": np.array([]),
            "slow_times": np.array([]),
            "status": f"error: {e}",
        }


def _save_pulse_csv(out_dir: str, session_name: str, fast_times: np.ndarray, slow_times: np.ndarray) -> str:
    """Save pulse times to a two-column CSV (NaN-padded to equal length)."""
    n = max(len(fast_times), len(slow_times))
    fast_padded = np.full(n, np.nan)
    slow_padded = np.full(n, np.nan)
    fast_padded[:len(fast_times)] = fast_times
    slow_padded[:len(slow_times)] = slow_times

    df = pd.DataFrame({"fast_times": fast_padded, "slow_times": slow_padded})
    csv_path = os.path.join(out_dir, f"{session_name}_tf_pulses.csv")
    df.to_csv(csv_path, index=False)
    return csv_path


def main():
    parser = argparse.ArgumentParser(
        description="Extract raw TF pulse times from session PKLs"
    )
    parser.add_argument(
        "--workers", type=int, default=6,
        help="Number of parallel workers (default: 6)",
    )
    parser.add_argument(
        "--qc-only", action="store_true",
        help="Only process non-excluded sessions (default: all sessions)",
    )
    parser.add_argument(
        "--no-constraints", action="store_true",
        help="Disable temporal constraints (min_after_baseline, etc.)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be done without executing",
    )
    args = parser.parse_args()

    # Build pulse config
    cfg = TFRespPulseConfig(
        kept_only=False,
        use_constraints=not args.no_constraints,
    )

    # Load manifest
    if args.qc_only:
        manifest = load_staging_manifest(qc_only=True)
    else:
        manifest = load_staging_manifest(qc_only=False, apply_filter=False)
        manifest = manifest[manifest["stage"] != "Excluded"]

    sessions = list(zip(
        manifest["session_name"].astype(str).tolist(),
        manifest["stage"].tolist(),
    ))

    # Prepare output directory
    os.makedirs(PULSE_CACHE_DIR, exist_ok=True)

    print("=" * 70)
    print("  TF Pulse Time Extraction")
    print("=" * 70)
    print(f"  Sessions:      {len(sessions)} ({'QC-only' if args.qc_only else 'all non-excluded'})")
    print(f"  Constraints:   {'ON' if cfg.use_constraints else 'OFF'}")
    print(f"  Workers:       {args.workers}")
    print(f"  Output:        {PULSE_CACHE_DIR}")
    print()

    if args.dry_run:
        print("  [DRY RUN] Would extract pulse times for:")
        for sname, stage in sessions:
            sname_padded = str(sname).zfill(8)
            print(f"    {SUBJECT}_{sname_padded}  [{stage:>10s}]")
        return

    t_start = time.time()
    results = []

    if args.workers > 1:
        # Parallel extraction
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(_process_session, sname, stage, cfg): sname
                for sname, stage in sessions
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting"):
                result = future.result()
                results.append(result)
                if result["status"] == "ok":
                    _save_pulse_csv(
                        PULSE_CACHE_DIR,
                        result["session_name"],
                        result["fast_times"],
                        result["slow_times"],
                    )
    else:
        # Sequential extraction
        for sname, stage in tqdm(sessions, desc="Extracting"):
            result = _process_session(sname, stage, cfg)
            results.append(result)
            if result["status"] == "ok":
                _save_pulse_csv(
                    PULSE_CACHE_DIR,
                    result["session_name"],
                    result["fast_times"],
                    result["slow_times"],
                )

    elapsed = time.time() - t_start

    # Sort results chronologically for summary
    results.sort(key=lambda r: r["session_name"])

    # Print summary
    ok = [r for r in results if r["status"] == "ok"]
    errors = [r for r in results if r["status"] != "ok"]

    print()
    print("=" * 70)
    print("  Summary")
    print("=" * 70)
    print(f"  Processed:   {len(ok)}/{len(sessions)} sessions")
    print(f"  Errors:      {len(errors)}")
    if errors:
        for r in errors:
            print(f"    {SUBJECT}_{r['session_name']}  {r['status']}")
    total_fast = sum(r["n_fast"] for r in ok)
    total_slow = sum(r["n_slow"] for r in ok)
    print(f"  Total fast:  {total_fast:,} pulses")
    print(f"  Total slow:  {total_slow:,} pulses")
    print(f"  Elapsed:     {elapsed:.1f} s ({elapsed / 60:.1f} min)")
    print(f"  Output:      {PULSE_CACHE_DIR}")

    # Save a manifest of extracted sessions
    summary_rows = [
        {
            "session_name": r["session_name"],
            "stage": r["stage"],
            "n_fast": r["n_fast"],
            "n_slow": r["n_slow"],
            "status": r["status"],
        }
        for r in results
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(PULSE_CACHE_DIR, "_extraction_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"  Manifest:    {summary_path}")


if __name__ == "__main__":
    main()
