#!/usr/bin/env python3
"""Check completion status of concat-sort KS4 runs.

Reads the ks4_run_manifest.json and checks for ks4_complete.txt markers
in each run directory to determine how many jobs completed successfully.

Usage:
    python scripts/pipelines/concat_sort/check_ks4_completion_status.py

    Or with custom paths:
    python scripts/pipelines/concat_sort/check_ks4_completion_status.py --manifest X:/path/to/ks4_run_manifest.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# Default paths based on the project structure
DEFAULT_MANIFEST = "X:/public/projects/BeJG_20230130_VisDetect/wEPhys/BG_046/concat_sort/ks4_runs/ks4_run_manifest.json"


def check_completion_status(manifest_path: str, verbose: bool = False):
    """Check completion status for all KS4 runs."""

    if not os.path.exists(manifest_path):
        print(f"ERROR: Manifest not found: {manifest_path}")
        return

    print(f"Reading manifest: {manifest_path}")

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    windows = manifest.get('windows', [])
    total_jobs = len(windows)

    print(f"Total jobs in manifest: {total_jobs}")
    print("="*60)

    completed = []
    failed = []
    missing_dirs = []

    for i, win in enumerate(windows):
        job_id = i + 1
        window_idx = win['window_idx']
        shank_id = win['shank_id']
        run_dir = Path(win['run_dir'])
        sessions = win['sessions']

        # Check if run directory exists
        if not run_dir.exists():
            missing_dirs.append({
                'job_id': job_id,
                'window_idx': window_idx,
                'shank_id': shank_id,
                'run_dir': str(run_dir),
                'sessions': sessions
            })
            continue

        # Check for completion marker
        complete_marker = run_dir / "ks4_complete.txt"

        if complete_marker.exists():
            # Read completion info
            try:
                with open(complete_marker, 'r') as f:
                    content = f.read().strip()
                completed_time = None
                elapsed_time = None
                for line in content.split('\n'):
                    if line.startswith('Completed:'):
                        completed_time = line.split('Completed: ')[1]
                    elif line.startswith('Elapsed:'):
                        elapsed_time = line.split('Elapsed: ')[1]

                completed.append({
                    'job_id': job_id,
                    'window_idx': window_idx,
                    'shank_id': shank_id,
                    'run_dir': str(run_dir),
                    'sessions': sessions,
                    'completed_time': completed_time,
                    'elapsed_time': elapsed_time
                })

                if verbose:
                    print(f"OK  Job {job_id:3d}: Window {window_idx}, Shank {shank_id} - COMPLETED ({completed_time})")

            except Exception as e:
                print(f"WARN Job {job_id:3d}: Window {window_idx}, Shank {shank_id} - Marker exists but unreadable: {e}")
        else:
            # Check if any KS4 output files exist (partial run)
            ks4_outputs = [
                'spike_times.npy', 'spike_clusters.npy', 'templates.npy', 'cluster_info.tsv'
            ]
            has_outputs = any((run_dir / f).exists() for f in ks4_outputs)

            failed.append({
                'job_id': job_id,
                'window_idx': window_idx,
                'shank_id': shank_id,
                'run_dir': str(run_dir),
                'sessions': sessions,
                'has_partial_output': has_outputs
            })

            if verbose:
                status_icon = "PART" if has_outputs else "FAIL"
                status_text = "PARTIAL OUTPUT" if has_outputs else "NO OUTPUT"
                print(f"{status_icon} Job {job_id:3d}: Window {window_idx}, Shank {shank_id} - {status_text}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY:")
    print(f"  Total jobs:        {total_jobs}")
    print(f"  OK Completed:       {len(completed):3d} ({100*len(completed)/total_jobs:.1f}%)")
    print(f"  FAIL Failed/Missing: {len(failed):3d} ({100*len(failed)/total_jobs:.1f}%)")
    print(f"  MISS Missing dirs:   {len(missing_dirs):3d} ({100*len(missing_dirs)/total_jobs:.1f}%)")

    if failed:
        print(f"\nFAILED JOBS ({len(failed)}):")
        failed_by_window = {}
        for f in failed:
            window_idx = f['window_idx']
            if window_idx not in failed_by_window:
                failed_by_window[window_idx] = []
            failed_by_window[window_idx].append(f)

        for window_idx in sorted(failed_by_window.keys()):
            jobs_in_window = failed_by_window[window_idx]
            shank_ids = [j['shank_id'] for j in jobs_in_window]
            job_ids = [j['job_id'] for j in jobs_in_window]
            print(f"   Window {window_idx}: Shanks {sorted(shank_ids)} (Jobs {sorted(job_ids)})")

        # Check for patterns
        failed_shanks = [f['shank_id'] for f in failed]
        from collections import Counter
        shank_counts = Counter(failed_shanks)
        if len(set(failed_shanks)) < len(failed_shanks):
            print(f"\n   Shank failure pattern:")
            for shank_id, count in sorted(shank_counts.items()):
                print(f"     Shank {shank_id}: {count} failures")

    if missing_dirs:
        print(f"\nMISSING DIRECTORIES ({len(missing_dirs)}):")
        for md in missing_dirs[:5]:  # Show first 5
            print(f"   Job {md['job_id']}: {md['run_dir']}")
        if len(missing_dirs) > 5:
            print(f"   ... and {len(missing_dirs)-5} more")

    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if completed:
        completed_df = pd.DataFrame(completed)
        completed_file = f"ks4_completed_jobs_{timestamp}.csv"
        completed_df.to_csv(completed_file, index=False)
        print(f"\nCOMPLETED jobs saved to: {completed_file}")

    if failed:
        failed_df = pd.DataFrame(failed)
        failed_file = f"ks4_failed_jobs_{timestamp}.csv"
        failed_df.to_csv(failed_file, index=False)
        print(f"FAILED jobs saved to: {failed_file}")

    return {
        'total': total_jobs,
        'completed': len(completed),
        'failed': len(failed),
        'missing_dirs': len(missing_dirs),
        'completion_rate': len(completed) / total_jobs if total_jobs > 0 else 0
    }


def main():
    parser = argparse.ArgumentParser(description="Check KS4 concat-sort completion status")
    parser.add_argument("--manifest", "-m", default=DEFAULT_MANIFEST,
                        help=f"Path to ks4_run_manifest.json (default: {DEFAULT_MANIFEST})")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show status for each individual job")

    args = parser.parse_args()

    print("KS4 Concat-Sort Completion Status Checker")
    print("="*60)

    status = check_completion_status(args.manifest, verbose=args.verbose)

    # Exit code based on completion
    if status['completion_rate'] == 1.0:
        print("\nSUCCESS: ALL JOBS COMPLETED!")
        return 0
    elif status['completion_rate'] >= 0.9:
        print(f"\nWARNING: Most jobs completed ({status['completion_rate']:.1%})")
        return 0
    else:
        print(f"\nERROR: Significant failures detected ({status['completion_rate']:.1%} complete)")
        return 1


if __name__ == "__main__":
    sys.exit(main())