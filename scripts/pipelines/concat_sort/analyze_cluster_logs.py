#!/usr/bin/env python3
"""Analyze cluster logs for failed KS4 concat-sort jobs.

Reads SLURM output/error logs to identify failure patterns and root causes.
Focuses on the failed jobs identified by check_ks4_completion_status.py.

Usage:
    python scripts/pipelines/concat_sort/analyze_cluster_logs.py [--log-dir path] [--failed-jobs-csv path]
"""

import argparse
import os
import re
import sys
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
import glob

# Default paths based on the SLURM script configuration
DEFAULT_LOG_DIR = "Z:/Documents/ks4/logs_ks4_resort"
DEFAULT_FAILED_JOBS = "ks4_failed_jobs_*.csv"


def find_log_files(log_dir, job_ids):
    """Find SLURM log files for specific job IDs."""
    log_files = {}

    if not os.path.exists(log_dir):
        print(f"Log directory not found: {log_dir}")
        return log_files

    # Pattern: ks4_resort-JOBID_ARRAYID.out/.err
    patterns = [
        "ks4_resort-*_{}.out",
        "ks4_resort-*_{}.err",
        "ks4_resort-{}_*.out",
        "ks4_resort-{}_*.err"
    ]

    for job_id in job_ids:
        for pattern in patterns:
            search_pattern = os.path.join(log_dir, pattern.format(job_id))
            matches = glob.glob(search_pattern)
            if matches:
                for match in matches:
                    basename = os.path.basename(match)
                    if match.endswith('.out'):
                        log_files[job_id] = log_files.get(job_id, {})
                        log_files[job_id]['stdout'] = match
                    elif match.endswith('.err'):
                        log_files[job_id] = log_files.get(job_id, {})
                        log_files[job_id]['stderr'] = match

    return log_files


def extract_error_patterns(log_content):
    """Extract common error patterns from log content."""
    errors = []

    # Common error patterns
    patterns = {
        'cuda_oom': r'(?i)(cuda|gpu).*out of memory|memory.*cuda|RuntimeError.*GPU',
        'timeout': r'(?i)time.*limit|timeout|CANCELLED.*TIME|DUE TO TIME LIMIT',
        'node_failure': r'(?i)node.*fail|hardware.*error|slurm.*node|NODE_FAIL',
        'file_not_found': r'(?i)no such file|file.*not found|FileNotFoundError',
        'permission_denied': r'(?i)permission denied|access denied|PermissionError',
        'import_error': r'(?i)ImportError|ModuleNotFoundError|No module named',
        'segfault': r'(?i)segmentation fault|segfault|core dumped',
        'numpy_error': r'(?i)numpy.*error|array.*error|invalid.*array',
        'kilosort_error': r'(?i)kilosort.*error|spike.*error|template.*error',
        'disk_space': r'(?i)no space left|disk.*full|quota.*exceeded',
        'network_error': r'(?i)network.*error|connection.*error|mount.*error',
        'python_error': r'(?i)python.*error|script.*error|syntax.*error'
    }

    for error_type, pattern in patterns.items():
        if re.search(pattern, log_content, re.MULTILINE):
            errors.append(error_type)

    # Extract specific error lines
    error_lines = []
    lines = log_content.split('\n')
    for i, line in enumerate(lines):
        if any(keyword in line.lower() for keyword in ['error', 'exception', 'fail', 'killed', 'cancelled']):
            # Get context around error
            start = max(0, i-2)
            end = min(len(lines), i+3)
            context = '\n'.join(lines[start:end])
            error_lines.append(context.strip())

    return errors, error_lines[:5]  # Limit to first 5 errors


def analyze_job_log(job_id, log_paths):
    """Analyze logs for a specific job."""
    result = {
        'job_id': job_id,
        'has_stdout': False,
        'has_stderr': False,
        'stdout_size': 0,
        'stderr_size': 0,
        'error_types': [],
        'error_excerpts': [],
        'exit_code': None,
        'runtime_hours': None,
        'last_output': None
    }

    # Analyze stdout
    if 'stdout' in log_paths:
        result['has_stdout'] = True
        try:
            with open(log_paths['stdout'], 'r', encoding='utf-8', errors='ignore') as f:
                stdout_content = f.read()
                result['stdout_size'] = len(stdout_content)

                # Extract runtime and exit status
                if 'Elapsed:' in stdout_content:
                    elapsed_match = re.search(r'Elapsed:\s*(\d+):(\d+):(\d+)', stdout_content)
                    if elapsed_match:
                        h, m, s = map(int, elapsed_match.groups())
                        result['runtime_hours'] = h + m/60 + s/3600

                # Get last few lines
                lines = stdout_content.strip().split('\n')
                if lines:
                    result['last_output'] = '\n'.join(lines[-5:])

                errors, excerpts = extract_error_patterns(stdout_content)
                result['error_types'].extend(errors)
                result['error_excerpts'].extend(excerpts)

        except Exception as e:
            result['error_excerpts'].append(f"Failed to read stdout: {e}")

    # Analyze stderr
    if 'stderr' in log_paths:
        result['has_stderr'] = True
        try:
            with open(log_paths['stderr'], 'r', encoding='utf-8', errors='ignore') as f:
                stderr_content = f.read()
                result['stderr_size'] = len(stderr_content)

                if stderr_content.strip():  # Non-empty stderr
                    errors, excerpts = extract_error_patterns(stderr_content)
                    result['error_types'].extend(errors)
                    result['error_excerpts'].extend(excerpts)

                    # If stderr has content, include it in excerpts
                    if len(stderr_content.strip()) > 0:
                        result['error_excerpts'].insert(0, f"STDERR:\n{stderr_content[:1000]}")

        except Exception as e:
            result['error_excerpts'].append(f"Failed to read stderr: {e}")

    # Remove duplicates
    result['error_types'] = list(set(result['error_types']))

    return result


def analyze_failure_patterns(failed_jobs_df, log_analyses):
    """Analyze patterns in job failures."""
    print("\n" + "="*80)
    print("FAILURE PATTERN ANALYSIS")
    print("="*80)

    # Map analyses by job_id
    analysis_by_job = {r['job_id']: r for r in log_analyses if r['job_id'] in failed_jobs_df['job_id'].values}

    # Overall log availability
    total_failed = len(failed_jobs_df)
    has_logs = sum(1 for r in analysis_by_job.values() if r['has_stdout'] or r['has_stderr'])

    print(f"Failed jobs with logs found: {has_logs}/{total_failed} ({100*has_logs/total_failed:.1f}%)")

    # Error type frequency
    all_error_types = []
    for analysis in analysis_by_job.values():
        all_error_types.extend(analysis['error_types'])

    if all_error_types:
        print(f"\nTop error types:")
        error_counts = Counter(all_error_types)
        for error_type, count in error_counts.most_common(10):
            pct = 100 * count / len(analysis_by_job) if analysis_by_job else 0
            print(f"  {error_type:20s}: {count:3d} jobs ({pct:.1f}%)")

    # Window-based analysis
    print(f"\nFailure analysis by window:")
    window_analysis = defaultdict(list)
    for _, row in failed_jobs_df.iterrows():
        job_id = row['job_id']
        window_idx = row['window_idx']
        analysis = analysis_by_job.get(job_id, {})
        window_analysis[window_idx].append({
            'job_id': job_id,
            'shank_id': row['shank_id'],
            'error_types': analysis.get('error_types', []),
            'has_logs': analysis.get('has_stdout', False) or analysis.get('has_stderr', False)
        })

    for window_idx in sorted(window_analysis.keys()):
        jobs = window_analysis[window_idx]
        error_types = set()
        for job in jobs:
            error_types.update(job['error_types'])

        has_logs_count = sum(1 for job in jobs if job['has_logs'])
        print(f"  Window {window_idx:2d}: {len(jobs)} jobs failed, {has_logs_count} have logs, errors: {sorted(error_types)}")

    # Timeline analysis
    print(f"\nEarly vs Late window comparison:")
    early_windows = [w for w in window_analysis.keys() if w <= 20]
    late_windows = [w for w in window_analysis.keys() if w > 20]

    early_jobs = sum(len(window_analysis[w]) for w in early_windows)
    late_jobs = sum(len(window_analysis[w]) for w in late_windows)

    print(f"  Early windows (0-20): {len(early_windows)} windows, {early_jobs} failed jobs")
    print(f"  Late windows (21+):   {len(late_windows)} windows, {late_jobs} failed jobs")

    return window_analysis, analysis_by_job


def main():
    parser = argparse.ArgumentParser(description="Analyze cluster logs for failed KS4 jobs")
    parser.add_argument("--log-dir", "-l",
                        help=f"SLURM log directory (default: {DEFAULT_LOG_DIR})")
    parser.add_argument("--failed-jobs-csv", "-f",
                        help="CSV file with failed job info (default: find latest ks4_failed_jobs_*.csv)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed error excerpts for each job")

    args = parser.parse_args()

    # Find log directory
    log_dir = args.log_dir or DEFAULT_LOG_DIR

    # Handle network path mapping if needed
    if log_dir.startswith("~/Documents") and not os.path.exists(log_dir):
        # Try Windows equivalent
        windows_log_dir = log_dir.replace("~/Documents", os.path.expanduser("~/Documents"))
        if os.path.exists(windows_log_dir):
            log_dir = windows_log_dir

    print(f"Cluster Log Analysis for Failed KS4 Jobs")
    print("="*60)
    print(f"Log directory: {log_dir}")

    # Find failed jobs CSV
    if args.failed_jobs_csv:
        failed_csv = args.failed_jobs_csv
    else:
        csv_files = glob.glob(DEFAULT_FAILED_JOBS)
        if not csv_files:
            print(f"ERROR: No failed jobs CSV found matching: {DEFAULT_FAILED_JOBS}")
            print("Run check_ks4_completion_status.py first to generate the failed jobs CSV.")
            return 1
        failed_csv = max(csv_files, key=os.path.getmtime)  # Most recent

    print(f"Failed jobs file: {failed_csv}")

    if not os.path.exists(failed_csv):
        print(f"ERROR: Failed jobs CSV not found: {failed_csv}")
        return 1

    # Load failed jobs
    failed_jobs_df = pd.read_csv(failed_csv)
    failed_job_ids = failed_jobs_df['job_id'].tolist()

    print(f"Analyzing {len(failed_job_ids)} failed jobs...")

    # Find log files
    log_files = find_log_files(log_dir, failed_job_ids)

    if not log_files:
        print(f"\nWARNING: No log files found in {log_dir}")
        print("This could mean:")
        print("  1. Logs are in a different directory")
        print("  2. Logs use a different naming pattern")
        print("  3. Jobs haven't been submitted yet")
        print("  4. Network path mapping issue")
        return 1

    print(f"Found logs for {len(log_files)}/{len(failed_job_ids)} failed jobs")

    # Analyze each job's logs
    log_analyses = []
    for job_id in failed_job_ids:
        if job_id in log_files:
            analysis = analyze_job_log(job_id, log_files[job_id])
            log_analyses.append(analysis)
        else:
            # Job with no logs found
            log_analyses.append({
                'job_id': job_id,
                'has_stdout': False,
                'has_stderr': False,
                'error_types': ['no_logs_found'],
                'error_excerpts': ['No log files found for this job'],
            })

    # Pattern analysis
    window_analysis, analysis_by_job = analyze_failure_patterns(failed_jobs_df, log_analyses)

    # Detailed job-by-job analysis
    if args.verbose:
        print(f"\n" + "="*80)
        print("DETAILED JOB ANALYSIS")
        print("="*80)

        for analysis in log_analyses:
            if analysis['error_types']:
                print(f"\nJob {analysis['job_id']}:")
                print(f"  Error types: {analysis['error_types']}")
                print(f"  Has stdout: {analysis['has_stdout']}, stderr: {analysis['has_stderr']}")
                if analysis['runtime_hours']:
                    print(f"  Runtime: {analysis['runtime_hours']:.2f} hours")

                for i, excerpt in enumerate(analysis['error_excerpts'][:2]):  # Limit output
                    print(f"  Error excerpt {i+1}:")
                    for line in excerpt.split('\n')[:5]:  # First 5 lines
                        print(f"    {line}")

    # Save analysis results
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    analysis_df = pd.DataFrame(log_analyses)
    output_file = f"log_analysis_{timestamp}.csv"
    analysis_df.to_csv(output_file, index=False)
    print(f"\nAnalysis saved to: {output_file}")

    # Summary recommendations
    print(f"\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    error_counts = Counter()
    for analysis in log_analyses:
        for error_type in analysis['error_types']:
            error_counts[error_type] += 1

    if error_counts:
        top_error = error_counts.most_common(1)[0][0]
        print(f"Primary failure cause: {top_error}")

        if top_error == 'cuda_oom':
            print("  -> Reduce batch_size or nblocks to lower GPU memory usage")
            print("  -> Request nodes with more GPU memory (A100-80GB)")
        elif top_error == 'timeout':
            print("  -> Increase SLURM time limit (-t)")
            print("  -> Consider shorter time windows or fewer sessions per window")
        elif top_error == 'file_not_found':
            print("  -> Check network storage connectivity and file paths")
            print("  -> Verify binary files exist for failed sessions")
        elif top_error == 'no_logs_found':
            print("  -> Check if jobs were actually submitted to SLURM")
            print("  -> Verify log directory path and network access")

    return 0


if __name__ == "__main__":
    sys.exit(main())