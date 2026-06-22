"""Generate a staging manifest CSV for a subject's session PKL files.

Scans a directory of session PKLs, computes SDT performance metrics for each
session, applies QC gates, and assigns each session a learning stage using a
chronological sliding-window algorithm.

QC Gates (sessions failing any gate are marked 'Excluded'):
  Gate 1 — Minimum trial counts : n_go >= 20, n_catch >= 10
  Gate 2 — Minimum total trials : n_go + n_catch >= 100
  Gate 3 — Minimum engagement   : hit_rate >= 0.10 OR fa_rate >= 0.10
  Gate 4 — Minimum performance  : d' >= threshold (default: 0.8, optional with --skip-dprime-gate)

Stage assignment (one-way, chronological):
  Naive  →  Learning  : 3 of last 4 valid sessions have d' > 1.0
  Learning  →  Expert : 3 of last 4 valid sessions have d' > 1.5
  Expert sessions with d' < 0.5 are labelled 'Disengaged'.

Usage (from project root):
  python scripts/analysis/stage_sessions.py \\
      --subject_dir  data/pkls/BG_046 \\
      --subject_name BG_046 \\
      --output       data/BG_046_staging_manifest.csv

  # With custom d' threshold:
  python scripts/analysis/stage_sessions.py \
      --subject_dir  data/pkls/BG_046 \
      --subject_name BG_046 \
      --output       data/BG_046_staging_manifest.csv \
      --dprime-threshold 1.0

  # Skip d' threshold gate (old behavior):
  python scripts/analysis/stage_sessions.py \
      --subject_dir  data/pkls/BG_046 \
      --subject_name BG_046 \
      --output       data/BG_046_staging_manifest.csv \
      --skip-dprime-gate
Output CSV columns:
  session_name, date, path, hits, misses, fas, crs, n_go, n_catch,
  hit_rate, fa_rate, d_prime, qc_fail, early_licks, aborts, stage

A plain-text staging_log.txt is written to the working directory listing
QC failures and per-session d' values.
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
from datetime import datetime

# Ensure project root is in path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
src_dir = os.path.join(repo_root, 'src')

from visdetect.core.session import load_session
from visdetect.analysis.behavior import compute_session_performance
from visdetect.analysis.config import session_date_key

def stage_sessions(subject_dir, subject_name, output_csv, dprime_threshold=0.8, skip_dprime_gate=False):
    """Generate staging manifest with QC gates including optional d' threshold.
    
    Parameters
    ----------
    subject_dir : str
        Directory containing session PKL files
    subject_name : str
        Subject identifier (e.g., 'BG_046')
    output_csv : str
        Path to output CSV manifest
    dprime_threshold : float, default=0.8
        Minimum d' for session inclusion in QC (ignored if skip_dprime_gate=True)
    skip_dprime_gate : bool, default=False
        If True, skip Gate 4 (d' threshold) completely
    """
    pkl_files = glob.glob(os.path.join(subject_dir, f"{subject_name}_*.pkl"))
    pkl_files.sort()
    
    records = []
    
    log_file = open('staging_log.txt', 'w')
    def log(msg):
        print(msg)
        log_file.write(msg + '\n')
        log_file.flush()
    
    log(f"Found {len(pkl_files)} sessions for {subject_name}")
    
    for pkl_path in pkl_files:
        try:
            session = load_session(pkl_path)
            
            # ── SDT metrics via canonical behavior module ─────────────
            perf = compute_session_performance(session)

            if not perf:
                log(f"Skipping {pkl_path}: no trials in session")
                continue

            true_hits   = perf['n_sdt_hits']
            true_misses = perf['n_sdt_misses']
            true_fas    = perf['n_sdt_fas']
            true_crs    = perf['n_sdt_crs']
            n_go        = perf['n_go']
            n_catch     = perf['n_catch']
            hit_rate    = perf['hit_rate']
            fa_rate     = perf['fa_rate_total']
            early_licks = perf['n_fa']       # behavioral "FA" = early/anticipatory lick
            aborts      = perf['n_abort']

            # QC: Minimum trial constraints + engagement gate + performance threshold
            MIN_GO = 20
            MIN_CATCH = 10       # lowered from 20; 10+ catch trials is sufficient
                                 # for a stable FA rate estimate (was excluding
                                 # sessions like 18082025 with 18 catch trials
                                 # despite excellent performance)
            MIN_TOTAL = 100      # minimum total trials (n_go + n_catch)

            # Calculate d' first (needed for Gate 4)
            d_prime = perf['d_prime']
            
            # Apply QC gates
            qc_fail = False
            qc_reasons = []

            # Gate 1: Minimum trial counts
            if n_go < MIN_GO or n_catch < MIN_CATCH:
                qc_fail = True
                qc_reasons.append(f"low trial count (n_go={n_go}, n_catch={n_catch})")

            # Gate 2: Minimum total trials
            if (n_go + n_catch) < MIN_TOTAL:
                qc_fail = True
                qc_reasons.append(f"insufficient total trials (n_go+n_catch={n_go+n_catch}, min={MIN_TOTAL})")

            # Gate 3: Behavioural engagement — require at least 10% response
            # rate on go OR catch trials. Ensures d' reflects genuine task
            # engagement rather than near-zero licking throughout the session.
            MIN_LICK_RATE = 0.10
            if hit_rate < MIN_LICK_RATE and fa_rate < MIN_LICK_RATE:
                qc_fail = True
                qc_reasons.append(f"low engagement (hit_rate={hit_rate:.3f}, fa_rate={fa_rate:.3f}, min={MIN_LICK_RATE})")

            # Gate 4: Minimum d' threshold for performance quality (optional)
            if not skip_dprime_gate and d_prime < dprime_threshold:
                qc_fail = True
                qc_reasons.append(f"d' below threshold (d'={d_prime:.3f}, min={dprime_threshold})")

            # Mark failed sessions
            if qc_fail:
                d_prime = np.nan
                log(f"QC Fail for {session.session_name}: {'; '.join(qc_reasons)}")
            
            records.append({
                'session_name': session.session_name or os.path.basename(pkl_path).replace('.pkl',''),
                'date': session.session_name.split('_')[-1] if session.session_name else "Unknown",
                'path': pkl_path,
                'hits': true_hits,
                'misses': true_misses,
                'fas': true_fas,
                'crs': true_crs,
                'n_go': n_go,
                'n_catch': n_catch,
                'hit_rate': hit_rate,
                'fa_rate': fa_rate,
                'd_prime': d_prime,
                'qc_fail': qc_fail,
                'early_licks': early_licks,
                'aborts': aborts,
            })
            if not qc_fail:
                log(f"Processed {session.session_name}: d'={d_prime:.2f}")
            
        except Exception as e:
            log(f"Error processing {pkl_path}: {e}")
            continue

    log_file.close()

    df = pd.DataFrame(records)

    if df.empty:
        print(f"No sessions found or processed for {subject_name} in {subject_dir}")
        return

    # ── Staging Logic: Chronological + Performance Threshold ──────────
    #
    # Algorithm (Option 2 — hybrid chronological + performance):
    # 1. Exclude QC-failed sessions (d' = NaN → 'Excluded')
    # 2. Sort remaining sessions chronologically
    # 3. Walk through in order using sustained-performance transitions:
    #    - Start in NAIVE
    #    - Transition to LEARNING when REQUIRED_ABOVE out of the last
    #      WINDOW_SIZE sessions have d' > NAIVE_CEILING
    #    - Transition to EXPERT when REQUIRED_ABOVE out of the last
    #      WINDOW_SIZE sessions have d' > EXPERT_FLOOR
    #    - Transitions are one-way (no reverting to earlier stages)
    # 4. In EXPERT window: sessions with d' < DISENGAGE_CUTOFF → 'Disengaged'
    #
    # Thresholds (chosen based on SDT interpretation):
    #   d' = 1.0  → modest discrimination (above chance)
    #   d' = 1.5  → strong discrimination (reliable expert performance)
    #   d' < 0.5  → near-chance (disengaged / sick day)
    NAIVE_CEILING    = 1.0
    EXPERT_FLOOR     = 1.5
    DISENGAGE_CUTOFF = 0.5
    WINDOW_SIZE      = 4
    REQUIRED_ABOVE   = 3

    # Parse dates for chronological ordering. Use the subject-aware parser so
    # 6-digit DDMMYY tokens (BG_031/038/039) and prefixed/suffixed tokens sort
    # correctly — the old zfill(8)+strptime sent 6-digit names to datetime.min,
    # scrambling the chronological staging.
    def parse_date_for_sort(date_str):
        try:
            return session_date_key(date_str)
        except Exception:
            return (0, 0, 0)

    # Mark QC failures as Excluded
    df['stage'] = ''
    df.loc[df['qc_fail'] == True, 'stage'] = 'Excluded'

    # Sort chronologically
    df['date_parsed'] = df['date'].apply(parse_date_for_sort)
    df = df.sort_values('date_parsed').reset_index(drop=True)

    # Indices of valid (non-QC-fail) sessions in chronological order
    valid_idx = df.index[df['qc_fail'] == False].tolist()

    # Walk through valid sessions and assign stages
    current_stage = 'Naive'
    for pos, i in enumerate(valid_idx):
        d = df.loc[i, 'd_prime']

        # Check for Naive → Learning transition
        if current_stage == 'Naive' and pos >= WINDOW_SIZE - 1:
            window_positions = valid_idx[pos - WINDOW_SIZE + 1 : pos + 1]
            window_vals = df.loc[window_positions, 'd_prime'].values
            if (window_vals > NAIVE_CEILING).sum() >= REQUIRED_ABOVE:
                current_stage = 'Learning'

        # Check for Learning → Expert transition
        elif current_stage == 'Learning' and pos >= WINDOW_SIZE - 1:
            window_positions = valid_idx[pos - WINDOW_SIZE + 1 : pos + 1]
            window_vals = df.loc[window_positions, 'd_prime'].values
            if (window_vals > EXPERT_FLOOR).sum() >= REQUIRED_ABOVE:
                current_stage = 'Expert'

        # Assign stage (flag disengaged Expert sessions)
        if current_stage == 'Expert' and d < DISENGAGE_CUTOFF:
            df.loc[i, 'stage'] = 'Disengaged'
        else:
            df.loc[i, 'stage'] = current_stage

    # Drop helper column and save
    df = df.drop(columns=['date_parsed'])
    df.to_csv(output_csv, index=False)

    # Print summary
    print(f"\nSaved manifest to {output_csv}")
    print(f"\nStaging method: Chronological + Performance Threshold")
    print(f"  Thresholds: Naive ceiling d'={NAIVE_CEILING}, Expert floor d'={EXPERT_FLOOR}")
    print(f"  Transition rule: {REQUIRED_ABOVE}/{WINDOW_SIZE} sessions must exceed threshold")
    print(f"  Disengaged cutoff: d' < {DISENGAGE_CUTOFF} in Expert window")
    print(f"\nStage counts:")
    for stage in ['Naive', 'Learning', 'Expert', 'Disengaged', 'Excluded']:
        n = (df['stage'] == stage).sum()
        if n > 0:
            print(f"  {stage:>12}: {n}")
    print(f"\nChronological assignment:")
    for _, row in df.iterrows():
        sn = str(row['session_name']).zfill(8)
        d_str = f"{row['d_prime']:.3f}" if pd.notna(row['d_prime']) else "  NaN"
        print(f"  {sn} d'={d_str} -> {row['stage']}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_dir', default='data/pkls/BG_046')
    parser.add_argument('--subject_name', default='BG_046')
    parser.add_argument('--output', default='data/BG_046_staging_manifest.csv')
    parser.add_argument('--dprime-threshold', type=float, default=0.8, 
                        help='Minimum d\' for session inclusion (default: 0.8)')
    parser.add_argument('--skip-dprime-gate', action='store_true',
                        help='Skip Gate 4 (d\' threshold) to replicate old behavior')
    args = parser.parse_args()
    
    stage_sessions(args.subject_dir, args.subject_name, args.output, 
                   dprime_threshold=args.dprime_threshold,
                   skip_dprime_gate=args.skip_dprime_gate)

