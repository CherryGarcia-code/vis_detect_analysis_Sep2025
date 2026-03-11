
"""
Build a session manifest for a subject and produce per-session behavioral summaries and learning-curve plots.

.. deprecated::
    This script is NOT part of the active analysis pipeline.  Session inclusion
    is now determined by ``scripts/analysis/stage_sessions.py`` which writes the
    staging manifest (``data/BG_046_staging_manifest.csv``).  The QC criteria
    here (min_trials, min_hit_rate) are different from the staging manifest's
    d'-based thresholds.  Kept for reference only.

Usage (legacy):
    python scripts/build_manifest_and_behavior_summary.py --subject BG_046 --data_dir pkls --min_trials 500 --min_hit_rate 0.4 --max_miss_rate 1.0
Outputs:
    - data/{subject}_sessions_manifest.csv (included sessions)
    - data/{subject}_excluded_sessions.csv (excluded sessions with reasons)
    - png_output/learning_analysis/{subject}_learning_curves.png
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure repo and src are on sys.path
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / 'src'))

from visdetect.core.session import load_session


def summarize_session(pkl_path):
    session = load_session(str(pkl_path))

    subject = getattr(session, 'subject', None)
    session_name = getattr(session, 'session_name', None)
    date = getattr(session, 'date', session_name)

    n_trials = len(getattr(session, 'trials', []))
    outcomes, rts, change_sizes = [], [], []

    for t in session.trials:
        out = getattr(t, 'trialoutcome', None)
        outcomes.append(out)
        cs = getattr(t, 'change_size', None)
        change_sizes.append(cs if cs is not None else 1.0)

        rt_val = None
        reactiontimes = getattr(t, 'reactiontimes', None) if not isinstance(t, dict) else t.get('reactiontimes', None)
        if reactiontimes:
            if out == 'Hit':
                rt_val = reactiontimes.get('RT', None)
            elif out == 'Miss':
                rt_val = reactiontimes.get('Miss', None)
            elif out in ('FA', 'abort'):
                rt_val = reactiontimes.get(out, None)
        if rt_val is None:
            rt_val = getattr(t, 'rt', None) or getattr(t, 'reaction_time', None) or getattr(t, 'rt_true', None)

        rts.append(np.nan if rt_val is None else float(rt_val))

    outcomes = np.array(outcomes, dtype=object)
    rts = np.array(rts, dtype=float)
    change_sizes = np.array(change_sizes, dtype=float)

    # --- SDT classification using change_size ---
    # Go trial: change_size > 1 (stimulus changed).  Catch trial: change_size ≈ 1.
    is_go = (change_sizes - 1.0) > 0.01
    is_catch = ~is_go

    sdt_hits   = int(np.sum(is_go    & (outcomes == 'Hit')))
    sdt_misses = int(np.sum(is_go    & (outcomes == 'Miss')))
    sdt_fas    = int(np.sum(is_catch & (outcomes == 'Hit')))   # licked on catch = SDT FA
    sdt_crs    = int(np.sum(is_catch & (outcomes == 'Miss')))  # withheld on catch = SDT CR

    n_go    = sdt_hits + sdt_misses
    n_catch = sdt_fas  + sdt_crs

    sdt_hit_rate = sdt_hits / n_go    if n_go    > 0 else 0.0
    sdt_fa_rate  = sdt_fas  / n_catch if n_catch > 0 else 0.0

    # Behavioral label fractions (for QC filtering)
    def frac(name):
        return float(np.sum(outcomes == name)) / max(1.0, n_trials)

    stats = {
        'subject': subject,
        'session_name': session_name,
        'date': date,
        'pkl_path': str(pkl_path.resolve()),
        'n_trials': n_trials,
        'n_hits': int(np.sum(outcomes == 'Hit')),   # behavioral label count
        'n_miss': int(np.sum(outcomes == 'Miss')),   # behavioral label count
        'n_fa': int(np.sum(outcomes == 'FA')),       # early/anticipatory licks (NOT SDT FA)
        'n_abort': int(np.sum(outcomes == 'Abort')),
        'hit_rate': sdt_hit_rate,                    # SDT hit rate (go trials)
        'miss_rate': sdt_misses / n_go if n_go > 0 else 0.0,  # SDT miss rate
        'fa_rate': sdt_fa_rate,                      # SDT FA rate (catch trials)
        'abort_rate': frac('Abort'),
        'n_go': n_go,
        'n_catch': n_catch,
        'median_rt': float(np.nanmedian(rts)) if np.any(~np.isnan(rts)) else np.nan,
        'mean_rt': float(np.nanmean(rts)) if np.any(~np.isnan(rts)) else np.nan,
    }

    return stats


def build_manifest(subject='BG_046', data_dir='data', min_trials=500, min_hit_rate=0.4, max_miss_rate=0.3):
    root = Path(__file__).resolve().parents[2]
    data_dir = root / data_dir
    out_dir = root / 'png_output' / 'learning_analysis'
    out_dir.mkdir(parents=True, exist_ok=True)

    pattern = f"{subject}_*.pkl"
    files = sorted(data_dir.glob(pattern))

    included_rows = []
    excluded_rows = []

    for p in files:
        try:
            stats = summarize_session(p)
        except Exception as e:
            excluded_rows.append({'session_name': p.stem, 'reason': f'Failed to load: {e}'})
            continue

        # Apply quality filter
        reasons = []
        if stats['n_trials'] < min_trials:
            reasons.append(f"n_trials<{min_trials}")
        if stats['hit_rate'] < min_hit_rate:
            reasons.append(f"hit_rate<{min_hit_rate}")
        if stats['miss_rate'] > max_miss_rate:
            reasons.append(f"miss_rate>{max_miss_rate}")

        if reasons:
            excluded_rows.append({'session_name': stats['session_name'], 'reason': ', '.join(reasons)})
            continue

        included_rows.append(stats)

    manifest_df = pd.DataFrame(included_rows)
    excluded_df = pd.DataFrame(excluded_rows)

    if manifest_df.empty:
        print('No sessions passed quality criteria for', subject)
        return None

    # Extract and parse date from session_name
    manifest_df['date'] = manifest_df['session_name'].str.extract(r'_(\\d{2}\\d{2}\\d{4})')[0]
    manifest_df['date'] = pd.to_datetime(manifest_df['date'], format='%d%m%Y', errors='coerce')

    # Sort by date and reset index
    manifest_df_sorted = manifest_df.sort_values('date').reset_index(drop=True)

    # Save included and excluded CSVs
    manifest_csv = data_dir / f'{subject}_sessions_manifest.csv'
    manifest_df_sorted.to_csv(manifest_csv, index=False)
    print('Wrote manifest:', manifest_csv)

    excluded_csv = data_dir / f'{subject}_excluded_sessions.csv'
    excluded_df.to_csv(excluded_csv, index=False)
    print('Wrote excluded sessions list:', excluded_csv)

    # Plot learning curves with session names (chronologically sorted)
    x = np.arange(len(manifest_df_sorted))
    plt.figure(figsize=(10, 4))
    plt.plot(x, manifest_df_sorted['fa_rate'], marker='o', label='FA rate')
    plt.plot(x, manifest_df_sorted['hit_rate'], marker='o', label='Hit rate')
    plt.plot(x, manifest_df_sorted['median_rt'], marker='o', label='Median RT (s)')
    plt.xticks(x, manifest_df_sorted['session_name'], rotation=45, ha='right')  # Keep session names
    plt.xlabel('Session')
    plt.ylabel('Value')
    plt.title(f'Learning curves for {subject} (Filtered)')
    plt.legend()
    plt.tight_layout()
    out_png = out_dir / f'{subject}_learning_curves.png'
    plt.savefig(out_png, dpi=150)
    print('Wrote learning curve plot:', out_png)

    return manifest_csv, excluded_csv, out_png


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', default='BG_046')
    parser.add_argument('--data_dir', default='data', help='Directory containing session .pkl files')
    parser.add_argument('--min_trials', type=int, default=500, help='Minimum number of trials for session inclusion')
    parser.add_argument('--min_hit_rate', type=float, default=0.4, help='Minimum hit rate for session inclusion')
    parser.add_argument('--max_miss_rate', type=float, default=0.3, help='Maximum miss rate for session inclusion')
    args = parser.parse_args()
    build_manifest(subject=args.subject, data_dir=args.data_dir,
                   min_trials=args.min_trials, min_hit_rate=args.min_hit_rate, max_miss_rate=args.max_miss_rate)
