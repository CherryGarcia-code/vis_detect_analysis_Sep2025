"""
Build a session manifest for a subject (BG_046) and produce per-session behavioral summaries and learning-curve plots.

Usage:
    python scripts/build_manifest_and_behavior_summary.py --subject BG_046

Outputs:
    - data/{subject}_sessions_manifest.csv
    - png_output/learning_analysis/{subject}_learning_curves.png

This script expects the repository package to be importable (run from repo root or set PYTHONPATH).
"""

import argparse
from pathlib import Path
import sys
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ensure repo and src are on sys.path so pickles referencing package modules can be imported
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / 'src'))

# local imports
from src.session_io import load_session
from src import align as align_mod


def summarize_session(pkl_path):
    session = load_session(str(pkl_path))
    # basic metadata
    subject = getattr(session, 'subject', None)
    session_name = getattr(session, 'session_name', None)
    date = getattr(session, 'date', session_name)

    # trials assumed available as session.trials (list-like) with attributes
    n_trials = len(getattr(session, 'trials', []))
    outcomes = []
    rts = []
    change_sizes = []
    for idx, t in enumerate(session.trials):
        out = getattr(t, 'trialoutcome', None)
        outcomes.append(out)
        # Prefer reactiontimes dict used throughout the repo (see src.align.compute_true_reaction_time)
        rt_val = None
        reactiontimes = None
        if not isinstance(t, dict):
            reactiontimes = getattr(t, 'reactiontimes', None)
        else:
            reactiontimes = t.get('reactiontimes', None)

        if reactiontimes:
            # reactiontimes is expected to be a dict with keys like 'RT', 'Miss', 'FA', 'abort'
            if out == 'Hit':
                rt_val = reactiontimes.get('RT', None)
            elif out == 'Miss':
                rt_val = reactiontimes.get('Miss', None)
            elif out in ('FA', 'abort'):
                rt_val = reactiontimes.get(out, None)
        # fallbacks to older attribute names if dict missing
        if rt_val is None:
            rt_val = getattr(t, 'rt', None) or getattr(t, 'reaction_time', None) or getattr(t, 'rt_true', None)

        rts.append(np.nan if rt_val is None else float(rt_val))
        change_sizes.append(getattr(t, 'change_size', np.nan))

    outcomes = np.array(outcomes, dtype=object)
    rts = np.array(rts, dtype=float)
    change_sizes = np.array(change_sizes, dtype=float)

    def frac(name):
        return float(np.sum(outcomes == name)) / max(1.0, n_trials)

    stats = {
        'subject': subject,
        'session_name': session_name,
        'date': date,
        'pkl_path': str(pkl_path.resolve()),
        'n_trials': n_trials,
        'n_hits': int(np.sum(outcomes == 'Hit')),
        'n_miss': int(np.sum(outcomes == 'Miss')),
        'n_fa': int(np.sum(outcomes == 'FA')),
        'n_abort': int(np.sum(outcomes == 'Abort')),
        'hit_rate': frac('Hit'),
        'miss_rate': frac('Miss'),
        'fa_rate': frac('FA'),
        'abort_rate': frac('Abort'),
        'median_rt': float(np.nanmedian(rts)) if np.any(~np.isnan(rts)) else np.nan,
        'mean_rt': float(np.nanmean(rts)) if np.any(~np.isnan(rts)) else np.nan,
    }

    # psychometric: compute Hit probability per unique change size
    unique_sizes = np.unique(change_sizes[~np.isnan(change_sizes)])
    psych = []
    for s in unique_sizes:
        mask = (change_sizes == s)
        if np.sum(mask) == 0:
            continue
        p_hit = float(np.sum(outcomes[mask] == 'Hit')) / float(np.sum(mask))
        psych.append((s, int(np.sum(mask)), p_hit))

    # return rts (latencies in seconds, where available) for downstream histograms
    return stats, psych, rts


def build_manifest(subject='BG_046'):
    root = Path(__file__).resolve().parents[1]
    data_dir = root / 'data'
    out_dir = root / 'png_output' / 'learning_analysis'
    out_dir.mkdir(parents=True, exist_ok=True)

    pattern = f"{subject}_*.pkl"
    files = sorted(data_dir.glob(pattern))
    manifest_rows = []
    psych_by_session = {}
    for p in files:
        try:
            stats, psych, session_rts = summarize_session(p)
        except Exception as e:
            print(f'Failed to load {p}: {e}')
            continue
        # initialize png path placeholders
        stats['psych_png'] = None
        stats['rthist_png'] = None
        manifest_rows.append(stats)
        psych_by_session[stats['session_name']] = psych
        # save per-session psychometric plot
        if psych:
            sizes = [s for s, n, ph in psych]
            ns = [n for s, n, ph in psych]
            phs = [ph for s, n, ph in psych]
            plt.figure(figsize=(4,3))
            plt.plot(sizes, phs, marker='o')
            plt.xlabel('Change size')
            plt.ylabel('P(hit)')
            plt.title(f"Psychometric: {stats['session_name']}")
            plt.tight_layout()
            out_png = out_dir / f"{stats['session_name']}_psychometric.png"
            plt.savefig(out_png, dpi=120)
            plt.close()
            # record in manifest row
            manifest_rows[-1]['psych_png'] = str(out_png)
        # save per-session RT histogram (use rts from summarize_session)
        try:
            rts = np.array([float(x) for x in session_rts if x is not None and not (isinstance(x, float) and np.isnan(x))])
            if rts.size > 0:
                plt.figure(figsize=(4,2.5))
                plt.hist(rts, bins=30)
                plt.xlabel('RT (s)')
                plt.ylabel('Count')
                plt.title(f"RTs: {stats['session_name']}")
                plt.tight_layout()
                out_png = out_dir / f"{stats['session_name']}_rthist.png"
                plt.savefig(out_png, dpi=120)
                plt.close()
                manifest_rows[-1]['rthist_png'] = str(out_png)
        except Exception:
            pass

    manifest_df = pd.DataFrame(manifest_rows)
    if manifest_df.empty:
        print('No sessions found for', subject)
        return None

    manifest_csv = data_dir / f'{subject}_sessions_manifest.csv'
    manifest_df.to_csv(manifest_csv, index=False)
    print('Wrote manifest:', manifest_csv)

    # plot learning curves: aligned by date/session order
    manifest_df_sorted = manifest_df.sort_values('date')
    x = np.arange(len(manifest_df_sorted))

    plt.figure(figsize=(10,4))
    plt.plot(x, manifest_df_sorted['fa_rate'], marker='o', label='FA rate')
    plt.plot(x, manifest_df_sorted['hit_rate'], marker='o', label='Hit rate')
    plt.plot(x, manifest_df_sorted['median_rt'], marker='o', label='Median RT (s)')
    plt.xticks(x, manifest_df_sorted['session_name'], rotation=45, ha='right')
    plt.xlabel('Session')
    plt.ylabel('Value')
    plt.title(f'Learning curves for {subject}')
    plt.legend()
    plt.tight_layout()
    out_png = out_dir / f'{subject}_learning_curves.png'
    plt.savefig(out_png, dpi=150)
    print('Wrote learning curve plot:', out_png)

    return manifest_csv, out_png


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', default='BG_046')
    args = parser.parse_args()
    build_manifest(subject=args.subject)
