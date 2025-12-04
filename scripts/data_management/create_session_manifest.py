"""
Create a session manifest CSV from a directory of session .pkl files.

Usage:
    python scripts/misc_utils/create_session_manifest.py --pkl-dir pkls/BG_046
"""
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import sys

# Ensure repo root is in path to import visdetect
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from visdetect.core.session import load_session
from visdetect.analysis.behavior import compute_session_performance

def parse_date_from_name(name):
    # Try to parse DDMMYYYY from the end of the string
    # e.g. BG_046_01072025 -> 01072025
    try:
        parts = name.split('_')
        date_str = parts[-1]
        if len(date_str) == 8 and date_str.isdigit():
            return datetime.strptime(date_str, "%d%m%Y")
    except:
        pass
    return datetime.min

def get_session_metrics(pkl_path):
    try:
        session = load_session(pkl_path)
    except Exception as e:
        print(f"Error loading {pkl_path}: {e}")
        return None

    # Use the centralized behavior analysis function
    perf = compute_session_performance(session)
    if not perf:
        return None

    return {
        'subject': session.subject if session.subject else 'Unknown',
        'session_name': session.session_name if session.session_name else pkl_path.stem,
        'date_obj': parse_date_from_name(session.session_name if session.session_name else pkl_path.stem),
        'pkl_path': str(pkl_path.absolute()),
        'n_trials': perf['n_trials'],
        'n_hits': perf['n_hits'],
        'n_miss': perf['n_miss'],
        'n_fa': perf['n_fa'],
        'n_fa_early': perf['n_fa_early'],
        'n_fa_late': perf['n_fa_late'],
        'n_abort': perf['n_abort'],
        'hit_rate': perf['hit_rate'],
        'miss_rate': perf['miss_rate'],
        'hit_rate_no_size_1': perf['hit_rate_no_size_1'],
        'fa_rate': perf['fa_rate_total'],
        'abort_rate': perf['abort_rate'],
        'fraction_hit': perf['fraction_hit'],
        'fraction_miss': perf['fraction_miss'],
        'fraction_fa': perf['fraction_fa'],
        'fraction_abort': perf['fraction_abort'],
        'median_rt': perf['median_rt_hit'],
        'mean_rt': perf['mean_rt_hit'],
        'mean_rt_fa_early': perf['mean_rt_fa_early'],
        'mean_rt_fa_late': perf['mean_rt_fa_late'],
        'sem_rt_hit': perf['sem_rt_hit'],
        'sem_rt_fa_early': perf['sem_rt_fa_early'],
        'sem_rt_fa_late': perf['sem_rt_fa_late'],
        'd_prime': perf['d_prime']
    }

def main():
    parser = argparse.ArgumentParser(description="Create session manifest from pkl directory.")
    parser.add_argument('--pkl-dir', required=True, help='Directory containing .pkl files')
    parser.add_argument('--out', help='Output CSV path (optional)')
    parser.add_argument('--recursive', action='store_true', help='Search recursively in subdirectories')
    args = parser.parse_args()

    pkl_dir = Path(args.pkl_dir)
    if not pkl_dir.exists():
        print(f"Directory not found: {pkl_dir}")
        return

    pattern = "**/*.pkl" if args.recursive else "*.pkl"
    pkl_files = list(pkl_dir.glob(pattern))
    
    if not pkl_files:
        print(f"No .pkl files found in {pkl_dir}")
        return

    print(f"Found {len(pkl_files)} sessions. Processing...")
    
    rows = []
    for pkl in tqdm(pkl_files):
        metrics = get_session_metrics(pkl)
        if metrics:
            rows.append(metrics)

    if not rows:
        print("No valid session data extracted.")
        return

    df = pd.DataFrame(rows)
    
    # Ensure session_name is string and padded
    df['session_name'] = df['session_name'].astype(str)
    # If session_name is numeric and 7 digits, pad with 0
    df['session_name'] = df['session_name'].apply(lambda x: x.zfill(8) if x.isdigit() and len(x) == 7 else x)
    
    # Sort by date
    df = df.sort_values('date_obj').drop(columns=['date_obj'])
    
    # Define output path
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        # If multiple subjects are found, maybe name it generic, but usually this is run per subject folder
        # The user requested saving it in the same folder.
        # Let's try to guess a good name. If all subjects are the same, use that.
        subjects = df['subject'].unique()
        if len(subjects) == 1:
            filename = f"{subjects[0]}_sessions_manifest.csv"
        else:
            filename = "sessions_manifest.csv"
            
        out_path = pkl_dir / filename
        
    df.to_csv(out_path, index=False)
    
    print(f"Manifest saved to {out_path}")
    print(df[['session_name', 'n_trials', 'hit_rate']].head())

if __name__ == "__main__":
    main()
