"""
Compare early vs late pre-lick ramp dynamics across sessions.

This script quantifies the shape of the lick response ramp using distinct metrics for different phases:
1. Early Phase (-1.0s to -0.5s): Mean Amplitude (Z-score). 
   - Captures the "level of preparatory recruitment" in the non-linear early phase.
2. Late Phase (-0.4s to -0.2s): Linear Slope (Z-score/s).
   - Captures the "ballistic acceleration" of the pre-lick drive.

Generates plots with independent scales to properly visualize the evolution of both metrics.

Usage:
    python scripts/analysis/lick/compare_slopes.py --manifest data/pkls/BG_046/BG_046_sessions_manifest.csv --out FIGURES/lick/slopes_BG_046
"""
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

# Ensure repo root/src is in path
repo_root = Path(__file__).resolve().parents[3]
if str(repo_root / 'src') not in sys.path:
    sys.path.insert(0, str(repo_root / 'src'))

from visdetect.analysis.config import load_staging_manifest

# --- Helper Functions ---

def parse_date(session_name):
    """Extract date from session name (format *_DDMMYYYY)."""
    try:
        date_str = session_name.split('_')[-1]
        return datetime.strptime(date_str, "%d%m%Y")
    except Exception:
        return datetime.min

def get_session_folder(root, s_name):
    """Robustly find session folder (handles padded/unpadded variations)."""
    p = root / s_name
    if p.exists(): return p
    
    if s_name.startswith('0'):
        p_unpadded = root / s_name.lstrip('0')
        if p_unpadded.exists(): return p_unpadded
        
    return None

def calculate_slope(trace, time_axis, window):
    """
    Calculate the linear slope (Z/s) of the trace within the specified time window.
    """
    t_start, t_end = window
    mask = (time_axis >= t_start) & (time_axis <= t_end)
    
    if np.sum(mask) < 2:
        return np.nan
        
    x = time_axis[mask]
    y = trace[mask]
    
    try:
        slope, intercept = np.polyfit(x, y, 1)
        return slope
    except Exception:
        return np.nan

def calculate_mean_amplitude(trace, time_axis, window):
    """
    Calculate the mean amplitude (Z-score) within the specified time window.
    """
    t_start, t_end = window
    mask = (time_axis >= t_start) & (time_axis <= t_end)
    
    if np.sum(mask) == 0:
        return np.nan
        
    return np.mean(trace[mask])

def load_session_and_compute_metrics(args):
    """
    Worker function: Loads trace and computes metrics for a single session.
    """
    session_name, session_dir, windows = args
    early_win = windows['early']
    late_win = windows['late']
    
    csv_path = session_dir / "lick_responsiveness.csv"
    npz_path = session_dir / "lick_responsiveness.npz"

    if not csv_path.exists() or not npz_path.exists():
        return None

    try:
        # Load metadata and traces
        df = pd.read_csv(csv_path)
        data = np.load(npz_path)
        z_traces = data['z_traces']
        cluster_ids = data['cluster_ids']
        time_axis = data['time_axis']
        
        # Identify "Excited" units
        excited_units = df[(df['is_significant']) & (df['delta_mean'] > 0)]['cluster_id'].values
        
        if len(excited_units) == 0:
            return None
            
        # Compute mean population trace
        mask = np.isin(cluster_ids, excited_units)
        mean_trace = np.nanmean(z_traces[mask], axis=0)

        # Compute Metrics
        mean_early = calculate_mean_amplitude(mean_trace, time_axis, early_win)
        slope_late = calculate_slope(mean_trace, time_axis, late_win)

        return {
            'session': session_name,
            'mean_early': mean_early,
            'slope_late': slope_late,
            'n_units': len(excited_units)
        }
    except Exception as e:
        print(f"Error processing {session_name}: {e}")
        return None

# --- Main Analysis ---

def main():
    parser = argparse.ArgumentParser(description="Compare early vs late pre-lick metrics.")
    parser.add_argument('--manifest', default=None, help='Path to sessions manifest CSV (default: canonical)')
    parser.add_argument('--figures-root', default='FIGURES/lick', help='Root directory for session outputs')
    parser.add_argument('--out', required=True, help='Output directory for plots')
    parser.add_argument('--workers', type=int, default=4, help='Parallel workers')
    
    parser.add_argument('--early-window', nargs=2, type=float, default=[-1.0, -0.5], 
                        help='Early window (s) (default: -1.0 -0.5)')
    parser.add_argument('--late-window', nargs=2, type=float, default=[-0.4, -0.2], 
                        help='Late window (s) (default: -0.4 -0.2)')
    parser.add_argument('--no-filter', action='store_true', help='Bypass SESSION_FILTER')
    
    args = parser.parse_args()

    # 1. Setup
    manifest = load_staging_manifest(manifest_path=args.manifest,
                                     apply_filter=not args.no_filter)
    subject = manifest.iloc[0]['subject']
    
    # Enforce formatting
    manifest['session_name'] = manifest['session_name'].apply(lambda x: x.zfill(8) if x.isdigit() and len(x) == 7 else x)
    manifest['folder_name'] = manifest['session_name'] 
    
    # Parse dates
    manifest['parsed_date'] = manifest['session_name'].apply(lambda x: parse_date(f"{subject}_{x}"))
    manifest = manifest.sort_values('parsed_date').reset_index(drop=True)

    figures_root = Path(args.figures_root)
    subject_folder = figures_root / subject
    search_root = subject_folder if subject_folder.exists() else figures_root
    
    print(f"Comparing Ramp Dynamics for {subject}")
    print(f"  Early Phase (Mean Amp): {args.early_window}")
    print(f"  Late Phase (Slope): {args.late_window}")

    # 2. Prepare Tasks
    windows = {'early': args.early_window, 'late': args.late_window}
    tasks = []
    session_map = {} 
    
    for idx, row in manifest.iterrows():
        s_name = row['folder_name']
        folder = get_session_folder(search_root, s_name)
        
        if folder:
            tasks.append((s_name, folder, windows))
            session_map[s_name] = {
                'date': row['parsed_date'],
                'index': idx,
                'full_name': f"{subject}_{s_name}"
            }

    # 3. Process
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(load_session_and_compute_metrics, t): t[0] for t in tasks}
        
        for future in tqdm(as_completed(futures), total=len(tasks)):
            res = future.result()
            if res is not None:
                meta = session_map[res['session']]
                res['date'] = meta['date']
                res['sort_index'] = meta['index']
                res['label'] = meta['full_name']
                results.append(res)
    
    if not results:
        print("No valid data loaded.")
        return

    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values('sort_index')
    
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 4. Plotting
    sns.set_theme(style="whitegrid")
    
    # --- Dual Subplot: Separate Scales ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    x_axis = np.arange(len(df_res))
    
    # Late Slope (Top)
    ax1.plot(x_axis, df_res['slope_late'], marker='s', color='crimson', linewidth=2, label='Late Slope')
    ax1.set_ylabel('Slope (Z / s)', color='crimson', fontsize=12)
    ax1.set_title(f'Late Phase: Ballistic Acceleration ({args.late_window[0]} to {args.late_window[1]}s)', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.tick_params(axis='y', labelcolor='crimson')
    
    # Early Mean (Bottom)
    ax2.plot(x_axis, df_res['mean_early'], marker='o', color='dodgerblue', linewidth=2, label='Early Amplitude')
    ax2.set_ylabel('Mean Amplitude (Z-score)', color='dodgerblue', fontsize=12)
    ax2.set_title(f'Early Phase: Preparatory Level ({args.early_window[0]} to {args.early_window[1]}s)', fontsize=14)
    ax2.set_xlabel('Session Index', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.tick_params(axis='y', labelcolor='dodgerblue')
    
    plt.tight_layout()
    plt.savefig(output_dir / "ramp_dynamics_metrics.png", dpi=300)
    plt.close()
    
    # --- Scatter ---
    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=df_res, x='mean_early', y='slope_late', 
                    hue='sort_index', palette='coolwarm', s=100, edgecolor='black')
    
    plt.title('Relationship: Early Level vs Late Acceleration')
    plt.xlabel('Early Mean Amplitude (Z)')
    plt.ylabel('Late Slope (Z / s)')
    plt.legend(title='Session', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_dir / "ramp_dynamics_scatter.png", dpi=300)
    plt.close()
    
    df_res.to_csv(output_dir / "ramp_dynamics_results.csv", index=False)
    print(f"Analysis complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
