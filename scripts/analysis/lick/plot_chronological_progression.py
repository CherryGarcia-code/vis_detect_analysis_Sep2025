"""
Plot chronological progression of lick responses across sessions.
Replicates "Population mean traces" and "Pre-0 peak times" visualizations.
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
import matplotlib.cm as cm
import sys


from visdetect.analysis.config import load_staging_manifest

# --- Helper Functions (Shared Logic) ---

def parse_date(session_name):
    """Extract date from session name (format *_DDMMYYYY)."""
    try:
        date_str = session_name.split('_')[-1]
        return datetime.strptime(date_str, "%d%m%Y")
    except Exception:
        return datetime.min

def get_session_folder(root, s_name):
    """Robustly find session folder (handles padded/unpadded variations)."""
    # 1. Try exact match (e.g. 01072025)
    p = root / s_name
    if p.exists(): return p
    
    # 2. Try unpadded match (e.g. 1072025)
    if s_name.startswith('0'):
        p_unpadded = root / s_name.lstrip('0')
        if p_unpadded.exists(): return p_unpadded
        
    return None

def load_session_trace(args):
    """
    Worker function: Loads the mean Z-trace for EXCITED units for a single session.
    """
    session_name, session_dir = args
    
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
        
        # Identify "Excited" units (Significant & Positive Modulation)
        # Note: We rely on the 'delta_mean' metric calculated in the pipeline
        excited_units = df[(df['is_significant']) & (df['delta_mean'] > 0)]['cluster_id'].values
        
        if len(excited_units) == 0:
            # If no excited units, return a zero trace (or None if strict)
            mean_trace = np.zeros_like(time_axis)
        else:
            # Mask for excited units in the Z-trace matrix
            mask = np.isin(cluster_ids, excited_units)
            mean_trace = np.nanmean(z_traces[mask], axis=0)

        return {
            'session': session_name,
            'mean_trace': mean_trace,
            'time_axis': time_axis,
            'n_units': len(excited_units)
        }
    except Exception as e:
        print(f"Error loading {session_name}: {e}")
        return None

# --- Main Analysis ---

def main():
    parser = argparse.ArgumentParser(description="Plot chronological progression of lick responses.")
    parser.add_argument('--manifest', default=None, help='Path to sessions manifest CSV (default: canonical)')
    parser.add_argument('--figures-root', default='FIGURES/lick', help='Root directory for session outputs')
    parser.add_argument('--out', required=True, help='Output directory for plots')
    parser.add_argument('--workers', type=int, default=4, help='Parallel workers')
    
    # Peak detection parameters
    parser.add_argument('--peak-window', nargs=2, type=float, default=[-0.4, 0.0], 
                        help='Time window (s) to search for peak response (default: -0.4 0.0)')
    parser.add_argument('--no-filter', action='store_true', help='Bypass SESSION_FILTER')
    
    args = parser.parse_args()

    # 1. Setup Paths and Manifest
    manifest = load_staging_manifest(manifest_path=args.manifest,
                                     apply_filter=not args.no_filter)
    subject = manifest.iloc[0]['subject']
    
    # Enforce formatting
    manifest['session_name'] = manifest['session_name'].apply(lambda x: x.zfill(8) if x.isdigit() and len(x) == 7 else x)
    manifest['folder_name'] = manifest['session_name'] # Assumes folder names match padded session names (mostly)
    
    # Parse dates for sorting
    manifest['parsed_date'] = manifest['session_name'].apply(lambda x: parse_date(f"{subject}_{x}"))
    manifest = manifest.sort_values('parsed_date').reset_index(drop=True)

    # Resolve search root
    figures_root = Path(args.figures_root)
    subject_folder = figures_root / subject
    search_root = subject_folder if subject_folder.exists() else figures_root
    print(f"Searching for session data in: {search_root}")

    # 2. Prepare Loading Tasks
    tasks = []
    session_map = {} # Map session name back to manifest index/date
    
    for idx, row in manifest.iterrows():
        s_name = row['folder_name']
        folder = get_session_folder(search_root, s_name)
        
        if folder:
            tasks.append((s_name, folder))
            session_map[s_name] = {
                'date': row['parsed_date'],
                'index': idx,
                'full_name': f"{subject}_{s_name}"
            }
        else:
            print(f"Warning: Session folder not found: {s_name}")

    # 3. Load Data
    print(f"Loading traces for {len(tasks)} sessions...")
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(load_session_trace, t): t[0] for t in tasks}
        
        for future in tqdm(as_completed(futures), total=len(tasks)):
            res = future.result()
            if res is not None:
                # Attach sorting metadata
                meta = session_map[res['session']]
                res['date'] = meta['date']
                res['sort_index'] = meta['index']
                res['label'] = meta['full_name']
                results.append(res)
    
    if not results:
        print("No valid data loaded.")
        return

    # Sort results by chronological index
    results.sort(key=lambda x: x['sort_index'])
    
    print(f"Successfully loaded {len(results)} sessions.")
    
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 4. Generate Plots
    generate_traces_plot(results, output_dir)
    generate_peak_latency_plot(results, output_dir, window=args.peak_window)


def generate_traces_plot(results, out_dir):
    """
    Plot 1: Population mean traces across sessions (sorted by date).
    """
    time_axis = results[0]['time_axis']
    n_sessions = len(results)
    
    # Setup Colormap (Cool -> Warm)
    cmap = cm.get_cmap('coolwarm', n_sessions)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate Grand Mean
    all_traces = np.array([r['mean_trace'] for r in results])
    grand_mean = np.nanmean(all_traces, axis=0)

    # Plot individual sessions
    for i, res in enumerate(results):
        color = cmap(i)
        ax.plot(time_axis, res['mean_trace'], 
                color=color, linewidth=1.2, alpha=0.8, 
                label=res['label']) # Label for legend

    # Plot Grand Mean
    ax.plot(time_axis, grand_mean, color='black', linewidth=3, label='Grand Mean')
    
    # Decoration
    ax.axvline(0, color='gray', linestyle='--')
    ax.set_title('Population Mean Traces (Excited Units) - Chronological', fontsize=14)
    ax.set_xlabel('Time from Lick (s)')
    ax.set_ylabel('Mean Z-Score')
    ax.set_xlim([-2.0, 0.8]) # Match reference image range approx
    
    # Legend
    # Put legend outside to the right
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    
    # Create a cleaner legend with just dates? 
    # Or just show colorbar? The reference image has a full legend.
    # Given 30+ sessions, legend might be tall. Let's try font scaling.
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='x-small', ncol=1)
    
    plt.savefig(out_dir / "chronological_traces.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved traces plot to {out_dir / 'chronological_traces.png'}")


def generate_peak_latency_plot(results, out_dir, window=[-0.4, 0.0]):
    """
    Plot 2: Peak times in the pre-lick window (flipped/session-vs-time).
    """
    time_axis = results[0]['time_axis']
    
    # Filter indices for the window
    win_start, win_end = window
    mask = (time_axis >= win_start) & (time_axis <= win_end)
    window_time = time_axis[mask]
    
    peak_latencies = []
    session_indices = []
    colors = []
    
    n_sessions = len(results)
    cmap = cm.get_cmap('coolwarm', n_sessions)

    for i, res in enumerate(results):
        trace_segment = res['mean_trace'][mask]
        
        # Find peak
        if len(trace_segment) > 0:
            peak_idx = np.argmax(trace_segment)
            peak_time = window_time[peak_idx]
            
            peak_latencies.append(peak_time)
            session_indices.append(i) # Use index as "y-axis" (progression)
            colors.append(cmap(i))
            
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Scatter plot: X = Peak Time, Y = Session Index
    # Edgecolors black to match the "dot" look in reference
    ax.scatter(peak_latencies, session_indices, c=colors, s=60, edgecolors='black', linewidth=1, zorder=3)
    
    # Decoration
    ax.axvline(0, color='gray', linestyle='--', linewidth=2)
    ax.set_title(f'Peak Response Latency (Window: {win_start}s to {win_end}s)', fontsize=14)
    ax.set_xlabel('Peak Time Relative to Lick (s)')
    ax.set_ylabel('Session Index (Early -> Late)')
    ax.set_xlim([win_start - 0.05, 0.05]) # Slightly padded around window and 0
    ax.grid(True, axis='y', linestyle=':', alpha=0.5)
    
    # Invert Y axis? 
    # Reference image title says "flipped". 
    # Usually "flipped" might mean Y-axis is 0 at top?
    # Or X and Y swapped?
    # Reference image has dots scattered horizontally. 
    # The reference image 2 dots are scattered in X (time) and spread in Y.
    # The blue dots (early) are at the bottom? Let's check shading.
    # If standard coolwarm, blue=low index (early).
    # In reference: Blue dots are low on Y. Red dots are high on Y. 
    # So Y axis is indeed Session Index (increasing upwards).
    
    plt.tight_layout()
    plt.savefig(out_dir / "peak_latency_progression.png", dpi=300)
    plt.close()
    print(f"Saved peak latency plot to {out_dir / 'peak_latency_progression.png'}")

if __name__ == "__main__":
    main()
