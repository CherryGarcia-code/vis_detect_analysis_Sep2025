"""
Compare lick responsiveness across sessions (learning analysis).

Generates:
1. Recruitment plot: % of responsive units over sessions.
2. Potentiation plot (Split): Strength of response (delta_mean) for Excited vs Inhibited units.
3. Session Heatmap (Split): Evolution of the population mean lick response for Excited vs Inhibited units.

Usage:
    python scripts/analysis/lick/compare_sessions_lick.py --manifest data/BG_046_sessions_manifest.csv --out FIGURES/lick/comparison_BG_046
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


from visdetect.analysis.config import load_staging_manifest

def parse_date(session_name):
    # Assumes format BG_046_DDMMYYYY
    try:
        date_str = session_name.split('_')[-1]
        return datetime.strptime(date_str, "%d%m%Y")
    except Exception:
        return datetime.min

def load_session_data(args):
    """
    Worker function to load data for a single session.
    """
    session_name, session_dir = args
    # session_dir is now the full path to the session folder
    
    csv_path = session_dir / "lick_responsiveness.csv"
    npz_path = session_dir / "lick_responsiveness.npz"

    if not csv_path.exists() or not npz_path.exists():
        return None

    try:
        # Load CSV
        df = pd.read_csv(csv_path)
        
        # Load NPZ
        data = np.load(npz_path)
        z_traces = data['z_traces']
        cluster_ids = data['cluster_ids']
        time_axis = data['time_axis']
        
        # Filter for significant units
        sig_df = df[df['is_significant']]
        
        # Split IDs
        excited_ids = sig_df[sig_df['delta_mean'] > 0]['cluster_id'].values
        inhibited_ids = sig_df[sig_df['delta_mean'] < 0]['cluster_id'].values
        
        # Create masks
        excited_mask = np.isin(cluster_ids, excited_ids)
        inhibited_mask = np.isin(cluster_ids, inhibited_ids)
        
        # Compute mean traces
        if np.any(excited_mask):
            mean_z_excited = np.nanmean(z_traces[excited_mask], axis=0)
        else:
            mean_z_excited = np.zeros_like(time_axis)
            
        if np.any(inhibited_mask):
            mean_z_inhibited = np.nanmean(z_traces[inhibited_mask], axis=0)
        else:
            mean_z_inhibited = np.zeros_like(time_axis)

        return {
            'session': session_name,
            'df': df,
            'mean_z_excited': mean_z_excited,
            'mean_z_inhibited': mean_z_inhibited,
            'time_axis': time_axis
        }
    except Exception as e:
        print(f"Error loading {session_name}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Compare lick responsiveness across sessions.")
    parser.add_argument('--manifest', default=None, help='Path to sessions manifest CSV (default: canonical)')
    parser.add_argument('--figures-root', default='FIGURES/lick', help='Root directory where session subfolders are located')
    parser.add_argument('--out', required=True, help='Output directory for comparison plots')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers for file loading')
    parser.add_argument('--no-filter', action='store_true', help='Bypass SESSION_FILTER')
    args = parser.parse_args()

    # 1. Load Manifest and Sort
    manifest = load_staging_manifest(manifest_path=args.manifest,
                                     apply_filter=not args.no_filter)
    
    subject = manifest.iloc[0]['subject']
    # Ensure session_name is string and padded
    manifest['session_name'] = manifest['session_name'].apply(lambda x: x.zfill(8) if x.isdigit() and len(x) == 7 else x)
    
    # Construct full session name if needed, but usually the folder name is just the session name (e.g. 25062025)
    # The batch script outputs to FIGURES/lick/BG_046/25062025
    # So we should look for folders named '25062025' inside FIGURES/lick/BG_046
    # But args.figures_root is 'FIGURES/lick'.
    # So we need to know the subject folder too?
    # Or maybe args.figures_root should be 'FIGURES/lick/BG_046'?
    # The user passed --out FIGURES/lick/comparison_BG_046, but didn't specify figures root.
    # Default is FIGURES/lick.
    
    # Let's try to guess the subject folder from the manifest subject
    figures_root = Path(args.figures_root)
    subject_folder = figures_root / subject
    
    if subject_folder.exists():
        print(f"Looking for session folders in {subject_folder}")
        search_root = subject_folder
    else:
        print(f"Subject folder {subject_folder} not found. Looking in {figures_root}")
        search_root = figures_root

    # manifest['full_session_name'] = manifest['session_name'].apply(lambda x: f"{subject}_{x}")
    # The folder name is likely just the session_name (e.g. 25062025) based on batch_run_lick_analysis.py
    
    manifest['folder_name'] = manifest['session_name']
    manifest['parsed_date'] = manifest['session_name'].apply(lambda x: parse_date(f"{subject}_{x}"))
    
    manifest = manifest.sort_values('parsed_date').reset_index(drop=True)
    
    print(f"Found {len(manifest)} sessions in manifest.")

    # 2. Load Data in Parallel
    results = []
    
    # Helper to find correct folder
    def get_session_folder(root, s_name):
        # Try exact match (padded)
        p = root / s_name
        if p.exists(): return p
        
        # Try unpadded (if s_name starts with 0)
        if s_name.startswith('0'):
            p_unpadded = root / s_name.lstrip('0')
            if p_unpadded.exists(): return p_unpadded
            
        return None

    # Check for missing folders before loading
    for _, row in manifest.iterrows():
        s_name = row['folder_name']
        folder = get_session_folder(search_root, s_name)
        if not folder:
            print(f"Warning: Folder for session {s_name} not found in {search_root}")

    print("Loading session data...")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # We need to pass the actual folder name found on disk
        # But we want to report the session name from manifest (padded) for consistency.
        
        # Let's change tasks to be: (manifest_session_name, actual_folder_path)
        real_tasks = []
        for _, row in manifest.iterrows():
            s_name = row['folder_name']
            folder = get_session_folder(search_root, s_name)
            if folder:
                real_tasks.append((s_name, folder))
        
        futures = {executor.submit(load_session_data, task): task[0] for task in real_tasks}
        
        for future in tqdm(as_completed(futures), total=len(real_tasks)):
            res = future.result()
            if res is not None:
                results.append(res)

    if not results:
        print("No data loaded. Check paths.")
        return

    # Re-sort results based on manifest order
    session_order = {name: i for i, name in enumerate(manifest['folder_name'])}
    results.sort(key=lambda x: session_order.get(x['session'], 999))
    
    # Filter out sessions that weren't found
    results = [r for r in results if r['session'] in session_order]
    
    print(f"Successfully loaded {len(results)} sessions.")

    # 3. Aggregate Data
    stats_rows = []
    heatmap_rows_excited = []
    heatmap_rows_inhibited = []
    all_sig_units = []

    time_axis = results[0]['time_axis'] # Assume same time axis for all

    for res in results:
        session = res['session']
        df = res['df']
        
        n_total = len(df)
        if n_total == 0:
            continue
            
        n_sig = df['is_significant'].sum()
        pct_sig = (n_sig / n_total) * 100
        
        # Split into Pos/Neg
        sig_df = df[df['is_significant']]
        n_pos = (sig_df['delta_mean'] > 0).sum()
        n_neg = (sig_df['delta_mean'] < 0).sum()
        
        pct_pos = (n_pos / n_total) * 100
        pct_neg = (n_neg / n_total) * 100
        
        stats_rows.append({
            'session': session,
            'date': parse_date(session),
            'pct_responsive': pct_sig,
            'pct_excited': pct_pos,
            'pct_inhibited': pct_neg
        })
        
        heatmap_rows_excited.append(res['mean_z_excited'])
        heatmap_rows_inhibited.append(res['mean_z_inhibited'])
        
        # Collect individual unit data for potentiation plot
        for _, row in sig_df.iterrows():
            all_sig_units.append({
                'session': session,
                'date': parse_date(session),
                'delta_mean': row['delta_mean'],
                'direction': 'Excited' if row['delta_mean'] > 0 else 'Inhibited'
            })

    stats_df = pd.DataFrame(stats_rows)
    units_df = pd.DataFrame(all_sig_units)
    
    # Create output directory
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 4. Plotting
    sns.set_theme(style="whitegrid")

    # --- Plot 1: Recruitment (Fraction Responsive) ---
    plt.figure(figsize=(12, 6))
    melted_stats = stats_df.melt(id_vars=['session', 'date'], 
                                 value_vars=['pct_responsive', 'pct_excited', 'pct_inhibited'],
                                 var_name='Type', value_name='Percentage')
    name_map = {'pct_responsive': 'Total', 'pct_excited': 'Excited', 'pct_inhibited': 'Inhibited'}
    melted_stats['Type'] = melted_stats['Type'].map(name_map)
    
    sns.lineplot(data=melted_stats, x='session', y='Percentage', hue='Type', marker='o', linewidth=2)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.title('Recruitment: Fraction of Lick-Responsive Units Across Sessions')
    plt.ylabel('% of Units')
    plt.xlabel('Session')
    plt.tight_layout()
    plt.savefig(out_dir / "recruitment_over_sessions.png", dpi=150)
    plt.close()

    # --- Plot 2: Potentiation Split (Excited vs Inhibited) ---
    if not units_df.empty:
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # Excited
        sns.boxplot(data=units_df[units_df['direction'] == 'Excited'], x='session', y='delta_mean', ax=axes[0], color='dodgerblue', showfliers=False)
        axes[0].set_title('Excited Units: Response Strength')
        axes[0].set_ylabel('Delta Mean (spikes/ms)')
        axes[0].set_xlabel('')
        
        # Inhibited
        sns.boxplot(data=units_df[units_df['direction'] == 'Inhibited'], x='session', y='delta_mean', ax=axes[1], color='crimson', showfliers=False)
        axes[1].set_title('Inhibited Units: Response Strength')
        axes[1].set_ylabel('Delta Mean (spikes/ms)')
        axes[1].set_xlabel('Session')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(out_dir / "potentiation_split.png", dpi=150)
        plt.close()

    # --- Plot 3: Session Heatmap Split (Excited vs Inhibited) ---
    if len(heatmap_rows_excited) > 0:
        matrix_exc = np.array(heatmap_rows_excited)
        matrix_inh = np.array(heatmap_rows_inhibited)
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
        
        # Common extent
        extent = [time_axis[0], time_axis[-1], len(matrix_exc), 0]
        
        # Excited Heatmap
        v_exc = np.nanpercentile(matrix_exc, 99)
        im1 = axes[0].imshow(matrix_exc, aspect='auto', cmap='RdBu_r', vmin=-v_exc, vmax=v_exc, extent=extent, interpolation='nearest')
        axes[0].set_title('Excited Population Response (Mean Z-Score)')
        axes[0].set_ylabel('Session')
        plt.colorbar(im1, ax=axes[0], label='Z-score')
        
        # Inhibited Heatmap
        v_inh = np.nanpercentile(np.abs(matrix_inh), 99)
        im2 = axes[1].imshow(matrix_inh, aspect='auto', cmap='RdBu_r', vmin=-v_inh, vmax=v_inh, extent=extent, interpolation='nearest')
        axes[1].set_title('Inhibited Population Response (Mean Z-Score)')
        axes[1].set_ylabel('Session')
        axes[1].set_xlabel('Time from Lick (s)')
        plt.colorbar(im2, ax=axes[1], label='Z-score')
        
        # Ticks
        yticks = np.arange(0.5, len(matrix_exc), max(1, len(matrix_exc)//10))
        yticklabels = [stats_df.iloc[int(y)]['session'] for y in yticks]
        axes[0].set_yticks(yticks)
        axes[0].set_yticklabels(yticklabels, fontsize=8)
        axes[1].set_yticks(yticks)
        axes[1].set_yticklabels(yticklabels, fontsize=8)
        
        plt.tight_layout()
        plt.savefig(out_dir / "session_heatmap_split.png", dpi=150)
        plt.close()

    print(f"Analysis complete. Plots saved to {out_dir}")

if __name__ == "__main__":
    main()
