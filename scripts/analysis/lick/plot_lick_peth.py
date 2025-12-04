
"""
Plot peristimulus time histogram (PETH) for significant lick-responsive clusters from .npz traces file.

Usage:
    python scripts/analysis/lick/plot_lick_peth.py --npz <lick_traces.npz> --csv <lick_responsiveness.csv> --outdir <FIGURES/session_folder> [--cluster-ids 1 2 3 ...]

If --cluster-ids is omitted, only significant clusters are plotted by default.
Each plot is saved as <outdir>/lick_peth_cluster_<id>.png
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

def main():

    parser = argparse.ArgumentParser(description="Plot PETHs for significant lick-responsive clusters.")
    parser.add_argument('--npz', required=True, help='Input .npz file from find_lick_responsive_neurons.py')
    parser.add_argument('--csv', required=True, help='CSV file with significance info (from find_lick_responsive_neurons.py)')
    parser.add_argument('--outdir', required=True, help='Output directory for plots')
    parser.add_argument('--cluster-ids', type=int, nargs='*', help='Cluster IDs to plot (default: significant only)')
    args = parser.parse_args()

    data = np.load(args.npz)
    cluster_ids = data['cluster_ids']
    z_traces = data['z_traces']
    sem_traces = data['sem_traces']
    t = data['time_axis']

    # Load significance info
    df = pd.read_csv(args.csv)
    sig_ids = set(df.loc[df['is_significant'], 'cluster_id'])

    # Determine which clusters to plot
    if args.cluster_ids:
        plot_ids = set(args.cluster_ids)
    else:
        plot_ids = sig_ids
    mask = np.isin(cluster_ids, list(plot_ids))
    cluster_ids = cluster_ids[mask]
    z_traces = z_traces[mask]
    sem_traces = sem_traces[mask]

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    for i, cid in enumerate(cluster_ids):
        plt.figure(figsize=(6, 3))
        plt.plot(t, z_traces[i], label=f'Cluster {cid}', color='dodgerblue')
        plt.fill_between(t, z_traces[i] - sem_traces[i], z_traces[i] + sem_traces[i], color='dodgerblue', alpha=0.3)
        plt.axvline(0, color='k', linestyle='--', lw=1)
        plt.xlabel('Time (s)')
        plt.ylabel('Z-scored firing rate')
        plt.title(f'Lick PETH: Cluster {cid}')
        plt.tight_layout()
        outpath = Path(args.outdir) / f'lick_peth_cluster_{cid}.png'
        plt.savefig(outpath, dpi=150)
        plt.close()
        print(f"Saved {outpath}")

if __name__ == '__main__':
    main()
