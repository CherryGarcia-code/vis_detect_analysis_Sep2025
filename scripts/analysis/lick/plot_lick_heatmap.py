
"""
Plot heatmap of z-scored PSTHs for significant lick-responsive clusters from .npz traces file.

Usage:
    python scripts/analysis/lick/plot_lick_heatmap.py --npz <lick_traces.npz> --csv <lick_responsiveness.csv> --out <output_png>

The heatmap is saved as <output_png> (should be in the session's FIGURES folder).
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

def main():

    parser = argparse.ArgumentParser(description="Plot heatmap of z-scored PSTHs for significant clusters.")
    parser.add_argument('--npz', required=True, help='Input .npz file from find_lick_responsive_neurons.py')
    parser.add_argument('--csv', required=True, help='CSV file with significance info (from find_lick_responsive_neurons.py)')
    parser.add_argument('--out', required=True, help='Output PNG file for heatmap')
    args = parser.parse_args()

    data = np.load(args.npz)
    z_traces = data['z_traces']
    cluster_ids = data['cluster_ids']
    t = data['time_axis']

    # Load significance info
    df = pd.read_csv(args.csv)
    sig_ids = set(df.loc[df['is_significant'], 'cluster_id'])
    mask = np.isin(cluster_ids, list(sig_ids))
    z_traces = z_traces[mask]
    cluster_ids = cluster_ids[mask]


    # Sort clusters by delta_mean (positive first, then negative)
    df = pd.read_csv(args.csv)
    sig_df = df[df['is_significant']].copy()
    # Map cluster_id to delta_mean
    delta_map = dict(zip(sig_df['cluster_id'], sig_df['delta_mean']))
    delta_means = np.array([delta_map.get(cid, 0) for cid in cluster_ids])
    # Sort: positive delta_mean first (descending), then negative (ascending)
    pos_idx = np.where(delta_means > 0)[0]
    neg_idx = np.where(delta_means < 0)[0]
    pos_order = pos_idx[np.argsort(-delta_means[pos_idx])]
    neg_order = neg_idx[np.argsort(delta_means[neg_idx])]
    order = np.concatenate([pos_order, neg_order])
    z_traces_sorted = z_traces[order]
    cluster_ids_sorted = cluster_ids[order]

    import seaborn as sns
    sns.set_context('poster', font_scale=2.2)
    plt.figure(figsize=(10, max(8, 0.1 * len(cluster_ids_sorted))))
    vmin, vmax = -8, 8
    label_fontsize = 32
    tick_fontsize = 26
    im = plt.imshow(
        np.clip(z_traces_sorted, vmin, vmax),
        aspect='auto',
        cmap='coolwarm',
        vmin=vmin,
        vmax=vmax,
        extent=[t[0], t[-1], 0, len(cluster_ids_sorted)]
    )
    # Add vertical dashed line at time=0 (lick event)
    plt.axvline(x=0, color='k', linestyle='--', linewidth=2, alpha=0.8)
    cbar = plt.colorbar(im)
    cbar.set_label('z-score', fontsize=label_fontsize)
    cbar.ax.tick_params(labelsize=tick_fontsize)
    plt.xlabel('Time (s)', fontsize=label_fontsize, labelpad=12)
    plt.ylabel('Cluster (sorted by delta_mean)', fontsize=label_fontsize, labelpad=12)
    plt.title('Lick-responsive cluster heatmap \n (significant only)', fontsize=label_fontsize+2, pad=18)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    plt.close()
    print(f"Saved heatmap to {args.out}")

if __name__ == '__main__':
    main()
