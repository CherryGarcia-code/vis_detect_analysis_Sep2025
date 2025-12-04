"""
Plot mean PSTH (with error shading) for positive and negative significant lick-responsive neurons.

Usage:
    python scripts/analysis/lick/plot_lick_mean_psth_posneg.py --npz <lick_responsiveness.npz> --csv <lick_responsiveness.csv> --out <output_png>

The plot overlays the mean PSTH of positive (blue) and negative (red) responsive units, with SEM shading.
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Plot mean PSTH for positive and negative significant lick-responsive neurons.")
    parser.add_argument('--npz', required=True, help='Input .npz file from find_lick_responsive_neurons.py')
    parser.add_argument('--csv', required=True, help='CSV file with significance info (from find_lick_responsive_neurons.py)')
    parser.add_argument('--out', required=True, help='Output PNG file for plot')
    args = parser.parse_args()

    data = np.load(args.npz)
    cluster_ids = data['cluster_ids']
    z_traces = data['z_traces']
    t = data['time_axis']

    df = pd.read_csv(args.csv)
    sig_df = df[df['is_significant']].copy()
    pos_ids = set(sig_df[sig_df['delta_mean'] > 0]['cluster_id'])
    neg_ids = set(sig_df[sig_df['delta_mean'] < 0]['cluster_id'])

    pos_mask = np.isin(cluster_ids, list(pos_ids))
    neg_mask = np.isin(cluster_ids, list(neg_ids))

    pos_traces = z_traces[pos_mask]
    neg_traces = z_traces[neg_mask]

    plt.figure(figsize=(7, 4))
    if len(pos_traces) > 0:
        pos_mean = np.nanmean(pos_traces, axis=0)
        pos_sem = np.nanstd(pos_traces, axis=0, ddof=1) / np.sqrt(pos_traces.shape[0])
        plt.plot(t, pos_mean, color='dodgerblue', label='Positive Δ (n={})'.format(len(pos_traces)))
        plt.fill_between(t, pos_mean - pos_sem, pos_mean + pos_sem, color='dodgerblue', alpha=0.3)
    if len(neg_traces) > 0:
        neg_mean = np.nanmean(neg_traces, axis=0)
        neg_sem = np.nanstd(neg_traces, axis=0, ddof=1) / np.sqrt(neg_traces.shape[0])
        plt.plot(t, neg_mean, color='crimson', label='Negative Δ (n={})'.format(len(neg_traces)))
        plt.fill_between(t, neg_mean - neg_sem, neg_mean + neg_sem, color='crimson', alpha=0.3)
    plt.axvline(0, color='k', linestyle='--', lw=1)
    plt.xlabel('Time (s)')
    plt.ylabel('Z-scored firing rate')
    plt.title('Lick-excited vs Lick-inhibited Neuron Mean PSTH\n(FA trials only, significant units, blue=excited, red=inhibited)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Wrote plot to {args.out}")

if __name__ == '__main__':
    main()
