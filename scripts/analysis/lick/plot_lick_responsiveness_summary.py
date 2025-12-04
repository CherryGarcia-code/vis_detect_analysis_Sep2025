"""
Plot mean lick responsiveness (with error shading) for a session from a CSV table.

Usage:
    python scripts/analysis/lick/plot_lick_responsiveness_summary.py --csv <lick_responsiveness.csv> --out <output_png>

The input CSV should be the output of find_lick_responsive_neurons.py.
Output plot is saved in the session's figures directory.
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Plot mean lick responsiveness with error shading.")
    parser.add_argument('--csv', required=True, help='Input CSV file from find_lick_responsive_neurons.py')
    parser.add_argument('--out', required=True, help='Output PNG file for plot')
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if df.empty:
        print("No data to plot.")
        return
    # Only plot significant units
    sig = df['is_significant'] if 'is_significant' in df.columns else np.ones(len(df), dtype=bool)
    sig_df = df.loc[sig].copy()
    # Split into positive and negative delta_mean
    pos_df = sig_df[sig_df['delta_mean'] > 0]
    neg_df = sig_df[sig_df['delta_mean'] < 0]
    plt.figure(figsize=(8, 4))
    # Plot positive (blue) and negative (red) bars, sorted by delta_mean
    pos_sorted = pos_df.sort_values('delta_mean', ascending=False)
    neg_sorted = neg_df.sort_values('delta_mean')
    all_sorted = pd.concat([pos_sorted, neg_sorted], ignore_index=True)
    colors = ['dodgerblue'] * len(pos_sorted) + ['crimson'] * len(neg_sorted)
    plt.bar(range(len(all_sorted)), all_sorted['delta_mean'], color=colors, alpha=0.8)
    plt.axhline(0, color='k', linestyle='--', lw=1)
    plt.xlabel('Unit (lick-responsive, sorted)')
    plt.ylabel('Delta mean (post - baseline) [spikes/ms]')
    plt.title('Lick Responsiveness: Positive (blue) vs Negative (red)')
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Wrote plot to {args.out}")

if __name__ == '__main__':
    main()
