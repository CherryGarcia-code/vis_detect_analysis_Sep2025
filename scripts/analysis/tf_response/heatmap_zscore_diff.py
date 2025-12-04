"""
Heatmap of (z_max_fast - z_max_slow) for top N splitters.
Usage:
    python scripts/analysis/tf_response/heatmap_zscore_diff.py --csv <input.csv> [--top N] [--out <output.png>]
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Heatmap of (z_max_fast - z_max_slow) for top N splitters.")
    parser.add_argument('--csv', required=True, help='Input CSV file with z_max_fast/z_max_slow')
    parser.add_argument('--top', type=int, default=20, help='Number of top splitters to plot')
    parser.add_argument('--out', default=None, help='Output PNG file (optional)')
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df = df.sort_values('splitting_score', ascending=False).head(args.top)
    split1 = df['split1']
    split2 = df['split2']
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    vlim = max(np.max(np.abs(split1)), np.max(np.abs(split2)))
    im0 = axes[0].imshow([split1], aspect='auto', cmap='bwr', vmin=-vlim, vmax=vlim)
    axes[0].set_yticks([])
    axes[0].set_title(f'split1: z_max_fast - z_min_slow (Top {args.top})')
    fig.colorbar(im0, ax=axes[0], orientation='vertical', label='split1')
    im1 = axes[1].imshow([split2], aspect='auto', cmap='bwr', vmin=-vlim, vmax=vlim)
    axes[1].set_yticks([])
    axes[1].set_title('split2: z_max_slow - z_min_fast')
    fig.colorbar(im1, ax=axes[1], orientation='vertical', label='split2')
    plt.xticks(range(len(df)), df['cluster_id'].astype(str), rotation=90)
    plt.tight_layout()
    if args.out:
        plt.savefig(args.out, dpi=150)
        print(f"Saved heatmap to {args.out}")
    else:
        plt.show()

if __name__ == '__main__':
    main()
