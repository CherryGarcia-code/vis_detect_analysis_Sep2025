"""
Pairwise line plot for top N splitters: (z_max_fast, z_max_slow) and (z_min_fast, z_min_slow).
Usage:
    python scripts/analysis/tf_response/pairwise_lineplot_splitters.py --csv <input.csv> [--top N] [--out <output.png>]
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Pairwise line plot for top N splitters.")
    parser.add_argument('--csv', required=True, help='Input CSV file with z_max/z_min for fast/slow')
    parser.add_argument('--top', type=int, default=20, help='Number of top splitters to plot')
    parser.add_argument('--out', default=None, help='Output PNG file (optional)')
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df = df.sort_values('splitting_score', ascending=False).head(args.top)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    # Left: split1 (z_max_fast, z_min_slow)
    for idx, row in df.iterrows():
        axes[0].plot([0, 1], [row['z_max_fast'], row['z_min_slow']], 'b-', alpha=0.7)
        axes[0].text(1.02, row['z_min_slow'], str(int(row['cluster_id'])), va='center', fontsize=8)
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(['z_max_fast', 'z_min_slow'])
    axes[0].set_ylabel('z-score')
    axes[0].set_title(f'split1: z_max_fast to z_min_slow (Top {args.top})')
    # Right: split2 (z_max_slow, z_min_fast)
    for idx, row in df.iterrows():
        axes[1].plot([0, 1], [row['z_max_slow'], row['z_min_fast']], 'crimson', alpha=0.7)
        axes[1].text(1.02, row['z_min_fast'], str(int(row['cluster_id'])), va='center', fontsize=8)
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(['z_max_slow', 'z_min_fast'])
    axes[1].set_title('split2: z_max_slow to z_min_fast')
    plt.tight_layout()
    if args.out:
        plt.savefig(args.out, dpi=150)
        print(f"Saved pairwise line plot to {args.out}")
    else:
        plt.show()

if __name__ == '__main__':
    main()
