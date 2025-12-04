"""
Bar plot of top N splitters by splitting score.
Usage:
    python scripts/analysis/tf_response/barplot_top_splitters.py --csv <input.csv> [--top N] [--out <output.png>]
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Bar plot of top N splitters by splitting score.")
    parser.add_argument('--csv', required=True, help='Input CSV file with splitting scores')
    parser.add_argument('--top', type=int, default=20, help='Number of top splitters to plot')
    parser.add_argument('--out', default=None, help='Output PNG file (optional)')
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df = df.sort_values('splitting_score', ascending=False).head(args.top)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].bar(df['cluster_id'].astype(str), df['split1'], color='dodgerblue')
    axes[0].set_ylabel('z_max_fast - z_min_slow')
    axes[0].set_title(f'Top {args.top} Splitters: split1 (fast>slow)')
    axes[1].bar(df['cluster_id'].astype(str), df['split2'], color='crimson')
    axes[1].set_ylabel('z_max_slow - z_min_fast')
    axes[1].set_title('split2 (slow>fast)')
    axes[1].set_xlabel('Cluster ID')
    plt.xticks(rotation=90)
    plt.tight_layout()
    if args.out:
        plt.savefig(args.out, dpi=150)
        print(f"Saved bar plot to {args.out}")
    else:
        plt.show()

if __name__ == '__main__':
    main()
