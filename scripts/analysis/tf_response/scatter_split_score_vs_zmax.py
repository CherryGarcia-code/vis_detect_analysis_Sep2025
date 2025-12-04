"""
Scatter plot: Splitting score vs z_max_fast and z_max_slow.
Usage:
    python scripts/analysis/tf_response/scatter_split_score_vs_zmax.py --csv <input.csv> [--out <output.png>]
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Scatter plot: Splitting score vs z_max_fast/z_max_slow.")
    parser.add_argument('--csv', required=True, help='Input CSV file with splitting scores')
    parser.add_argument('--out', default=None, help='Output PNG file (optional)')
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    # split1
    axes[0].scatter(df['splitting_score'], df['split1'], label='split1', alpha=0.7, color='blue')
    axes[0].set_xlabel('Splitting Score')
    axes[0].set_ylabel('split1 (z_max_fast - z_min_slow)')
    axes[0].set_title('Splitting Score vs split1')
    axes[0].legend()
    # split2
    axes[1].scatter(df['splitting_score'], df['split2'], label='split2', alpha=0.7, color='crimson')
    axes[1].set_xlabel('Splitting Score')
    axes[1].set_ylabel('split2 (z_max_slow - z_min_fast)')
    axes[1].set_title('Splitting Score vs split2')
    axes[1].legend()
    plt.tight_layout()
    if args.out:
        plt.savefig(args.out, dpi=150)
        print(f"Saved scatter plot to {args.out}")
    else:
        plt.show()

if __name__ == '__main__':
    main()
