"""
Scatter plot: z_max_fast vs z_max_slow for all clusters.
Usage:
    python scripts/analysis/tf_response/scatter_zmax_fast_vs_slow.py --csv <input.csv> [--out <output.png>]
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Scatter plot: z_max_fast vs z_max_slow for all clusters.")
    parser.add_argument('--csv', required=True, help='Input CSV file with z_max_fast/z_max_slow')
    parser.add_argument('--out', default=None, help='Output PNG file (optional)')
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # split1: z_max_fast vs z_min_slow
    sc0 = axes[0].scatter(df['z_max_fast'], df['z_min_slow'], alpha=0.7, c=df['splitting_score'], cmap='coolwarm', edgecolor='k')
    axes[0].set_xlabel('z_max_fast')
    axes[0].set_ylabel('z_min_slow')
    axes[0].set_title('split1: z_max_fast vs z_min_slow')
    fig.colorbar(sc0, ax=axes[0], label='Splitting Score')
    # split2: z_max_slow vs z_min_fast
    sc1 = axes[1].scatter(df['z_max_slow'], df['z_min_fast'], alpha=0.7, c=df['splitting_score'], cmap='coolwarm', edgecolor='k')
    axes[1].set_xlabel('z_max_slow')
    axes[1].set_ylabel('z_min_fast')
    axes[1].set_title('split2: z_max_slow vs z_min_fast')
    fig.colorbar(sc1, ax=axes[1], label='Splitting Score')
    plt.tight_layout()
    if args.out:
        plt.savefig(args.out, dpi=150)
        print(f"Saved scatter plot to {args.out}")
    else:
        plt.show()

if __name__ == '__main__':
    main()
