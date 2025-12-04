"""
Identify clusters with divergent ("splitting") responses to fast vs slow TF pulses.

Usage:
    python scripts/analysis/tf_response/find_splitters_from_tf_grid_csv.py --csv <tf_pulse_grid_both.csv> [--out <output.csv>] [--top N]

This script reads a CSV file with per-cluster z-score summary statistics for fast and slow pulses (as output by plot_tf_pulse_grid.py),
computes a splitting score for each cluster, and outputs a ranked list of clusters with the most divergent responses.

Splitting score = |z_max_fast - z_max_slow| + |z_min_fast - z_min_slow|

"""
import argparse
import pandas as pd


def compute_splitting_score(row):
    # Absolute difference in max and min z-score between fast and slow
    return abs(row['z_max_fast'] - row['z_max_slow']) + abs(row['z_min_fast'] - row['z_min_slow'])


def main():
    parser = argparse.ArgumentParser(description="Identify clusters with divergent fast/slow pulse responses from tf_pulse_grid_both.csv")
    parser.add_argument('--csv', required=True, help='Input CSV file (from plot_tf_pulse_grid.py --which both)')
    parser.add_argument('--out', default=None, help='Optional output CSV file for ranked splitters')
    parser.add_argument('--top', type=int, default=20, help='Number of top splitters to print (default: 20)')
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df['splitting_score'] = df.apply(compute_splitting_score, axis=1)
    # Add split1 and split2 columns for plotting compatibility
    df['split1'] = df['z_max_fast'] - df['z_min_slow']
    df['split2'] = df['z_max_slow'] - df['z_min_fast']
    df_sorted = df.sort_values('splitting_score', ascending=False).reset_index(drop=True)

    print(f"Top {args.top} clusters with most divergent fast/slow responses:")
    print(df_sorted[['cluster_id', 'splitting_score', 'z_max_fast', 'z_max_slow', 'z_min_fast', 'z_min_slow']].head(args.top).to_string(index=False))

    if args.out:
        df_sorted.to_csv(args.out, index=False)
        print(f"Wrote ranked splitters to {args.out}")

if __name__ == '__main__':
    main()
