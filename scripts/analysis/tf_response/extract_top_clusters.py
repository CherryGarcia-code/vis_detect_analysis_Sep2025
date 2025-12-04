"""
Extract top N cluster IDs from a tf_splitters.csv file for downstream analysis/plotting.

Usage:
    python scripts/analysis/tf_response/extract_top_clusters.py --csv <tf_splitters.csv> --out <cluster_ids.txt> --top N [--metric splitting_score|split1|split2]

The output file will contain one cluster_id per line.
"""
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Extract top N cluster IDs from tf_splitters.csv")
    parser.add_argument('--csv', required=True, help='Input CSV file (with splitting_score, split1, split2 columns)')
    parser.add_argument('--out', required=True, help='Output text file for cluster IDs (one per line)')
    parser.add_argument('--top', type=int, default=20, help='Number of top clusters to extract')
    parser.add_argument('--metric', default='splitting_score', choices=['splitting_score', 'split1', 'split2'], help='Metric to rank clusters')
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if args.metric not in df.columns:
        raise ValueError(f"Metric '{args.metric}' not found in CSV columns: {df.columns.tolist()}")
    top_ids = df.sort_values(args.metric, ascending=False).head(args.top)['cluster_id'].astype(int)
    with open(args.out, 'w') as f:
        for cid in top_ids:
            f.write(f"{cid}\n")
    print(f"Wrote top {args.top} cluster IDs to {args.out} (metric: {args.metric})")

if __name__ == '__main__':
    main()
