from __future__ import annotations
from pathlib import Path
import pandas as pd
import sys

def main():
    root = Path("table_output/unit_qc").resolve()
    if not root.exists():
        print("No unit_qc root found.")
        return 1
    csvs = sorted(root.glob("*/unit_selection.csv"))
    if not csvs:
        print("No unit_selection.csv files found under table_output/unit_qc.")
        return 0
    print("session\tn_total\tn_kept\tkept_ids_sample")
    for csv in csvs:
        sess = csv.parent.name
        try:
            df = pd.read_csv(csv)
        except Exception as e:
            print(f"{sess}\tERROR reading CSV: {e}")
            continue
        if "keep" not in df.columns or "cluster_id" not in df.columns:
            print(f"{sess}\tmissing columns")
            continue
        kept_df = df.loc[df["keep"].astype(bool)]
        kept_ids = kept_df["cluster_id"].astype(int).tolist()
        sample = ",".join(map(str, kept_ids[:6]))
        print(f"{sess}\t{df.shape[0]}\t{len(kept_ids)}\t{sample}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
