"""Convert DANT IdxCluster.npy + unit_lookup.csv into a long registry CSV."""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import registry  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dant-output", default="FIGURES/tracking_dant/BG_046/dant_output")
    ap.add_argument("--input-dir", default="data/cache/dant/BG_046/input")
    ap.add_argument("--out", default="data/cache/dant/BG_046/dant_registry.csv")
    args = ap.parse_args()

    idx = np.load(os.path.join(args.dant_output, "IdxCluster.npy"))
    lookup = pd.read_csv(os.path.join(args.input_dir, "unit_lookup.csv"), dtype={"session": str})
    lookup["session"] = lookup["session"].str.zfill(8)
    reg = registry.idxcluster_to_registry(idx, lookup)
    reg.to_csv(args.out, index=False)
    n_tracked = (reg["dant_uid"] > 0).sum()
    n_clusters = int(reg.loc[reg["dant_uid"] > 0, "dant_uid"].nunique())
    print(f"wrote {args.out}: {len(reg)} units, {n_clusters} clusters, {n_tracked} tracked units")


if __name__ == "__main__":
    main()
