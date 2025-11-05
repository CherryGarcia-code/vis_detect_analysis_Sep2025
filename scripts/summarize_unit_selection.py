"""Cross-session summary for unit selection outputs.

Scans table_output/unit_qc/*/unit_selection.csv and builds a summary CSV
with kept/total counts and fractions. Also produces quick overview plots.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="table_output/unit_qc")
    p.add_argument("--out-csv", default="table_output/summary/unit_selection_summary.csv")
    p.add_argument("--png-root", default="png_output/summary")
    args = p.parse_args()

    root = Path(args.root)
    rows = []
    for csv in sorted(root.rglob("unit_selection.csv")):
        sess = csv.parent.name
        try:
            df = pd.read_csv(csv)
            if "keep" not in df.columns or "cluster_id" not in df.columns:
                continue
            n_total = df.shape[0]
            n_keep = int(df["keep"].sum())
            frac = (n_keep / n_total) if n_total > 0 else 0.0
            rows.append({"session": sess, "n_total": n_total, "n_keep": n_keep, "kept_frac": frac, "csv": str(csv)})
        except Exception:
            continue

    if not rows:
        print("No unit_selection.csv files found.")
        return

    summary = pd.DataFrame(rows).sort_values("session").reset_index(drop=True)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    png_root = Path(args.png_root)
    png_root.mkdir(parents=True, exist_ok=True)

    # Bar plot of kept fraction per session
    try:
        fig, ax = plt.subplots(1, 1, figsize=(max(8, 0.35 * len(summary)), 4))
        ax.bar(range(len(summary)), summary["kept_frac"].values, color="#4C78A8")
        ax.set_xticks(range(len(summary)))
        ax.set_xticklabels(summary["session"].tolist(), rotation=90, fontsize=8)
        ax.set_ylabel("Kept fraction")
        ax.set_title("Kept unit fraction across sessions")
        fig.tight_layout()
        pth = png_root / "kept_fraction_bar.png"
        fig.savefig(pth, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {pth}")
    except Exception:
        pass

    # Histogram of kept fraction
    try:
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        ax.hist(summary["kept_frac"].values, bins=20, color="#72B7B2")
        ax.set_xlabel("Kept fraction")
        ax.set_ylabel("Sessions")
        ax.set_title("Distribution of kept unit fraction")
        fig.tight_layout()
        pth = png_root / "kept_fraction_hist.png"
        fig.savefig(pth, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {pth}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
