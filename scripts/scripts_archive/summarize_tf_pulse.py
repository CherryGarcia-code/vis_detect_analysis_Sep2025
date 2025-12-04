"""Summarize TF pulse responsiveness across sessions.

Reads per-session CSVs written by run_tf_pulse_screening and aggregates counts
and proportions of fast- and slow-responsive units per session, plus totals.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(argv=None):
    ap = argparse.ArgumentParser(description="Summarize TF pulse responsiveness")
    ap.add_argument("--tf-root", default="table_output/tf_pulse", help="Root folder containing per-session CSVs")
    ap.add_argument("--out-root", default="table_output/tf_pulse_summary", help="Output folder for summary CSVs")
    ap.add_argument("--png-root", default="png_output/tf_pulse_summary", help="Output folder for PNG plots")
    args = ap.parse_args(argv)

    tf_root = Path(args.tf_root)
    out_root = Path(args.out_root)
    png_root = Path(args.png_root)
    out_root.mkdir(parents=True, exist_ok=True)
    png_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for sub in sorted(tf_root.glob("*")):
        if not sub.is_dir():
            continue
        csv = sub / "tf_pulse_units.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        n = len(df)
        n_fast = int(df["fast_responsive"].sum()) if "fast_responsive" in df.columns else 0
        n_slow = int(df["slow_responsive"].sum()) if "slow_responsive" in df.columns else 0
        rows.append({
            "session": sub.name,
            "n_units": n,
            "n_fast": n_fast,
            "n_slow": n_slow,
            "pct_fast": (100.0 * n_fast / n) if n > 0 else np.nan,
            "pct_slow": (100.0 * n_slow / n) if n > 0 else np.nan,
        })

    summary = pd.DataFrame(rows).sort_values("session").reset_index(drop=True)
    out_csv = out_root / "tf_pulse_summary.csv"
    summary.to_csv(out_csv, index=False)

    # Simple bar plot
    if len(summary) > 0:
        x = np.arange(len(summary))
        w = 0.35
        fig, ax = plt.subplots(1, 1, figsize=(max(6.0, 0.25 * len(summary) + 4), 4))
        ax.bar(x - w/2, summary["pct_fast"].values, width=w, label="fast %", color="#1f77b4")
        ax.bar(x + w/2, summary["pct_slow"].values, width=w, label="slow %", color="#d62728")
        ax.set_xticks(x)
        ax.set_xticklabels(summary["session"].values, rotation=45, ha="right")
        ax.set_ylabel("Responsive units (% of kept)")
        ax.set_title("TF pulse responsiveness across sessions")
        ax.legend(fontsize="small")
        fig.tight_layout()
        fig.savefig(png_root / "tf_pulse_responsive_percent.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"Wrote summary CSV: {out_csv}")


if __name__ == "__main__":
    raise SystemExit(main())
